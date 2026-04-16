#!/usr/bin/env python3
"""
Train a one-stage joint baseline:
- train on source-train (80%) + target-train (from split file)
- validate on source-val (20%) + target-val (from split file)
- test on target-test (from split file)
"""

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# Ensure repository's `scripts` directory is on the Python path
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from gnn.gnn_io import GraphDataset, collate_fn
from gnn.help_functions import GNN_Loss, validate_model_during_training
from training.help_functions import (
    EarlyStopping,
    EdgeFeatures,
    create_gnn_model,
    create_split_dataloader,
    get_available_gpus,
    get_paths,
    load_metadata_from_disk,
    normalize_dataset,
    normalize_dataset_with_scaler,
    select_best_gpu,
    set_cuda_visible_device,
    set_random_seeds,
    setup_wandb,
    str_to_bool,
)


def _resolve_path(p: str, project_root: Path, city_dir: Path) -> str:
    p = str(p)
    if os.path.isabs(p) and os.path.exists(p):
        return p
    # Common LRZ path prefix used in some splits/logs
    if p.startswith("/mnt/repo/"):
        candidate = str(project_root / p.replace("/mnt/repo/", ""))
        if os.path.exists(candidate):
            return candidate
    # Common absolute paths from other machines
    if "/gnn_data/" in p:
        rel = p.split("/gnn_data/", 1)[1]
        candidate = str(project_root / rel)
        if os.path.exists(candidate):
            return candidate
    candidate = str(project_root / p)
    if os.path.exists(candidate):
        return candidate
    return str(city_dir / Path(p).name)


def _load_target_split(split_file: Path, project_root: Path) -> Tuple[Dict, Dict, Dict]:
    with open(split_file, "r") as f:
        split_data = json.load(f)

    train_data = split_data.get("train_data") or {}
    val_data = split_data.get("val_data") or {}
    test_data = split_data.get("test_data") or {}

    required = ["path", "policy_region", "scenario", "city"]
    for name, d in (("train_data", train_data), ("val_data", val_data), ("test_data", test_data)):
        for k in required:
            if k not in d:
                raise ValueError(f"Split file missing '{name}.{k}'")

    split_parent = split_file.parent
    for d in (train_data, val_data, test_data):
        city_dir = (
            project_root
            / "data"
            / "bavaria"
            / "inductive_data"
            / "training_data"
            / "kreisfreistadt"
            / (str(d["city"][0]) if d.get("city") else "")
        )
        resolved = []
        for p in d["path"]:
            cand = _resolve_path(p, project_root=project_root, city_dir=split_parent)
            if not os.path.exists(cand):
                cand2 = str(city_dir / Path(p).name)
                cand = cand2
            resolved.append(cand)
        d["path"] = resolved

    return train_data, val_data, test_data


def _build_feature_filter(use_all_features: bool, use_destination_activity: bool):
    if use_all_features:
        node_features = []
        for feat in EdgeFeatures:
            name = feat.name
            if not use_destination_activity and feat.value >= 20:
                continue
            if name.startswith("ALLOWED_MODE"):
                continue
            if name.startswith("HIGHWAY"):
                continue
            node_features.append(name)
    else:
        node_features = [
            "VOL_BASE_CASE",
            "CAPACITY_BASE_CASE",
            "CAPACITY_REDUCTION",
            "FREESPEED",
            "LENGTH",
        ]

    node_feature_filter = [EdgeFeatures[n].value for n in node_features]
    filtered_feature_mapping = {EdgeFeatures[n].value: i for i, n in enumerate(node_features)}
    return node_feature_filter, filtered_feature_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one-stage joint source+target baseline.")
    parser.add_argument("--project_name", type=str, default="Benchmark_TL")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--gnn_arch", type=str, default="trans_encoder", choices=["gatv2", "trans_conv", "graphSAGE", "trans_encoder", "crossST", "citytrans"])
    parser.add_argument("--target_city", type=str, required=True)
    parser.add_argument("--source_cities", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)

    parser.add_argument("--use_all_features", type=str_to_bool, default=False)
    parser.add_argument("--use_destination_activity", type=str_to_bool, default=False)
    parser.add_argument("--fit_scaler_on", type=str, default="train", choices=["train", "train_val_test"])
    parser.add_argument(
        "--balance_domains",
        type=str_to_bool,
        default=False,
        help="If True, oversample target-train samples to counter source-target imbalance.",
    )
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--loss_fct", type=str, default="mse", choices=["mse", "l1"])
    parser.add_argument("--use_weighted_loss", type=str_to_bool, default=False)
    parser.add_argument("--target_type", type=str, default="abs_vol_car")
    parser.add_argument("--target_normalization", type=str, default="None", choices=["None", "relative_to_max_traffic_vol_base_case", "relative_standard_scaler"])
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--peak_lr", type=float, default=3e-4)
    parser.add_argument("--initial_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_fraction", type=float, default=0.1)
    parser.add_argument("--cosine_decay_rate", type=float, default=0.5)
    parser.add_argument("--min_lr_fraction", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--use_dropout", type=str_to_bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_gradient_clipping", type=str_to_bool, default=True)
    parser.add_argument("--continue_training", type=str_to_bool, default=False)
    parser.add_argument("--base_checkpoint_path", type=str, default=None)
    args = vars(parser.parse_args())

    project_root = Path(__file__).resolve().parents[2]
    dataset_root = project_root / "data" / "bavaria" / "inductive_data" / "training_data" / "kreisfreistadt"
    base_dir = project_root / "inductive_gnn_data_results" / "transductive"

    split_file = Path(args["split_file"])
    if not split_file.is_absolute():
        split_file = (project_root / split_file).resolve()
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    target_city = args["target_city"].strip()
    source_cities = [c.strip() for c in args["source_cities"].split(",") if c.strip()]
    if target_city in source_cities:
        raise ValueError("target_city must not be included in source_cities.")

    set_random_seeds()
    gpus = get_available_gpus()
    best_gpu = select_best_gpu(gpus)
    set_cuda_visible_device(best_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_meta = {"path": [], "policy_region": [], "scenario": [], "city": []}
    for city in sorted(source_cities):
        load_metadata_from_disk(source_meta, str(dataset_root / city / "metadata.json"))

    target_train, target_val, target_test = _load_target_split(split_file, project_root=project_root)

    combined_paths: List[str] = []
    combined_labels: List[str] = []

    def _append_block(block: Dict) -> List[int]:
        start = len(combined_paths)
        combined_paths.extend(list(block["path"]))
        combined_labels.extend([f"{c}_{pr}" for c, pr in zip(block["city"], block["policy_region"])])
        return list(range(start, len(combined_paths)))

    src_indices = _append_block(source_meta)
    tgt_train_indices = _append_block(target_train)
    tgt_val_indices = _append_block(target_val)
    tgt_test_indices = _append_block(target_test)

    dataset = GraphDataset(combined_paths, combined_labels)

    # One-stage protocol:
    # - source scenarios split 80/20 into train/val
    # - target scenarios from split file: train40/val10
    # - test target-only
    src_labels = [combined_labels[i] for i in src_indices]
    stratify = src_labels if len(set(src_labels)) > 1 else None
    src_train_indices, src_val_indices = train_test_split(
        list(src_indices), test_size=0.2, random_state=42, stratify=stratify
    )

    train_indices = list(src_train_indices) + list(tgt_train_indices)
    val_indices = list(src_val_indices) + list(tgt_val_indices)
    test_indices = list(tgt_test_indices)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    node_feature_filter, filtered_feature_mapping = _build_feature_filter(
        use_all_features=bool(args["use_all_features"]),
        use_destination_activity=bool(args["use_destination_activity"]),
    )
    collate_no_aug = partial(
        collate_fn,
        node_feature_filter=node_feature_filter,
        filtered_feature_mapping=filtered_feature_mapping,
        is_training=False,
    )
    collate_train = partial(
        collate_fn,
        node_feature_filter=node_feature_filter,
        filtered_feature_mapping=filtered_feature_mapping,
        is_training=True,
        augment_pos_rotation=False,
        augment_feature_noise=False,
        augment_node_masking_prob=0.0,
    )

    if args["fit_scaler_on"] == "train_val_test":
        combined_norm_set = Subset(dataset, train_indices + val_indices + test_indices)
    else:
        combined_norm_set = Subset(dataset, train_indices)
    train_set_norm, scalers = normalize_dataset(train_data_list=train_set, combined_data_list=combined_norm_set)
    val_set_norm = normalize_dataset_with_scaler(dataset_input=val_set, scalers=scalers)
    test_set_norm = normalize_dataset_with_scaler(dataset_input=test_set, scalers=scalers)

    if bool(args["balance_domains"]):
        num_source = len(src_train_indices)
        num_target = len(tgt_train_indices)
        if num_target <= 0:
            raise ValueError("Target train split is empty; cannot balance domains.")
        target_w = float(num_source) / float(num_target)
        weights = [1.0] * len(train_set_norm)
        for j in range(num_source, num_source + num_target):
            weights[j] = target_w
        sampler = WeightedRandomSampler(weights, num_samples=len(train_set_norm), replacement=True)
        train_dl = DataLoader(
            dataset=train_set_norm,
            batch_size=int(args["batch_size"]),
            sampler=sampler,
            shuffle=None,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=False,
            collate_fn=collate_train,
            worker_init_fn=None,
            drop_last=False,
        )
    else:
        train_dl = create_split_dataloader(train_set_norm, args["batch_size"], use_weighted_batches=False, collate_fn_type=collate_train)
    val_dl = create_split_dataloader(val_set_norm, args["batch_size"], use_weighted_batches=False, collate_fn_type=collate_no_aug)
    test_dl = create_split_dataloader(test_set_norm, args["batch_size"], use_weighted_batches=False, collate_fn_type=collate_no_aug)

    model_save_path, _ = get_paths(
        base_dir=str(base_dir / args["project_name"]),
        unique_model_description=args["run_name"],
        model_save_path="trained_model/model.pth",
    )

    # Keep WandB tracking close to common run_models-style runs:
    # same core training/model hyperparameters, without one-stage orchestration fields.
    args_for_wandb = dict(args)
    for k in ("target_city", "source_cities", "split_file", "fit_scaler_on", "balance_domains"):
        args_for_wandb.pop(k, None)
    args_for_wandb["unique_model_description"] = args["run_name"]
    args_for_wandb["target_normalization"] = None if args_for_wandb.get("target_normalization") == "None" else args_for_wandb.get("target_normalization")
    config = setup_wandb(args_for_wandb)

    sample_batch = next(iter(train_dl))
    actual_in_channels = int(sample_batch.x.shape[1])
    try:
        config.update({"in_channels": actual_in_channels}, allow_val_change=True)
    except Exception:
        pass

    model_kwargs = {"in_channels": actual_in_channels}
    if args["gnn_arch"] == "citytrans":
        model_kwargs["target_city"] = target_city

    model = create_gnn_model(gnn_arch=args["gnn_arch"], config=config, model_kwargs=model_kwargs, device=device).to(device)

    loss_fct = GNN_Loss(
        loss_fct=config.loss_fct,
        device=device,
        weighted=bool(getattr(config, "use_weighted_loss", False)),
        num_nodes=train_dl.dataset[0].x.shape[0],
    )
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, verbose=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.peak_lr, weight_decay=1e-3)
    best_val_loss, best_epoch = model.train_model(
        config=config,
        loss_fct=loss_fct,
        optimizer=optimizer,
        train_dl=train_dl,
        valid_dl=val_dl,
        device=device,
        early_stopping=early_stopping,
        model_save_path=model_save_path,
    )

    test_result = validate_model_during_training(
        config=config, model=model, dataset=test_dl, loss_func=loss_fct, device=device
    )

    run_dir = Path(base_dir) / args["project_name"] / args["run_name"]
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / f"{target_city}_metrics.json"

    if len(test_result) == 5:
        loss, r2, spearman, pearson, hit_rates = test_result
    else:
        loss, r2, spearman, pearson = test_result
        hit_rates = {}

    payload = {
        "project_name": args["project_name"],
        "run_name": args["run_name"],
        "gnn_arch": args["gnn_arch"],
        "target_city": target_city,
        "source_cities": source_cities,
        "split_file": str(split_file),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "test_loss": float(loss),
        "test_r2": float(r2),
        "test_spearman": float(spearman),
        "test_pearson": float(pearson),
        "test_hit_rates": {k: float(v) for k, v in (hit_rates or {}).items()},
        "protocol": {
            "source_train_ratio": 0.8,
            "source_val_ratio": 0.2,
            "target_train_from_split": len(tgt_train_indices),
            "target_val_from_split": len(tgt_val_indices),
            "target_test_from_split": len(tgt_test_indices),
        },
    }

    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
