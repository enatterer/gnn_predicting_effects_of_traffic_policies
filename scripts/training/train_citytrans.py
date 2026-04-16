#!/usr/bin/env python3
"""
Train CityTrans (end-to-end, no pretrain/finetune stages) for a leave-one-city-out benchmark.

This script is intentionally isolated from existing training/finetuning scripts so it does not
affect other algorithms. It:
  - loads all source-city graphs as "source domain"
  - loads a target-city split file (train/val/test) as "target domain"
  - trains CityTrans with prediction + domain-adversarial losses end-to-end
  - evaluates on the target-city test split
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

# Ensure repository's `scripts` directory is on the Python path
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from gnn.gnn_io import GraphDataset
from training.help_functions import (
    EarlyStopping,
    build_unique_model_description,
    balanced_subset_by_city,
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
from training.help_functions import EdgeFeatures  # re-exported enum import in that module
from gnn.help_functions import GNN_Loss
from torch.utils.data import Subset
from functools import partial
from gnn.gnn_io import collate_fn
from gnn.help_functions import validate_model_during_training


def _resolve_path(p: str, project_root: Path, city_dir: Path) -> str:
    p = str(p)
    if os.path.isabs(p) and os.path.exists(p):
        return p
    # Common LRZ path prefix used in some splits/logs
    if p.startswith("/mnt/repo/"):
        candidate = str(project_root / p.replace("/mnt/repo/", ""))
        if os.path.exists(candidate):
            return candidate
    # Common "data storage" absolute paths from other machines
    if "/gnn_data/" in p:
        # Map ".../gnn_data/XYZ" -> "<repo>/XYZ" (repo has `data/` and `data/splits/`)
        rel = p.split("/gnn_data/", 1)[1]
        candidate = str(project_root / rel)
        if os.path.exists(candidate):
            return candidate
    # If it's relative, interpret as relative to repo root
    candidate = str(project_root / p)
    if os.path.exists(candidate):
        return candidate
    # Otherwise, try relative to city directory (metadata.json location)
    candidate = str(city_dir / Path(p).name)
    return candidate


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

    # Resolve paths robustly.
    # Use split parent folder as a fallback base (where split file lives).
    split_parent = split_file.parent
    for d in (train_data, val_data, test_data):
        # The split path entries may be:
        # - absolute paths from another machine
        # - relative paths into this repo
        # - just filenames (e.g., 001338.pt) that live under the city's dataset folder
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
                # Try in the target city's dataset directory by filename.
                cand2 = str(city_dir / Path(p).name)
                cand = cand2
            resolved.append(cand)
        d["path"] = resolved

    return train_data, val_data, test_data


def _build_feature_filter(use_all_features: bool, use_destination_activity: bool) -> Tuple[List[int], Dict[int, int]]:
    if use_all_features:
        node_features = []
        for feat in EdgeFeatures:
            name = feat.name
            if not use_destination_activity and feat.value >= 20:
                continue
            # mirror training.help_functions module constants: allowed_modes/highway disabled by default
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

    filtered_feature_mapping: Dict[int, int] = {}
    current_idx = 0
    for feature_name in node_features:
        filtered_feature_mapping[EdgeFeatures[feature_name].value] = current_idx
        current_idx += 1

    return node_feature_filter, filtered_feature_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CityTrans end-to-end for a target city.")
    parser.add_argument("--project_name", type=str, default="Benchmark_TL")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--target_city", type=str, required=True)
    parser.add_argument("--source_cities", type=str, required=True, help="Comma-separated source cities (all treated as source domain).")
    parser.add_argument("--split_file", type=str, required=True, help="Target-city split JSON with train/val/test.")
    parser.add_argument(
        "--limit_source_graphs",
        type=int,
        default=0,
        help="If >0, city-balanced cap on number of source graphs (for fast smoke tests). Default 0 = no cap.",
    )

    # Training/common options
    parser.add_argument("--use_all_features", type=str_to_bool, default=False)
    parser.add_argument("--use_destination_activity", type=str_to_bool, default=False)
    parser.add_argument(
        "--balance_domains",
        type=str_to_bool,
        default=True,
        help="If True, oversample target-train graphs so source/target are ~balanced per epoch.",
    )
    parser.add_argument(
        "--fit_scaler_on",
        type=str,
        default="train",
        choices=["train", "train_val_test"],
        help="Which split(s) to fit the x-feature scaler on. 'train' avoids val/test leakage.",
    )
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--loss_fct", type=str, default="mse", choices=["mse", "l1"])
    parser.add_argument("--use_weighted_loss", type=str_to_bool, default=False)
    parser.add_argument("--target_type", type=str, default="abs_vol_car")
    parser.add_argument("--target_normalization", type=str, default="None", choices=["None", "relative_to_max_traffic_vol_base_case", "relative_standard_scaler"])
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers for CityTrans script. Default 0 for memory-safe runs.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Prefetch factor when num_workers > 0.",
    )
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
    parser.add_argument("--device_nr", type=int, default=0)

    # CityTrans hyperparams
    parser.add_argument("--citytrans_hidden_dim", type=int, default=128)
    parser.add_argument("--citytrans_num_gcn_layers", type=int, default=2)
    parser.add_argument("--citytrans_knowledge_gk", type=int, default=16)
    parser.add_argument("--citytrans_knowledge_pk", type=int, default=12)
    parser.add_argument("--citytrans_knowledge_dk", type=int, default=16)
    parser.add_argument("--citytrans_adv_lambda", type=float, default=0.5)
    parser.add_argument("--citytrans_grl_lambda", type=float, default=1.0)

    args_ns = parser.parse_args()
    args = vars(args_ns)

    project_root = Path(__file__).resolve().parents[2]
    dataset_root = project_root / "data" / "bavaria" / "inductive_data" / "training_data" / "kreisfreistadt"
    base_dir = project_root / "inductive_gnn_data_results" / "transductive"

    # Resolve split file path
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

    # GPU selection:
    # - If CUDA_VISIBLE_DEVICES is already set by caller, honor it.
    # - Otherwise, reuse existing auto-selection logic.
    preset_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if preset_cuda_visible:
        print(f"[CityTrans] Respecting preset CUDA_VISIBLE_DEVICES={preset_cuda_visible}", flush=True)
    else:
        gpus = get_available_gpus()
        best_gpu = select_best_gpu(gpus)
        set_cuda_visible_device(best_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CityTrans] device={device} target_city={target_city} sources={source_cities}", flush=True)

    # Load all source metadata
    print("[CityTrans] Loading source metadata...", flush=True)
    source_meta = {"path": [], "policy_region": [], "scenario": [], "city": []}
    for city in sorted(source_cities):
        load_metadata_from_disk(source_meta, str(dataset_root / city / "metadata.json"))
    if args.get("limit_source_graphs", 0) and int(args["limit_source_graphs"]) > 0:
        source_meta = balanced_subset_by_city(source_meta, int(args["limit_source_graphs"]))
    print(f"[CityTrans] Loaded source graphs: {len(source_meta['path'])}", flush=True)

    # Load target split (train/val/test)
    print(f"[CityTrans] Loading target split file: {split_file}", flush=True)
    target_train, target_val, target_test = _load_target_split(split_file, project_root=project_root)
    print(
        f"[CityTrans] Target split sizes: train={len(target_train['path'])} val={len(target_val['path'])} test={len(target_test['path'])}",
        flush=True,
    )

    # Build a combined dataset for explicit indexing
    combined_paths: List[str] = []
    combined_labels: List[str] = []

    def _append_block(block: Dict) -> List[int]:
        start = len(combined_paths)
        combined_paths.extend(list(block["path"]))
        combined_labels.extend([f"{c}_{pr}" for c, pr in zip(block["city"], block["policy_region"])])
        end = len(combined_paths)
        return list(range(start, end))

    # Source block
    src_indices = _append_block(source_meta)
    # Target blocks (train/val/test) for domain adaptation
    tgt_train_indices = _append_block(target_train)
    tgt_val_indices = _append_block(target_val)
    tgt_test_indices = _append_block(target_test)

    dataset = GraphDataset(combined_paths, combined_labels)

    # Protocol requested:
    # - split SOURCES into train/val with 80/20
    # - train includes target-train (40) from the split file
    # - val includes target-val (10) from the split file
    # - test: target-test (100) only
    src_labels = [combined_labels[i] for i in src_indices]
    # Stratify by city-policy label if possible to keep distribution similar.
    stratify = src_labels if len(set(src_labels)) > 1 else None
    src_train_indices, src_val_indices = train_test_split(
        list(src_indices),
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    train_indices = list(src_train_indices) + list(tgt_train_indices)
    val_indices = list(src_val_indices) + list(tgt_val_indices)
    test_indices = list(tgt_test_indices)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    print(
        f"[CityTrans] Combined dataset size={len(dataset)} train={len(train_set)} val={len(val_set)} test={len(test_set)}",
        flush=True,
    )

    # Feature filter + augmentation-collate (no extra augmentations by default in benchmark)
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

    # Normalization
    if args["fit_scaler_on"] == "train_val_test":
        print("[CityTrans] Normalizing datasets (fit scaler on train+val+test)...", flush=True)
        combined_norm_set = Subset(dataset, train_indices + val_indices + test_indices)
    else:
        print("[CityTrans] Normalizing datasets (fit scaler on train only)...", flush=True)
        combined_norm_set = Subset(dataset, train_indices)
    train_set_norm, scalers = normalize_dataset(train_data_list=train_set, combined_data_list=combined_norm_set)
    val_set_norm = normalize_dataset_with_scaler(dataset_input=val_set, scalers=scalers)
    test_set_norm = normalize_dataset_with_scaler(dataset_input=test_set, scalers=scalers)
    print("[CityTrans] Normalization done. Building dataloaders...", flush=True)

    loader_kwargs = {
        "batch_size": int(args["batch_size"]),
        "num_workers": int(args["num_workers"]),
        "pin_memory": False,
        "collate_fn": collate_train,
        "drop_last": False,
    }
    if int(args["num_workers"]) > 0:
        loader_kwargs["prefetch_factor"] = int(args["prefetch_factor"])

    if bool(args["balance_domains"]):
        # train_set_norm contains graphs in the exact order: src_train_indices + tgt_train_indices
        num_source = len(src_train_indices)
        num_target = len(tgt_train_indices)
        if num_target <= 0:
            raise ValueError("Target train split is empty; cannot balance domains.")

        target_w = float(num_source) / float(num_target)
        weights = [1.0] * len(train_set_norm)
        for j in range(num_source, num_source + num_target):
            weights[j] = target_w

        sampler = WeightedRandomSampler(weights, num_samples=len(train_set_norm), replacement=True)
        train_dl = DataLoader(dataset=train_set_norm, sampler=sampler, shuffle=None, **loader_kwargs)
        print(
            f"[CityTrans] Domain balancing enabled: source_train={num_source} target_train={num_target} target_weight={target_w:.2f}",
            flush=True,
        )
    else:
        # Keep deterministic simple loader here, but use script-specific worker settings.
        train_dl = DataLoader(dataset=train_set_norm, shuffle=True, sampler=None, **loader_kwargs)

    eval_loader_kwargs = {
        "batch_size": int(args["batch_size"]),
        "shuffle": True,
        "num_workers": int(args["num_workers"]),
        "pin_memory": False,
        "drop_last": False,
    }
    if int(args["num_workers"]) > 0:
        eval_loader_kwargs["prefetch_factor"] = int(args["prefetch_factor"])
    val_dl = DataLoader(dataset=val_set_norm, collate_fn=collate_no_aug, **eval_loader_kwargs)
    test_dl = DataLoader(dataset=test_set_norm, collate_fn=collate_no_aug, **eval_loader_kwargs)

    # Build run dir + wandb config
    unique_model_description = args.get("unique_model_description") or args["run_name"]
    model_save_path, path_to_save_dataloader = get_paths(
        base_dir=str(base_dir / args["project_name"]),
        unique_model_description=unique_model_description,
        model_save_path="trained_model/model.pth",
    )

    # Prepare config dict for wandb (mirror run_models style)
    args_for_wandb = dict(args)
    args_for_wandb["gnn_arch"] = "citytrans"
    args_for_wandb["unique_model_description"] = unique_model_description
    args_for_wandb["target_normalization"] = None if args_for_wandb.get("target_normalization") == "None" else args_for_wandb.get("target_normalization")
    config = setup_wandb(args_for_wandb)

    # Infer actual in_channels from a batch (after feature filtering)
    sample_batch = next(iter(train_dl))
    actual_in_channels = int(sample_batch.x.shape[1])
    try:
        config.update({"in_channels": actual_in_channels}, allow_val_change=True)
    except Exception:
        pass

    model_kwargs = {
        "in_channels": actual_in_channels,
        "hidden_dim": int(args["citytrans_hidden_dim"]),
        "num_gcn_layers": int(args["citytrans_num_gcn_layers"]),
        "knowledge_gk": int(args["citytrans_knowledge_gk"]),
        "knowledge_pk": int(args["citytrans_knowledge_pk"]),
        "knowledge_dk": int(args["citytrans_knowledge_dk"]),
        "adv_lambda": float(args["citytrans_adv_lambda"]),
        "grl_lambda": float(args["citytrans_grl_lambda"]),
        "target_city": target_city,
    }

    model = create_gnn_model(gnn_arch="citytrans", config=config, model_kwargs=model_kwargs, device=device).to(device)

    loss_fct = GNN_Loss(loss_fct=config.loss_fct, device=device, weighted=False, num_nodes=train_dl.dataset[0].x.shape[0])
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

    # Evaluate on target test set
    test_result = None
    try:
        test_result = validate_model_during_training(
            config=config,
            model=model,
            dataset=test_dl,
            loss_func=loss_fct,
            device=device,
        )
    except Exception as e:
        print(f"WARNING: test evaluation failed: {e}")

    run_dir = Path(base_dir) / args["project_name"] / unique_model_description
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / f"{target_city}_metrics.json"

    payload = {
        "project_name": args["project_name"],
        "run_name": unique_model_description,
        "target_city": target_city,
        "source_cities": source_cities,
        "split_file": str(split_file),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "model_path": str(model_save_path),
    }

    if test_result is not None:
        if len(test_result) == 5:
            loss, r2, spearman, pearson, hit_rates = test_result
        else:
            loss, r2, spearman, pearson = test_result
            hit_rates = {}
        payload.update(
            {
                "test_loss": float(loss),
                "test_r2": float(r2),
                "test_spearman": float(spearman),
                "test_pearson": float(pearson),
                "test_hit_rates": {k: float(v) for k, v in (hit_rates or {}).items()},
            }
        )

    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved CityTrans metrics to {metrics_path}")


if __name__ == "__main__":
    main()

