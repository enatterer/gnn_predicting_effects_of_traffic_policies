#!/usr/bin/env python3
"""
Investigate transfer learning performance as a function of pretraining data size.

For Regensburg as the target city, this script systematically varies the number
of source cities used for pretraining (1, 2, 3, or 4 cities) and evaluates
the resulting finetuning performance.

Experimental Design:
- 5 runs: pretrain on each of the 5 other cities individually
- 5 runs: pretrain on 5 random selections of 2 cities
- 5 runs: pretrain on 5 random selections of 3 cities
- 5 runs: pretrain on 5 random selections of 4 cities
Total: 20 pretrain→finetune experiments

Each pretraining run saves checkpoints, which are then used to finetune on Regensburg
using the same train/val/test split from run_pretrain_finetune_comparison.py.

IMPORTANT: This script requires an existing train/val/test split for Regensburg.
By default, it looks for: data/splits/regensburg/rs_1/t40_v10/regensburg_rs1_t40_v10_seed42_train40_val10_test100_random.json

To generate this split if it doesn't exist, run:
    python scripts/training/run_pretrain_finetune_comparison.py \
      --testing_cities regensburg --train_val_configs '40:10' --num_random_seeds 1 \
      --skip_scratch True --skip_finetuning True

Example usage:
    python scripts/training/run_varying_pretrain_sources_regensburg.py --gnn_arch trans_encoder

Example with custom split:
    python scripts/training/run_varying_pretrain_sources_regensburg.py \
      --train_val_size '80:20' --seed_idx 2

Example (nohup):
    nohup bash -lc 'source ~/.bashrc; PYTHONUNBUFFERED=1 stdbuf -oL -eL python -u scripts/training/run_varying_pretrain_sources_regensburg.py --project_name VaryingPretrainSources' > runs_varying_sources_regensburg.log 2>&1 & echo $!
"""

import argparse
import importlib
import json
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Ensure the repository's `scripts` directory is on the Python path
CURRENT_FILE = Path(__file__).resolve()
SCRIPTS_DIR = CURRENT_FILE.parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from training.help_functions import (
    str_to_bool,
)
from training import run_models as run_models_module

# Base directory used by both training and finetuning scripts
BASE_DIR = Path(run_models_module.base_dir).resolve()

# Target city for finetuning
TARGET_CITY = "regensburg"

# Source cities available for pretraining (excluding Regensburg)
SOURCE_CITIES: Sequence[str] = (
    "landshut",
    "bayreuth",
    "schweinfurt",
    "bamberg",
    "wuerzburg",
)

# Argument names for each script (reused from run_pretrain_finetune_comparison.py)
RUN_MODELS_ARGS = [
    "gnn_arch",
    "project_name",
    "use_inductive_variant",
    "unique_model_description",
    "in_channels",
    "use_all_features",
    "out_channels",
    "model_kwargs",
    "loss_fct",
    "use_weighted_loss",
    "use_city_balanced_loss",
    "use_target_standardization",
    "target_type",
    "use_weighted_batches",
    "num_epochs",
    "batch_size",
    "peak_lr",
    "initial_lr",
    "warmup_fraction",
    "cosine_decay_rate",
    "min_lr_fraction",
    "early_stopping_patience",
    "use_dropout",
    "dropout",
    "gradient_accumulation_steps",
    "use_gradient_clipping",
    "device_nr",
    "continue_training",
    "base_checkpoint_path",
    "use_nested_neighbor_loader",
    "neighbor_sizes",
    "subgraphs_per_graph",
    "seed_size",
    "sampling_strategy",
    "min_subgraph_nodes",
    "max_subgraph_nodes",
    "aug_pos_rotation",
    "aug_feature_noise",
    "aug_node_masking_probability",
    "limit_available_graphs",
]

FINETUNE_ARGS = [
    "run_name",
    "pretrain_run_name",
    "gnn_arch",
    "cities",
    "project_name",
    "pretraining_inductive",
    "in_channels",
    "use_all_features",
    "out_channels",
    "model_kwargs",
    "loss_fct",
    "use_weighted_loss",
    "target_normalization",
    "predict_mode_stats",
    "target_type",
    "use_bootstrapping",
    "use_weighted_sampling",
    "num_epochs",
    "batch_size",
    "peak_lr",
    "initial_lr",
    "warmup_fraction",
    "cosine_decay_rate",
    "min_lr_fraction",
    "early_stopping_patience",
    "use_dropout",
    "dropout",
    "gradient_accumulation_steps",
    "use_gradient_clipping",
    "device_nr",
    "use_nested_neighbor_loader",
    "neighbor_sizes",
    "subgraphs_per_graph",
    "seed_size",
    "sampling_strategy",
    "min_subgraph_nodes",
    "max_subgraph_nodes",
    "use_data_augmentation",
    "use_message_dropout_probability",
    "augment_feature_noise_prob",
    "use_node_masking_probability",
    "limit_train_graphs",
    "limit_val_graphs",
    "limit_test_graphs",
    "unique_model_description",
    "start_from_scratch",
    "split_file",
]

SHARED_ARG_NAMES = sorted(set(RUN_MODELS_ARGS) & set(FINETUNE_ARGS))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Investigate transfer learning as a function of pretraining data size."
    )

    # Shared arguments
    parser.add_argument("--gnn_arch", type=str, default="trans_encoder")
    parser.add_argument("--in_channels", type=int, default=5)
    parser.add_argument("--use_all_features", type=str_to_bool, default=False)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--model_kwargs", type=str, default=None)
    parser.add_argument("--loss_fct", type=str, default="mse")
    parser.add_argument("--use_weighted_loss", type=str_to_bool, default=False)
    parser.add_argument("--target_type", type=str, default="abs_vol_car")
    parser.add_argument("--warmup_fraction", type=float, default=0.1)
    parser.add_argument("--cosine_decay_rate", type=float, default=0.5)
    parser.add_argument("--min_lr_fraction", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--use_dropout", type=str_to_bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_gradient_clipping", type=str_to_bool, default=True)
    parser.add_argument("--device_nr", type=int, default=0)
    parser.add_argument("--use_nested_neighbor_loader", type=str_to_bool, default=False)
    parser.add_argument("--neighbor_sizes", type=str, default="5,5,5")
    parser.add_argument("--subgraphs_per_graph", type=int, default=2)
    parser.add_argument("--seed_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="neighbor_sampling",
        choices=["neighbor_sampling", "random_walk"],
    )
    parser.add_argument("--min_subgraph_nodes", type=int, default=500)
    parser.add_argument("--max_subgraph_nodes", type=int, default=50000)
    parser.add_argument("--unique_model_description", type=str, default=None)

    # Arguments specific to run_models.py
    parser.add_argument("--use_inductive_variant", type=str_to_bool, default=False)
    parser.add_argument("--use_weighted_batches", type=str_to_bool, default=False)
    parser.add_argument("--use_target_standardization", type=str_to_bool, default=False)
    parser.add_argument("--use_city_balanced_loss", type=str_to_bool, default=False)
    parser.add_argument("--aug_pos_rotation", type=str_to_bool, default=False)
    parser.add_argument("--aug_feature_noise", type=str_to_bool, default=False)
    parser.add_argument("--aug_node_masking_probability", type=float, default=0.0)
    parser.add_argument("--continue_training", type=str_to_bool, default=False)
    parser.add_argument("--base_checkpoint_path", type=str, default=None)
    parser.add_argument("--run_peak_lr", type=float, default=0.0003)
    parser.add_argument("--run_initial_lr", type=float, default=0.00003)
    parser.add_argument("--run_num_epochs", type=int, default=200)
    parser.add_argument("--run_limit_available_graphs", type=int, default=0)

    # Arguments specific to finetune_models.py
    parser.add_argument("--target_normalization", type=str, default="None",
                        choices=["None", "relative_to_max_traffic_vol_base_case", "relative_standard_scaler"])
    parser.add_argument("--predict_mode_stats", type=str_to_bool, default=False)
    parser.add_argument("--use_bootstrapping", type=str_to_bool, default=False)
    parser.add_argument("--use_weighted_sampling", type=str_to_bool, default=False)
    parser.add_argument("--use_data_augmentation", type=str_to_bool, default=False)
    parser.add_argument("--use_message_dropout_probability", type=float, default=0.0)
    parser.add_argument("--augment_feature_noise_prob", type=str_to_bool, default=False)
    parser.add_argument("--use_node_masking_probability", type=float, default=0.0)
    parser.add_argument("--pretraining_inductive", type=str_to_bool, default=False)
    parser.add_argument("--finetune_peak_lr", type=float, default=0.0003)
    parser.add_argument("--finetune_initial_lr", type=float, default=0.00003)
    parser.add_argument("--finetune_num_epochs", type=int, default=500)
    parser.add_argument("--finetune_limit_train_graphs", type=int, default=0)
    parser.add_argument("--finetune_limit_val_graphs", type=int, default=0)
    parser.add_argument("--finetune_limit_test_graphs", type=int, default=0)

    # Orchestrator-specific arguments
    parser.add_argument("--project_name", type=str, default="VaryingPretrainSources",
                        help="Project name for WandB and results directory.")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to dataset directory.")
    parser.add_argument("--splits_dir", type=str, default="data/splits",
                        help="Directory to load split files from (same as run_pretrain_finetune_comparison.py).")
    parser.add_argument("--split_file", type=str, default=None,
                        help="Path to existing train/val/test split file for Regensburg. If not provided, auto-detects from splits_dir.")
    parser.add_argument("--train_val_size", type=str, default="40:10",
                        help="Train:val size to use (e.g., '40:10'). Used to locate the correct split file.")
    parser.add_argument("--seed_idx", type=int, default=1,
                        help="Seed index to use (1-5). Used to locate the correct split file.")
    parser.add_argument("--test_count", type=int, default=100,
                        help="Test count used in the split file name.")
    parser.add_argument("--test_set_type", type=str, default="random",
                        choices=["distant_iou", "random"],
                        help="Test set type used in the split file name.")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for generating city combinations.")
    parser.add_argument("--skip_pretraining", type=str_to_bool, default=False,
                        help="Skip pretraining stage (assumes checkpoints already exist).")
    parser.add_argument("--skip_finetuning", type=str_to_bool, default=False,
                        help="Skip finetuning stage.")
    parser.add_argument("--skip_scratch", type=str_to_bool, default=False,
                        help="Skip scratch (from-scratch) training stage.")

    return parser


def parse_neighbor_sizes(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_cli_args(arg_values: Dict[str, object], include: Sequence[str]) -> List[str]:
    argv: List[str] = []
    for name in include:
        value = arg_values.get(name)
        if value is None:
            continue

        formatted = None
        if isinstance(value, bool):
            formatted = "True" if value else "False"
        elif isinstance(value, list):
            if name == "neighbor_sizes":
                formatted = ",".join(str(v) for v in value)
            else:
                formatted = ",".join(str(v) for v in value)
        else:
            formatted = str(value)

        argv.extend([f"--{name}", formatted])
    return argv


def pretrain_checkpoint_dir(project_name: str, run_name: str) -> Path:
    """Return the checkpoint directory for a pretraining run."""
    return BASE_DIR / project_name / run_name / "trained_model" / "checkpoints"


def finetuned_model_path(project_name: str, run_name: str) -> Path:
    """Return the expected model path for a finetuning run."""
    return BASE_DIR / project_name / run_name / "finetuned_model" / "model.pth"


def scratch_model_path(project_name: str, run_name: str) -> Path:
    """Return the expected model path for a scratch (from-scratch) training run."""
    return BASE_DIR / project_name / run_name / "finetuned_model" / "model.pth"


def has_checkpoints(ckpt_dir: Path) -> bool:
    """True if a checkpoint directory exists and has at least one epoch file."""
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        return False
    return any(f.name.startswith("checkpoint_epoch_") and f.name.endswith(".pt") for f in ckpt_dir.iterdir())


def call_run_models(
    base_args: Dict[str, object],
    train_cities: Sequence[str],
    val_cities: Sequence[str],
    test_cities: Sequence[str],
) -> None:
    """Call run_models.py to pretrain on specified cities."""
    module = importlib.import_module("training.run_models")
    module = importlib.reload(module)

    module.train_cities = list(train_cities)
    module.val_cities = list(val_cities)
    module.test_cities = list(test_cities)

    argv = build_cli_args(base_args, RUN_MODELS_ARGS)
    saved_argv = sys.argv
    try:
        sys.argv = [module.__file__ or "run_models.py", *argv]
        module.main()
    finally:
        sys.argv = saved_argv


def call_finetune_models(base_args: Dict[str, object]) -> None:
    """Call finetune_models.py to finetune on the target city."""
    module = importlib.import_module("training.finetune_models")
    module = importlib.reload(module)

    argv = build_cli_args(base_args, FINETUNE_ARGS)
    saved_argv = sys.argv
    try:
        sys.argv = [module.__file__ or "finetune_models.py", *argv]
        module.main()
    finally:
        sys.argv = saved_argv
        sys.stdout.flush()
        sys.stderr.flush()


def generate_city_combinations(source_cities: Sequence[str], seed: int) -> List[Tuple[int, List[str], str]]:
    """
    Generate combinations of source cities for pretraining.
    
    Returns a list of (num_cities, city_list, run_suffix) tuples:
    - 5 single cities (all 5 individually)
    - 5 random pairs of cities
    - 5 random triplets of cities
    - 5 random quadruplets of cities
    """
    rng = random.Random(seed)
    configs = []
    
    # 1. All single cities (5 combinations)
    for i, city in enumerate(source_cities, start=1):
        configs.append((1, [city], f"n1_c{i}_{city}"))
    
    # 2. Random selections of 2 cities (5 combinations)
    all_pairs = list(combinations(source_cities, 2))
    rng.shuffle(all_pairs)
    selected_pairs = all_pairs[:5]
    for i, pair in enumerate(selected_pairs, start=1):
        city_str = "_".join(sorted(pair))
        configs.append((2, list(pair), f"n2_c{i}_{city_str}"))
    
    # 3. Random selections of 3 cities (5 combinations)
    all_triplets = list(combinations(source_cities, 3))
    rng.shuffle(all_triplets)
    selected_triplets = all_triplets[:5]
    for i, triplet in enumerate(selected_triplets, start=1):
        city_str = "_".join(sorted(triplet))
        configs.append((3, list(triplet), f"n3_c{i}_{city_str}"))
    
    # 4. Random selections of 4 cities (5 combinations)
    all_quadruplets = list(combinations(source_cities, 4))
    rng.shuffle(all_quadruplets)
    selected_quadruplets = all_quadruplets[:5]
    for i, quad in enumerate(selected_quadruplets, start=1):
        city_str = "_".join(sorted(quad))
        configs.append((4, list(quad), f"n4_c{i}_{city_str}"))
    
    return configs


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parents[2]
    
    if args.dataset_path is None:
        dataset_path = project_root / 'data' / 'bavaria' / 'inductive_data' / 'training_data' / 'kreisfreistadt'
    else:
        dataset_path = Path(args.dataset_path)
    dataset_path = dataset_path.resolve()
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    if args.splits_dir and not Path(args.splits_dir).is_absolute():
        splits_dir = project_root / args.splits_dir
    else:
        splits_dir = Path(args.splits_dir) if args.splits_dir else project_root / "data/splits_varying_sources"
    splits_dir = splits_dir.resolve()
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    neighbor_sizes = parse_neighbor_sizes(args.neighbor_sizes)

    # Prepare base arguments for pretraining
    run_base_args: Dict[str, object] = {
        name: getattr(args, name) for name in RUN_MODELS_ARGS if hasattr(args, name)
    }
    run_base_args["neighbor_sizes"] = neighbor_sizes
    run_base_args["limit_available_graphs"] = args.run_limit_available_graphs
    run_base_args["project_name"] = args.project_name
    
    if args.run_peak_lr is not None:
        run_base_args["peak_lr"] = args.run_peak_lr
    if args.run_initial_lr is not None:
        run_base_args["initial_lr"] = args.run_initial_lr
    if args.run_num_epochs is not None:
        run_base_args["num_epochs"] = args.run_num_epochs

    # Prepare base arguments for finetuning
    finetune_base_args: Dict[str, object] = {
        name: getattr(args, name) for name in FINETUNE_ARGS if hasattr(args, name)
    }
    finetune_base_args["neighbor_sizes"] = neighbor_sizes
    finetune_base_args["project_name"] = args.project_name
    finetune_base_args["cities"] = TARGET_CITY
    finetune_base_args["limit_train_graphs"] = None
    finetune_base_args["limit_val_graphs"] = None
    finetune_base_args["limit_test_graphs"] = None
    
    if args.finetune_peak_lr is not None:
        finetune_base_args["peak_lr"] = args.finetune_peak_lr
    if args.finetune_initial_lr is not None:
        finetune_base_args["initial_lr"] = args.finetune_initial_lr
    if args.finetune_num_epochs is not None:
        finetune_base_args["num_epochs"] = args.finetune_num_epochs

    if finetune_base_args.get("pretraining_inductive") is None:
        finetune_base_args["pretraining_inductive"] = run_base_args.get("use_inductive_variant", False)

    # Set split file for Regensburg
    if args.split_file:
        split_file_path = Path(args.split_file).resolve()
        if not split_file_path.exists():
            raise ValueError(f"Split file does not exist: {split_file_path}")
    else:
        # Auto-detect split file from run_pretrain_finetune_comparison.py structure
        # Parse train_val_size
        if ":" not in args.train_val_size:
            raise ValueError(f"Invalid train_val_size '{args.train_val_size}'. Use format 'train:val' (e.g., '40:10')")
        train_size, val_size = args.train_val_size.split(":")
        train_size, val_size = int(train_size), int(val_size)
        
        # Calculate seed value (same logic as original script)
        shuffle_seed = 42  # default from original script
        seed = shuffle_seed + (args.seed_idx - 1)
        
        # Construct path following run_pretrain_finetune_comparison.py naming convention
        split_subdir = splits_dir / TARGET_CITY / f"rs_{args.seed_idx}" / f"t{train_size}_v{val_size}"
        split_filename = (
            f"{TARGET_CITY}_rs{args.seed_idx}_t{train_size}_v{val_size}_seed{seed}_"
            f"train{train_size}_val{val_size}_test{args.test_count}_{args.test_set_type}.json"
        )
        split_file_path = split_subdir / split_filename
        
        if not split_file_path.exists():
            print(f"\n⚠️  Expected split file not found: {split_file_path}")
            print(f"\nTo generate this split, run:")
            print(f"  python scripts/training/run_pretrain_finetune_comparison.py \\")
            print(f"    --testing_cities {TARGET_CITY} \\")
            print(f"    --train_val_configs '{train_size}:{val_size}' \\")
            print(f"    --num_random_seeds {args.seed_idx} \\")
            print(f"    --skip_scratch True --skip_finetuning True")
            print(f"\nOr provide a custom split file with --split_file")
            raise ValueError(f"Split file not found: {split_file_path}")
        
        print(f"\n✓ Auto-detected split file: {split_file_path}")

    finetune_base_args["split_file"] = str(split_file_path)

    # Generate all city combinations
    city_configs = generate_city_combinations(SOURCE_CITIES, args.random_seed)
    
    print("\n" + "=" * 80)
    print(f"VARYING PRETRAIN SOURCES EXPERIMENT")
    print("=" * 80)
    print(f"Target city: {TARGET_CITY}")
    print(f"Source cities: {list(SOURCE_CITIES)}")
    print(f"Total experiments: {len(city_configs)}")
    print(f"Project name: {args.project_name}")
    print(f"\n{TARGET_CITY.capitalize()} split file:")
    print(f"  {split_file_path}")
    with open(split_file_path, 'r') as f:
        split_info = json.load(f)
    print(f"  Train: {split_info.get('train_count', 'N/A')}, Val: {split_info.get('val_count', 'N/A')}, Test: {split_info.get('test_count', 'N/A')}")
    print("=" * 80)

    # Process each configuration
    for config_idx, (num_cities, pretrain_cities, run_suffix) in enumerate(city_configs, start=1):
        print("\n" + "=" * 80)
        print(f"Experiment {config_idx}/{len(city_configs)}")
        print(f"Pretraining on {num_cities} cities: {pretrain_cities}")
        print(f"Run suffix: {run_suffix}")
        print("=" * 80)

        pretrain_run_name = f"{TARGET_CITY}_pretrain_{run_suffix}"
        finetune_run_name = f"{TARGET_CITY}_finetune_{run_suffix}"

        # Step 1: Pretraining
        if not args.skip_pretraining:
            pretrain_ckpt_dir = pretrain_checkpoint_dir(args.project_name, pretrain_run_name)
            if has_checkpoints(pretrain_ckpt_dir):
                print(f"\n[Step 1/3] Skipping pretraining; found existing checkpoints: {pretrain_ckpt_dir}")
            else:
                print(f"\n[Step 1/3] Pretraining on {pretrain_cities}...")
                run_args = dict(run_base_args)
                run_args["unique_model_description"] = pretrain_run_name
                
                try:
                    # Transductive pretraining: use pretrain_cities for training
                    call_run_models(run_args, pretrain_cities, [], [TARGET_CITY])
                except KeyboardInterrupt:
                    print(f"\n  ⚠️  KeyboardInterrupt received during pretraining")
                    print(f"  Exiting gracefully...")
                    raise  # Re-raise to allow clean exit
                except Exception as e:
                    print(f"  ✗ Pretraining failed with exception: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"  Skipping this configuration and continuing...")
                    continue
                
                if not has_checkpoints(pretrain_ckpt_dir):
                    print(f"  ✗ Pretraining failed to create checkpoints: {pretrain_ckpt_dir}")
                    continue
        else:
            print(f"\n[Step 1/3] Skipping pretraining (--skip_pretraining)")

        # Step 2: Finetuning from checkpoint
        if not args.skip_finetuning:
            finetuned_model = finetuned_model_path(args.project_name, finetune_run_name)
            print(f"\n[Step 2/3] Checking for existing finetuned model: {finetuned_model}")
            if finetuned_model.exists():
                print(f"  ✓ Found existing model, skipping finetuning: {finetuned_model}")
            else:
                print(f"  ✗ Model not found, starting finetuning...")
                print(f"\n[Step 2/3] Finetuning {TARGET_CITY} from checkpoint...")
                finetune_args = dict(finetune_base_args)
                finetune_args["run_name"] = finetune_run_name
                finetune_args["pretrain_run_name"] = pretrain_run_name
                finetune_args["start_from_scratch"] = False
                finetune_args["unique_model_description"] = finetune_run_name
                
                try:
                    call_finetune_models(finetune_args)
                    print(f"  ✓ Finetuning completed: {finetune_run_name}")
                except KeyboardInterrupt:
                    print(f"\n  ⚠️  KeyboardInterrupt received during finetuning")
                    print(f"  Exiting gracefully...")
                    raise  # Re-raise to allow clean exit
                except Exception as e:
                    print(f"  ✗ Finetuning failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        else:
            print(f"\n[Step 2/3] Skipping finetuning (--skip_finetuning)")

        # Step 3: Scratch (from-scratch) training
        scratch_run_name = f"{TARGET_CITY}_scratch_{run_suffix}"
        if not args.skip_scratch:
            scratch_model = scratch_model_path(args.project_name, scratch_run_name)
            print(f"\n[Step 3/3] Checking for existing scratch model: {scratch_model}")
            if scratch_model.exists():
                print(f"  ✓ Found existing model, skipping scratch training: {scratch_model}")
            else:
                print(f"  ✗ Model not found, starting scratch training...")
                print(f"\n[Step 3/3] Training {TARGET_CITY} from scratch (no pretraining)...")
                scratch_args = dict(finetune_base_args)
                scratch_args["run_name"] = scratch_run_name
                scratch_args["pretrain_run_name"] = None  # No pretraining
                scratch_args["start_from_scratch"] = True
                scratch_args["unique_model_description"] = scratch_run_name
                # Use the exact same split_file as finetuning
                scratch_args["split_file"] = finetune_base_args["split_file"]
                
                try:
                    call_finetune_models(scratch_args)
                    print(f"  ✓ Scratch training completed: {scratch_run_name}")
                except KeyboardInterrupt:
                    print(f"\n  ⚠️  KeyboardInterrupt received during scratch training")
                    print(f"  Exiting gracefully...")
                    raise  # Re-raise to allow clean exit
                except Exception as e:
                    print(f"  ✗ Scratch training failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue to evaluation even if scratch fails
        else:
            print(f"\n[Step 3/3] Skipping scratch training (--skip_scratch)")

        print(f"\n✓ Completed experiment {config_idx}/{len(city_configs)}")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()

