#!/usr/bin/env python3
"""
Orchestrate pretraining, finetuning (scratch vs checkpoint), and result comparison.

For each city in the target list:
1. Pretrain transductively on the other cities (excluding test city) once
   → run name: "{city}_pretrain_without_{city}"
2. For each random seed (default 5) and each train/val size
   [(10,3), (20,5), (40,10), (80,20), (160,40)]:
   a. Generate random train/val split for the city (per seed & config)
   b. Run finetuning from scratch   → "{city}_scratch_rs_{k}_t{train}_v{val}"
   c. Run finetuning from checkpoint → "{city}_finetune_rs_{k}_t{train}_v{val}"
   d. Generate distant test set (default 100 graphs) per seed/config
   e. Evaluate scratch and finetune models on that test split

Example:
    python run_pretrain_finetune_comparison.py --gnn_arch trans_encoder --num_epochs 200

Example (nohup, defaults: 5 seeds, 5 train/val configs, test_count=100):
    nohup bash -lc 'source ~/.bashrc; PYTHONUNBUFFERED=1 stdbuf -oL -eL python -u scripts/training/run_pretrain_finetune_comparison.py --project_name PretrainFinetune_Comparison' > nohup_pretrain_finetune_comp.log 2>&1 & echo $!

Custom train/val via CLI (e.g., two configs 20:5 and 60:15):
    ... run_pretrain_finetune_comparison.py --train_val_configs "20:5,60:15" ...
"""

import argparse
import importlib
import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple
import torch

# Ensure the repository's `scripts` directory is on the Python path
CURRENT_FILE = Path(__file__).resolve()
SCRIPTS_DIR = CURRENT_FILE.parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from training.help_functions import (
    load_metadata_from_disk,
    prepare_data_with_graph_features,
    set_cuda_visible_device,
    str_to_bool,
)
from gnn.help_functions import GNN_Loss, validate_model_during_training
from training import run_models as run_models_module

# Base directory used by both training and finetuning scripts
BASE_DIR = Path(run_models_module.base_dir).resolve()

# Defines ALL POSSIBLE Citites
# Target cities
TARGET_CITIES: Sequence[str] = (
    "landshut",
    "regensburg",
    "bayreuth",
    "schweinfurt",
    "bamberg",
    "wuerzburg",
)

# Train/val split configurations (default set of five) and test count
DEFAULT_TRAIN_VAL_CONFIGS: Sequence[Tuple[int, int]] = (
    (10, 3),
    (20, 5),
    (40, 10),
    (80, 20),
    (160, 40),
)
DEFAULT_TEST_COUNT = 100
# Default number of random seeds for scratch vs finetune runs
DEFAULT_NUM_RANDOM_SEEDS = 5

# Argument names for each script
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
        description="Pretrain, finetune (scratch vs checkpoint), and compare results."
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
    parser.add_argument("--use_inductive_variant", type=str_to_bool, default=False)  # Transductive for pretraining
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
    parser.add_argument("--run_num_epochs", type=int, default=300)
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
    parser.add_argument("--pretraining_inductive", type=str_to_bool, default=False)  # Transductive pretraining
    parser.add_argument("--finetune_peak_lr", type=float, default=0.0003)
    parser.add_argument("--finetune_initial_lr", type=float, default=0.00003)
    parser.add_argument("--finetune_num_epochs", type=int, default=300)
    parser.add_argument("--finetune_limit_train_graphs", type=int, default=0)
    parser.add_argument("--finetune_limit_val_graphs", type=int, default=0)
    parser.add_argument("--finetune_limit_test_graphs", type=int, default=0)

    # Orchestrator-specific arguments
    parser.add_argument("--shuffle_seed", type=int, default=42,
                        help="Seed for random train/val split generation.")
    parser.add_argument("--selected_cities", type=str, default=None,
                        help="Optional comma-separated subset of cities to consider. Defaults to all. [FOR PRETRAINING, Defines the Universe]")
    parser.add_argument("--testing_cities", type=str, default=None,
                        help="Optional comma-separated subset of cities to test on. Defaults to all. [FOR FINETUNING, Run finetuning vs scratch on these cities one by one]")
    parser.add_argument("--skip_pretraining", type=str_to_bool, default=False,
                        help="Skip pretraining stage (assumes checkpoints already exist).")
    parser.add_argument("--skip_scratch", type=str_to_bool, default=False,
                        help="Skip training from scratch stage.")
    parser.add_argument("--skip_finetuning", type=str_to_bool, default=False,
                        help="Skip finetuning (from checkpoint) stage.")
    parser.add_argument("--skip_test_generation", type=str_to_bool, default=False,
                        help="Skip test set generation.")
    parser.add_argument("--splits_dir", type=str, default="data/splits",
                        help="Directory to save/load split files.")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to dataset directory.")
    parser.add_argument("--num_trials_test", type=int, default=2000,
                        help="Number of trials for finding distant test sets (default: 2000 for better results).")
    parser.add_argument("--test_count", type=int, default=DEFAULT_TEST_COUNT,
                        help="Number of test graphs to select for the distant split.")
    parser.add_argument("--force_test_regeneration", type=str_to_bool, default=False,
                        help="If True, regenerate test splits even if they already exist.")
    parser.add_argument("--project_name", type=str, default=None,
                        help="Project name for WandB and results directory.")
    parser.add_argument("--num_random_seeds", type=int, default=DEFAULT_NUM_RANDOM_SEEDS,
                        help="Number of random seeds (and corresponding scratch/finetune runs) per city.")
    parser.add_argument("--train_val_configs", type=str, default=None,
                        help="Comma-separated train:val pairs, e.g., '10:3,20:5'. "
                             "Defaults to 10:3,20:5,40:10,80:20,160:40.")

    return parser


def parse_neighbor_sizes(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_train_val_configs(value: Optional[str]) -> Sequence[Tuple[int, int]]:
    """Parse comma-separated train:val pairs into a sequence of tuples."""
    if not value:
        return DEFAULT_TRAIN_VAL_CONFIGS
    pairs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid train_val_configs entry '{item}'. Use 'train:val' format.")
        train_str, val_str = item.split(":", 1)
        train, val = int(train_str), int(val_str)
        if train <= 0 or val <= 0:
            raise ValueError(f"Train/val counts must be positive: got {train}:{val}")
        pairs.append((train, val))
    if not pairs:
        raise ValueError("No valid train_val_configs provided.")
    return tuple(pairs)


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


def pretrain_model_path(project_name: str, run_name: str) -> Path:
    """Return the expected model path for a pretraining run."""
    return BASE_DIR / project_name / run_name / "trained_model" / "model.pth"


def pretrain_checkpoint_dir(project_name: str, run_name: str) -> Path:
    """Return the checkpoint directory for a pretraining run."""
    return BASE_DIR / project_name / run_name / "trained_model" / "checkpoints"


def finetuned_model_path(project_name: str, run_name: str) -> Path:
    """Return the expected model path for a finetuning run."""
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
    module = importlib.import_module("training.finetune_models")
    module = importlib.reload(module)

    argv = build_cli_args(base_args, FINETUNE_ARGS)
    saved_argv = sys.argv
    try:
        sys.argv = [module.__file__ or "finetune_models.py", *argv]
        module.main()
    finally:
        sys.argv = saved_argv
        # Ensure buffered output is flushed so the orchestrator can proceed
        sys.stdout.flush()
        sys.stderr.flush()


def generate_random_train_val_split(
    city: str,
    dataset_path: Path,
    train_count: int,
    val_count: int,
    seed: int,
    output_path: Path,
) -> Path:
    """Generate a random train/val split and save to JSON file."""
    print(f"\n{'=' * 80}")
    print(f"Generating random train/val split for {city}")
    print(f"{'=' * 80}")
    
    # Load all data for the city
    city_metadata_path = dataset_path / city / 'metadata.json'
    if not city_metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {city_metadata_path}")
    
    all_data = {'path': list(), 'policy_region': list(), 'scenario': list(), 'city': list()}
    load_metadata_from_disk(all_data, str(city_metadata_path))
    
    print(f"Loaded {len(all_data['path'])} total graphs from {city}")
    
    if len(all_data['path']) < train_count + val_count:
        raise ValueError(
            f"Insufficient data for city {city}: "
            f"Requested {train_count} train + {val_count} val = {train_count + val_count} graphs, "
            f"but only {len(all_data['path'])} available."
        )
    
    # Random split
    rng = random.Random(seed)
    indices = list(range(len(all_data['path'])))
    rng.shuffle(indices)
    
    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count + val_count]
    
    train_paths = [all_data['path'][i] for i in train_indices]
    val_paths = [all_data['path'][i] for i in val_indices]
    
    # Create split data structure
    split_data = {
        'city': city,
        'train_count': len(train_paths),
        'val_count': len(val_paths),
        'train_paths': train_paths,
        'val_paths': val_paths,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'train_data': {
            'path': train_paths,
            'policy_region': [all_data['policy_region'][i] for i in train_indices],
            'scenario': [all_data['scenario'][i] for i in train_indices],
            'city': [city] * len(train_paths)
        },
        'val_data': {
            'path': val_paths,
            'policy_region': [all_data['policy_region'][i] for i in val_indices],
            'scenario': [all_data['scenario'][i] for i in val_indices],
            'city': [city] * len(val_paths)
        }
    }
    
    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"✓ Saved random split to: {output_path}")
    print(f"  Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    return output_path


def generate_distant_test_set(
    city: str,
    train_val_split_file: Path,
    dataset_path: Path,
    test_count: int,
    output_path: Path,
) -> Path:
    """Generate test set distant from train+val sets."""
    print(f"\n{'=' * 80}")
    print(f"Generating distant test set for {city}")
    print(f"{'=' * 80}")
    
    # Load train/val split
    with open(train_val_split_file, 'r') as f:
        split_data = json.load(f)

    # Fix PATHS for Retina, splits were created in LRZ
    split_data['train_paths'] = [x.replace('/mnt/repo/','/home/rrao/development/gnn_predicting_effects_of_traffic_policies/') for x in split_data['train_paths']]
    split_data['val_paths'] = [x.replace('/mnt/repo/','/home/rrao/development/gnn_predicting_effects_of_traffic_policies/') for x in split_data['val_paths']]
    split_data['train_data']['path'] = [x.replace('/mnt/repo/','/home/rrao/development/gnn_predicting_effects_of_traffic_policies/') for x in split_data['train_data']['path']]
    split_data['val_data']['path'] = [x.replace('/mnt/repo/','/home/rrao/development/gnn_predicting_effects_of_traffic_policies/') for x in split_data['val_data']['path']]
    
    train_paths = split_data['train_paths']
    val_paths = split_data['val_paths']
    
    # Load all data for the city
    city_metadata_path = dataset_path / city / 'metadata.json'
    if not city_metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {city_metadata_path}")
    
    all_data = {'path': list(), 'policy_region': list(), 'scenario': list(), 'city': list()}
    load_metadata_from_disk(all_data, str(city_metadata_path))
    
    # Import the function from generate_distant_splits
    generate_script = Path(__file__).resolve().parents[1] / "analysis" / "generate_distant_splits.py"
    spec = importlib.util.spec_from_file_location("generate_distant_splits", generate_script)
    generate_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate_module)
    
    # Find distant test split
    test_paths, test_distances_when_picked, test_distances_from_train, test_distances_from_val = generate_module.find_distant_iou_test_split(train_paths,
                                                                                                                                             val_paths,
                                                                                                                                             all_data['path'],
                                                                                                                                             test_count)
    
    # Create test data structure
    # Create a mapping for efficient lookup
    path_to_index = {path: idx for idx, path in enumerate(all_data['path'])}
    
    # Verify all test paths are in all_data (they should be, but check for safety)
    missing_paths = [p for p in test_paths if p not in path_to_index]
    if missing_paths:
        raise ValueError(f"Some test paths are not in all_data: {len(missing_paths)} paths missing. "
                        f"This should not happen - test paths should be a subset of all_data['path'].")
    
    test_data = {
        'path': test_paths,
        'policy_region': [all_data['policy_region'][path_to_index[p]] for p in test_paths],
        'scenario': [all_data['scenario'][path_to_index[p]] for p in test_paths],
        'city': [city] * len(test_paths)
    }
    
    # Update split file with test data and distance information
    split_data['test_count'] = len(test_paths)
    split_data['test_paths'] = test_paths
    split_data['test_distances_when_picked'] = test_distances_when_picked
    split_data['test_distances_from_train'] = test_distances_from_train
    split_data['test_distances_from_val'] = test_distances_from_val
    split_data['test_data'] = test_data
    
    # Save updated split file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"✓ Saved test set to: {output_path}")
    print(f"  Test set size: {len(test_paths)}")
    print(f"  Minimum distance from train: {min(test_distances_from_train):.6f}")
    print(f"  Minimum distance from val: {min(test_distances_from_val):.6f}")
    
    return output_path


_EVAL_MODULE = None


def _load_eval_module():
    """Lazily load the evaluation helper module to reuse its checkpoint loader."""
    global _EVAL_MODULE
    if _EVAL_MODULE is None:
        eval_script = Path(__file__).resolve().parents[1] / "evaluation" / "evaluate_pretrained_on_cities.py"
        spec = importlib.util.spec_from_file_location("evaluate_pretrained_on_cities", eval_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _EVAL_MODULE = module
    return _EVAL_MODULE


def evaluate_model_on_test_split(
    run_name: str,
    project_name: str,
    model_path: Path,
    split_file_path: Path,
    eval_params: Dict[str, object],
) -> Optional[Dict[str, object]]:
    """Load a trained model and evaluate it on the test split defined in split_file_path."""
    if not split_file_path.exists():
        print(f"  ✗ Test split missing: {split_file_path}")
        return None
    if not model_path.exists():
        print(f"  ✗ Model file missing: {model_path}")
        return None

    with open(split_file_path, "r") as f:
        split_data = json.load(f)

    test_data = split_data.get("test_data") or {}
    if not test_data.get("path"):
        print(f"  ✗ Split file has no test_data paths: {split_file_path}")
        return None

    train_data = split_data.get("train_data") or {}
    val_data = split_data.get("val_data") or {}

    eval_module = _load_eval_module()

    # GPU selection (reuse training helper)
    set_cuda_visible_device(eval_params["device_nr"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = BASE_DIR / project_name / run_name
    model, inferred_config = eval_module.load_model_from_checkpoint(
        checkpoint_path=model_path,
        run_dir=run_dir,
        gnn_arch=eval_params["gnn_arch"],
        device=device,
    )
    model = model.to(device)

    # Build config object used by validate_model_during_training
    config_dict = {
        "target_type": eval_params["target_type"],
        "target_normalization": eval_params.get("target_normalization"),
        "use_all_features": eval_params["use_all_features"],
        "use_destination_activity": inferred_config.get("use_destination_activity", False),
    }
    # Prefer values inferred from checkpoint when present
    for key, value in inferred_config.items():
        config_dict.setdefault(key, value)
    config_obj = SimpleNamespace(**config_dict)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        _, _, test_loader = prepare_data_with_graph_features(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            use_inductive_variant=False,
            batch_size=eval_params["batch_size"],
            path_to_save_dataloader=str(tmp_dir) + "/",
            use_all_features=config_obj.use_all_features,
            use_weighted_batches=False,
            use_nested_neighbor_loader=False,
            neighbor_sizes=eval_params["neighbor_sizes"],
            subgraphs_per_graph=eval_params["subgraphs_per_graph"],
            seed_size=eval_params["seed_size"],
            sampling_strategy=eval_params["sampling_strategy"],
            min_subgraph_nodes=eval_params["min_subgraph_nodes"],
            max_subgraph_nodes=eval_params["max_subgraph_nodes"],
            aug_pos_rotation=False,
            aug_feature_noise=False,
            aug_node_masking_probability=0.0,
            use_destination_activity_param=config_obj.use_destination_activity,
            return_test_loader=True,
            x_scaler_path=run_dir / "data_created_during_finetuning" / "train_x_scaler.pkl") # Use a previously saved x_scaler

        if test_loader is None or len(test_loader) == 0:
            print(f"  ✗ No test graphs available for split: {split_file_path}")
            return None
        else:
            print(f"  ✓ Test loader ready — graphs: {len(test_data.get('path', []))}, batches: {len(test_loader)}")

        loss_func = GNN_Loss(loss_fct="mse", device=device, weighted=False)
        loss, r2, spearman, pearson, hit_rates = validate_model_during_training(
            config=config_obj,
            model=model,
            dataset=test_loader,
            loss_func=loss_func,
            device=device,
        )

    metrics = {
        "run_name": run_name,
        "project_name": project_name,
        "model_path": str(model_path),
        "split_file": str(split_file_path),
        "city": split_data.get("city"),
        "test_graphs": len(test_data.get("path", [])),
        "loss": float(loss),
        "r2": float(r2),
        "spearman": float(spearman),
        "pearson": float(pearson),
        "hit_rates": {k: float(v) for k, v in (hit_rates or {}).items()},
    }

    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split_file_path.stem}_metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  ✓ Saved test metrics to: {output_path}")
    return metrics


def select_cities(selected: Optional[str]) -> List[str]:
    if not selected:
        return list(TARGET_CITIES)
    requested = [city.strip() for city in selected.split(",") if city.strip()]
    unknown = [city for city in requested if city not in TARGET_CITIES]
    if unknown:
        raise ValueError(f"Unknown cities requested: {unknown}")
    return requested


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
        splits_dir = Path(args.splits_dir) if args.splits_dir else project_root / "data/splits"
    splits_dir = splits_dir.resolve()
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    neighbor_sizes = parse_neighbor_sizes(args.neighbor_sizes)
    seeds = [args.shuffle_seed + i for i in range(args.num_random_seeds)]

    # Prepare base arguments
    run_base_args: Dict[str, object] = {
        name: getattr(args, name) for name in RUN_MODELS_ARGS if hasattr(args, name)
    }
    finetune_base_args: Dict[str, object] = {
        name: getattr(args, name) for name in FINETUNE_ARGS if hasattr(args, name)
    }

    run_base_args["neighbor_sizes"] = neighbor_sizes
    finetune_base_args["neighbor_sizes"] = neighbor_sizes
    run_base_args["limit_available_graphs"] = args.run_limit_available_graphs
    
    if args.run_peak_lr is not None:
        run_base_args["peak_lr"] = args.run_peak_lr
    if args.run_initial_lr is not None:
        run_base_args["initial_lr"] = args.run_initial_lr
    if args.run_num_epochs is not None:
        run_base_args["num_epochs"] = args.run_num_epochs

    if args.finetune_peak_lr is not None:
        finetune_base_args["peak_lr"] = args.finetune_peak_lr
    if args.finetune_initial_lr is not None:
        finetune_base_args["initial_lr"] = args.finetune_initial_lr
    if args.finetune_num_epochs is not None:
        finetune_base_args["num_epochs"] = args.finetune_num_epochs

    # Avoid redundant dataloader caps; splits already define exact graphs
    finetune_base_args["limit_train_graphs"] = None
    finetune_base_args["limit_val_graphs"] = None
    finetune_base_args["limit_test_graphs"] = None

    if finetune_base_args.get("pretraining_inductive") is None:
        finetune_base_args["pretraining_inductive"] = run_base_args.get("use_inductive_variant", False)

    eval_params = {
        "gnn_arch": args.gnn_arch,
        "use_all_features": args.use_all_features,
        "target_type": args.target_type,
        "target_normalization": finetune_base_args.get("target_normalization"),
        "batch_size": args.batch_size,
        "neighbor_sizes": neighbor_sizes,
        "subgraphs_per_graph": args.subgraphs_per_graph,
        "seed_size": args.seed_size,
        "sampling_strategy": args.sampling_strategy,
        "min_subgraph_nodes": args.min_subgraph_nodes,
        "max_subgraph_nodes": args.max_subgraph_nodes,
        "device_nr": args.device_nr,
    }

    # Set project name
    if args.project_name:
        run_base_args["project_name"] = args.project_name
        finetune_base_args["project_name"] = args.project_name
    else:
        default_project = "GNN_Inductive" if args.use_inductive_variant else "GNN_Transductive"
        run_base_args["project_name"] = default_project
        finetune_base_args["project_name"] = default_project

    # Process each city
    test_count = args.test_count
    train_val_configs = parse_train_val_configs(args.train_val_configs)

    for test_city in select_cities(args.testing_cities):
        remaining_cities = [city for city in select_cities(args.selected_cities) if city != test_city]
        
        # Transductive pretraining: use all remaining cities for train
        train_cities = remaining_cities
        val_cities = []
        
        pretrain_run_name = f"{test_city}_pretrain_without_{test_city}"
        project_name = run_base_args["project_name"]
        
        print("\n" + "=" * 80)
        print(f"Processing city: {test_city}")
        print(f"{'=' * 80}")
        print(f"Pretraining cities ({len(train_cities)}): {train_cities}")
        print(f"Pretraining run name: {pretrain_run_name}")
        print("=" * 80)

        # Step 1: Pretraining
        run_args = dict(run_base_args)
        if not args.skip_pretraining:
            existing_pretrain = pretrain_model_path(project_name, pretrain_run_name)
            pretrain_ckpt_dir = pretrain_checkpoint_dir(project_name, pretrain_run_name)
            if existing_pretrain.exists() and has_checkpoints(pretrain_ckpt_dir):
                print(f"\n[Step 1/6] Skipping pretraining; found existing model and checkpoints: {existing_pretrain}")
            else:
                run_args["unique_model_description"] = pretrain_run_name
                
                print(f"\n[Step 1/6] Pretraining on {len(train_cities)} cities (excluding {test_city})...")
                call_run_models(run_args, train_cities, val_cities, [test_city])
        else:
            print(f"\n[Step 1/6] Skipping pretraining (--skip_pretraining)")

        for seed_idx, seed in enumerate(seeds, start=1):
            print("\n" + "-" * 80)
            print(f" Seed {seed_idx}/{len(seeds)} (value={seed}) for city {test_city}")
            print("-" * 80)

            for train_count, val_count in train_val_configs:
                print("\n" + "." * 80)
                print(f"  Train/Val config: train={train_count}, val={val_count}")
                print("." * 80)

                # Step 2: Generate random train/val split per seed and config
                city_config_split_dir = splits_dir / test_city / f"rs_{seed_idx}" / f"t{train_count}_v{val_count}"
                city_config_split_dir.mkdir(parents=True, exist_ok=True)
                split_filename = (
                    f"{test_city}_rs{seed_idx}_t{train_count}_v{val_count}_seed{seed}_train{train_count}_val{val_count}_random.json"
                )
                split_file_path = city_config_split_dir / split_filename

                if split_file_path.exists():
                    print(f"\n[Step 2/6] Using existing split file: {split_file_path}")
                else:
                    print(f"\n[Step 2/6] Generating random train/val split...")
                    generate_random_train_val_split(
                        test_city,
                        dataset_path,
                        train_count,
                        val_count,
                        seed,
                        split_file_path
                    )

                scratch_run_name = f"{test_city}_scratch_rs_{seed_idx}_t{train_count}_v{val_count}"
                finetune_run_name = f"{test_city}_finetune_rs_{seed_idx}_t{train_count}_v{val_count}"

                shared_values = {name: run_args.get(name, getattr(args, name, None)) for name in SHARED_ARG_NAMES}
                
                # Step 3: Finetuning from scratch
                if not args.skip_scratch:
                    scratch_model_file = finetuned_model_path(project_name, scratch_run_name)
                    if scratch_model_file.exists():
                        print(f"\n[Step 3/6] Skipping finetuning from scratch; found existing model: {scratch_model_file}")
                    else:
                        print(f"\n[Step 3/6] Running finetuning from scratch...")
                        finetune_args_scratch = dict(finetune_base_args)
                        for key, value in shared_values.items():
                            if key not in finetune_args_scratch or finetune_args_scratch[key] is None:
                                finetune_args_scratch[key] = value
                        if finetune_args_scratch.get("pretraining_inductive") is None:
                            finetune_args_scratch["pretraining_inductive"] = run_args.get("use_inductive_variant", False)

                        finetune_args_scratch["run_name"] = scratch_run_name
                        finetune_args_scratch["cities"] = test_city
                        finetune_args_scratch["start_from_scratch"] = True
                        finetune_args_scratch["unique_model_description"] = scratch_run_name
                        finetune_args_scratch["split_file"] = str(split_file_path)

                        try:
                            call_finetune_models(finetune_args_scratch)
                        except ValueError as e:
                            if "Insufficient data" in str(e):
                                print(f"SKIPPING {test_city} (scratch, seed {seed_idx}, t{train_count}_v{val_count}): {e}")
                                continue
                            else:
                                raise
                else:
                    print(f"\n[Step 3/6] Skipping training from scratch (--skip_scratch)")

                # Step 4: Finetuning from checkpoint
                if not args.skip_finetuning:
                    finetuned_checkpoint_model = finetuned_model_path(project_name, finetune_run_name)
                    if finetuned_checkpoint_model.exists():
                        print(f"\n[Step 4/6] Skipping finetuning from checkpoint; found existing model: {finetuned_checkpoint_model}")
                    else:
                        pretrain_ckpt_dir = pretrain_checkpoint_dir(project_name, pretrain_run_name)
                        if not has_checkpoints(pretrain_ckpt_dir):
                            print(f"\n[Step 4/6] Pretraining checkpoints missing for {pretrain_run_name}; rerunning pretraining first.")
                            run_args["unique_model_description"] = pretrain_run_name
                            call_run_models(run_args, train_cities, val_cities, [test_city])
                            if not has_checkpoints(pretrain_ckpt_dir):
                                raise ValueError(f"Checkpoint directory still missing after rerun: {pretrain_ckpt_dir}")

                        print(f"\n[Step 4/6] Running finetuning from checkpoint...")
                        finetune_args_checkpoint = dict(finetune_base_args)
                        for key, value in shared_values.items():
                            if key not in finetune_args_checkpoint or finetune_args_checkpoint[key] is None:
                                finetune_args_checkpoint[key] = value
                        if finetune_args_checkpoint.get("pretraining_inductive") is None:
                            finetune_args_checkpoint["pretraining_inductive"] = run_args.get("use_inductive_variant", False)

                        # run_name is for saving the finetuned model, pretrain_run_name is for loading the checkpoint
                        finetune_args_checkpoint["run_name"] = finetune_run_name
                        finetune_args_checkpoint["pretrain_run_name"] = pretrain_run_name
                        finetune_args_checkpoint["cities"] = test_city
                        finetune_args_checkpoint["start_from_scratch"] = False
                        finetune_args_checkpoint["unique_model_description"] = finetune_run_name
                        finetune_args_checkpoint["split_file"] = str(split_file_path)

                        try:
                            call_finetune_models(finetune_args_checkpoint)
                            print(f"✓ Finetuning from checkpoint completed for {test_city} (seed {seed_idx}, t{train_count}_v{val_count})")
                        except ValueError as e:
                            if "Insufficient data" in str(e):
                                print(f"SKIPPING {test_city} (finetune, seed {seed_idx}, t{train_count}_v{val_count}): {e}")
                                continue
                            else:
                                raise
                else:
                    print(f"\n[Step 4/6] Skipping finetuning (--skip_finetuning)")

                # Step 5: Generate or reuse distant test set per seed/config
                test_split_filename = (
                    f"{test_city}_rs{seed_idx}_t{train_count}_v{val_count}_seed{seed}_train{train_count}_val{val_count}_test{test_count}_distant_iou.json"
                )
                test_split_file_path = city_config_split_dir / test_split_filename
                test_split_exists = test_split_file_path.exists()

                if not args.skip_test_generation:
                    if test_split_exists and not args.force_test_regeneration:
                        print(f"\n[Step 5/6] Test split already exists; reusing without regeneration: {test_split_file_path}")
                    else:
                        print(f"\n[Step 5/6] Generating distant test set...")
                        try:
                            generate_distant_test_set(
                                test_city,
                                split_file_path,
                                dataset_path,
                                test_count,
                                test_split_file_path)
                            test_split_exists = True
                        except Exception as e:
                            print(f"ERROR generating test set for {test_city} (seed {seed_idx}, t{train_count}_v{val_count}): {e}")
                            import traceback
                            traceback.print_exc()
                else:
                    if test_split_exists:
                        print(f"\n[Step 5/6] Skipping test generation (--skip_test_generation); existing split will be used: {test_split_file_path}")
                    else:
                        print(f"\n[Step 5/6] Skipping test generation (--skip_test_generation) and no split exists; evaluation will be skipped.")

                # Step 6: Evaluate scratch and finetune models on test set (if available)
                if not test_split_exists or not test_split_file_path.exists():
                    print(f"\n[Step 6/6] Skipping evaluation; test split unavailable for {test_city} (seed {seed_idx}, t{train_count}_v{val_count}).")
                    print(f"\n✓ Completed processing for {test_city} seed {seed_idx} t{train_count}_v{val_count}")
                    print("." * 80)
                    continue

                print(f"\n[Step 6/6] Evaluating models on test split: {test_split_file_path}")

                scratch_metrics = evaluate_model_on_test_split(
                    run_name=scratch_run_name,
                    project_name=project_name,
                    model_path=finetuned_model_path(project_name, scratch_run_name),
                    split_file_path=test_split_file_path,
                    eval_params=eval_params,
                )

                finetune_metrics = evaluate_model_on_test_split(
                    run_name=finetune_run_name,
                    project_name=project_name,
                    model_path=finetuned_model_path(project_name, finetune_run_name),
                    split_file_path=test_split_file_path,
                    eval_params=eval_params,
                )

                if scratch_metrics or finetune_metrics:
                    print("  Evaluation results (loss, r2, spearman, pearson):")
                    if scratch_metrics:
                        print(f"    Scratch: loss={scratch_metrics['loss']:.4f}, r2={scratch_metrics['r2']:.4f}, "
                              f"spearman={scratch_metrics['spearman']:.4f}, pearson={scratch_metrics['pearson']:.4f}")
                        if scratch_metrics.get("hit_rates"):
                            print(f"      Scratch hit rates: {scratch_metrics['hit_rates']}")
                    else:
                        print("    Scratch: not evaluated (missing model or data)")

                    if finetune_metrics:
                        print(f"    Finetune: loss={finetune_metrics['loss']:.4f}, r2={finetune_metrics['r2']:.4f}, "
                              f"spearman={finetune_metrics['spearman']:.4f}, pearson={finetune_metrics['pearson']:.4f}")
                        if finetune_metrics.get("hit_rates"):
                            print(f"      Finetune hit rates: {finetune_metrics['hit_rates']}")
                    else:
                        print("    Finetune: not evaluated (missing model or data)")
                else:
                    print("  No evaluation results available (models or data missing).")

                print(f"\n✓ Completed processing for {test_city} seed {seed_idx} t{train_count}_v{val_count}")
                print("." * 80)

        print(f"\n✓ Completed processing for {test_city}")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("All cities processed!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Evaluate models on test sets")
    print("2. Compare results between scratch and finetune runs")
    print("=" * 80)


if __name__ == "__main__":
    main()

