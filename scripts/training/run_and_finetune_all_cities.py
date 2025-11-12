#!/usr/bin/env python3
"""
Orchestrate sequential training and finetuning runs for each city.

Example:
    python run_and_finetune_all_cities.py --gnn_arch gatv2 --num_epochs 200

For every city in the global list, this script:
  1. Uses the city as test city for `run_models.py` while sampling an
     80/20 random split of the remaining cities into train/validation sets.
  2. Launches finetuning twice via `finetune_models.py`:
        - `run_and_finetune_{city}`: continue from the freshly trained checkpoint.
        - `finetune_{city}`: comparison run starting from scratch.

CLI arguments mirror those in `run_models.py` and `finetune_models.py`.
Shared arguments are passed with identical values to both scripts, while
arguments that exist only in one script are scoped accordingly.
"""

import argparse
import importlib
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Ensure the repository's `scripts` directory is on the Python path so that
# we can import `training.*` modules just like the standalone scripts do.
CURRENT_FILE = Path(__file__).resolve()
SCRIPTS_DIR = CURRENT_FILE.parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from training.help_functions import str_to_bool  # noqa: E402

# Global city pool used for all splits.
ALL_CITIES: Sequence[str] = (
    "aschaffenburg",
    "regensburg",
    "landshut",
    "bayreuth",
    "erlangen",
    "fuerth",
    "kempten",
    "neuulm",
    # "muenchen",
    "augsburg",
    "rosenheim",
    "schweinfurt",
    "bamberg",
    "nuernberg",
    "ingolstadt",
    "wuerzburg",
)

# Argument names for each script (matching their respective parsers).
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
    "limit_train_graphs",
    "limit_val_graphs",
    "limit_test_graphs",
]

FINETUNE_ARGS = [
    "run_name",
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
]

SHARED_ARG_NAMES = sorted(set(RUN_MODELS_ARGS) & set(FINETUNE_ARGS))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequentially run training and finetuning across all cities."
    )

    # ------------------------------------------------------------------
    # Arguments shared between run_models.py and finetune_models.py
    # ------------------------------------------------------------------
    parser.add_argument("--gnn_arch", type=str, default="trans_encoder")
    parser.add_argument("--in_channels", type=int, default=5)
    parser.add_argument("--use_all_features", type=str_to_bool, default=True)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--model_kwargs", type=str, default=None)
    parser.add_argument("--loss_fct", type=str, default="mse")
    parser.add_argument("--use_weighted_loss", type=str_to_bool, default=False)
    parser.add_argument("--target_type", type=str, default="abs_vol_car")
    parser.add_argument("--warmup_fraction", type=float, default=0.1)
    parser.add_argument("--cosine_decay_rate", type=float, default=0.5)
    parser.add_argument("--min_lr_fraction", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=40)
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

    # ------------------------------------------------------------------
    # Arguments specific to run_models.py
    # ------------------------------------------------------------------
    parser.add_argument("--use_inductive_variant", type=str_to_bool, default=True)
    parser.add_argument("--use_weighted_batches", type=str_to_bool, default=False)
    parser.add_argument("--use_target_standardization", type=str_to_bool, default=False)
    parser.add_argument("--use_city_balanced_loss", type=str_to_bool, default=False)
    parser.add_argument("--aug_pos_rotation", type=str_to_bool, default=False)
    parser.add_argument("--aug_feature_noise", type=str_to_bool, default=False)
    parser.add_argument("--aug_node_masking_probability", type=float, default=0.0)
    parser.add_argument("--continue_training", type=str_to_bool, default=False)
    parser.add_argument("--base_checkpoint_path", type=str, default=None)
    parser.add_argument("--run_peak_lr", type=float, default=0.002)
    parser.add_argument("--run_initial_lr", type=float, default=0.0005)
    parser.add_argument("--run_num_epochs", type=int, default=500)
    parser.add_argument("--run_limit_train_graphs", type=int, default=5000)
    parser.add_argument("--run_limit_val_graphs", type=int, default=0) # Small thing to know: the split into train/val/test is done in the run_models.py script, so we don't need to specify it here.
    parser.add_argument("--run_limit_test_graphs", type=int, default=0) # Small thing to know: the split into train/val/test is done in the run_models.py script, so we don't need to specify it here.

    # ------------------------------------------------------------------
    # Arguments specific to finetune_models.py
    # ------------------------------------------------------------------
    parser.add_argument("--target_normalization", type=str_to_bool, default=False)
    parser.add_argument("--predict_mode_stats", type=str_to_bool, default=False)
    parser.add_argument("--use_bootstrapping", type=str_to_bool, default=False)
    parser.add_argument("--use_weighted_sampling", type=str_to_bool, default=False)
    parser.add_argument("--use_data_augmentation", type=str_to_bool, default=False)
    parser.add_argument("--use_message_dropout_probability", type=float, default=0.0)
    parser.add_argument("--augment_feature_noise_prob", type=str_to_bool, default=False)
    parser.add_argument("--use_node_masking_probability", type=float, default=0.0)
    parser.add_argument("--start_from_scratch", type=str_to_bool, default=False)
    parser.add_argument("--pretraining_inductive", type=str_to_bool, default=None,
                        help="Set to True if finetuning should load inductive pretraining checkpoints, False for transductive. Defaults to the value used in run_models.")
    parser.add_argument("--finetune_peak_lr", type=float, default=0.001)
    parser.add_argument("--finetune_initial_lr", type=float, default=0.0005)
    parser.add_argument("--finetune_num_epochs", type=int, default=500)
    parser.add_argument("--finetune_limit_train_graphs", type=int, default=48)
    parser.add_argument("--finetune_limit_val_graphs", type=int, default=12) # Here, we do need to specify it. 
    parser.add_argument("--finetune_limit_test_graphs", type=int, default=0) # Here, we do need to specify it. 

    # ------------------------------------------------------------------
    # Orchestrator-specific arguments
    # ------------------------------------------------------------------
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.8,
        help="Fraction of remaining cities used for training (validation gets the rest).",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Seed controlling the random city splits.",
    )
    parser.add_argument(
        "--selected_cities",
        type=str,
        default=None,
        help="Optional comma-separated subset of cities to run. Defaults to all.",
    )
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


def validate_fractions(train_fraction: float) -> None:
    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be between 0 and 1 (exclusive).")


def select_cities(selected: Optional[str]) -> List[str]:
    if not selected:
        return list(ALL_CITIES)
    requested = [city.strip() for city in selected.split(",") if city.strip()]
    unknown = [city for city in requested if city not in ALL_CITIES]
    if unknown:
        raise ValueError(f"Unknown cities requested: {unknown}")
    return requested


def split_cities(
    rng: random.Random,
    available_cities: Sequence[str],
    train_fraction: float,
) -> Tuple[List[str], List[str]]:
    others = list(available_cities)
    rng.shuffle(others)
    total = len(others)
    if total == 0:
        raise ValueError("No remaining cities to split into train/validation.")

    num_train = max(1, int(round(train_fraction * total)))
    num_train = min(num_train, total - 1) if total > 1 else total

    train_split = others[:num_train]
    val_split = others[num_train:] if total > num_train else []
    if not val_split and total > 1:
        val_split = [train_split.pop()]
    return train_split, val_split


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    validate_fractions(args.train_fraction)
    neighbor_sizes = parse_neighbor_sizes(args.neighbor_sizes)

    city_sequence = select_cities(args.selected_cities)
    rng = random.Random(args.shuffle_seed)

    run_base_args: Dict[str, object] = {
        name: getattr(args, name) for name in RUN_MODELS_ARGS if hasattr(args, name)
    }
    finetune_base_args: Dict[str, object] = {
        name: getattr(args, name) for name in FINETUNE_ARGS if hasattr(args, name)
    }

    run_base_args["neighbor_sizes"] = neighbor_sizes
    finetune_base_args["neighbor_sizes"] = neighbor_sizes
    run_base_args["limit_train_graphs"] = (
        args.run_limit_train_graphs
        if args.run_limit_train_graphs is not None
        else getattr(args, "limit_train_graphs", 0)
    )
    run_base_args["limit_val_graphs"] = (
        args.run_limit_val_graphs
        if args.run_limit_val_graphs is not None
        else getattr(args, "limit_val_graphs", 0)
    )
    run_base_args["limit_test_graphs"] = (
        args.run_limit_test_graphs
        if args.run_limit_test_graphs is not None
        else getattr(args, "limit_test_graphs", 0)
    )

    finetune_base_args["limit_train_graphs"] = (
        args.finetune_limit_train_graphs
        if args.finetune_limit_train_graphs is not None
        else getattr(args, "limit_train_graphs", 0)
    )
    finetune_base_args["limit_val_graphs"] = (
        args.finetune_limit_val_graphs
        if args.finetune_limit_val_graphs is not None
        else getattr(args, "limit_val_graphs", 0)
    )
    finetune_base_args["limit_test_graphs"] = (
        args.finetune_limit_test_graphs
        if args.finetune_limit_test_graphs is not None
        else getattr(args, "limit_test_graphs", 0)
    )
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

    if finetune_base_args.get("pretraining_inductive") is None:
        finetune_base_args["pretraining_inductive"] = run_base_args.get("use_inductive_variant", True)

    for idx, test_city in enumerate(city_sequence, start=1):
        remaining_cities = [city for city in city_sequence if city != test_city]
        train_cities, val_cities = split_cities(rng, remaining_cities, args.train_fraction)

        pretrain_run_name = f"pretrain_without_{test_city}"
        finetune_run_name = f"finetune_{test_city}"
        scratch_run_name = f"run_from_scratch_{test_city}"

        print("=" * 80)
        print(f"[{idx}/{len(city_sequence)}] Test city: {test_city}")
        print(f"Train cities ({len(train_cities)}): {train_cities}")
        print(f"Validation cities ({len(val_cities)}): {val_cities}")
        print(f"Pretraining (without test city) run name: {pretrain_run_name}")
        print("=" * 80)

        run_args = dict(run_base_args)
        run_args["unique_model_description"] = pretrain_run_name

        call_run_models(run_args, train_cities, val_cities, [test_city])

        shared_values = {name: run_args.get(name, getattr(args, name, None)) for name in SHARED_ARG_NAMES}

        finetune_args_checkpoint = dict(finetune_base_args)
        for key, value in shared_values.items():
            if key not in finetune_args_checkpoint or finetune_args_checkpoint[key] is None:
                finetune_args_checkpoint[key] = value
        if finetune_args_checkpoint.get("pretraining_inductive") is None:
            finetune_args_checkpoint["pretraining_inductive"] = run_args.get("use_inductive_variant", True)
        finetune_args_checkpoint["run_name"] = pretrain_run_name
        finetune_args_checkpoint["cities"] = test_city
        finetune_args_checkpoint["start_from_scratch"] = False
        finetune_args_checkpoint["unique_model_description"] = finetune_run_name

        call_finetune_models(finetune_args_checkpoint)

        print("-" * 80)
        print(f"Comparison finetune from scratch: {scratch_run_name}")
        print("-" * 80)

        finetune_args_scratch = dict(finetune_base_args)
        for key, value in shared_values.items():
            if key not in finetune_args_scratch or finetune_args_scratch[key] is None:
                finetune_args_scratch[key] = value
        if finetune_args_scratch.get("pretraining_inductive") is None:
            finetune_args_scratch["pretraining_inductive"] = run_args.get("use_inductive_variant", True)
        finetune_args_scratch["run_name"] = scratch_run_name
        finetune_args_scratch["cities"] = test_city
        finetune_args_scratch["start_from_scratch"] = True
        finetune_args_scratch["unique_model_description"] = scratch_run_name

        call_finetune_models(finetune_args_scratch)


if __name__ == "__main__":
    main()

