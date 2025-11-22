#!/usr/bin/env python3
"""
Generate distant train/val splits and run finetuning experiments with them.

This script:
1. Generates distant splits for all target cities and training graph counts
2. Runs finetuning experiments using those distant splits

Usage:
    python scripts/training/generate_and_run_distant_splits.py \
        --gnn_arch trans_encoder \
        --num_epochs 200 \
        --num_trials 1000
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add the 'scripts' directory to Python Path
scripts_path = Path(__file__).resolve().parents[1]
if str(scripts_path) not in sys.path:
    sys.path.append(str(scripts_path))

from training.help_functions import str_to_bool

# Target cities for finetuning/from-scratch experiments
TARGET_CITIES = ['erlangen', 'bamberg', 'muenchen', 'neuulm']

# Number of training graphs to loop over
TRAIN_GRAPH_COUNTS = [10]

# Number of validation graphs (fixed)
VAL_GRAPH_COUNT = 100

# Number of trials for finding distant splits
NUM_TRIALS = 1000

# Output directory for splits
SPLITS_DIR = "data/splits"


def generate_all_splits(target_cities, train_graph_counts, val_graph_count, 
                        splits_dir, num_trials, dataset_path=None, use_all_features=True):
    """Generate distant splits for all city/train_count combinations."""
    generate_script = Path(__file__).resolve().parents[1] / "analysis" / "generate_distant_splits.py"
    
    print("=" * 80)
    print("Step 1: Generating Distant Splits")
    print("=" * 80)
    
    split_files = {}
    
    for city in target_cities:
        for train_count in train_graph_counts:
            print(f"\n{'=' * 80}")
            print(f"Generating split for {city}, train={train_count}, val={val_graph_count}")
            print(f"{'=' * 80}")
            
            split_filename = f"{city}_train{train_count}_val{val_graph_count}_distant.json"
            split_file_path = Path(splits_dir) / split_filename
            
            # Check if split file already exists
            if split_file_path.exists():
                print(f"  Split file already exists: {split_file_path}")
                print(f"  Skipping generation. Delete the file to regenerate.")
                split_files[(city, train_count)] = str(split_file_path)
                continue
            
            # Build command to generate split
            cmd = [
                sys.executable,
                str(generate_script),
                "--city", city,
                "--train_count", str(train_count),
                "--val_count", str(val_graph_count),
                "--output_dir", str(splits_dir),
                "--num_trials", str(num_trials),
                "--use_all_features", str(use_all_features),
            ]
            
            if dataset_path:
                cmd.extend(["--dataset_path", str(dataset_path)])
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=False)
                split_files[(city, train_count)] = str(split_file_path)
                print(f"  ✓ Generated split: {split_file_path}")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to generate split for {city}, train={train_count}: {e}")
                continue
    
    print(f"\n{'=' * 80}")
    print(f"Generated {len(split_files)} split files")
    print(f"{'=' * 80}")
    
    return split_files


def run_finetuning_experiments(args, splits_dir):
    """Run finetuning experiments using pre-generated splits."""
    run_script = Path(__file__).parent / "run_finetuning_experiments.py"
    
    print("\n" + "=" * 80)
    print("Step 2: Running Finetuning Experiments with Distant Splits")
    print("=" * 80)
    
    # Build command to run experiments
    cmd = [
        sys.executable,
        str(run_script),
        "--gnn_arch", args.gnn_arch,
        "--use_distant_splits", "True",
        "--splits_dir", str(splits_dir),
    ]
    
    # Add all other arguments from the original parser
    if args.project_name:
        cmd.extend(["--project_name", args.project_name])
    if args.pretraining_inductive is not None:
        cmd.extend(["--pretraining_inductive", str(args.pretraining_inductive)])
    cmd.extend(["--in_channels", str(args.in_channels)])
    cmd.extend(["--use_all_features", str(args.use_all_features)])
    cmd.extend(["--use_destination_activity", str(args.use_destination_activity)])
    cmd.extend(["--out_channels", str(args.out_channels)])
    cmd.extend(["--loss_fct", args.loss_fct])
    cmd.extend(["--use_weighted_loss", str(args.use_weighted_loss)])
    cmd.extend(["--target_type", args.target_type])
    cmd.extend(["--num_epochs", str(args.num_epochs)])
    cmd.extend(["--batch_size", str(args.batch_size)])
    cmd.extend(["--peak_lr", str(args.peak_lr)])
    cmd.extend(["--initial_lr", str(args.initial_lr)])
    cmd.extend(["--warmup_fraction", str(args.warmup_fraction)])
    cmd.extend(["--cosine_decay_rate", str(args.cosine_decay_rate)])
    cmd.extend(["--min_lr_fraction", str(args.min_lr_fraction)])
    cmd.extend(["--early_stopping_patience", str(args.early_stopping_patience)])
    cmd.extend(["--use_dropout", str(args.use_dropout)])
    cmd.extend(["--dropout", str(args.dropout)])
    cmd.extend(["--gradient_accumulation_steps", str(args.gradient_accumulation_steps)])
    cmd.extend(["--use_gradient_clipping", str(args.use_gradient_clipping)])
    cmd.extend(["--device_nr", str(args.device_nr)])
    
    # GraphSAGE parameters
    cmd.extend(["--use_nested_neighbor_loader", str(args.use_nested_neighbor_loader)])
    cmd.extend(["--neighbor_sizes", args.neighbor_sizes])
    cmd.extend(["--subgraphs_per_graph", str(args.subgraphs_per_graph)])
    cmd.extend(["--seed_size", str(args.seed_size)])
    cmd.extend(["--sampling_strategy", args.sampling_strategy])
    cmd.extend(["--min_subgraph_nodes", str(args.min_subgraph_nodes)])
    cmd.extend(["--max_subgraph_nodes", str(args.max_subgraph_nodes)])
    
    # Data augmentation parameters
    cmd.extend(["--use_data_augmentation", str(args.use_data_augmentation)])
    cmd.extend(["--use_message_dropout_probability", str(args.use_message_dropout_probability)])
    cmd.extend(["--augment_feature_noise_prob", str(args.augment_feature_noise_prob)])
    cmd.extend(["--use_node_masking_probability", str(args.use_node_masking_probability)])
    
    # Run configuration
    if args.pretrain_run_name:
        cmd.extend(["--pretrain_run_name", args.pretrain_run_name])
    if args.target_cities:
        cmd.extend(["--target_cities", args.target_cities])
    if args.train_graph_counts:
        cmd.extend(["--train_graph_counts", args.train_graph_counts])
    cmd.extend(["--val_graph_count", str(args.val_graph_count)])
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✓ All experiments completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiments failed: {e}")
        raise


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate distant splits and run finetuning experiments."
    )
    
    # Required arguments
    parser.add_argument("--gnn_arch", type=str, required=True,
                        help="The GNN architecture to use.",
                        choices=["point_net_transf_gat", "gat", "gatv2", "gatv3", "gcn", "gcn2", 
                                "trans_conv", "pnc", "fc_nn", "graphSAGE", "eign", "xgboost", "trans_encoder"])
    
    # Optional arguments (mostly passed through to run_finetuning_experiments.py)
    parser.add_argument("--project_name", type=str, default=None,
                        help="Project name for the pretrained model.")
    parser.add_argument("--pretraining_inductive", type=str_to_bool, default=False,
                        help="Whether the pretraining was inductive (True) or transductive (False).")
    parser.add_argument("--in_channels", type=int, default=5, help="The number of input channels.")
    parser.add_argument("--use_all_features", type=str_to_bool, default=True, 
                        help="Whether to use all features (True) or a subset (False).")
    parser.add_argument("--use_destination_activity", type=str_to_bool, default=False,
                        help="Whether to include destination/activity features (20-27). Default: False (excludes features 20-27 with NaNs).")
    parser.add_argument("--out_channels", type=int, default=1, help="The number of output channels.")
    parser.add_argument("--loss_fct", type=str, default="mse", help="The loss function to use.")
    parser.add_argument("--use_weighted_loss", type=str_to_bool, default=False, 
                        help="Whether to use weighted loss.")
    parser.add_argument("--target_type", type=str, default="abs_vol_car", 
                        help="Which target to use for training.")
    parser.add_argument("--num_epochs", type=int, default=600, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--peak_lr", type=float, default=0.001, 
                        help="The peak learning rate (after warmup).")
    parser.add_argument("--initial_lr", type=float, default=0.0005, 
                        help="The initial learning rate (used during warmup).")
    parser.add_argument("--warmup_fraction", type=float, default=0.1, 
                        help="Fraction of total training steps to use for linear warmup.")
    parser.add_argument("--cosine_decay_rate", type=float, default=0.5, 
                        help="The rate at which the learning rate decays after warmup.")
    parser.add_argument("--min_lr_fraction", type=float, default=0.01, 
                        help="The minimum learning rate fraction.")
    parser.add_argument("--early_stopping_patience", type=int, default=30, 
                        help="The early stopping patience.")
    parser.add_argument("--use_dropout", type=str_to_bool, default=False, 
                        help="Whether to use dropout.")
    parser.add_argument("--dropout", type=float, default=0.3, help="The dropout rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="After how many steps the gradient should be updated.")
    parser.add_argument("--use_gradient_clipping", type=str_to_bool, default=True, 
                        help="Whether to use gradient clipping.")
    parser.add_argument("--device_nr", type=int, default=0, 
                        help="The device number (0 or 1 for Retina Roaster's two GPUs).")
    
    # GraphSAGE parameters
    parser.add_argument("--use_nested_neighbor_loader", type=str_to_bool, default=False, 
                        help="Whether to use nested neighbor loader.")
    parser.add_argument("--neighbor_sizes", type=str, default="5,5,5", 
                        help="The neighbor sizes for the nested neighbor loader (comma-separated).")
    parser.add_argument("--subgraphs_per_graph", type=int, default=2, 
                        help="The number of subgraphs to sample per graph.")
    parser.add_argument("--seed_size", type=int, default=10, 
                        help="The number of seed nodes in each subgraph.")
    parser.add_argument("--sampling_strategy", type=str, default="neighbor_sampling", 
                        choices=["neighbor_sampling", "random_walk"],
                        help="The sampling strategy to use for the nested neighbor loader.")
    parser.add_argument("--min_subgraph_nodes", type=int, default=500, 
                        help="The minimum number of nodes in a subgraph.")
    parser.add_argument("--max_subgraph_nodes", type=int, default=50000, 
                        help="The maximum number of nodes in a subgraph.")
    
    # Data augmentation parameters
    parser.add_argument("--use_data_augmentation", type=str_to_bool, default=False, 
                        help="Whether to use data augmentation.")
    parser.add_argument("--use_message_dropout_probability", type=float, default=0.0, 
                        help="The probability of message dropout.")
    parser.add_argument("--augment_feature_noise_prob", type=str_to_bool, default=False, 
                        help="Whether to use Gaussian noise addition to node features.")
    parser.add_argument("--use_node_masking_probability", type=float, default=0.0, 
                        help="The probability of masking all features of a node to 0.")
    
    # Run configuration
    parser.add_argument("--pretrain_run_name", type=str, default=None,
                        help="Name of the pretrained run.")
    parser.add_argument("--target_cities", type=str, default=None,
                        help="Comma-separated list of cities to run. Defaults to: " + ", ".join(TARGET_CITIES))
    parser.add_argument("--train_graph_counts", type=str, default=None,
                        help=f"Comma-separated list of training graph counts. Defaults to: {','.join(map(str, TRAIN_GRAPH_COUNTS))}")
    parser.add_argument("--val_graph_count", type=int, default=VAL_GRAPH_COUNT,
                        help=f"Number of validation graphs (default: {VAL_GRAPH_COUNT}).")
    
    # Split generation parameters
    parser.add_argument("--num_trials", type=int, default=NUM_TRIALS,
                        help=f"Number of random splits to try when finding distant splits (default: {NUM_TRIALS}).")
    parser.add_argument("--splits_dir", type=str, default=SPLITS_DIR,
                        help=f"Directory to save/load split JSON files (default: {SPLITS_DIR}).")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to dataset directory. Defaults to data/bavaria/inductive_data/training_data/kreisfreistadt")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip split generation and only run experiments (assumes splits already exist).")
    parser.add_argument("--skip_experiments", action="store_true",
                        help="Skip experiments and only generate splits.")
    
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).resolve().parents[2]
    
    if args.splits_dir and not Path(args.splits_dir).is_absolute():
        splits_dir = project_root / args.splits_dir
    else:
        splits_dir = Path(args.splits_dir) if args.splits_dir else project_root / SPLITS_DIR
    splits_dir = splits_dir.resolve()
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse target cities
    if args.target_cities:
        target_cities = [c.strip() for c in args.target_cities.split(',') if c.strip()]
    else:
        target_cities = TARGET_CITIES
    
    # Parse training graph counts
    if args.train_graph_counts:
        train_graph_counts = [int(x.strip()) for x in args.train_graph_counts.split(',') if x.strip()]
    else:
        train_graph_counts = TRAIN_GRAPH_COUNTS
    
    val_graph_count = args.val_graph_count
    
    print("=" * 80)
    print("Generate Distant Splits and Run Experiments")
    print("=" * 80)
    print(f"Target cities: {target_cities}")
    print(f"Training graph counts: {train_graph_counts}")
    print(f"Validation graph count: {val_graph_count}")
    print(f"Splits directory: {splits_dir}")
    print(f"Number of trials for split generation: {args.num_trials}")
    print("=" * 80)
    
    # Step 1: Generate splits
    if not args.skip_generation:
        split_files = generate_all_splits(
            target_cities, 
            train_graph_counts, 
            val_graph_count,
            splits_dir,
            args.num_trials,
            dataset_path=args.dataset_path,
            use_all_features=args.use_all_features
        )
        
        if not split_files:
            print("ERROR: No split files were generated!")
            return
    else:
        print("\nSkipping split generation (--skip_generation flag set)")
    
    # Step 2: Run experiments
    if not args.skip_experiments:
        run_finetuning_experiments(args, splits_dir)
    else:
        print("\nSkipping experiments (--skip_experiments flag set)")
    
    print("\n" + "=" * 80)
    print("All tasks completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

