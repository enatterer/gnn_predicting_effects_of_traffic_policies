#!/usr/bin/env python3
"""
Run finetuning and from-scratch experiments for specified cities.

For each city in the target list, this script:
  1. Runs finetuning from a pretrained checkpoint
  2. Runs training from scratch (no checkpoint)

For each run type, it loops over different numbers of training graphs (25, 50, 100)
while keeping validation graphs fixed (default: 2, configurable via --val_graph_count).

Example usage:
    python run_finetuning_experiments.py --gnn_arch trans_encoder --num_epochs 200
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add the 'scripts' directory to Python Path
scripts_path = Path(__file__).resolve().parents[1]
if str(scripts_path) not in sys.path:
    sys.path.append(str(scripts_path))

from training.help_functions import str_to_bool

# Target cities for finetuning/from-scratch experiments
TARGET_CITIES = ['erlangen', 'bamberg', 'neuulm', 'muenchen']
# TARGET_CITIES = ['muenchen']

# Number of training graphs to loop over
TRAIN_GRAPH_COUNTS = [10]

# Number of validation graphs (fixed)
VAL_GRAPH_COUNT = 100

# Pretrained checkpoint configuration
# Note: The checkpoint path structure is: base_dir/project_name/run_name/trained_model/checkpoints/
# User specified checkpoint path: /home/enatterer/Development/elena_gnn_predicting_effects_of_traffic_policies/data/inductive_gnn_data_results/transductive/Bavaria_Test
# Checkpoint location: base_dir/Bavaria_Test/general_surrogate_v0/trained_model/checkpoints/
# Note: finetune_models.py uses base_dir = project_root/inductive_gnn_data_results/transductive
# A symlink has been created: inductive_gnn_data_results/transductive/Bavaria_Test -> data/inductive_gnn_data_results/transductive/Bavaria_Test
# This allows finetune_models.py to find the checkpoint at the expected location.
PRETRAIN_RUN_NAME = "general_surrogate_v0"  # Run name for the pretrained model
PRETRAIN_PROJECT_NAME = "Bavaria_Test"  # Project name for the pretrained model

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with shared parameters for finetuning."""
    parser = argparse.ArgumentParser(
        description="Run finetuning and from-scratch experiments for target cities."
    )
    
    # Required arguments
    parser.add_argument("--gnn_arch", type=str, required=True,
                        help="The GNN architecture to use (must match the pretrained model).",
                        choices=["point_net_transf_gat", "gat", "gatv2", "gatv3", "gcn", "gcn2", 
                                "trans_conv", "pnc", "fc_nn", "graphSAGE", "eign", "xgboost", "trans_encoder"])
    
    # Optional arguments with defaults
    parser.add_argument("--project_name", type=str, default=PRETRAIN_PROJECT_NAME,
                        help=f"Project name for the pretrained model (default: {PRETRAIN_PROJECT_NAME}).")
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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
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
    parser.add_argument("--pretrain_run_name", type=str, default=PRETRAIN_RUN_NAME,
                        help=f"Name of the pretrained run (default: {PRETRAIN_RUN_NAME}).")
    parser.add_argument("--target_cities", type=str, default=None,
                        help="Comma-separated list of cities to run. Defaults to: " + ", ".join(TARGET_CITIES))
    parser.add_argument("--train_graph_counts", type=str, default=None,
                        help=f"Comma-separated list of training graph counts. Defaults to: {','.join(map(str, TRAIN_GRAPH_COUNTS))}")
    parser.add_argument("--val_graph_count", type=int, default=VAL_GRAPH_COUNT,
                        help=f"Number of validation graphs (default: {VAL_GRAPH_COUNT}).")
    
    # Split file configuration
    parser.add_argument("--splits_dir", type=str, default="data/splits",
                        help="Directory containing pre-generated split JSON files. If provided, will use split files instead of random splitting.")
    parser.add_argument("--use_distant_splits", type=str_to_bool, default=False,
                        help="If True, use pre-generated distant splits from splits_dir instead of random splits.")
    
    return parser


def build_finetune_command(args, city: str, train_graphs: int, val_graphs: int, 
                          start_from_scratch: bool, split_file: str = None) -> list:
    """Build the command to run finetune_models.py."""
    script_path = Path(__file__).parent / "finetune_models.py"
    
    # For finetuning: use unique run_name for saving, pass pretrain_run_name for checkpoint loading
    # For scratch runs: use unique run_name (no checkpoint needed)
    if start_from_scratch:
        run_name = f"scratch_{city}_train{train_graphs}_val{val_graphs}_high_dist_train_val"
    else:
        # For finetuning: use unique run_name per configuration to avoid overwriting
        run_name = f"finetuned_{city}_train{train_graphs}_val{val_graphs}_high_dist_train_val"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--run_name", run_name,
        "--gnn_arch", args.gnn_arch,
        "--cities", city,
        "--project_name", args.project_name,
    ]
    
    # For finetuning runs, add pretrain_run_name so checkpoint can be loaded from the correct location
    if not start_from_scratch:
        cmd.extend(["--pretrain_run_name", args.pretrain_run_name])
    
    cmd.extend([
        "--pretraining_inductive", str(args.pretraining_inductive),
        "--in_channels", str(args.in_channels),
        "--use_all_features", str(args.use_all_features),
        "--use_destination_activity", str(args.use_destination_activity),
        "--out_channels", str(args.out_channels),
        "--loss_fct", args.loss_fct,
        "--use_weighted_loss", str(args.use_weighted_loss),
        "--target_type", args.target_type,
        "--num_epochs", str(args.num_epochs),
        "--batch_size", str(args.batch_size),
        "--peak_lr", str(args.peak_lr),
        "--initial_lr", str(args.initial_lr),
        "--warmup_fraction", str(args.warmup_fraction),
        "--cosine_decay_rate", str(args.cosine_decay_rate),
        "--min_lr_fraction", str(args.min_lr_fraction),
        "--early_stopping_patience", str(args.early_stopping_patience),
        "--use_dropout", str(args.use_dropout),
        "--dropout", str(args.dropout),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--use_gradient_clipping", str(args.use_gradient_clipping),
        "--device_nr", str(args.device_nr),
        "--use_nested_neighbor_loader", str(args.use_nested_neighbor_loader),
        "--neighbor_sizes", args.neighbor_sizes,
        "--subgraphs_per_graph", str(args.subgraphs_per_graph),
        "--seed_size", str(args.seed_size),
        "--sampling_strategy", args.sampling_strategy,
        "--min_subgraph_nodes", str(args.min_subgraph_nodes),
        "--max_subgraph_nodes", str(args.max_subgraph_nodes),
        "--use_data_augmentation", str(args.use_data_augmentation),
        "--use_message_dropout_probability", str(args.use_message_dropout_probability),
        "--augment_feature_noise_prob", str(args.augment_feature_noise_prob),
        "--use_node_masking_probability", str(args.use_node_masking_probability),
        "--limit_train_graphs", str(train_graphs),
        "--limit_val_graphs", str(val_graphs),
        "--limit_test_graphs", "0",
        "--start_from_scratch", str(start_from_scratch),
        "--unique_model_description", run_name,  # Pass run_name so it's used directly without duplication
    ])
    
    # Add split file if provided
    if split_file:
        cmd.extend(["--split_file", split_file])
    
    return cmd


def main():
    parser = create_parser()
    args = parser.parse_args()
    
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
    
    # Determine if we should use pre-generated splits
    splits_dir = None
    if args.use_distant_splits:
        if not os.path.isabs(args.splits_dir):
            splits_dir = Path(__file__).resolve().parents[2] / args.splits_dir
        else:
            splits_dir = Path(args.splits_dir)
        splits_dir = splits_dir.resolve()
        if not splits_dir.exists():
            print(f"Warning: Splits directory does not exist: {splits_dir}")
            print("  Will generate splits on-the-fly or use random splits.")
            splits_dir = None
    
    print("=" * 80)
    print("Finetuning and From-Scratch Experiments")
    print("=" * 80)
    print(f"Target cities: {target_cities}")
    print(f"Training graph counts: {train_graph_counts}")
    print(f"Validation graph count: {val_graph_count}")
    print(f"Pretrained run name: {args.pretrain_run_name}")
    print(f"Project name: {args.project_name}")
    print(f"GNN architecture: {args.gnn_arch}")
    if splits_dir:
        print(f"Using pre-generated distant splits from: {splits_dir}")
    else:
        print("Using random splits (default behavior)")
    print("=" * 80)
    
    total_runs = len(target_cities) * len(train_graph_counts) * 2  # 2 = finetune + scratch
    run_counter = 0
    
    for city_idx, city in enumerate(target_cities, 1):
        print(f"\n{'=' * 80}")
        print(f"[City {city_idx}/{len(target_cities)}] Processing city: {city}")
        print(f"{'=' * 80}")
        
        for train_graphs in train_graph_counts:
            print(f"\n{'-' * 80}")
            print(f"Training graphs: {train_graphs}, Validation graphs: {val_graph_count}")
            print(f"{'-' * 80}")
            
            # Check if we have a pre-generated split file
            split_file = None
            if splits_dir:
                split_filename = f"{city}_train{train_graphs}_val{val_graph_count}_distant.json"
                split_file_path = splits_dir / split_filename
                if split_file_path.exists():
                    split_file = str(split_file_path)
                    print(f"Using pre-generated split file: {split_file}")
                else:
                    print(f"Warning: Split file not found: {split_file_path}")
                    print("  Will use random splitting instead.")
            
            # Run finetuning (from checkpoint)
            run_counter += 1
            print(f"\n[{run_counter}/{total_runs}] Running FINETUNING for {city} "
                  f"(train={train_graphs}, val={val_graph_count})")
            print(f"Command: finetune_models.py --cities {city} --limit_train_graphs {train_graphs} "
                  f"--limit_val_graphs {val_graph_count} --start_from_scratch False")
            if split_file:
                print(f"  Using split file: {split_file}")
            
            cmd_finetune = build_finetune_command(
                args, city, train_graphs, val_graph_count, start_from_scratch=False, split_file=split_file
            )
            
            try:
                result = subprocess.run(cmd_finetune, check=True, capture_output=False)
                print(f"✓ Finetuning completed successfully for {city} (train={train_graphs})")
            except subprocess.CalledProcessError as e:
                print(f"✗ Finetuning failed for {city} (train={train_graphs}): {e}")
                print("Continuing to next run...")
                continue
            except ValueError as e:
                if "Insufficient data" in str(e):
                    print(f"✗ Skipping {city} (train={train_graphs}): {e}")
                    continue
                else:
                    raise
            
            # Run from scratch (no checkpoint)
            run_counter += 1
            print(f"\n[{run_counter}/{total_runs}] Running FROM SCRATCH for {city} "
                  f"(train={train_graphs}, val={val_graph_count})")
            print(f"Command: finetune_models.py --cities {city} --limit_train_graphs {train_graphs} "
                  f"--limit_val_graphs {val_graph_count} --start_from_scratch True")
            if split_file:
                print(f"  Using split file: {split_file}")
            
            cmd_scratch = build_finetune_command(
                args, city, train_graphs, val_graph_count, start_from_scratch=True, split_file=split_file
            )
            
            try:
                result = subprocess.run(cmd_scratch, check=True, capture_output=False)
                print(f"✓ From-scratch training completed successfully for {city} (train={train_graphs})")
            except subprocess.CalledProcessError as e:
                print(f"✗ From-scratch training failed for {city} (train={train_graphs}): {e}")
                print("Continuing to next run...")
                continue
            except ValueError as e:
                if "Insufficient data" in str(e):
                    print(f"✗ Skipping {city} (train={train_graphs}): {e}")
                    continue
                else:
                    raise
    
    print("\n" + "=" * 80)
    print("All experiments completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()

