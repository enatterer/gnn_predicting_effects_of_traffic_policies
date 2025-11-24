#!/usr/bin/env python3
"""
Evaluate hit rates for all existing model checkpoints.

This script:
1. Scans the results directory for all model checkpoints
2. Loads each checkpoint and evaluates on validation data
3. Computes hit rates (top 1%, bottom 1%, minus top 1%, top 5%, bottom 5%, minus top 5%)
4. Saves results to CSV/JSON for visualization

Usage:
    python scripts/evaluation/evaluate_hit_rates_for_existing_runs.py \
        --results_dir data/inductive_gnn_data_results/transductive/Bavaria_Test \
        --dataset_path data/bavaria/inductive_data/training_data/kreisfreistadt \
        --output_dir data/analysis_results \
        --device_nr 0
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add the 'scripts' directory to Python Path
scripts_path = Path(__file__).resolve().parents[1]
if str(scripts_path) not in sys.path:
    sys.path.append(str(scripts_path))

from gnn.help_functions import (
    compute_hit_rates,
    compute_r2_torch,
    compute_spearman_pearson,
    select_target_tensor,
    validate_model_during_training,
)
from training.help_functions import (
    create_gnn_model,
    get_available_gpus,
    load_metadata_from_disk,
    prepare_data_with_graph_features,
    select_best_gpu,
    set_cuda_visible_device,
    str_to_bool,
)


def extract_run_info(run_name: str) -> Optional[Dict[str, any]]:
    """
    Extract city, train_count, val_count, and method from run name.
    Handles runs with optional "_high_dist_train_val" suffix.
    
    Examples:
    - "finetuned_erlangen_train5_val200" -> {city: "erlangen", train: 5, val: 200, method: "finetuned"}
    - "scratch_erlangen_train5_val200" -> {city: "erlangen", train: 5, val: 200, method: "scratch"}
    - "finetuned_erlangen_train5_val200_high_dist_train_val" -> {city: "erlangen", train: 5, val: 200, method: "finetuned"}
    """
    # Pattern matches with optional _high_dist_train_val suffix
    pattern = r'(finetuned|scratch)_(\w+)_train(\d+)_val(\d+)(?:_high_dist_train_val)?$'
    match = re.match(pattern, run_name)
    
    if not match:
        return None
    
    method = match.group(1)
    city = match.group(2)
    train_count = int(match.group(3))
    val_count = int(match.group(4))
    
    return {
        'city': city,
        'train_count': train_count,
        'val_count': val_count,
        'method': method
    }


def find_checkpoints(results_dir: Path, wandb_csv_path: Optional[Path] = None) -> List[Dict]:
    """
    Find all model checkpoints in the results directory.
    Uses wandb CSV to determine which runs are finished (State == "finished").
    Only includes runs ending with "high_dist_train_val".
    
    Args:
        results_dir: Directory containing run directories
        wandb_csv_path: Path to wandb CSV file with run states
    
    Returns:
        List of dictionaries with checkpoint information:
        {
            'run_name': str,
            'checkpoint_path': Path,
            'run_info': dict (from extract_run_info),
            'run_dir': Path
        }
    """
    # Load finished runs from wandb CSV if provided
    finished_runs = set()
    if wandb_csv_path and wandb_csv_path.exists():
        try:
            df = pd.read_csv(wandb_csv_path)
            # Check for State column (case-insensitive)
            state_col = None
            name_col = None
            for col in df.columns:
                if col.lower() == 'state':
                    state_col = col
                if col.lower() in ['name', 'run_name']:
                    name_col = col
            
            if state_col and name_col:
                # Filter to finished runs ending with "high_dist_train_val"
                # Handle potential whitespace/quote issues
                finished_df = df[df[state_col].astype(str).str.strip().str.lower() == 'finished'].copy()
                for _, row in finished_df.iterrows():
                    run_name = str(row[name_col]).strip().strip('"').strip("'")
                    if run_name.endswith('high_dist_train_val'):
                        finished_runs.add(run_name)
                print(f"  Loaded {len(finished_runs)} finished runs from wandb CSV (ending with 'high_dist_train_val')")
            else:
                print(f"  Warning: Could not find 'State' or 'Name' column in wandb CSV")
        except Exception as e:
            print(f"  Warning: Could not read wandb CSV: {e}")
    
    checkpoints = []
    
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        run_name = run_dir.name
        
        # Only process runs ending with "high_dist_train_val"
        if not run_name.endswith('high_dist_train_val'):
            continue
        
        # Extract run info - must match pattern (finetuned|scratch)_city_trainX_valY_high_dist_train_val
        run_info = extract_run_info(run_name)
        if run_info is None:
            continue
        
        # Check if run is finished (from wandb CSV)
        if wandb_csv_path:
            # If we have wandb CSV but no finished runs found, skip all runs
            if not finished_runs:
                continue
            # Only process runs that are in the finished_runs set
            if run_name not in finished_runs:
                continue
        else:
            # If no wandb CSV provided, skip (we require wandb CSV for completion check)
            print(f"  Warning: No wandb CSV provided, skipping run {run_name}")
            continue
        
        # Find checkpoint (prefer model.pth, otherwise latest checkpoint)
        checkpoint_path = None
        
        # Check for model.pth first (the best model file)
        model_paths = [
            run_dir / 'finetuned_model' / 'model.pth',
            run_dir / 'trained_model' / 'model.pth',
        ]
        for model_path in model_paths:
            if model_path.exists():
                checkpoint_path = model_path
                break
        
        # If no model.pth, look for checkpoints
        if checkpoint_path is None:
            checkpoint_dirs = [
                run_dir / 'finetuned_model' / 'checkpoints',
                run_dir / 'trained_model' / 'checkpoints',
            ]
            for chk_dir in checkpoint_dirs:
                if not chk_dir.exists():
                    continue
                
                checkpoint_files = list(chk_dir.glob('checkpoint_epoch_*.pt'))
                if checkpoint_files:
                    def extract_epoch(filename):
                        match = re.search(r'checkpoint_epoch_(\d+)\.pt', filename.name)
                        return int(match.group(1)) if match else -1
                    
                    latest_checkpoint = max(checkpoint_files, key=extract_epoch)
                    checkpoint_path = latest_checkpoint
                    break
        
        # Only add if we found a checkpoint
        if checkpoint_path is not None and checkpoint_path.exists():
            checkpoints.append({
                'run_name': run_name,
                'checkpoint_path': checkpoint_path,
                'run_info': run_info,
                'run_dir': run_dir
            })
    
    return checkpoints


def load_model_from_checkpoint(
    checkpoint_path: Path,
    run_dir: Path,
    gnn_arch: str,
    device: torch.device
) -> Tuple[torch.nn.Module, Dict]:
    """
    Load model from checkpoint and infer configuration.
    
    Returns:
        (model, config_dict)
    """
    # Try to load config from run directory
    config_path = run_dir / 'config.json'
    config = {}
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Infer model config from checkpoint if needed
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Also check for target normalization in checkpoint
        if 'target_normalization' in checkpoint:
            config['target_normalization'] = checkpoint['target_normalization']
    else:
        state_dict = checkpoint
    
    # Infer model configuration based on architecture
    if gnn_arch == "trans_encoder":
        # Infer effective input channels from first graph conv layer
        if 'graph_convs.0.lin_key.weight' in state_dict:
            key_weight_shape = state_dict['graph_convs.0.lin_key.weight'].shape
            effective_in_channels = key_weight_shape[1]
            config['effective_in_channels'] = effective_in_channels
            
            # Infer ff_dim from transformer layers
            ff_dim = None
            if 'transformer.layers.0.linear1.weight' in state_dict:
                linear1_shape = state_dict['transformer.layers.0.linear1.weight'].shape
                ff_dim = linear1_shape[0]
                config['ff_dim'] = ff_dim
            else:
                # Fallback: try to get from config.json or use default
                ff_dim = config.get('ff_dim', 256)
                config['ff_dim'] = ff_dim
            
            # Infer feature configuration from effective_in_channels
            # effective_in_channels = node_features + pos_dim (if use_pos) + lap_pe_dim (if use_lap_pe)
            # Common patterns:
            # - 7 = 1 feature + 6 pos_dim (use_all_features=False, but only 1 feature used - unusual)
            # - 11 = 5 base features + 6 pos_dim (use_all_features=False, use_pos=True)
            # - 26 = 20 features + 6 pos_dim (use_all_features=True, use_pos=True, use_destination_activity=False)
            # - 27 = 21 features + 6 pos_dim (use_all_features=True, use_pos=True, use_destination_activity=True)
            # Override config.json values with inferred values (model weights are the source of truth)
            print(f"  Inferred effective_in_channels: {effective_in_channels}, ff_dim: {ff_dim}")
            
            # Infer pos_dim and use_pos (assume pos_dim=6 if use_pos, otherwise 0)
            # Try to infer from effective_in_channels
            pos_dim = 6  # Default assumption
            use_pos = True  # Default assumption
            use_lap_pe = False
            lap_pe_dim = 0
            
            # Calculate base in_channels from effective_in_channels
            # effective_in_channels = in_channels + pos_dim (if use_pos) + lap_pe_dim (if use_lap_pe)
            # Common patterns:
            # - 7 = 7 features without pos (use_pos=False) OR 1 feature + 6 pos (unusual)
            # - 11 = 5 base features + 6 pos_dim (use_all_features=False, use_pos=True)
            # - 26 = 20 features + 6 pos_dim (use_all_features=True, use_pos=True, use_destination_activity=False)
            # - 27 = 21 features + 6 pos_dim (use_all_features=True, use_pos=True, use_destination_activity=True)
            
            base_in_channels_with_pos = effective_in_channels - pos_dim  # Assuming use_pos=True and pos_dim=6
            
            # Set feature configuration based on effective_in_channels
            if effective_in_channels == 7:
                # Could be: 5 base features + 2 pos_dim (most likely) OR 1 feature + 6 pos_dim (unlikely) OR 7 features without pos
                # Since use_all_features=False gives 5 features, try: 5 + 2 = 7
                config['use_all_features'] = False  # 5 base features
                config['use_pos'] = True
                config['pos_dim'] = 2  # 5 + 2 = 7
                config['in_channels'] = 5
                print(f"  Inferred config: in_channels=5, use_all_features=False, use_pos=True, pos_dim=2 (effective_in_channels={effective_in_channels})")
            elif base_in_channels_with_pos == 5:
                # 5 base features
                config['use_all_features'] = False
                config['use_pos'] = True
                config['pos_dim'] = pos_dim
                config['in_channels'] = 5
                print(f"  Inferred config: in_channels=5, use_all_features=False, use_pos=True, pos_dim={pos_dim}")
            elif base_in_channels_with_pos == 20:
                # 20 features (0-19)
                config['use_all_features'] = True
                config['use_destination_activity'] = False
                config['use_pos'] = True
                config['pos_dim'] = pos_dim
                config['in_channels'] = 20
                print(f"  Inferred config: in_channels=20, use_all_features=True, use_destination_activity=False, use_pos=True, pos_dim={pos_dim}")
            elif base_in_channels_with_pos == 21:
                # 21 features (0-20, if features 20-27 were included)
                config['use_all_features'] = True
                config['use_destination_activity'] = True
                config['use_pos'] = True
                config['pos_dim'] = pos_dim
                config['in_channels'] = 21
                print(f"  Inferred config: in_channels=21, use_all_features=True, use_destination_activity=True, use_pos=True, pos_dim={pos_dim}")
            else:
                # Unknown pattern - try use_pos=False first (more common)
                if effective_in_channels in [5, 7, 20, 21, 26, 27]:
                    # Could be these features without pos
                    config['use_pos'] = False
                    config['pos_dim'] = 0
                    config['in_channels'] = effective_in_channels
                    print(f"  WARNING: Unknown pattern. Trying use_pos=False: in_channels={effective_in_channels}, use_pos=False")
                else:
                    # Use the calculated base_in_channels_with_pos
                    config['in_channels'] = base_in_channels_with_pos
                    config['use_pos'] = use_pos
                    config['pos_dim'] = pos_dim
                    print(f"  WARNING: Unknown base_in_channels={base_in_channels_with_pos} (from effective_in_channels={effective_in_channels} - pos_dim={pos_dim})")
                    print(f"  Using: in_channels={base_in_channels_with_pos}, use_pos={use_pos}, pos_dim={pos_dim}")
            
            # Store ff_dim in config for model creation
            config['ff_dim'] = ff_dim
    else:
        # For other architectures, try to infer in_channels from first layer
        if 'in_channels' not in config:
            for key in state_dict.keys():
                if 'lin' in key or 'conv' in key or 'embed' in key:
                    if 'weight' in key:
                        weight_shape = state_dict[key].shape
                        if len(weight_shape) == 2:
                            # Assume first dimension is output, second is input
                            config['in_channels'] = weight_shape[1]
                            break
    
    # Set defaults (only if not already set by inference)
    config.setdefault('in_channels', 5)
    config.setdefault('out_channels', 1)
    config.setdefault('dropout', 0.3)
    config.setdefault('use_dropout', False)
    config.setdefault('target_type', 'abs_vol_car')
    config.setdefault('use_all_features', True)
    config.setdefault('use_destination_activity', False)
    config.setdefault('target_normalization', None)
    config.setdefault('ff_dim', 256)  # Default ff_dim
    config.setdefault('use_pos', True)
    config.setdefault('pos_dim', 6)
    
    # Prepare model_kwargs with inferred parameters
    model_kwargs = {
        'in_channels': config.get('in_channels', 5),
        'ff_dim': config.get('ff_dim', 256),
        'use_pos': config.get('use_pos', True),
        'pos_dim': config.get('pos_dim', 6),
    }
    
    # Create model with inferred parameters
    model = create_gnn_model(
        gnn_arch=gnn_arch,
        config=type('Config', (), config)(),  # Convert dict to object
        model_kwargs=model_kwargs,
        device=device
    )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    # Load target statistics if available (for standard scaler)
    if hasattr(model, 'target_normalization') and model.target_normalization == "relative_standard_scaler":
        if 'target_mean' in checkpoint and 'target_std' in checkpoint:
            model.target_mean = checkpoint['target_mean'].to(device)
            model.target_std = checkpoint['target_std'].to(device)
    
    model.eval()
    
    return model, config


def evaluate_checkpoint(
    checkpoint_path: Path,
    run_dir: Path,
    run_info: Dict,
    dataset_path: Path,
    gnn_arch: str,
    device: torch.device,
    splits_dir: Path,
    seed: int = 42
) -> Dict:
    """
    Evaluate a single checkpoint on validation data.
    
    Returns:
        Dictionary with evaluation metrics including hit rates.
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {run_info['method']} {run_info['city']} train={run_info['train_count']} val={run_info['val_count']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}")
    
    try:
        # Load model
        model, config = load_model_from_checkpoint(
            checkpoint_path, run_dir, gnn_arch, device
        )
        model = model.to(device)
        
        # Prepare validation data
        # REQUIRED: Load saved split file (runs with _high_dist_train_val suffix must have split files)
        city = run_info['city']
        train_count = run_info['train_count']
        val_count = run_info['val_count']
        
        # Split file is REQUIRED - no fallback to random split
        if splits_dir is None:
            print(f"  ERROR: --splits_dir is required but not provided. Skipping this run.")
            print(f"  For runs with '_high_dist_train_val' suffix, split files are mandatory.")
            return None
        
        split_filename = f"{city}_train{train_count}_val{val_count}_distant.json"
        split_file_path = splits_dir / split_filename
        
        if not split_file_path.exists():
            print(f"  ERROR: Split file not found: {split_file_path}")
            print(f"  Skipping this run. Split file is required for accurate evaluation.")
            return None
        
        # Load and validate split file
        print(f"  Loading saved split file: {split_file_path}")
        try:
            with open(split_file_path, 'r') as f:
                split_data = json.load(f)
        except Exception as e:
            print(f"  ERROR: Failed to load split file: {e}")
            return None
        
        # Verify the split matches
        if split_data.get('city') != city:
            print(f"  ERROR: Split file city '{split_data.get('city')}' doesn't match '{city}'")
            return None
        if split_data.get('train_count') != train_count:
            print(f"  ERROR: Split file train_count {split_data.get('train_count')} doesn't match {train_count}")
            return None
        if split_data.get('val_count') != val_count:
            print(f"  ERROR: Split file val_count {split_data.get('val_count')} doesn't match {val_count}")
            return None
        
        # Use the exact split from the file
        train_data_dict = split_data.get('train_data', {})
        val_data_dict = split_data.get('val_data', {})
        
        if not train_data_dict or not val_data_dict:
            print(f"  ERROR: Split file missing train_data or val_data")
            return None
        
        required_fields = ['path', 'policy_region', 'scenario', 'city']
        if not all(field in train_data_dict for field in required_fields):
            print(f"  ERROR: Split file train_data missing required fields: {required_fields}")
            return None
        if not all(field in val_data_dict for field in required_fields):
            print(f"  ERROR: Split file val_data missing required fields: {required_fields}")
            return None
        
        print(f"  ✓ Using exact split from file: {len(train_data_dict['path'])} train, {len(val_data_dict['path'])} val")
        if 'distance' in split_data:
            print(f"  Split Wasserstein distance: {split_data['distance']:.6f}")
        
        test_data_dict = {'path': [], 'policy_region': [], 'scenario': [], 'city': []}
        
        # Create a temporary directory for scalers (won't be used but required by prepare_data_with_graph_features)
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Use prepare_data_with_graph_features to create validation dataloader
            # This handles normalization, feature filtering, etc.
            # We include train data so the scaler can be fitted properly (for transductive, scaler uses train+val)
            _, val_dl, _ = prepare_data_with_graph_features(
                train_data=train_data_dict,
                val_data=val_data_dict,
                test_data=test_data_dict,
                use_inductive_variant=False,  # Transductive
                batch_size=8,
                path_to_save_dataloader=temp_dir,
                use_all_features=config.get('use_all_features', True),
                use_weighted_batches=False,
                use_nested_neighbor_loader=False,
                neighbor_sizes=[5, 5, 5],
                subgraphs_per_graph=1,
                seed_size=10,
                min_subgraph_nodes=500,
                max_subgraph_nodes=50000,
                sampling_strategy='neighbor_sampling',
                aug_pos_rotation=False,
                aug_feature_noise=False,
                aug_node_masking_probability=0.0,
                use_destination_activity_param=config.get('use_destination_activity', False)
            )
        finally:
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Evaluate model
        from gnn.help_functions import GNN_Loss
        loss_func = GNN_Loss(
            loss_fct='mse',
            device=device,
            weighted=False
        )
        
        # Use validate_model_during_training to get all metrics
        config_obj = type('Config', (), config)()
        val_result = validate_model_during_training(
            config=config_obj,
            model=model,
            dataset=val_dl,
            loss_func=loss_func,
            device=device
        )
        
        # Unpack results
        if len(val_result) == 5:
            val_loss, r_squared, spearman_corr, pearson_corr, hit_rates = val_result
        else:
            val_loss, r_squared, spearman_corr, pearson_corr = val_result
            hit_rates = {}
        
        # Prepare result dictionary
        result = {
            'run_name': run_info['method'] + '_' + run_info['city'] + f'_train{run_info["train_count"]}_val{run_info["val_count"]}',
            'method': run_info['method'],
            'city': run_info['city'],
            'train_count': run_info['train_count'],
            'val_count': run_info['val_count'],
            'val_loss': float(val_loss),
            'r_squared': float(r_squared),
            'spearman': float(spearman_corr),
            'pearson': float(pearson_corr),
        }
        
        # Add hit rates
        for key, value in hit_rates.items():
            result[key] = float(value)
        
        print(f"  Results: R²={r_squared:.4f}, Spearman={spearman_corr:.4f}")
        if hit_rates:
            print(f"  Hit rates: {', '.join([f'{k}={v:.4f}' for k, v in hit_rates.items()])}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR evaluating checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hit rates for all existing model checkpoints."
    )
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Path to results directory containing model checkpoints")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="data/analysis_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--gnn_arch", type=str, default="trans_encoder",
                       help="GNN architecture (default: trans_encoder)")
    parser.add_argument("--device_nr", type=int, default=0,
                       help="Device number (0 or 1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for data splitting (default: 42)")
    parser.add_argument("--wandb_csv", type=str, default=None,
                       help="Path to wandb CSV file with run states (default: data/analysis_data/wandb_current_runs.csv)")
    parser.add_argument("--splits_dir", type=str, required=True,
                       help="Directory containing saved split JSON files (e.g., data/splits). REQUIRED for runs with '_high_dist_train_val' suffix. Runs without matching split files will be skipped.")
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).resolve().parents[2]
    
    if not Path(args.results_dir).is_absolute():
        results_dir = project_root / args.results_dir
    else:
        results_dir = Path(args.results_dir)
    results_dir = results_dir.resolve()
    
    if not results_dir.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")
    
    if not Path(args.dataset_path).is_absolute():
        dataset_path = project_root / args.dataset_path
    else:
        dataset_path = Path(args.dataset_path)
    dataset_path = dataset_path.resolve()
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # Resolve splits_dir (required)
    if not Path(args.splits_dir).is_absolute():
        splits_dir = project_root / args.splits_dir
    else:
        splits_dir = Path(args.splits_dir)
    splits_dir = splits_dir.resolve()
    
    if not splits_dir.exists():
        raise ValueError(f"Splits directory does not exist: {splits_dir}")
    
    print(f"Using splits directory: {splits_dir}")
    
    if not Path(args.output_dir).is_absolute():
        output_dir = project_root / args.output_dir
    else:
        output_dir = Path(args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GPU setup
    gpus = get_available_gpus()
    if args.device_nr < len(gpus):
        set_cuda_visible_device(gpus[args.device_nr]['index'])  # Fix: pass index, not the whole dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("WARNING: CUDA not available, using CPU. This will be very slow for large graphs!")
    
    # Resolve wandb CSV path
    wandb_csv_path = None
    if args.wandb_csv:
        if not Path(args.wandb_csv).is_absolute():
            wandb_csv_path = project_root / args.wandb_csv
        else:
            wandb_csv_path = Path(args.wandb_csv)
        wandb_csv_path = wandb_csv_path.resolve()
    else:
        # Default path
        default_wandb_csv = project_root / "data" / "analysis_data" / "wandb_current_runs.csv"
        if default_wandb_csv.exists():
            wandb_csv_path = default_wandb_csv
            print(f"Using default wandb CSV: {wandb_csv_path}")
    
    # Find all checkpoints (only finished runs from wandb CSV, ending with "high_dist_train_val")
    print("=" * 80)
    print("Finding all completed runs...")
    if wandb_csv_path:
        print(f"  Using wandb CSV: {wandb_csv_path}")
        print("  Filtering to runs with State='finished' and ending with 'high_dist_train_val'")
    else:
        print("  ERROR: No wandb CSV provided. Wandb CSV is required to determine finished runs.")
        print("  Please provide --wandb_csv or ensure default path exists.")
    print("=" * 80)
    checkpoints = find_checkpoints(results_dir, wandb_csv_path=wandb_csv_path)
    print(f"Found {len(checkpoints)} completed runs to evaluate")
    
    # Group by method to show statistics
    finetuned_count = sum(1 for c in checkpoints if c['run_info']['method'] == 'finetuned')
    scratch_count = sum(1 for c in checkpoints if c['run_info']['method'] == 'scratch')
    print(f"  Finetuned runs: {finetuned_count}")
    print(f"  Scratch runs: {scratch_count}")
    
    # Evaluate each checkpoint
    print("\n" + "=" * 80)
    print("Evaluating checkpoints...")
    print("=" * 80)
    
    results = []
    for i, checkpoint_info in enumerate(checkpoints):
        print(f"\n[{i+1}/{len(checkpoints)}]")
        result = evaluate_checkpoint(
            checkpoint_path=checkpoint_info['checkpoint_path'],
            run_dir=checkpoint_info['run_dir'],
            run_info=checkpoint_info['run_info'],
            dataset_path=dataset_path,
            gnn_arch=args.gnn_arch,
            device=device,
            splits_dir=splits_dir,
            seed=args.seed
        )
        
        if result is not None:
            results.append(result)
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)
    
    if results:
        df = pd.DataFrame(results)
        
        # Save CSV
        csv_path = output_dir / 'hit_rates_evaluation.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to: {csv_path}")
        
        # Save JSON
        json_path = output_dir / 'hit_rates_evaluation.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON to: {json_path}")
        
        print(f"\nEvaluated {len(results)}/{len(checkpoints)} checkpoints successfully")
        print(f"Results include columns: {list(df.columns)}")
        
        # Show pair statistics
        print("\n" + "=" * 80)
        print("Pair Statistics:")
        print("=" * 80)
        
        # Group by configuration
        from collections import defaultdict
        configs = defaultdict(list)
        for result in results:
            key = (result['city'], result['train_count'], result['val_count'])
            configs[key].append(result['method'])
        
        # Count pairs
        pairs_found = 0
        finetuned_only = 0
        scratch_only = 0
        
        for (city, train_count, val_count), methods in configs.items():
            has_finetuned = 'finetuned' in methods
            has_scratch = 'scratch' in methods
            
            if has_finetuned and has_scratch:
                pairs_found += 1
                print(f"  ✓ Pair found: {city}, train={train_count}, val={val_count}")
            elif has_finetuned:
                finetuned_only += 1
                print(f"  ⚠ Finetuned only: {city}, train={train_count}, val={val_count}")
            elif has_scratch:
                scratch_only += 1
                print(f"  ⚠ Scratch only: {city}, train={train_count}, val={val_count}")
        
        print(f"\n  Total pairs found: {pairs_found}")
        print(f"  Finetuned-only configs: {finetuned_only}")
        print(f"  Scratch-only configs: {scratch_only}")
    else:
        print("ERROR: No results to save!")


if __name__ == "__main__":
    main()

