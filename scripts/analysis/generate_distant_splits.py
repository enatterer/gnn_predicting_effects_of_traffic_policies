#!/usr/bin/env python3
"""
Generate training and validation splits that maximize Wasserstein distance.

This script:
1. Loads all graphs for a given city
2. Computes graph-level features for all graphs
3. Tries multiple random splits to find one that maximizes Wasserstein distance
4. Saves the split to a JSON file for use in finetuning experiments

Usage:
    python scripts/analysis/generate_distant_splits.py \
        --city erlangen \
        --train_count 10 \
        --val_count 100 \
        --num_trials 1000 \
        --output_dir data/splits
"""

import os
import sys
import json
import random as _rnd
import argparse
import importlib.util
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from scipy.stats import wasserstein_distance
import torch
from torch_geometric.data import Data

# Add the 'scripts' directory to Python Path
scripts_path = Path(__file__).resolve().parents[1]
if str(scripts_path) not in sys.path:
    sys.path.append(str(scripts_path))

from training.help_functions import load_metadata_from_disk

# Import the exact same function from analyze_pretraining_benefit_vs_distance.py
# to ensure we compute distances exactly the same way
analysis_script_path = Path(__file__).parent / "analyze_pretraining_benefit_vs_distance.py"
if str(analysis_script_path.parent) not in sys.path:
    sys.path.insert(0, str(analysis_script_path.parent))

# Import using importlib to handle the module name
import importlib.util
spec = importlib.util.spec_from_file_location("analyze_pretraining_benefit_vs_distance", analysis_script_path)
analyze_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_module)

# Import the functions we need
compute_graph_features_analysis = analyze_module.compute_graph_features
compute_wasserstein_distance_analysis = analyze_module.compute_wasserstein_distance

# We now use compute_wasserstein_distance_analysis() directly which:
# 1. Takes train_paths and val_paths (not pre-computed features)
# 2. Loads graphs, computes features, handles NaN/inf correctly
# 3. Extracts mean_node_feat (feature index 4)
# 4. Computes Wasserstein distance exactly as in the analysis script
# This ensures 100% consistency with analyze_pretraining_benefit_vs_distance.py


def find_distant_split(all_paths: List[str], train_count: int, val_count: int, 
                      num_trials: int = 1000, seed: int = 42, use_all_features: bool = True) -> Tuple[List[str], List[str], float]:
    """
    Find a train/val split that maximizes Wasserstein distance.
    
    Uses the EXACT same compute_wasserstein_distance() function from the analysis script
    to ensure 100% consistency. The feature selection matches training exactly:
    - If use_all_features=False: Uses only features [0, 1, 2, 3, 10] (5 base features)
    - If use_all_features=True: Uses features [0-19] (excludes 20-27 with NaNs)
    
    Args:
        use_all_features: If False, use only 5 base features. If True, use features 0-19.
                         Must match the feature selection used during training.
    
    Returns:
        train_paths, val_paths, best_distance
    """
    # Filter out invalid paths first
    valid_paths = []
    for path in all_paths:
        if os.path.exists(path):
            valid_paths.append(path)
    
    if len(valid_paths) < train_count + val_count:
        raise ValueError(
            f"Insufficient valid graphs: {len(valid_paths)} < {train_count} + {val_count}"
        )
    
    print(f"Found {len(valid_paths)} valid graphs from {len(all_paths)} total paths")
    
    # Try multiple random splits to find the one with maximum distance
    # Use the EXACT same compute_wasserstein_distance() function from analysis script
    print(f"\nTrying {num_trials} random splits to find maximum distance...")
    print(f"Using EXACT same distance computation as analyze_pretraining_benefit_vs_distance.py")
    print(f"Feature selection: {'features 0-19' if use_all_features else 'features [0,1,2,3,10] (5 base features)'}")
    print(f"(This may take a while as each trial loads graphs and computes features...)")
    
    best_distance = -1.0
    best_train_paths = None
    best_val_paths = None
    
    _rnd.seed(seed)
    for trial in range(num_trials):
        # Random shuffle
        indices = list(range(len(valid_paths)))
        _rnd.shuffle(indices)
        
        # Split paths (not indices)
        train_indices = indices[:train_count]
        val_indices = indices[train_count:train_count + val_count]
        
        train_paths_trial = [valid_paths[i] for i in train_indices]
        val_paths_trial = [valid_paths[i] for i in val_indices]
        
        # Compute distance using EXACT same function from analysis script
        # This ensures 100% consistency with the correlation analysis
        # Pass verbose=False to suppress detailed output
        # Pass use_all_features to match the exact feature selection used in training
        try:
            distance = compute_wasserstein_distance_analysis(
                train_paths_trial, 
                val_paths_trial,
                use_capacity_reduction_only=False,  # Use mean_node_feat (default)
                verbose=False,  # Suppress verbose output during trials
                use_all_features=use_all_features  # Use same features as training
            )
        except Exception as e:
            print(f"  Trial {trial + 1}/{num_trials}: Failed to compute distance: {e}")
            continue
        
        # Print trial number and distance for all trials (not just best ones)
        print(f"Trial number: {trial + 1}/{num_trials}, distance: {distance:.6f}")
        
        if distance > best_distance:
            best_distance = distance
            best_train_paths = train_paths_trial
            best_val_paths = val_paths_trial
            print(f"  → New best distance found: {best_distance:.6f}")
    
    if best_train_paths is None or best_val_paths is None:
        raise ValueError("Failed to find any valid split with computable distance")
    
    print(f"\n✓ Best distance found: {best_distance:.6f}")
    print(f"  Train set size: {len(best_train_paths)}")
    print(f"  Val set size: {len(best_val_paths)}")
    
    return best_train_paths, best_val_paths, best_distance


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/val splits that maximize Wasserstein distance."
    )
    parser.add_argument("--city", type=str, required=True,
                       help="City name (e.g., 'erlangen')")
    parser.add_argument("--train_count", type=int, required=True,
                       help="Number of training graphs")
    parser.add_argument("--val_count", type=int, required=True,
                       help="Number of validation graphs")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to dataset directory. Defaults to data/bavaria/inductive_data/training_data/kreisfreistadt")
    parser.add_argument("--output_dir", type=str, default="data/splits",
                       help="Directory to save split JSON files")
    parser.add_argument("--num_trials", type=int, default=1000,
                       help="Number of random splits to try (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--use_all_features", type=lambda x: x.lower() == 'true', default=True,
                       help="If False, use only 5 base features (0,1,2,3,10). If True, use features 0-19. Default: True.")
    
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
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Generating Distant Train/Val Split")
    print("=" * 80)
    print(f"City: {args.city}")
    print(f"Train count: {args.train_count}")
    print(f"Val count: {args.val_count}")
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Use all features: {args.use_all_features} ({'features 0-19' if args.use_all_features else 'features [0,1,2,3,10] (5 base)'})")
    print("=" * 80)
    
    # Load all data for the city
    city_metadata_path = dataset_path / args.city / 'metadata.json'
    if not city_metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {city_metadata_path}")
    
    all_data = {'path': list(), 'policy_region': list(), 'scenario': list(), 'city': list()}
    load_metadata_from_disk(all_data, str(city_metadata_path))
    
    print(f"\nLoaded {len(all_data['path'])} total graphs from {args.city}")
    
    if len(all_data['path']) < args.train_count + args.val_count:
        raise ValueError(
            f"Insufficient data for city {args.city}: "
            f"Requested {args.train_count} train + {args.val_count} val = {args.train_count + args.val_count} graphs, "
            f"but only {len(all_data['path'])} available."
        )
    
    # Find distant split
    train_paths, val_paths, distance = find_distant_split(
        all_data['path'], 
        args.train_count, 
        args.val_count, 
        num_trials=args.num_trials,
        seed=args.seed,
        use_all_features=args.use_all_features
    )
    
    # Create split data structure
    split_data = {
        'city': args.city,
        'train_count': len(train_paths),
        'val_count': len(val_paths),
        'distance': float(distance),
        'train_paths': train_paths,
        'val_paths': val_paths,
        'train_indices': [all_data['path'].index(p) for p in train_paths],
        'val_indices': [all_data['path'].index(p) for p in val_paths],
        # Also store full metadata for convenience
        'train_data': {
            'path': train_paths,
            'policy_region': [all_data['policy_region'][all_data['path'].index(p)] for p in train_paths],
            'scenario': [all_data['scenario'][all_data['path'].index(p)] for p in train_paths],
            'city': [args.city] * len(train_paths)
        },
        'val_data': {
            'path': val_paths,
            'policy_region': [all_data['policy_region'][all_data['path'].index(p)] for p in val_paths],
            'scenario': [all_data['scenario'][all_data['path'].index(p)] for p in val_paths],
            'city': [args.city] * len(val_paths)
        }
    }
    
    # Save split to JSON file
    output_filename = f"{args.city}_train{args.train_count}_val{args.val_count}_distant.json"
    output_path = output_dir / output_filename
    
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"\n✓ Saved split to: {output_path}")
    print(f"  Distance: {distance:.6f}")
    
    return output_path


if __name__ == "__main__":
    main()

