#!/usr/bin/env python3
"""
Verify and display Wasserstein distances for splits.

This script:
1. Shows what splits will be generated
2. Can verify distances for existing split files
3. Computes distances using the EXACT same method as analyze_pretraining_benefit_vs_distance.py

Usage:
    # Show planned splits and their target distances
    python scripts/analysis/verify_splits_distances.py \
        --target_cities "erlangen,bamberg,muenchen,neuulm" \
        --train_graph_counts "10" \
        --val_graph_count 100 \
        --num_trials 100
    
    # Verify existing split files
    python scripts/analysis/verify_splits_distances.py \
        --splits_dir data/splits \
        --verify_existing
"""

import argparse
import json
import sys
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

# Import the exact same functions from analyze_pretraining_benefit_vs_distance.py
sys.path.insert(0, str(Path(__file__).parent))
from analyze_pretraining_benefit_vs_distance import (
    compute_graph_features,
    load_graph as load_graph_analysis,
    recover_train_val_split
)


def compute_distance_for_split(train_paths: List[str], val_paths: List[str]) -> float:
    """
    Compute Wasserstein distance for a split using the EXACT same method as 
    analyze_pretraining_benefit_vs_distance.py.
    """
    # Use the exact same approach as analyze_pretraining_benefit_vs_distance.py
    train_features = []
    val_features = []
    
    # Load and compute features for training graphs
    for path in train_paths:
        try:
            graph = load_graph_analysis(path)
            features = compute_graph_features(
                graph,
                exclude_activity_features=True,
                use_only_first_5_features=True
            )
            train_features.append(features)
        except Exception as e:
            print(f"    Warning: Failed to load {path}: {e}")
            continue
    
    # Load and compute features for validation graphs
    for path in val_paths:
        try:
            graph = load_graph_analysis(path)
            features = compute_graph_features(
                graph,
                exclude_activity_features=True,
                use_only_first_5_features=True
            )
            val_features.append(features)
        except Exception as e:
            print(f"    Warning: Failed to load {path}: {e}")
            continue
    
    if len(train_features) == 0 or len(val_features) == 0:
        raise ValueError("No valid graphs found in train or val sets")
    
    train_features = np.array(train_features)
    val_features = np.array(val_features)
    
    # Replace NaN/inf with 0 (same as analysis script)
    train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
    val_features = np.nan_to_num(val_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Extract only mean_node_feat (feature index 4) for Wasserstein distance
    mean_node_feat_idx = 4
    train_dist = train_features[:, mean_node_feat_idx]
    val_dist = val_features[:, mean_node_feat_idx]
    
    # Compute Wasserstein distance (exact same as analysis script)
    train_is_constant = np.all(train_dist == train_dist[0])
    val_is_constant = np.all(val_dist == val_dist[0])
    
    if train_is_constant and val_is_constant:
        # Constant arrays - distance is just the difference
        dist = abs(train_dist[0] - val_dist[0])
    else:
        try:
            dist = wasserstein_distance(train_dist, val_dist)
            if not np.isfinite(dist):
                # Fallback: use mean absolute difference
                dist = np.mean(np.abs(train_dist - val_dist))
        except Exception:
            # Fallback: use mean absolute difference
            dist = np.mean(np.abs(train_dist - val_dist))
    
    return dist


def verify_existing_splits(splits_dir: Path):
    """Verify distances for all existing split files."""
    print("=" * 80)
    print("Verifying Existing Split Files")
    print("=" * 80)
    
    split_files = list(splits_dir.glob("*_distant.json"))
    
    if not split_files:
        print(f"No split files found in {splits_dir}")
        return
    
    print(f"Found {len(split_files)} split files\n")
    
    results = []
    
    for split_file in sorted(split_files):
        print(f"Verifying: {split_file.name}")
        
        try:
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            
            city = split_data.get('city', 'unknown')
            train_count = split_data.get('train_count', 0)
            val_count = split_data.get('val_count', 0)
            stored_distance = split_data.get('distance', 0.0)
            train_paths = split_data.get('train_paths', [])
            val_paths = split_data.get('val_paths', [])
            
            if not train_paths or not val_paths:
                print(f"  ERROR: Missing paths in split file")
                continue
            
            print(f"  City: {city}, Train: {train_count}, Val: {val_count}")
            print(f"  Stored distance: {stored_distance:.6f}")
            print(f"  Computing distance...", end=" ", flush=True)
            
            computed_distance = compute_distance_for_split(train_paths, val_paths)
            
            print(f"Computed: {computed_distance:.6f}")
            
            # Check if they match (allow small floating point differences)
            diff = abs(stored_distance - computed_distance)
            match = diff < 1e-5
            
            status = "✓ MATCH" if match else "✗ MISMATCH"
            print(f"  {status} (difference: {diff:.10f})")
            
            results.append({
                'city': city,
                'train_count': train_count,
                'val_count': val_count,
                'stored_distance': stored_distance,
                'computed_distance': computed_distance,
                'match': match,
                'diff': diff
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        
        print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total splits verified: {len(results)}")
    matches = sum(1 for r in results if r['match'])
    print(f"Matches: {matches}/{len(results)}")
    
    if results:
        print("\nDistances:")
        print(f"{'City':<15} {'Train':<8} {'Val':<8} {'Stored':<12} {'Computed':<12} {'Match':<8}")
        print("-" * 80)
        for r in results:
            match_str = "✓" if r['match'] else "✗"
            print(f"{r['city']:<15} {r['train_count']:<8} {r['val_count']:<8} "
                  f"{r['stored_distance']:<12.6f} {r['computed_distance']:<12.6f} {match_str:<8}")


def show_planned_splits(target_cities: List[str], train_graph_counts: List[int], 
                       val_graph_count: int, dataset_path: Path, num_trials: int = 100):
    """Show what splits will be generated and estimate their distances."""
    print("=" * 80)
    print("Planned Splits for Generation")
    print("=" * 80)
    print(f"Target cities: {target_cities}")
    print(f"Training graph counts: {train_graph_counts}")
    print(f"Validation graph count: {val_graph_count}")
    print(f"Number of trials: {num_trials}")
    print(f"Dataset path: {dataset_path}")
    print("=" * 80)
    
    # For each city and train_count, show:
    # 1. Available graphs
    # 2. Estimated distance range (by trying a few random splits)
    # 3. Target: maximize distance
    
    for city in target_cities:
        city_metadata_path = dataset_path / city / 'metadata.json'
        
        if not city_metadata_path.exists():
            print(f"\n{city}: ERROR - metadata file not found")
            continue
        
        all_data = {'path': list(), 'policy_region': list(), 'scenario': list(), 'city': list()}
        load_metadata_from_disk(all_data, str(city_metadata_path))
        
        print(f"\n{city}:")
        print(f"  Total graphs available: {len(all_data['path'])}")
        
        for train_count in train_graph_counts:
            if len(all_data['path']) < train_count + val_graph_count:
                print(f"  Train={train_count}, Val={val_graph_count}: INSUFFICIENT DATA "
                      f"({len(all_data['path'])} < {train_count + val_graph_count})")
                continue
            
            print(f"  Train={train_count}, Val={val_graph_count}:")
            
            # Try a few random splits to get a sense of distance range
            distances_sample = []
            import random as _rnd
            _rnd.seed(42)
            
            for trial in range(min(10, num_trials // 10)):  # Sample 10 splits
                indices = list(range(len(all_data['path'])))
                _rnd.shuffle(indices)
                
                train_indices = indices[:train_count]
                val_indices = indices[train_count:train_count + val_graph_count]
                
                train_paths = [all_data['path'][i] for i in train_indices]
                val_paths = [all_data['path'][i] for i in val_indices]
                
                try:
                    distance = compute_distance_for_split(train_paths, val_paths)
                    distances_sample.append(distance)
                except Exception as e:
                    print(f"    Warning: Failed to compute distance for trial {trial}: {e}")
                    continue
            
            if distances_sample:
                min_dist = min(distances_sample)
                max_dist = max(distances_sample)
                mean_dist = np.mean(distances_sample)
                print(f"    Sample distance range: [{min_dist:.6f}, {max_dist:.6f}] "
                      f"(mean: {mean_dist:.6f}, std: {np.std(distances_sample):.6f})")
                print(f"    Target: Find split with distance >= {max_dist:.6f} (will try {num_trials} splits)")


def main():
    parser = argparse.ArgumentParser(
        description="Verify and display Wasserstein distances for splits."
    )
    parser.add_argument("--target_cities", type=str, default=None,
                       help="Comma-separated list of cities (for showing planned splits)")
    parser.add_argument("--train_graph_counts", type=str, default=None,
                       help="Comma-separated list of training graph counts")
    parser.add_argument("--val_graph_count", type=int, default=100,
                       help="Number of validation graphs")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to dataset directory")
    parser.add_argument("--splits_dir", type=str, default="data/splits",
                       help="Directory containing split JSON files")
    parser.add_argument("--verify_existing", action="store_true",
                       help="Verify distances for existing split files")
    parser.add_argument("--num_trials", type=int, default=100,
                       help="Number of trials for distance estimation (used with --target_cities)")
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).resolve().parents[2]
    
    if args.verify_existing:
        # Verify existing splits
        if args.splits_dir and not Path(args.splits_dir).is_absolute():
            splits_dir = project_root / args.splits_dir
        else:
            splits_dir = Path(args.splits_dir) if args.splits_dir else project_root / "data/splits"
        splits_dir = splits_dir.resolve()
        
        if not splits_dir.exists():
            print(f"ERROR: Splits directory does not exist: {splits_dir}")
            return
        
        verify_existing_splits(splits_dir)
    
    elif args.target_cities:
        # Show planned splits
        target_cities = [c.strip() for c in args.target_cities.split(',') if c.strip()]
        
        if args.train_graph_counts:
            train_graph_counts = [int(x.strip()) for x in args.train_graph_counts.split(',') if x.strip()]
        else:
            train_graph_counts = [10]  # Default
        
        if args.dataset_path is None:
            dataset_path = project_root / 'data' / 'bavaria' / 'inductive_data' / 'training_data' / 'kreisfreistadt'
        else:
            dataset_path = Path(args.dataset_path)
        dataset_path = dataset_path.resolve()
        
        if not dataset_path.exists():
            print(f"ERROR: Dataset path does not exist: {dataset_path}")
            return
        
        show_planned_splits(target_cities, train_graph_counts, args.val_graph_count, 
                           dataset_path, args.num_trials)
    
    else:
        parser.print_help()
        print("\nERROR: Must specify either --verify_existing or --target_cities")


if __name__ == "__main__":
    main()

