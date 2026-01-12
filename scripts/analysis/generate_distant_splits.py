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
from tqdm import tqdm
import torch

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


def find_distant_test_split(train_paths: List[str], val_paths: List[str], all_paths: List[str], 
                           test_count: int, num_trials: int = 2000, seed: int = 42, 
                           use_all_features: bool = True) -> Tuple[List[str], float, float, float]:
    """
    Find a test set that maximizes Wasserstein distance from both train and val sets.
    
    Uses a combined scoring strategy to ensure test set is VERY different from both:
    1. Primary: Maximize minimum distance (ensures both distances are large)
    2. Secondary: Maximize sum of distances (favors cases where both are large)
    3. Returns both individual distances for verification
    
    Args:
        train_paths: Already selected training paths
        val_paths: Already selected validation paths
        all_paths: All available paths (excluding train and val)
        test_count: Number of test graphs to select
        num_trials: Number of random trials (default: 2000 for better results)
        seed: Random seed
        use_all_features: Feature selection flag
    
    Returns:
        test_paths, min_distance, dist_test_train, dist_test_val
    """
    # Filter out train and val paths from all_paths
    train_set = set(train_paths)
    val_set = set(val_paths)
    available_paths = [p for p in all_paths if p not in train_set and p not in val_set]
    
    # Filter out invalid paths
    valid_paths = []
    for path in available_paths:
        if os.path.exists(path):
            valid_paths.append(path)
    
    if len(valid_paths) < test_count:
        raise ValueError(
            f"Insufficient valid graphs for test: {len(valid_paths)} < {test_count}"
        )
    
    print(f"Found {len(valid_paths)} valid test graphs (excluding {len(train_paths)} train + {len(val_paths)} val)")
    
    # Try multiple random splits to find the one with maximum distance from BOTH train and val
    print(f"\nTrying {num_trials} random splits to find test set with MAXIMUM distance from train+val...")
    print(f"Using EXACT same distance computation as analyze_pretraining_benefit_vs_distance.py")
    print(f"Feature selection: {'features 0-19' if use_all_features else 'features [0,1,2,3,10] (5 base features)'}")
    print(f"Optimization strategy: Maximize minimum distance, with sum as tiebreaker")
    
    best_min_distance = -1.0
    best_sum_distance = -1.0
    best_test_paths = None
    best_dist_test_train = 0.0
    best_dist_test_val = 0.0
    
    _rnd.seed(seed)
    for trial in range(num_trials):
        # Random shuffle
        indices = list(range(len(valid_paths)))
        _rnd.shuffle(indices)
        
        # Select test paths
        test_indices = indices[:test_count]
        test_paths_trial = [valid_paths[i] for i in test_indices]
        
        # Compute distances from test to train and test to val
        try:
            dist_test_train = compute_wasserstein_distance_analysis(
                test_paths_trial,
                train_paths,
                use_capacity_reduction_only=False,
                verbose=False,
                use_all_features=use_all_features
            )
            dist_test_val = compute_wasserstein_distance_analysis(
                test_paths_trial,
                val_paths,
                use_capacity_reduction_only=False,
                verbose=False,
                use_all_features=use_all_features
            )
            
            # Primary: minimum distance (ensures test is distant from BOTH)
            min_distance = min(dist_test_train, dist_test_val)
            # Secondary: sum of distances (favors cases where both are large)
            sum_distance = dist_test_train + dist_test_val
            
        except Exception as e:
            print(f"  Trial {trial + 1}/{num_trials}: Failed to compute distance: {e}")
            continue
        
        # Update best if:
        # 1. Minimum distance is better, OR
        # 2. Minimum distance is equal but sum is better (tiebreaker)
        is_better = (
            min_distance > best_min_distance or
            (min_distance == best_min_distance and sum_distance > best_sum_distance)
        )
        
        # Print progress
        if (trial + 1) % 200 == 0 or is_better:
            print(f"Trial {trial + 1}/{num_trials}, min: {min_distance:.6f}, sum: {sum_distance:.6f} "
                  f"(test-train: {dist_test_train:.6f}, test-val: {dist_test_val:.6f})")
        
        if is_better:
            best_min_distance = min_distance
            best_sum_distance = sum_distance
            best_test_paths = test_paths_trial
            best_dist_test_train = dist_test_train
            best_dist_test_val = dist_test_val
            print(f"  → New best found! min_distance: {best_min_distance:.6f}, "
                  f"test-train: {best_dist_test_train:.6f}, test-val: {best_dist_test_val:.6f}")
    
    if best_test_paths is None:
        raise ValueError("Failed to find any valid test split with computable distance")
    
    print(f"\n✓ Best test set found:")
    print(f"  Minimum distance (from both train and val): {best_min_distance:.6f}")
    print(f"  Distance from train: {best_dist_test_train:.6f}")
    print(f"  Distance from val: {best_dist_test_val:.6f}")
    print(f"  Sum of distances: {best_sum_distance:.6f}")
    print(f"  Test set size: {len(best_test_paths)}")
    
    return best_test_paths, best_min_distance, best_dist_test_train, best_dist_test_val

def compute_iou_dist(set_a: np.ndarray, set_b: np.ndarray) -> float:
    """Compute IoU distance (1 - IoU) between two boolean arrays."""
    intersection = np.logical_and(set_a, set_b).sum()
    union = np.logical_or(set_a, set_b).sum()

    # If no reductions in both, then they are identical → IoU = 1 → distance = 0
    if union == 0:
        return 0.0
    
    # IoU distance
    return 1 - (intersection / union)

def compute_iou_distance_from_set(anchor_paths: List[str], valid_paths: List[str],
                                  reduction_graphs: Dict[str, np.ndarray]) -> None:
    
    candidates = {path: np.inf for path in valid_paths}

    # Compute distances from train/val set to all candidates
    for path in anchor_paths:
        
        reduction = reduction_graphs[path]
        
        for candidate_path in candidates.keys():
            
            candidate_reduction = reduction_graphs[candidate_path]

            iou_dist = compute_iou_dist(reduction, candidate_reduction)
            candidates[candidate_path] = min(candidates[candidate_path], iou_dist)

    return candidates

def find_distant_iou_test_split(train_paths: List[str], val_paths: List[str], all_paths: List[str],
                               test_count: int) -> Tuple[List[str], List[float], List[float], List[float]]:
    
    # Filter out train and val paths from all_paths
    train_set = set(train_paths)
    val_set = set(val_paths)
    available_paths = [p for p in all_paths if p not in train_set and p not in val_set]
    
    # Filter out invalid paths
    valid_paths = []
    for path in available_paths:
        if os.path.exists(path):
            valid_paths.append(path)
    
    if len(valid_paths) < test_count:
        raise ValueError(
            f"Insufficient valid graphs for test: {len(valid_paths)} < {test_count}"
        )
    
    print(f"Found {len(valid_paths)} valid test graphs (excluding {len(train_paths)} train + {len(val_paths)} val)")

    # Get capacity reduction features for all graphs
    reduction_graphs = dict()
    for path in train_paths + val_paths + valid_paths:
        
        graph = torch.load(path, map_location='cpu')
        node_features = graph.x.cpu().numpy()
        
        # Extract feature 2 (CAPACITY_REDUCTION) for all nodes
        capacity_reduction = node_features[:, 2]  # Feature index 2
        reduction_graphs[path] = capacity_reduction.flatten().astype(bool)

    test_paths = []
    test_distances_from_train = []
    test_distances_from_val = []
    test_distances_when_picked = []

    # Compute distances from train set
    candidate_train_distances = compute_iou_distance_from_set(train_paths, valid_paths, reduction_graphs)
    
    # Compute distances from val set
    candidate_val_distances = compute_iou_distance_from_set(val_paths, valid_paths, reduction_graphs)

    # Initialize candidates with min distance from train and val
    candidates = dict()
    for path in valid_paths:
        candidates[path] = min(candidate_train_distances[path], candidate_val_distances[path])
            
    # Greedyly select test graphs with maximin IoU distance
    for _ in tqdm(range(test_count), desc="Selecting distant test graphs ..."):
        
        # Select best candidate
        best_path = max(candidates.items(), key=lambda x: x[1])[0]
        best_distance = candidates[best_path]
        
        test_paths.append(best_path)
        test_distances_from_train.append(candidate_train_distances[best_path])
        test_distances_from_val.append(candidate_val_distances[best_path])
        test_distances_when_picked.append(best_distance)
        
        # Remove selected path from candidates
        del candidates[best_path]
        
        # Update minimum distances for remaining candidates, improves intra test diversity
        best_reduction = reduction_graphs[best_path]
        
        for candidate_path in candidates.keys():
            candidate_reduction = reduction_graphs[candidate_path]
            iou_dist = compute_iou_dist(best_reduction, candidate_reduction)
            candidates[candidate_path] = min(candidates[candidate_path], iou_dist)

    return test_paths, test_distances_when_picked, test_distances_from_train, test_distances_from_val

def find_random_test_split(train_paths: List[str], val_paths: List[str], all_paths: List[str],
                           test_count: int, seed: int = 42) -> Tuple[List[str], List[float], List[float]]:
    
    # Filter out train and val paths from all_paths
    train_set = set(train_paths)
    val_set = set(val_paths)
    available_paths = [p for p in all_paths if p not in train_set and p not in val_set]
    
    # Filter out invalid paths
    valid_paths = []
    for path in available_paths:
        if os.path.exists(path):
            valid_paths.append(path)
    
    if len(valid_paths) < test_count:
        raise ValueError(
            f"Insufficient valid graphs for test: {len(valid_paths)} < {test_count}"
        )
    
    print(f"Found {len(valid_paths)} valid test graphs (excluding {len(train_paths)} train + {len(val_paths)} val)")

    # Randomly select test graphs
    rng = _rnd.Random(seed)
    selected_paths = rng.sample(valid_paths, test_count)

    # Get capacity reduction features for all graphs
    reduction_graphs = dict()
    for path in train_paths + val_paths + selected_paths:
        
        graph = torch.load(path, map_location='cpu')
        node_features = graph.x.cpu().numpy()
        
        # Extract feature 2 (CAPACITY_REDUCTION) for all nodes
        capacity_reduction = node_features[:, 2]  # Feature index 2
        reduction_graphs[path] = capacity_reduction.flatten().astype(bool)

    test_distances_from_train = []
    test_distances_from_val = []

    # Compute distances from train set
    candidate_train_distances = compute_iou_distance_from_set(train_paths, selected_paths, reduction_graphs)

    # Compute distances from val set
    candidate_val_distances = compute_iou_distance_from_set(val_paths, selected_paths, reduction_graphs)

    for path in selected_paths:
        test_distances_from_train.append(candidate_train_distances[path])
        test_distances_from_val.append(candidate_val_distances[path])

    return selected_paths, test_distances_from_train, test_distances_from_val

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

