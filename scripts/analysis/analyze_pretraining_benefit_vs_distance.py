#!/usr/bin/env python3
"""
Analyze the relationship between train-val graph distance and pretraining benefit.

This script:
1. Parses wandb CSV results to extract finetuned vs scratch performance
2. Recovers the exact train/val splits used (using fixed seed 42)
3. Computes Wasserstein distance between training and validation graph sets
4. Creates plots showing performance increase (finetuned - scratch) vs distance

Usage Examples:
---------------

Default (uses graph-level features - mean_node_feat):
    python scripts/analysis/analyze_pretraining_benefit_vs_distance.py \
        --wandb_csv scripts/analysis/wandb_export.csv \
        --dataset_path data/bavaria/inductive_data/training_data/kreisfreistadt \
        --output_dir data/analysis_results \
        --seed 42

Use CAPACITY_REDUCTION only instead:
    python scripts/analysis/analyze_pretraining_benefit_vs_distance.py \
        --wandb_csv scripts/analysis/wandb_export.csv \
        --dataset_path data/bavaria/inductive_data/training_data/kreisfreistadt \
        --output_dir data/analysis_results \
        --seed 42 \
        --use_only_capacity_reduction
"""

import os
import sys
import csv
import json
import random as _rnd  # Match finetune_models.py exactly
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
import torch
from torch_geometric.data import Data

# Add the 'scripts' directory to Python Path
scripts_path = Path(__file__).resolve().parents[1]
if str(scripts_path) not in sys.path:
    sys.path.append(str(scripts_path))

from training.help_functions import load_metadata_from_disk

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def parse_wandb_csv(csv_path: str, only_finished: bool = True) -> pd.DataFrame:
    """
    Parse the wandb export CSV file.
    
    Args:
        only_finished: If True, only include runs where State == "finished"
    
    Returns a DataFrame with columns: Name, r^2, val_loss, spearman, pearson
    """
    df = pd.read_csv(csv_path)
    
    # Debug: Print column names to help diagnose issues
    print(f"  CSV columns: {list(df.columns)}")
    print(f"  Number of rows: {len(df)}")
    
    # Filter to only finished runs if requested
    if only_finished:
        if 'State' in df.columns:
            original_count = len(df)
            df = df[df['State'] == 'finished'].copy()
            print(f"  Filtered to finished runs: {len(df)}/{original_count} runs")
        else:
            print("  Warning: 'State' column not found, cannot filter by state")
    
    # Check for required columns
    required_cols = ['Name']  # At minimum, we need the run name
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # Try case-insensitive matching
        name_cols = [col for col in df.columns if col.lower() == 'name']
        if name_cols:
            df = df.rename(columns={name_cols[0]: 'Name'})
        else:
            raise ValueError(f"Required column 'Name' not found in CSV. Available columns: {list(df.columns)}")
    
    return df


def extract_run_info(run_name: str) -> Optional[Dict[str, any]]:
    """
    Extract city, train_count, val_count from run name.
    
    Examples:
    - "finetuned_erlangen_train5_val200" -> {city: "erlangen", train: 5, val: 200, is_finetuned: True}
    - "scratch_erlangen_train5_val200" -> {city: "erlangen", train: 5, val: 200, is_finetuned: False}
    """
    import re
    
    # Pattern: (finetuned|scratch)_(city)_train(\d+)_val(\d+)
    pattern = r'(finetuned|scratch)_(\w+)_train(\d+)_val(\d+)'
    match = re.match(pattern, run_name)
    
    if not match:
        return None
    
    is_finetuned = match.group(1) == 'finetuned'
    city = match.group(2)
    train_count = int(match.group(3))
    val_count = int(match.group(4))
    
    return {
        'city': city,
        'train_count': train_count,
        'val_count': val_count,
        'is_finetuned': is_finetuned
    }


def recover_train_val_split(city: str, train_count: int, val_count: int, 
                           dataset_path: str, seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Recover the exact train/val split used in finetune_models.py.
    
    This replicates the logic from finetune_models.py lines 239-290:
    - Load all metadata for the city
    - Shuffle with seed 42
    - Take first train_count for train, next val_count for val
    """
    # Load all data from the city
    all_data = {'path': list(), 'policy_region': list(), 'scenario': list(), 'city': list()}
    city_metadata_path = os.path.join(dataset_path, city, 'metadata.json')
    
    if not os.path.exists(city_metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {city_metadata_path}")
    
    load_metadata_from_disk(all_data, city_metadata_path)
    
    # Shuffle and split (same logic as finetune_models.py lines 247-249)
    # CRITICAL: Must use module-level random exactly as finetune_models.py does
    _rnd.seed(seed)  # For reproducibility - matches finetune_models.py line 247
    indices = list(range(len(all_data['path'])))
    _rnd.shuffle(indices)  # Matches finetune_models.py line 249
    
    # Calculate split sizes
    limit_train = train_count if train_count > 0 else len(indices)
    limit_val = val_count if val_count > 0 else len(indices)
    
    # Ensure we don't exceed available data
    total_needed = limit_train + limit_val
    if total_needed > len(indices):
        raise ValueError(
            f"Insufficient data for city {city}: "
            f"Requested {limit_train} train + {limit_val} val = {total_needed} graphs, "
            f"but only {len(indices)} available."
        )
    
    # Split indices
    train_indices = indices[:limit_train]
    val_indices = indices[limit_train:limit_train + limit_val]
    
    # Extract paths
    train_paths = [all_data['path'][idx] for idx in train_indices]
    val_paths = [all_data['path'][idx] for idx in val_indices]
    
    return train_paths, val_paths


def load_graph(path: str) -> Data:
    """Load a graph from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")
    return torch.load(path, map_location='cpu')


def compute_graph_features(graph: Data, exclude_activity_features: bool = True, 
                          use_only_first_5_features: bool = False,
                          node_feature_indices: Optional[List[int]] = None) -> np.ndarray:
    """
    Compute a feature vector for a graph.
    
    Features:
    - Number of nodes
    - Number of edges
    - Average node degree
    - Graph density
    - Mean node features (if available) - EXCLUDING activity features (20-27) if exclude_activity_features=True
    - Std node features (if available) - EXCLUDING activity features (20-27) if exclude_activity_features=True
    
    Args:
        exclude_activity_features: If True, exclude features 20-27 (activity/destination features) 
                                   which are often NaN. Default: True.
        use_only_first_5_features: If True, only use the first 5 graph-level features (nodes, edges, 
                                   degree, density, mean_node_feat). Excludes std_node_feat. Default: False.
        node_feature_indices: Optional list of node feature indices (0-19) to use when computing 
                             mean_node_feat. If None, uses all features 0-19 (or 0-19 excluding activity 
                             features if exclude_activity_features=True). Default: None.
    """
    num_nodes = graph.num_nodes
    num_edges = graph.num_edges
    
    # Average degree
    avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0.0
    
    # Graph density (for undirected graph)
    max_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 0
    density = num_edges / max_edges if max_edges > 0 else 0.0
    
    features = [num_nodes, num_edges, avg_degree, density]
    
    # Add node feature statistics if available
    if hasattr(graph, 'x') and graph.x is not None:
        node_features = graph.x.cpu().numpy() if torch.is_tensor(graph.x) else graph.x
        if node_features.size > 0 and len(node_features.shape) == 2:
            num_nodes_actual, num_features = node_features.shape
            
            # Apply feature filtering
            if node_feature_indices is not None:
                # Use only specified feature indices
                valid_indices = [idx for idx in node_feature_indices if 0 <= idx < num_features]
                if len(valid_indices) == 0:
                    raise ValueError(f"No valid feature indices in {node_feature_indices} for graph with {num_features} features")
                node_features_filtered = node_features[:, valid_indices]
            elif exclude_activity_features and num_features > 20:
                # Use only features 0-19 (network features, always valid)
                node_features_filtered = node_features[:, :20]
            else:
                node_features_filtered = node_features
            
            # Flatten filtered node features
            node_features_flat = node_features_filtered.flatten()
            
            # Remove NaN and inf values before computing statistics
            mask = ~np.isnan(node_features_flat) & ~np.isinf(node_features_flat)
            node_features_clean = node_features_flat[mask]
            
            if len(node_features_clean) > 0:
                mean_feat = np.mean(node_features_clean)
                std_feat = np.std(node_features_clean)
                # Replace NaN/inf with 0 (shouldn't happen after filtering, but be safe)
                mean_feat = mean_feat if np.isfinite(mean_feat) else 0.0
                std_feat = std_feat if np.isfinite(std_feat) else 0.0
                features.append(mean_feat)
                features.append(std_feat)
            else:
                # All values were NaN/inf, use 0
                features.extend([0.0, 0.0])
        elif node_features.size > 0:
            # Handle 1D case (shouldn't happen, but be safe)
            node_features_flat = node_features.flatten()
            mask = ~np.isnan(node_features_flat) & ~np.isinf(node_features_flat)
            node_features_clean = node_features_flat[mask]
            if len(node_features_clean) > 0:
                mean_feat = np.mean(node_features_clean)
                std_feat = np.std(node_features_clean)
                mean_feat = mean_feat if np.isfinite(mean_feat) else 0.0
                std_feat = std_feat if np.isfinite(std_feat) else 0.0
                features.append(mean_feat)
                features.append(std_feat)
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
    else:
        features.extend([0.0, 0.0])
    
    # Ensure all features are finite
    features = [f if np.isfinite(f) else 0.0 for f in features]
    
    # If requested, only use first 5 features (nodes, edges, degree, density, mean_node_feat)
    if use_only_first_5_features:
        features = features[:5]
    
    return np.array(features, dtype=np.float32)


def compute_wasserstein_distance(train_paths: List[str], val_paths: List[str], 
                                 use_capacity_reduction_only: bool = False) -> float:
    """
    Compute Wasserstein distance between training and validation graph sets.
    
    If use_capacity_reduction_only=True, extracts feature 2 (CAPACITY_REDUCTION) for all nodes
    in each graph and computes Wasserstein distance on the distribution of these values.
    If False (default), uses graph-level features (mean_node_feat).
    
    Args:
        use_capacity_reduction_only: If True, compute distance only on feature 2 (CAPACITY_REDUCTION).
                                    If False, use graph-level features (mean_node_feat). Default: False.
    """
    if use_capacity_reduction_only:
        # Extract feature 2 (CAPACITY_REDUCTION) for all nodes across all graphs
        train_capacity_reduction = []
        val_capacity_reduction = []
        
        print(f"  Loading {len(train_paths)} training graphs...")
        print(f"  Extracting feature 2 (CAPACITY_REDUCTION) for all nodes...")
        for path in train_paths:
            try:
                graph = load_graph(path)
                if hasattr(graph, 'x') and graph.x is not None:
                    node_features = graph.x.cpu().numpy() if torch.is_tensor(graph.x) else graph.x
                    if node_features.size > 0 and len(node_features.shape) == 2:
                        num_nodes, num_features = node_features.shape
                        if num_features > 2:
                            # Extract feature 2 (CAPACITY_REDUCTION) for all nodes
                            capacity_reduction = node_features[:, 2]  # Feature index 2
                            # Remove NaN/inf values
                            capacity_reduction = capacity_reduction[~np.isnan(capacity_reduction) & ~np.isinf(capacity_reduction)]
                            train_capacity_reduction.extend(capacity_reduction.tolist())
            except Exception as e:
                print(f"    Warning: Failed to load {path}: {e}")
                continue
        
        print(f"  Loading {len(val_paths)} validation graphs...")
        for path in val_paths:
            try:
                graph = load_graph(path)
                if hasattr(graph, 'x') and graph.x is not None:
                    node_features = graph.x.cpu().numpy() if torch.is_tensor(graph.x) else graph.x
                    if node_features.size > 0 and len(node_features.shape) == 2:
                        num_nodes, num_features = node_features.shape
                        if num_features > 2:
                            # Extract feature 2 (CAPACITY_REDUCTION) for all nodes
                            capacity_reduction = node_features[:, 2]  # Feature index 2
                            # Remove NaN/inf values
                            capacity_reduction = capacity_reduction[~np.isnan(capacity_reduction) & ~np.isinf(capacity_reduction)]
                            val_capacity_reduction.extend(capacity_reduction.tolist())
            except Exception as e:
                print(f"    Warning: Failed to load {path}: {e}")
                continue
        
        # Convert to numpy arrays
        train_capacity_reduction = np.array(train_capacity_reduction)
        val_capacity_reduction = np.array(val_capacity_reduction)
        
        if len(train_capacity_reduction) == 0 or len(val_capacity_reduction) == 0:
            print(f"    Warning: Empty arrays (train: {len(train_capacity_reduction)}, val: {len(val_capacity_reduction)})")
            return 0.0
        
        # Debug: Print statistics
        print(f"    CAPACITY_REDUCTION feature:")
        print(f"      Train: min={train_capacity_reduction.min():.4f}, max={train_capacity_reduction.max():.4f}, "
              f"mean={train_capacity_reduction.mean():.4f}, std={train_capacity_reduction.std():.4f}, "
              f"n={len(train_capacity_reduction)}")
        print(f"      Val:   min={val_capacity_reduction.min():.4f}, max={val_capacity_reduction.max():.4f}, "
              f"mean={val_capacity_reduction.mean():.4f}, std={val_capacity_reduction.std():.4f}, "
              f"n={len(val_capacity_reduction)}")
        print(f"      Train unique values: {len(np.unique(train_capacity_reduction))}, "
              f"Val unique values: {len(np.unique(val_capacity_reduction))}")
        
        # Check if all values are the same
        train_is_constant = len(np.unique(train_capacity_reduction)) == 1
        val_is_constant = len(np.unique(val_capacity_reduction)) == 1
        
        if train_is_constant and val_is_constant:
            # Constant arrays - distance is just the difference
            dist = abs(train_capacity_reduction[0] - val_capacity_reduction[0])
            print(f"    CAPACITY_REDUCTION: Constant arrays (train={train_capacity_reduction[0]:.4f}, "
                  f"val={val_capacity_reduction[0]:.4f}), distance = {dist}")
            return dist
        else:
            try:
                dist = wasserstein_distance(train_capacity_reduction, val_capacity_reduction)
                # Check if result is valid
                if np.isfinite(dist):
                    print(f"    CAPACITY_REDUCTION: Wasserstein distance = {dist}")
                    return dist
                else:
                    print(f"    Warning: Wasserstein distance returned non-finite value: {dist}")
                    # Fallback: use mean absolute difference
                    dist = np.abs(train_capacity_reduction.mean() - val_capacity_reduction.mean())
                    print(f"    Using mean absolute difference as fallback: {dist}")
                    return dist
            except Exception as e:
                print(f"    Warning: Error computing Wasserstein distance: {e}")
                # Fallback: use mean absolute difference
                dist = np.abs(train_capacity_reduction.mean() - val_capacity_reduction.mean())
                print(f"    Using mean absolute difference as fallback: {dist}")
                return dist
    else:
        # Fallback: use graph-level features (mean_node_feat approach)
        train_features = []
        val_features = []
        
        print(f"  Loading {len(train_paths)} training graphs...")
        print(f"  Computing graph-level features (mean_node_feat)...")
        for path in train_paths:
            try:
                graph = load_graph(path)
                features = compute_graph_features(graph, 
                                                 exclude_activity_features=True,
                                                 use_only_first_5_features=True)
                train_features.append(features)
            except Exception as e:
                print(f"    Warning: Failed to load {path}: {e}")
                continue
        
        print(f"  Loading {len(val_paths)} validation graphs...")
        for path in val_paths:
            try:
                graph = load_graph(path)
                features = compute_graph_features(graph,
                                                 exclude_activity_features=True,
                                                 use_only_first_5_features=True)
                val_features.append(features)
            except Exception as e:
                print(f"    Warning: Failed to load {path}: {e}")
                continue
        
        if len(train_features) == 0 or len(val_features) == 0:
            raise ValueError("No valid graphs found in train or val sets")
        
        train_features = np.array(train_features)
        val_features = np.array(val_features)
        
        # Check for NaN or inf values and provide detailed diagnostics
        train_has_nan = np.any(~np.isfinite(train_features))
        val_has_nan = np.any(~np.isfinite(val_features))
        
        if train_has_nan or val_has_nan:
            print(f"    Warning: Found NaN/inf in features")
            if train_has_nan:
                nan_count = np.isnan(train_features).sum()
                inf_count = np.isinf(train_features).sum()
                nan_per_feature = [np.isnan(train_features[:, i]).sum() for i in range(train_features.shape[1])]
                print(f"      Train: {nan_count} NaN, {inf_count} Inf values")
                print(f"      Train NaN per feature: {nan_per_feature}")
            if val_has_nan:
                nan_count = np.isnan(val_features).sum()
                inf_count = np.isinf(val_features).sum()
                nan_per_feature = [np.isnan(val_features[:, i]).sum() for i in range(val_features.shape[1])]
                print(f"      Val: {nan_count} NaN, {inf_count} Inf values")
                print(f"      Val NaN per feature: {nan_per_feature}")
            print(f"      Replacing with 0...")
            train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
            val_features = np.nan_to_num(val_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute Wasserstein distance ONLY for mean_node_feat (feature index 4)
        # Other features (nodes, edges, degree, density) are constant for graphs from the same city
        num_features = train_features.shape[1]
        mean_node_feat_idx = 4  # Index of mean_node_feat in the feature array
        
        if num_features <= mean_node_feat_idx:
            raise ValueError(f"Feature array has only {num_features} features, but need index {mean_node_feat_idx} for mean_node_feat")
        
        print(f"    Train features shape: {train_features.shape}, Val features shape: {val_features.shape}")
        print(f"    Computing Wasserstein distance ONLY for mean_node_feat (feature index {mean_node_feat_idx})")
        
        # Extract only mean_node_feat
        train_dist = train_features[:, mean_node_feat_idx]
        val_dist = val_features[:, mean_node_feat_idx]
        
        # Debug: Print statistics for mean_node_feat
        train_min, train_max = train_dist.min(), train_dist.max()
        train_mean, train_std = train_dist.mean(), train_dist.std()
        val_min, val_max = val_dist.min(), val_dist.max()
        val_mean, val_std = val_dist.mean(), val_dist.std()
        print(f"      mean_node_feat:")
        print(f"        Train: min={train_min:.4f}, max={train_max:.4f}, mean={train_mean:.4f}, std={train_std:.4f}")
        print(f"        Val:   min={val_min:.4f}, max={val_max:.4f}, mean={val_mean:.4f}, std={val_std:.4f}")
        print(f"        Train unique values: {len(np.unique(train_dist))}, Val unique values: {len(np.unique(val_dist))}")
        
        # Check if all values are the same
        train_is_constant = np.all(train_dist == train_dist[0])
        val_is_constant = np.all(val_dist == val_dist[0])
        
        if train_is_constant and val_is_constant:
            # Constant arrays - distance is just the difference
            dist = abs(train_dist[0] - val_dist[0])
            print(f"    mean_node_feat: Constant arrays (train={train_dist[0]:.4f}, val={val_dist[0]:.4f}), distance = {dist:.6f}")
            return dist
        else:
            try:
                dist = wasserstein_distance(train_dist, val_dist)
                # Check if result is valid
                if np.isfinite(dist):
                    print(f"    mean_node_feat: Wasserstein distance = {dist:.6f}")
                    return dist
                else:
                    print(f"    Warning: mean_node_feat: Wasserstein distance is {dist}, using fallback")
                    # Fallback: use mean absolute difference
                    fallback_dist = np.mean(np.abs(train_dist - val_dist))
                    print(f"    mean_node_feat: Using fallback (mean abs diff) = {fallback_dist:.6f}")
                    return fallback_dist
            except Exception as e:
                print(f"    Warning: mean_node_feat: Error computing Wasserstein distance: {e}, using fallback")
                # Fallback: use mean absolute difference
                fallback_dist = np.mean(np.abs(train_dist - val_dist))
                print(f"    mean_node_feat: Using fallback (mean abs diff) = {fallback_dist:.6f}")
                return fallback_dist


def match_finetuned_scratch_pairs(df: pd.DataFrame) -> List[Dict]:
    """
    Match finetuned and scratch runs with the same configuration.
    
    Returns a list of dictionaries with:
    - city, train_count, val_count
    - finetuned metrics (r^2, mse, spearman, pearson)
    - scratch metrics
    - performance differences
    """
    # Group runs by configuration
    runs_by_config = defaultdict(dict)
    unmatched_count = 0
    
    for _, row in df.iterrows():
        # Handle case-insensitive column access
        run_name = row.get('Name') or row.get('name') or row.get('run_name')
        if pd.isna(run_name) or run_name == '':
            continue
            
        run_info = extract_run_info(str(run_name))
        
        if run_info is None:
            unmatched_count += 1
            if unmatched_count == 1:  # Only print first unmatched run name as example
                print(f"  Note: Some run names don't match expected pattern (e.g., '{run_name}')")
            continue
        
        key = (run_info['city'], run_info['train_count'], run_info['val_count'])
        
        if run_info['is_finetuned']:
            runs_by_config[key]['finetuned'] = row
        else:
            runs_by_config[key]['scratch'] = row
    
    # Extract matched pairs
    matched_pairs = []
    
    for (city, train_count, val_count), runs in runs_by_config.items():
        if 'finetuned' not in runs or 'scratch' not in runs:
            print(f"  Warning: Missing pair for {city}, train={train_count}, val={val_count}")
            continue
        
        finetuned = runs['finetuned']
        scratch = runs['scratch']
        
        # Extract metrics (handle missing values and case variations)
        def safe_get(row, possible_cols, default=np.nan):
            """Try multiple possible column names (case variations)."""
            for col in possible_cols:
                val = row.get(col)
                if val is not None and not pd.isna(val) and val != '':
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue
            return default
        
        pair = {
            'city': city,
            'train_count': train_count,
            'val_count': val_count,
            'finetuned_r2': safe_get(finetuned, ['r^2', 'r2', 'R^2', 'R2', 'r_squared']),
            'scratch_r2': safe_get(scratch, ['r^2', 'r2', 'R^2', 'R2', 'r_squared']),
            'finetuned_mse': safe_get(finetuned, ['val_loss', 'val_loss', 'validation_loss', 'mse']),  # Using val_loss as MSE proxy
            'scratch_mse': safe_get(scratch, ['val_loss', 'val_loss', 'validation_loss', 'mse']),
            'finetuned_spearman': safe_get(finetuned, ['spearman', 'Spearman', 'spearman_corr']),
            'scratch_spearman': safe_get(scratch, ['spearman', 'Spearman', 'spearman_corr']),
            'finetuned_pearson': safe_get(finetuned, ['pearson', 'Pearson', 'pearson_corr']),
            'scratch_pearson': safe_get(scratch, ['pearson', 'Pearson', 'pearson_corr']),
        }
        
        # Compute performance differences
        # For R2, Spearman, Pearson: higher is better, so positive difference = finetuning helps
        # For MSE: lower is better, so we compute (scratch - finetuned) so positive = finetuning helps
        pair['r2_diff'] = pair['finetuned_r2'] - pair['scratch_r2']  # Positive = finetuning better
        pair['mse_diff'] = pair['scratch_mse'] - pair['finetuned_mse']  # Positive = finetuning better (lower MSE)
        pair['spearman_diff'] = pair['finetuned_spearman'] - pair['scratch_spearman']
        pair['pearson_diff'] = pair['finetuned_pearson'] - pair['scratch_pearson']
        
        matched_pairs.append(pair)
    
    return matched_pairs


def create_plots(matched_pairs: List[Dict], distances: Dict[Tuple[str, int, int], float],
                output_dir: str):
    """Create 4 plots (one for each metric) showing performance increase vs distance."""
    
    # Prepare data for plotting
    plot_data = []
    for pair in matched_pairs:
        key = (pair['city'], pair['train_count'], pair['val_count'])
        if key in distances:
            plot_data.append({
                'city': pair['city'],
                'distance': distances[key],
                'r2_diff': pair['r2_diff'],
                'mse_diff': pair['mse_diff'],
                'spearman_diff': pair['spearman_diff'],
                'pearson_diff': pair['pearson_diff'],
            })
    
    if not plot_data:
        print("  WARNING: No data to plot! (no matched pairs with distances)")
        print(f"    Matched pairs: {len(matched_pairs)}")
        print(f"    Distances computed: {len(distances)}")
        return
    
    print(f"  Preparing to plot {len(plot_data)} data points...")
    
    df_plot = pd.DataFrame(plot_data)
    
    # Get unique cities for color mapping
    cities = sorted(df_plot['city'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(cities)))
    city_colors = {city: colors[i] for i, city in enumerate(cities)}
    
    # Create 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Pretraining Benefit vs Train-Val Graph Distance', fontsize=16, fontweight='bold')
    
    metrics = [
        ('r2_diff', 'R² Difference (Finetuned - Scratch)', axes[0, 0]),
        ('mse_diff', 'MSE Difference (Scratch - Finetuned)', axes[0, 1]),
        ('spearman_diff', 'Spearman Difference (Finetuned - Scratch)', axes[1, 0]),
        ('pearson_diff', 'Pearson Difference (Finetuned - Scratch)', axes[1, 1]),
    ]
    
    for metric_key, metric_label, ax in metrics:
        # Plot scatter points for each city
        for city in cities:
            city_data = df_plot[df_plot['city'] == city]
            if len(city_data) > 0:
                ax.scatter(city_data['distance'], city_data[metric_key],
                          label=city, color=city_colors[city], alpha=0.7, s=100)
                
                # Add best fit line for this city
                if len(city_data) > 1:
                    x_city = city_data['distance'].values
                    y_city = city_data[metric_key].values
                    # Check for valid data (no NaN/inf and not all same x values)
                    if (len(np.unique(x_city)) > 1 and 
                        np.all(np.isfinite(x_city)) and np.all(np.isfinite(y_city))):
                        try:
                            # Fit linear regression
                            coeffs = np.polyfit(x_city, y_city, 1)
                            x_line = np.linspace(x_city.min(), x_city.max(), 100)
                            y_line = np.polyval(coeffs, x_line)
                            ax.plot(x_line, y_line, color=city_colors[city], linestyle='--', 
                                   alpha=0.6, linewidth=1.5, label=f'{city} fit')
                        except (np.linalg.LinAlgError, ValueError) as e:
                            # Skip if fit fails (e.g., constant x values)
                            pass
        
        # Add overall best fit line for all cities
        if len(df_plot) > 1:
            x_all = df_plot['distance'].values
            y_all = df_plot[metric_key].values
            # Check for valid data (no NaN/inf and not all same x values)
            if (len(np.unique(x_all)) > 1 and 
                np.all(np.isfinite(x_all)) and np.all(np.isfinite(y_all))):
                try:
                    coeffs_all = np.polyfit(x_all, y_all, 1)
                    x_line_all = np.linspace(x_all.min(), x_all.max(), 100)
                    y_line_all = np.polyval(coeffs_all, x_line_all)
                    ax.plot(x_line_all, y_line_all, color='black', linestyle='-', 
                           alpha=0.8, linewidth=2, label='Overall fit')
                except (np.linalg.LinAlgError, ValueError) as e:
                    # Skip if fit fails (e.g., constant x values)
                    pass
        
        ax.set_xlabel('Wasserstein Distance (Train vs Val Graphs)', fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(df_plot) > 1:
            corr = df_plot['distance'].corr(df_plot[metric_key])
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'pretraining_benefit_vs_distance.png')
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"\nSaved plot to: {output_path} (size: {file_size} bytes)")
        else:
            print(f"\nERROR: Plot file was not created at: {output_path}")
    except Exception as e:
        print(f"\nERROR saving plot: {e}")
        import traceback
        traceback.print_exc()
    
    # Also save individual plots
    for metric_key, metric_label, ax in metrics:
        fig_single, ax_single = plt.subplots(figsize=(10, 6))
        
        # Plot scatter points for each city
        for city in cities:
            city_data = df_plot[df_plot['city'] == city]
            if len(city_data) > 0:
                ax_single.scatter(city_data['distance'], city_data[metric_key],
                                 label=city, color=city_colors[city], alpha=0.7, s=100)
                
                # Add best fit line for this city
                if len(city_data) > 1:
                    x_city = city_data['distance'].values
                    y_city = city_data[metric_key].values
                    # Check for valid data (no NaN/inf and not all same x values)
                    if (len(np.unique(x_city)) > 1 and 
                        np.all(np.isfinite(x_city)) and np.all(np.isfinite(y_city))):
                        try:
                            # Fit linear regression
                            coeffs = np.polyfit(x_city, y_city, 1)
                            x_line = np.linspace(x_city.min(), x_city.max(), 100)
                            y_line = np.polyval(coeffs, x_line)
                            ax_single.plot(x_line, y_line, color=city_colors[city], linestyle='--', 
                                          alpha=0.6, linewidth=1.5, label=f'{city} fit')
                        except (np.linalg.LinAlgError, ValueError) as e:
                            # Skip if fit fails (e.g., constant x values)
                            pass
        
        # Add overall best fit line for all cities
        if len(df_plot) > 1:
            x_all = df_plot['distance'].values
            y_all = df_plot[metric_key].values
            # Check for valid data (no NaN/inf and not all same x values)
            if (len(np.unique(x_all)) > 1 and 
                np.all(np.isfinite(x_all)) and np.all(np.isfinite(y_all))):
                try:
                    coeffs_all = np.polyfit(x_all, y_all, 1)
                    x_line_all = np.linspace(x_all.min(), x_all.max(), 100)
                    y_line_all = np.polyval(coeffs_all, x_line_all)
                    ax_single.plot(x_line_all, y_line_all, color='black', linestyle='-', 
                                  alpha=0.8, linewidth=2, label='Overall fit')
                except (np.linalg.LinAlgError, ValueError) as e:
                    # Skip if fit fails (e.g., constant x values)
                    pass
        
        ax_single.set_xlabel('Wasserstein Distance (Train vs Val Graphs)', fontsize=12)
        ax_single.set_ylabel(metric_label, fontsize=12)
        ax_single.set_title(metric_label, fontsize=14, fontweight='bold')
        ax_single.legend(loc='best', fontsize=10)
        ax_single.grid(True, alpha=0.3)
        
        if len(df_plot) > 1:
            corr = df_plot['distance'].corr(df_plot[metric_key])
            ax_single.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                          transform=ax_single.transAxes, fontsize=11,
                          verticalalignment='top', 
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        metric_name = metric_key.replace('_diff', '')
        output_path_single = os.path.join(output_dir, f'pretraining_benefit_vs_distance_{metric_name}.png')
        try:
            plt.savefig(output_path_single, dpi=300, bbox_inches='tight')
            if os.path.exists(output_path_single):
                print(f"  Saved individual plot: {output_path_single}")
            else:
                print(f"  ERROR: Individual plot file was not created at: {output_path_single}")
        except Exception as e:
            print(f"  ERROR saving individual plot {metric_name}: {e}")
        finally:
            plt.close(fig_single)
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze relationship between train-val graph distance and pretraining benefit."
    )
    parser.add_argument("--wandb_csv", type=str, required=True,
                       help="Path to wandb export CSV file")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset directory (e.g., data/bavaria/inductive_data/training_data/kreisfreistadt). Can be relative to project root or absolute.")
    parser.add_argument("--output_dir", type=str, default="analysis_results",
                       help="Directory to save plots and results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed used for train/val split (default: 42)")
    parser.add_argument("--use_only_first_5_features", action="store_true",
                       help="If set, only use first 5 graph-level features (nodes, edges, degree, density, mean_node_feat)")
    parser.add_argument("--include_non_finished", action="store_true",
                       help="If set, include runs that are not 'finished' (default: only finished runs)")
    parser.add_argument("--node_feature_indices", type=str, default=None,
                       help="Comma-separated list of node feature indices (0-19) to use for mean_node_feat computation. "
                            "Example: '0,1,2,3,10' to use only VOL_BASE_CASE, CAPACITY_BASE_CASE, CAPACITY_REDUCTION, "
                            "FREESPEED, and LENGTH. If not specified, uses all features 0-19 (excluding 20-27).")
    parser.add_argument("--use_only_capacity_reduction", action="store_true", default=False,
                       help="If set, compute Wasserstein distance only on feature 2 (CAPACITY_REDUCTION). "
                            "Default (if not set): use graph-level features (mean_node_feat).")
    
    args = parser.parse_args()
    
    # Determine use_capacity_reduction_only based on flags
    # Default is False (use graph-level features), unless --use_only_capacity_reduction is set
    use_capacity_reduction_only = args.use_only_capacity_reduction
    
    # Parse node_feature_indices if provided
    node_feature_indices = None
    if args.node_feature_indices:
        try:
            node_feature_indices = [int(x.strip()) for x in args.node_feature_indices.split(',')]
            # Validate indices are in range 0-19
            invalid_indices = [idx for idx in node_feature_indices if idx < 0 or idx > 19]
            if invalid_indices:
                raise ValueError(f"Invalid feature indices: {invalid_indices}. Must be between 0 and 19.")
            print(f"Using node feature indices: {node_feature_indices}")
        except ValueError as e:
            raise ValueError(f"Error parsing --node_feature_indices '{args.node_feature_indices}': {e}")
    
    # Resolve dataset path relative to project root (same as finetune_models.py)
    # Repo root: scripts/analysis/analyze_pretraining_benefit_vs_distance.py → go two levels up
    project_root = Path(__file__).resolve().parents[2]
    
    # If dataset_path is relative, resolve it relative to project root
    if not os.path.isabs(args.dataset_path):
        dataset_path = project_root / args.dataset_path
    else:
        dataset_path = Path(args.dataset_path)
    
    dataset_path = dataset_path.resolve()
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    print(f"Using dataset path: {dataset_path}")
    
    # Resolve wandb_csv path (can be relative to current directory or absolute)
    if not os.path.isabs(args.wandb_csv):
        wandb_csv_path = Path(args.wandb_csv).resolve()
    else:
        wandb_csv_path = Path(args.wandb_csv)
    
    if not wandb_csv_path.exists():
        raise ValueError(f"WandB CSV file does not exist: {wandb_csv_path}")
    
    # Resolve output_dir relative to project root
    if not os.path.isabs(args.output_dir):
        output_dir = project_root / args.output_dir
    else:
        output_dir = Path(args.output_dir)
    
    output_dir = output_dir.resolve()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Analyzing Pretraining Benefit vs Train-Val Graph Distance")
    print("=" * 80)
    
    # Step 1: Parse CSV
    print("\n[Step 1] Parsing wandb CSV...")
    df = parse_wandb_csv(str(wandb_csv_path), only_finished=not args.include_non_finished)
    print(f"  Loaded {len(df)} runs from CSV")
    
    # Step 2: Match finetuned/scratch pairs
    print("\n[Step 2] Matching finetuned and scratch runs...")
    matched_pairs = match_finetuned_scratch_pairs(df)
    print(f"  Found {len(matched_pairs)} matched pairs")
    
    # Filter to only include pairs with at least 100 validation graphs
    print("\n[Step 2.5] Filtering pairs with val_count >= 100...")
    original_count = len(matched_pairs)
    matched_pairs = [pair for pair in matched_pairs if pair['val_count'] >= 100]
    print(f"  Filtered to {len(matched_pairs)}/{original_count} pairs with val_count >= 100")
    
    # Step 3: Recover splits and compute distances
    print("\n[Step 3] Recovering train/val splits and computing Wasserstein distances...")
    distances = {}
    
    for i, pair in enumerate(matched_pairs):
        city = pair['city']
        train_count = pair['train_count']
        val_count = pair['val_count']
        key = (city, train_count, val_count)
        
        print(f"\n[{i+1}/{len(matched_pairs)}] Processing {city}, train={train_count}, val={val_count}")
        
        try:
            train_paths, val_paths = recover_train_val_split(
                city, train_count, val_count, str(dataset_path), seed=args.seed
            )
            
            distance = compute_wasserstein_distance(train_paths, val_paths,
                                                   use_capacity_reduction_only=use_capacity_reduction_only)
            distances[key] = distance
            # Print with full precision (no rounding)
            print(f"  Wasserstein distance: {distance}")
            
        except Exception as e:
            print(f"  Error processing {city}: {e}")
            continue
    
    # Step 4: Create plots
    print("\n[Step 4] Creating plots...")
    try:
        create_plots(matched_pairs, distances, str(output_dir))
    except Exception as e:
        print(f"  ERROR creating plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Save summary statistics
    print("\n[Step 5] Saving summary statistics...")
    summary_data = []
    for pair in matched_pairs:
        key = (pair['city'], pair['train_count'], pair['val_count'])
        if key in distances:
            summary_data.append({
                'city': pair['city'],
                'train_count': pair['train_count'],
                'val_count': pair['val_count'],
                'distance': distances[key],
                'r2_diff': pair['r2_diff'],
                'mse_diff': pair['mse_diff'],
                'spearman_diff': pair['spearman_diff'],
                'pearson_diff': pair['pearson_diff'],
                'finetuned_r2': pair['finetuned_r2'],
                'scratch_r2': pair['scratch_r2'],
            })
    
    summary_df = None
    if len(summary_data) == 0:
        print("  WARNING: No summary data to save (no matched pairs with distances)")
    else:
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / 'summary_statistics.csv'
        try:
            summary_df.to_csv(summary_path, index=False)
            if os.path.exists(summary_path):
                file_size = os.path.getsize(summary_path)
                print(f"  Saved summary to: {summary_path} (size: {file_size} bytes)")
                print(f"  Summary contains {len(summary_df)} rows")
                print(f"  Summary columns: {list(summary_df.columns)}")
            else:
                print(f"  ERROR: Summary file was not created at: {summary_path}")
        except Exception as e:
            print(f"  ERROR saving summary: {e}")
            import traceback
            traceback.print_exc()
    
    # Print correlation statistics
    if summary_df is not None and len(summary_df) > 1:
        print("\n" + "=" * 80)
        print("Correlation Analysis:")
        print("=" * 80)
        for metric in ['r2_diff', 'mse_diff', 'spearman_diff', 'pearson_diff']:
            if metric in summary_df.columns:
                corr = summary_df['distance'].corr(summary_df[metric])
                print(f"  Distance vs {metric}: {corr:.4f}")
    elif summary_df is not None and len(summary_df) == 1:
        print(f"\n  Note: Only {len(summary_df)} data point(s), cannot compute correlation")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print(f"  Total matched pairs: {len(matched_pairs)}")
    print(f"  Pairs with distances: {len(distances)}")
    print(f"  Summary data points: {len(summary_data)}")


if __name__ == "__main__":
    main()

