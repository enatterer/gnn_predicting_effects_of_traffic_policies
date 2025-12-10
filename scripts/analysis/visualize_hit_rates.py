#!/usr/bin/env python3
"""
Visualize hit rates vs train-val graph distance for finetuned vs scratch models.

This script:
1. Loads hit rate evaluation results
2. Matches finetuned and scratch runs
3. Computes Wasserstein distances between train and validation sets
4. Creates visualizations with distance on x-axis and hit rates on y-axis
   (similar to pretraining_benefit_vs_distance.png)

Usage:
    python scripts/analysis/visualize_hit_rates.py \
        --results_csv data/analysis_results/hit_rates_evaluation.csv \
        --dataset_path data/bavaria/inductive_data/training_data/kreisfreistadt \
        --output_dir data/analysis_results \
        --seed 42
"""

import argparse
import os
import sys
import random as _rnd  # Match finetune_models.py exactly
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

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
plt.rcParams['figure.figsize'] = (14, 10)


def load_results(csv_path: str) -> pd.DataFrame:
    """Load hit rate evaluation results from CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from CSV")
    print(f"Columns: {list(df.columns)}")
    return df


def match_finetuned_scratch_pairs(df: pd.DataFrame) -> List[Dict]:
    """
    Match finetuned and scratch runs with the same configuration.
    
    Returns a list of dictionaries with:
    - city, train_count, val_count
    - finetuned hit rates
    - scratch hit rates
    - hit rate differences
    """
    runs_by_config = defaultdict(dict)
    
    for _, row in df.iterrows():
        method = row.get('method', '')
        city = row.get('city', '')
        train_count = row.get('train_count', 0)
        val_count = row.get('val_count', 0)
        
        if not method or not city:
            continue
        
        key = (city, train_count, val_count)
        
        if method == 'finetuned':
            runs_by_config[key]['finetuned'] = row
        elif method == 'scratch':
            runs_by_config[key]['scratch'] = row
    
    # Extract matched pairs
    matched_pairs = []
    
    for (city, train_count, val_count), runs in runs_by_config.items():
        if 'finetuned' not in runs or 'scratch' not in runs:
            continue
        
        finetuned = runs['finetuned']
        scratch = runs['scratch']
        
        # Extract hit rate metrics
        hit_rate_metrics = [
            'top_1_hit_rate', 'bottom_1_hit_rate',
            'top_5_hit_rate', 'bottom_5_hit_rate',
            'top_10_hit_rate', 'bottom_10_hit_rate'
        ]
        
        pair = {
            'city': city,
            'train_count': train_count,
            'val_count': val_count,
        }
        
        # Add finetuned and scratch hit rates
        for metric in hit_rate_metrics:
            if metric in finetuned and metric in scratch:
                finetuned_val = finetuned[metric]
                scratch_val = scratch[metric]
                pair[f'finetuned_{metric}'] = finetuned_val
                pair[f'scratch_{metric}'] = scratch_val
                pair[f'{metric}_diff'] = finetuned_val - scratch_val  # Positive = finetuning better
        
        # Also add other metrics
        for metric in ['r_squared', 'spearman', 'pearson', 'val_loss']:
            if metric in finetuned and metric in scratch:
                pair[f'finetuned_{metric}'] = finetuned[metric]
                pair[f'scratch_{metric}'] = scratch[metric]
                if metric == 'val_loss':
                    pair[f'{metric}_diff'] = scratch[metric] - finetuned[metric]  # Positive = finetuning better (lower loss)
                else:
                    pair[f'{metric}_diff'] = finetuned[metric] - scratch[metric]  # Positive = finetuning better
        
        matched_pairs.append(pair)
    
    return matched_pairs


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
                          node_feature_indices: list = None) -> np.ndarray:
    """
    Compute a feature vector for a graph (same as analyze_pretraining_benefit_vs_distance.py).
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
                valid_indices = [idx for idx in node_feature_indices if 0 <= idx < num_features]
                if len(valid_indices) == 0:
                    raise ValueError(f"No valid feature indices in {node_feature_indices} for graph with {num_features} features")
                node_features_filtered = node_features[:, valid_indices]
            elif exclude_activity_features and num_features > 20:
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
                mean_feat = mean_feat if np.isfinite(mean_feat) else 0.0
                std_feat = std_feat if np.isfinite(std_feat) else 0.0
                features.append(mean_feat)
                features.append(std_feat)
            else:
                features.extend([0.0, 0.0])
        else:
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
    
    # Ensure all features are finite
    features = [f if np.isfinite(f) else 0.0 for f in features]
    
    # If requested, only use first 5 features (nodes, edges, degree, density, mean_node_feat)
    if use_only_first_5_features:
        features = features[:5]
    
    return np.array(features, dtype=np.float32)


def compute_wasserstein_distance(train_paths: List[str], val_paths: List[str], 
                                 use_all_features: bool = True,
                                 verbose: bool = False) -> float:
    """
    Compute Wasserstein distance between training and validation graph sets.
    Uses graph-level features (mean_node_feat approach).
    """
    train_features = []
    val_features = []
    
    # Determine which node feature indices to use
    if use_all_features:
        node_feature_indices = list(range(20))  # Features 0-19
    else:
        node_feature_indices = [0, 1, 2, 3, 10]  # 5 base features
    
    for path in train_paths:
        try:
            graph = load_graph(path)
            features = compute_graph_features(graph, 
                                             exclude_activity_features=True,
                                             use_only_first_5_features=True,
                                             node_feature_indices=node_feature_indices)
            train_features.append(features)
        except Exception as e:
            if verbose:
                print(f"    Warning: Failed to load {path}: {e}")
            continue
    
    for path in val_paths:
        try:
            graph = load_graph(path)
            features = compute_graph_features(graph,
                                             exclude_activity_features=True,
                                             use_only_first_5_features=True,
                                             node_feature_indices=node_feature_indices)
            val_features.append(features)
        except Exception as e:
            if verbose:
                print(f"    Warning: Failed to load {path}: {e}")
            continue
    
    if len(train_features) == 0 or len(val_features) == 0:
        raise ValueError("No valid graphs found in train or val sets")
    
    train_features = np.array(train_features)
    val_features = np.array(val_features)
    
    # Replace NaN/inf with 0
    train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
    val_features = np.nan_to_num(val_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute Wasserstein distance ONLY for mean_node_feat (feature index 4)
    num_features = train_features.shape[1]
    mean_node_feat_idx = 4
    
    if num_features <= mean_node_feat_idx:
        raise ValueError(f"Feature array has only {num_features} features, but need index {mean_node_feat_idx}")
    
    train_dist = train_features[:, mean_node_feat_idx]
    val_dist = val_features[:, mean_node_feat_idx]
    
    # Check if all values are the same
    train_is_constant = np.all(train_dist == train_dist[0])
    val_is_constant = np.all(val_dist == val_dist[0])
    
    if train_is_constant and val_is_constant:
        dist = abs(train_dist[0] - val_dist[0])
        return dist
    else:
        try:
            dist = wasserstein_distance(train_dist, val_dist)
            if np.isfinite(dist):
                return dist
            else:
                return np.mean(np.abs(train_dist - val_dist))
        except Exception as e:
            return np.mean(np.abs(train_dist - val_dist))


def create_plots(matched_pairs: List[Dict], distances: Dict[Tuple[str, int, int], float], 
                 output_dir: str):
    """Create plots showing hit rates vs Wasserstein distance (like pretraining_benefit_vs_distance.png)."""
    
    if not matched_pairs:
        print("  WARNING: No matched pairs to plot!")
        return
    
    # Prepare data for plotting
    plot_data = []
    for pair in matched_pairs:
        key = (pair['city'], pair['train_count'], pair['val_count'])
        if key in distances:
            plot_data.append({
                'city': pair['city'],
                'distance': distances[key],
                **pair  # Include all pair data
            })
    
    if not plot_data:
        print("  WARNING: No data points with distances to plot!")
        return
    
    df_plot = pd.DataFrame(plot_data)
    
    # Get unique cities for color mapping
    cities = sorted(df_plot['city'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(cities)))
    city_colors = {city: colors[i] for i, city in enumerate(cities)}
    
    # Hit rate metrics to plot
    hit_rate_metrics = [
        ('top_1_hit_rate', 'Top 1% Hit Rate Difference'),
        ('bottom_1_hit_rate', 'Bottom 1% Hit Rate Difference'),
        ('top_5_hit_rate', 'Top 5% Hit Rate Difference'),
        ('bottom_5_hit_rate', 'Bottom 5% Hit Rate Difference'),
        ('top_10_hit_rate', 'Top 10% Hit Rate Difference'),
        ('bottom_10_hit_rate', 'Bottom 10% Hit Rate Difference'),
    ]
    
    # Create 6 subplots (one for each hit rate metric)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle('Hit Rate Difference (Finetuned - Scratch) vs Train-Val Graph Distance', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (metric_key, metric_label) in enumerate(hit_rate_metrics):
        ax = axes[idx]
        diff_key = f'{metric_key}_diff'
        
        if diff_key not in df_plot.columns:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Plot scatter points for each city
        for city in cities:
            city_data = df_plot[df_plot['city'] == city]
            if len(city_data) > 0 and diff_key in city_data.columns:
                x_vals = city_data['distance'].values
                y_vals = city_data[diff_key].values
                
                # Filter out NaN values
                mask = ~np.isnan(y_vals) & ~np.isnan(x_vals) & np.isfinite(x_vals) & np.isfinite(y_vals)
                if np.any(mask):
                    ax.scatter(x_vals[mask], y_vals[mask],
                              label=city, color=city_colors[city], alpha=0.7, s=100)
                    
                    # Add best fit line for this city
                    if np.sum(mask) > 1:
                        x_city = x_vals[mask]
                        y_city = y_vals[mask]
                        if len(np.unique(x_city)) > 1:
                            try:
                                coeffs = np.polyfit(x_city, y_city, 1)
                                x_line = np.linspace(x_city.min(), x_city.max(), 100)
                                y_line = np.polyval(coeffs, x_line)
                                ax.plot(x_line, y_line, color=city_colors[city], linestyle='--', 
                                       alpha=0.6, linewidth=1.5, label=f'{city} fit')
                            except (np.linalg.LinAlgError, ValueError):
                                pass
        
        # Add overall best fit line for all cities
        if len(df_plot) > 1:
            x_all = df_plot['distance'].values
            y_all = df_plot[diff_key].values
            mask_all = ~np.isnan(y_all) & ~np.isnan(x_all) & np.isfinite(x_all) & np.isfinite(y_all)
            if np.sum(mask_all) > 1 and len(np.unique(x_all[mask_all])) > 1:
                try:
                    coeffs_all = np.polyfit(x_all[mask_all], y_all[mask_all], 1)
                    x_line_all = np.linspace(x_all[mask_all].min(), x_all[mask_all].max(), 100)
                    y_line_all = np.polyval(coeffs_all, x_line_all)
                    ax.plot(x_line_all, y_line_all, color='black', linestyle='-', 
                           alpha=0.8, linewidth=2, label='Overall fit')
                except (np.linalg.LinAlgError, ValueError):
                    pass
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        ax.set_xlabel('Wasserstein Distance (Train vs Val Graphs)', fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(df_plot) > 1:
            corr = df_plot['distance'].corr(df_plot[diff_key])
            if not np.isnan(corr):
                ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'hit_rates_vs_distance.png')
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    except Exception as e:
        print(f"ERROR saving plot: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize hit rates vs train-val graph distance for finetuned vs scratch models."
    )
    parser.add_argument("--results_csv", type=str, required=True,
                       help="Path to hit rate evaluation CSV file")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset directory (e.g., data/bavaria/inductive_data/training_data/kreisfreistadt)")
    parser.add_argument("--output_dir", type=str, default="data/analysis_results",
                       help="Directory to save plots")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed used for train/val split (default: 42)")
    parser.add_argument("--use_all_features", action="store_true", default=True,
                       help="Use all features (0-19) for distance computation. Default: True")
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).resolve().parents[2]
    
    if not Path(args.results_csv).is_absolute():
        csv_path = project_root / args.results_csv
    else:
        csv_path = Path(args.results_csv)
    csv_path = csv_path.resolve()
    
    if not csv_path.exists():
        raise ValueError(f"Results CSV file does not exist: {csv_path}")
    
    if not Path(args.dataset_path).is_absolute():
        dataset_path = project_root / args.dataset_path
    else:
        dataset_path = Path(args.dataset_path)
    dataset_path = dataset_path.resolve()
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    if not Path(args.output_dir).is_absolute():
        output_dir = project_root / args.output_dir
    else:
        output_dir = Path(args.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Visualizing Hit Rates vs Train-Val Graph Distance")
    print("=" * 80)
    
    # Load results
    print("\n[Step 1] Loading results...")
    df = load_results(str(csv_path))
    
    # Match pairs
    print("\n[Step 2] Matching finetuned and scratch runs...")
    matched_pairs = match_finetuned_scratch_pairs(df)
    print(f"  Found {len(matched_pairs)} matched pairs")
    
    # Compute Wasserstein distances
    print("\n[Step 3] Computing Wasserstein distances...")
    distances = {}
    
    for i, pair in enumerate(matched_pairs):
        city = pair['city']
        train_count = pair['train_count']
        val_count = pair['val_count']
        key = (city, train_count, val_count)
        
        print(f"  [{i+1}/{len(matched_pairs)}] Computing distance for {city}, train={train_count}, val={val_count}")
        
        try:
            train_paths, val_paths = recover_train_val_split(
                city, train_count, val_count, str(dataset_path), seed=args.seed
            )
            
            distance = compute_wasserstein_distance(
                train_paths, val_paths,
                use_all_features=args.use_all_features,
                verbose=False
            )
            distances[key] = distance
            print(f"    Distance: {distance:.6f}")
            
        except Exception as e:
            print(f"    Error computing distance: {e}")
            continue
    
    # Create plots
    print("\n[Step 4] Creating plots...")
    try:
        create_plots(matched_pairs, distances, str(output_dir))
    except Exception as e:
        print(f"  ERROR creating plots: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

