#!/usr/bin/env python3
"""
Diagnostic script to investigate NaN values in graph features.

This script loads graphs and analyzes where NaN values come from.
"""

import os
import sys
import random as _rnd
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# Add the 'scripts' directory to Python Path
scripts_path = Path(__file__).resolve().parents[1]
if str(scripts_path) not in sys.path:
    sys.path.append(str(scripts_path))

from training.help_functions import load_metadata_from_disk

# Import from the same directory
import importlib.util
analysis_script_path = Path(__file__).parent / "analyze_pretraining_benefit_vs_distance.py"
spec = importlib.util.spec_from_file_location("analyze_pretraining_benefit_vs_distance", analysis_script_path)
ana_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ana_module)  # Execute the module to load functions

# Extract the functions we need
recover_train_val_split = ana_module.recover_train_val_split
load_graph = ana_module.load_graph
compute_graph_features = ana_module.compute_graph_features


def analyze_graph_features(graph: Data, graph_id: str = "unknown") -> dict:
    """Analyze a single graph for NaN/inf values."""
    stats = {
        'graph_id': graph_id,
        'num_nodes': graph.num_nodes,
        'num_edges': graph.num_edges,
        'has_x': hasattr(graph, 'x') and graph.x is not None,
        'x_shape': None,
        'x_nan_count': 0,
        'x_inf_count': 0,
        'x_total_elements': 0,
        'x_nan_percentage': 0.0,
        'x_inf_percentage': 0.0,
        'x_nan_per_feature': None,  # NaN count per feature dimension
        'x_nan_per_feature_pct': None,  # NaN percentage per feature dimension
        'nodes_with_any_nan': 0,  # Number of nodes that have at least one NaN
        'nodes_with_all_nan': 0,  # Number of nodes that have all NaN features
        'computed_features': None,
        'computed_features_nan_count': 0,
        'computed_features_inf_count': 0,
    }
    
    # Analyze node features
    if stats['has_x']:
        node_features = graph.x.cpu().numpy() if torch.is_tensor(graph.x) else graph.x
        stats['x_shape'] = node_features.shape
        stats['x_total_elements'] = node_features.size
        
        if node_features.size > 0:
            nan_count = np.isnan(node_features).sum()
            inf_count = np.isinf(node_features).sum()
            stats['x_nan_count'] = nan_count
            stats['x_inf_count'] = inf_count
            stats['x_nan_percentage'] = (nan_count / node_features.size) * 100
            stats['x_inf_percentage'] = (inf_count / node_features.size) * 100
            
            # Analyze NaN per feature dimension
            if len(node_features.shape) == 2:
                num_nodes, num_features = node_features.shape
                nan_per_feature = [np.isnan(node_features[:, i]).sum() for i in range(num_features)]
                nan_per_feature_pct = [(nan_per_feature[i] / num_nodes) * 100 for i in range(num_features)]
                stats['x_nan_per_feature'] = nan_per_feature
                stats['x_nan_per_feature_pct'] = nan_per_feature_pct
                
                # Count nodes with any NaN
                nodes_with_any_nan = np.any(np.isnan(node_features), axis=1).sum()
                stats['nodes_with_any_nan'] = nodes_with_any_nan
                
                # Count nodes with all NaN
                nodes_with_all_nan = np.all(np.isnan(node_features), axis=1).sum()
                stats['nodes_with_all_nan'] = nodes_with_all_nan
    
    # Compute features and check for NaN/inf
    try:
        computed_features = compute_graph_features(graph)
        stats['computed_features'] = computed_features
        stats['computed_features_nan_count'] = np.isnan(computed_features).sum()
        stats['computed_features_inf_count'] = np.isinf(computed_features).sum()
    except Exception as e:
        stats['error'] = str(e)
    
    return stats


def diagnose_city(city: str, dataset_path: str, sample_size: int = 10):
    """Diagnose NaN values for a specific city."""
    print(f"\n{'='*80}")
    print(f"Diagnosing city: {city}")
    print(f"{'='*80}")
    
    # Recover a sample split
    try:
        train_paths, val_paths = recover_train_val_split(
            city, sample_size, sample_size, dataset_path, seed=42
        )
    except Exception as e:
        print(f"Error recovering split: {e}")
        return None
    
    # Analyze graphs
    all_stats = []
    
    print(f"\nAnalyzing {len(train_paths)} training graphs...")
    for i, path in enumerate(train_paths[:sample_size]):  # Limit to sample_size
        try:
            graph = load_graph(path)
            stats = analyze_graph_features(graph, f"train_{i}")
            stats['split'] = 'train'
            stats['path'] = os.path.basename(path)
            all_stats.append(stats)
        except Exception as e:
            print(f"  Error loading {os.path.basename(path)}: {e}")
    
    print(f"\nAnalyzing {len(val_paths)} validation graphs...")
    for i, path in enumerate(val_paths[:sample_size]):  # Limit to sample_size
        try:
            graph = load_graph(path)
            stats = analyze_graph_features(graph, f"val_{i}")
            stats['split'] = 'val'
            stats['path'] = os.path.basename(path)
            all_stats.append(stats)
        except Exception as e:
            print(f"  Error loading {os.path.basename(path)}: {e}")
    
    # Summary statistics
    if all_stats:
        df = pd.DataFrame(all_stats)
        
        print(f"\n{'='*80}")
        print(f"Summary for {city}:")
        print(f"{'='*80}")
        
        print(f"\nNode Features (x):")
        print(f"  Graphs with x: {df['has_x'].sum()}/{len(df)}")
        if df['has_x'].any():
            print(f"  Average x shape: {df[df['has_x']]['x_shape'].iloc[0] if df['has_x'].any() else 'N/A'}")
            print(f"  Graphs with NaN in x: {(df['x_nan_count'] > 0).sum()}")
            print(f"  Graphs with Inf in x: {(df['x_inf_count'] > 0).sum()}")
            print(f"  Average NaN percentage: {df['x_nan_percentage'].mean():.2f}%")
            print(f"  Average Inf percentage: {df['x_inf_percentage'].mean():.2f}%")
            print(f"  Max NaN percentage: {df['x_nan_percentage'].max():.2f}%")
            print(f"  Max Inf percentage: {df['x_inf_percentage'].max():.2f}%")
        
        print(f"\nComputed Features:")
        print(f"  Graphs with NaN in computed features: {(df['computed_features_nan_count'] > 0).sum()}")
        print(f"  Graphs with Inf in computed features: {(df['computed_features_inf_count'] > 0).sum()}")
        
        # Show detailed breakdown for graphs with issues
        problematic = df[(df['x_nan_count'] > 0) | (df['x_inf_count'] > 0) | 
                        (df['computed_features_nan_count'] > 0) | (df['computed_features_inf_count'] > 0)]
        
        if len(problematic) > 0:
            print(f"\n{'='*80}")
            print(f"Problematic graphs ({len(problematic)}):")
            print(f"{'='*80}")
            for idx, row in problematic.iterrows():
                print(f"\n  Graph: {row['path']} ({row['split']})")
                print(f"    Nodes: {row['num_nodes']}, Edges: {row['num_edges']}")
                if row['has_x']:
                    print(f"    x shape: {row['x_shape']}")
                    print(f"    x NaN: {row['x_nan_count']} ({row['x_nan_percentage']:.2f}% of all feature values)")
                    print(f"    x Inf: {row['x_inf_count']} ({row['x_inf_percentage']:.2f}%)")
                    
                    # Show detailed breakdown
                    if row['x_nan_per_feature'] is not None and row['x_nan_count'] > 0:
                        print(f"    NaN breakdown:")
                        print(f"      Nodes with any NaN: {row['nodes_with_any_nan']} ({row['nodes_with_any_nan']/row['num_nodes']*100:.1f}% of nodes)")
                        print(f"      Nodes with all NaN: {row['nodes_with_all_nan']} ({row['nodes_with_all_nan']/row['num_nodes']*100:.1f}% of nodes)")
                        print(f"      NaN per feature dimension:")
                        nan_per_feat = row['x_nan_per_feature']
                        nan_per_feat_pct = row['x_nan_per_feature_pct']
                        for feat_idx, (nan_count, nan_pct) in enumerate(zip(nan_per_feat, nan_per_feat_pct)):
                            if nan_count > 0:
                                print(f"        Feature {feat_idx}: {nan_count} NaN ({nan_pct:.1f}% of nodes)")
                    
                print(f"    Computed features NaN: {row['computed_features_nan_count']}")
                print(f"    Computed features Inf: {row['computed_features_inf_count']}")
                if row['computed_features'] is not None:
                    print(f"    Computed features: {row['computed_features']}")
        
        return df
    else:
        print("No graphs analyzed successfully")
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Diagnose NaN values in graph features."
    )
    parser.add_argument("--city", type=str, required=False,
                       help="City to diagnose (required if --all_cities is not used)")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--sample_size", type=int, default=10,
                       help="Number of graphs to sample from each split")
    parser.add_argument("--all_cities", action="store_true",
                       help="Diagnose all cities found in dataset")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all_cities and not args.city:
        parser.error("Either --city or --all_cities must be provided")
    
    # Resolve paths
    project_root = Path(__file__).resolve().parents[2]
    if not os.path.isabs(args.dataset_path):
        dataset_path = project_root / args.dataset_path
    else:
        dataset_path = Path(args.dataset_path)
    dataset_path = dataset_path.resolve()
    
    print("="*80)
    print("NaN Value Diagnosis")
    print("="*80)
    print(f"Dataset path: {dataset_path}")
    
    if args.all_cities:
        # Find all cities
        cities = [d.name for d in dataset_path.iterdir() 
                 if d.is_dir() and (d / "metadata.json").exists()]
        print(f"\nFound {len(cities)} cities: {cities}")
        
        all_results = {}
        for city in sorted(cities):
            try:
                df = diagnose_city(city, str(dataset_path), args.sample_size)
                if df is not None:
                    all_results[city] = df
            except Exception as e:
                print(f"\nError diagnosing {city}: {e}")
        
        # Overall summary
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")
        
        total_graphs = sum(len(df) for df in all_results.values())
        total_with_nan_x = sum((df['x_nan_count'] > 0).sum() for df in all_results.values())
        total_with_inf_x = sum((df['x_inf_count'] > 0).sum() for df in all_results.values())
        total_with_nan_computed = sum((df['computed_features_nan_count'] > 0).sum() for df in all_results.values())
        total_with_inf_computed = sum((df['computed_features_inf_count'] > 0).sum() for df in all_results.values())
        
        print(f"\nTotal graphs analyzed: {total_graphs}")
        print(f"Graphs with NaN in x: {total_with_nan_x} ({total_with_nan_x/total_graphs*100:.1f}%)")
        print(f"Graphs with Inf in x: {total_with_inf_x} ({total_with_inf_x/total_graphs*100:.1f}%)")
        print(f"Graphs with NaN in computed features: {total_with_nan_computed} ({total_with_nan_computed/total_graphs*100:.1f}%)")
        print(f"Graphs with Inf in computed features: {total_with_inf_computed} ({total_with_inf_computed/total_graphs*100:.1f}%)")
        
        print(f"\nPer-city breakdown:")
        for city, df in sorted(all_results.items()):
            nan_pct = (df['x_nan_count'] > 0).sum() / len(df) * 100
            inf_pct = (df['x_inf_count'] > 0).sum() / len(df) * 100
            print(f"  {city}: {len(df)} graphs, NaN: {nan_pct:.1f}%, Inf: {inf_pct:.1f}%")
    else:
        diagnose_city(args.city, str(dataset_path), args.sample_size)


if __name__ == "__main__":
    main()

