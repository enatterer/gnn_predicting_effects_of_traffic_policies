"""
Compute scenario overlaps for varying capacity reduction thresholds.

This script uses the exact same logic as spatial_maps.ipynb for computing overlaps,
but varies the capacity reduction filter threshold from 0.5% to 5% in 0.1% increments.

Parameters:
  - x = 20: Top 20 overlap (k=20)
  - N = 100: test_count = 100
  - When N=100: train_count=80, val_count=20

How to run:
  From the project root directory:
    python scripts/evaluation/compute_overlaps_varying_capacity_reduction.py
  
  Or from the scripts/evaluation directory:
    python compute_overlaps_varying_capacity_reduction.py

Output:
  Results are saved in scripts/evaluation/overlaps/ directory as a JSON file.
  The JSON file includes:
    - metadata: Configuration parameters (thresholds, tolerance, filter_mode, etc.)
    - results: Overlap results for each capacity reduction percentage and seed
"""

import numpy as np
from typing import Tuple, Dict, List
import json
import os
import sys
import tempfile
import joblib
import argparse
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.data_preprocessing.process_simulations_for_gnn import *
from scripts.gnn.models.trans_encoder import TransEncoder
from scripts.training.help_functions import prepare_data_with_graph_features, set_cuda_visible_device
from scripts.gnn.help_functions import select_target_tensor


def get_overlaps(s_preds, f_preds, targs, rdx,
                 k=20, direction='bottom', verbose=False,
                 capacity_reduction_threshold=0.001, tolerance=0.001,
                 filter_mode='upper_bound', cap_red_on_all_roads=True,
                 highway_types=None, overlap_normalization='k'):
    """
    Calculate overlap between model predictions and targets for top/bottom k scenarios.
    
    This is the exact same function as in spatial_maps.ipynb, but with an adjustable
    capacity reduction threshold parameter and support for filtering by road types.
    
    Args:
        s_preds: Scratch model predictions (num_scenarios, num_links)
        f_preds: Finetune model predictions (num_scenarios, num_links)
        targs: Target values (num_scenarios, num_links)
        rdx: Capacity reduction mask (num_scenarios, num_links)
        k: Number of top/bottom scenarios to consider
        direction: 'top' for highest values, 'bottom' for lowest values
        verbose: Whether to print selected scenarios
        capacity_reduction_threshold: Percentage threshold for capacity reduction (e.g., 0.005 for 0.5%)
        tolerance: Tolerance around the threshold (e.g., 0.001 means ±0.1%)
        filter_mode: 'exact' to filter scenarios approximately equal to threshold,
                     'upper_bound' to filter scenarios <= threshold (like original)
        cap_red_on_all_roads: If True, filter by total % of all roads. If False, filter by % of 
                             reducible roads (roads that can have capacity reductions).
        highway_types: Highway type features (num_scenarios, num_links, num_highway_types) or None.
                      If None and cap_red_on_all_roads=False, will identify reducible roads from rdx.
    
    Returns:
        Tuple of (scratch_overlap, finetune_overlap) as proportions
    """
    num_scenarios, num_links = rdx.shape
    
    # Determine which roads can have capacity reductions
    if not cap_red_on_all_roads:
        # Identify reducible roads: roads that have capacity reductions in at least one scenario
        reducible_roads_mask = (rdx.sum(axis=0) > 0)  # Shape: (num_links,)
        num_reducible_roads = reducible_roads_mask.sum()
        
        if num_reducible_roads == 0:
            raise ValueError("No reducible roads found. Cannot filter by reducible roads percentage.")
        
        if verbose:
            print(f"Found {num_reducible_roads} reducible roads out of {num_links} total roads "
                  f"({num_reducible_roads/num_links:.1%})")
        
        # Calculate percentage of reducible roads affected for each scenario
        reducible_rdx = rdx[:, reducible_roads_mask]  # Shape: (num_scenarios, num_reducible_roads)
        capacity_reduction_per_scenario = reducible_rdx.sum(axis=1) / num_reducible_roads
    else:
        # Original behavior: use all roads
        capacity_reduction_per_scenario = rdx.sum(axis=1) / num_links
    
    # Filter scenarios based on capacity reduction threshold
    if filter_mode == 'upper_bound':
        # Filter scenarios with capacity reduction <= threshold
        redux_filter = capacity_reduction_per_scenario <= capacity_reduction_threshold
        lower_bound = 0.0
        upper_bound = capacity_reduction_threshold
    elif filter_mode == 'exact':
        # Filter scenarios where capacity reduction is approximately equal to threshold
        lower_bound = max(0.0, capacity_reduction_threshold - tolerance)
        upper_bound = capacity_reduction_threshold + tolerance
        redux_filter = (capacity_reduction_per_scenario >= lower_bound) & \
                      (capacity_reduction_per_scenario <= upper_bound)
    else:
        raise ValueError(f"filter_mode must be 'exact' or 'upper_bound', got '{filter_mode}'")
    
    # Check if we have enough filtered scenarios
    num_filtered = redux_filter.sum()
    if num_filtered == 0:
        raise ValueError(f"No scenarios pass the capacity reduction filter "
                        f"(threshold={capacity_reduction_threshold:.1%}, "
                        f"tolerance={tolerance:.1%})")
    
    if verbose:
        # Debug: Show actual capacity reduction percentages for filtered scenarios
        filtered_percentages = capacity_reduction_per_scenario[redux_filter]
        print(f"  Filtered {num_filtered} scenarios (threshold={capacity_reduction_threshold:.1%}, "
              f"tolerance={tolerance:.1%}, range=[{lower_bound:.1%}, {upper_bound:.1%}])")
        if num_filtered <= 10:
            print(f"  Actual percentages of filtered scenarios: {filtered_percentages * 100}")
        else:
            print(f"  Actual percentages range: [{filtered_percentages.min()*100:.1f}%, "
                  f"{filtered_percentages.max()*100:.1f}%]")
    
    if k > num_filtered:
        k_original = k
        k = num_filtered
        if verbose:
            print(f"Warning: k reduced from {k_original} to {k} (number of filtered scenarios)")
    
    # Get the original scenario indices that pass the filter
    filtered_scenario_indices = np.where(redux_filter)[0]  # Original scenario indices
    
    # Sum predictions/targets across all links for each filtered scenario
    scratch_preds_total = s_preds[redux_filter].sum(axis=1)
    finetune_preds_total = f_preds[redux_filter].sum(axis=1)
    targets_total = targs[redux_filter].sum(axis=1)
    
    # Get top/bottom k scenario indices (relative to filtered array)
    if direction == 'top':
        # Get k scenarios with highest total change (most impact)
        scratch_top_filtered_indices = scratch_preds_total.argsort()[-k:]
        finetune_top_filtered_indices = finetune_preds_total.argsort()[-k:]
        targets_top_filtered_indices = targets_total.argsort()[-k:]
    elif direction == 'bottom':
        # Get k scenarios with lowest total change (least impact)
        scratch_top_filtered_indices = scratch_preds_total.argsort()[:k]
        finetune_top_filtered_indices = finetune_preds_total.argsort()[:k]
        targets_top_filtered_indices = targets_total.argsort()[:k]
    else:
        raise ValueError("direction must be 'top' or 'bottom'")
    
    # Map filtered indices back to original scenario indices
    scratch_top_scenarios = set(filtered_scenario_indices[scratch_top_filtered_indices])
    finetune_top_scenarios = set(filtered_scenario_indices[finetune_top_filtered_indices])
    targets_top_scenarios = set(filtered_scenario_indices[targets_top_filtered_indices])
    
    if verbose:
        print(f"Scratch {direction.capitalize()} {k} Scenarios:")
        print(scratch_top_scenarios)
        print(f"Finetune {direction.capitalize()} {k} Scenarios:")
        print(finetune_top_scenarios)
        print(f"Target {direction.capitalize()} {k} Scenarios:")
        print(targets_top_scenarios)
    
    # Calculate overlap proportions
    scratch_overlap_count = len(scratch_top_scenarios.intersection(targets_top_scenarios))
    finetune_overlap_count = len(finetune_top_scenarios.intersection(targets_top_scenarios))

    if overlap_normalization == 'k':
        denominator = k
    elif overlap_normalization == 'eligible':
        denominator = num_filtered
    elif overlap_normalization == 'test_set':
        denominator = num_scenarios
    else:
        raise ValueError(f"overlap_normalization must be 'k', 'eligible', or 'test_set', got '{overlap_normalization}'")

    if denominator == 0:
        raise ValueError("Overlap normalization denominator is zero.")

    scratch_overlap = scratch_overlap_count / denominator
    finetune_overlap = finetune_overlap_count / denominator
    
    return round(scratch_overlap, 2), round(finetune_overlap, 2)


def compute_overlaps_for_capacity_reductions(
    scratch_predictions: Dict[int, np.ndarray],
    finetune_predictions: Dict[int, np.ndarray],
    targets: Dict[int, np.ndarray],
    reduxs: Dict[int, np.ndarray],
    k: int = 20,
    capacity_reduction_percentages: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    tolerance: float = 0.001,
    direction: str = 'bottom',
    filter_mode: str = 'upper_bound',
    cap_red_on_all_roads: bool = True,
    highway_types: Dict[int, np.ndarray] = None,
    verbose: bool = False,
    overlap_normalization: str = 'k'
) -> Dict[float, Dict[int, Tuple[float, float]]]:
    """
    Compute overlaps for multiple capacity reduction thresholds.
    
    Args:
        scratch_predictions: Dictionary mapping seed_idx to predictions array
        finetune_predictions: Dictionary mapping seed_idx to predictions array
        targets: Dictionary mapping seed_idx to targets array
        reduxs: Dictionary mapping seed_idx to capacity reduction masks
        k: Number of top scenarios to consider (default: 20)
        capacity_reduction_percentages: List of percentages (e.g., [0.1, 0.2, 0.3, ...])
        tolerance: Tolerance around each percentage (default: 0.001 = 0.1%)
        direction: 'top' or 'bottom' (default: 'bottom')
        filter_mode: 'exact' to filter scenarios approximately equal to threshold,
                     'upper_bound' to filter scenarios <= threshold (like original)
        cap_red_on_all_roads: If True, filter by total % of all roads. If False, filter by 
                             % of reducible roads (roads that can have capacity reductions).
        highway_types: Dictionary mapping seed_idx to highway type features (num_scenarios, num_links, 6)
                      or None. Only used if cap_red_on_all_roads=False (for future extensions).
        verbose: Whether to print detailed information
        overlap_normalization: 'k', 'eligible', or 'test_set' for overlap denominator
    
    Returns:
        Dictionary mapping capacity_reduction_percentage to results dict:
        {
            percentage: {
                seed_idx: (scratch_overlap, finetune_overlap),
                ...
            },
            ...
        }
    """
    results = {}
    
    for pct in capacity_reduction_percentages:
        threshold = pct / 100.0  # Convert percentage to fraction
        results[pct] = {}
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Computing overlaps for capacity reduction = {pct}%")
            print(f"{'='*60}")
        
        for seed_idx in sorted(scratch_predictions.keys()):
            s_preds = scratch_predictions[seed_idx]
            f_preds = finetune_predictions[seed_idx]
            targs = targets[seed_idx]
            rdx = reduxs[seed_idx]
            
            try:
                hw_types = highway_types[seed_idx] if highway_types is not None else None
                s_overlap, f_overlap = get_overlaps(
                    s_preds, f_preds, targs, rdx,
                    k=k,
                    direction=direction,
                    verbose=verbose,
                    capacity_reduction_threshold=threshold,
                    tolerance=tolerance,
                    filter_mode=filter_mode,
                    cap_red_on_all_roads=cap_red_on_all_roads,
                    highway_types=hw_types,
                    overlap_normalization=overlap_normalization
                )
                results[pct][seed_idx] = (s_overlap, f_overlap)
                
                if verbose:
                    print(f"Seed {seed_idx}: Scratch={s_overlap:.2f}, Finetune={f_overlap:.2f}")
            except ValueError as e:
                if verbose:
                    print(f"Seed {seed_idx}: {e}")
                results[pct][seed_idx] = None
    
    return results

def print_results_summary(results: Dict[float, Dict[int, Tuple[float, float]]]):
    """Print a summary of the results."""
    print("\n" + "="*80)
    print("OVERLAP RESULTS SUMMARY")
    print("="*80)
    print(f"{'Capacity Reduction':<20} {'Seed':<8} {'Scratch Overlap':<18} {'Finetune Overlap':<18}")
    print("-"*80)
    
    for pct in sorted(results.keys()):
        for seed_idx in sorted(results[pct].keys()):
            if results[pct][seed_idx] is not None:
                s_overlap, f_overlap = results[pct][seed_idx]
                print(f"{pct:>6.1f}%{'':<12} {seed_idx:<8} {s_overlap:>17.2f} {f_overlap:>17.2f}")
            else:
                print(f"{pct:>6.1f}%{'':<12} {seed_idx:<8} {'N/A':<18} {'N/A':<18}")
    
    print("\n" + "="*80)
    print("STATISTICS ACROSS SEEDS")
    print("="*80)
    print(f"{'Capacity Reduction':<20} {'Scratch Mean±Std':<20} {'Finetune Mean±Std':<20}")
    print("-"*80)
    
    for pct in sorted(results.keys()):
        scratch_overlaps = []
        finetune_overlaps = []
        
        for seed_idx in sorted(results[pct].keys()):
            if results[pct][seed_idx] is not None:
                s_overlap, f_overlap = results[pct][seed_idx]
                scratch_overlaps.append(s_overlap)
                finetune_overlaps.append(f_overlap)
        
        if scratch_overlaps:
            s_mean = np.mean(scratch_overlaps)
            s_std = np.std(scratch_overlaps)
            f_mean = np.mean(finetune_overlaps)
            f_std = np.std(finetune_overlaps)
            print(f"{pct:>6.1f}%{'':<12} {s_mean:>6.2f}±{s_std:<6.2f} {f_mean:>6.2f}±{f_std:<6.2f}")
        else:
            print(f"{pct:>6.1f}%{'':<12} {'N/A':<20} {'N/A':<20}")


def save_results(results: Dict[float, Dict[int, Tuple[float, float]]], 
                 output_path: str,
                 metadata: Dict = None):
    """
    Save results to a JSON file with metadata.
    
    Args:
        results: Dictionary mapping capacity_reduction_percentage to results
        output_path: Path to save the JSON file
        metadata: Dictionary containing metadata about the computation (thresholds, tolerance, etc.)
    """
    # Convert to JSON-serializable format
    json_results = {
        'metadata': metadata or {},
        'results': {}
    }
    
    for pct in results:
        json_results['results'][str(pct)] = {}
        for seed_idx in results[pct]:
            if results[pct][seed_idx] is not None:
                json_results['results'][str(pct)][str(seed_idx)] = {
                    'scratch_overlap': float(results[pct][seed_idx][0]),
                    'finetune_overlap': float(results[pct][seed_idx][1])
                }
            else:
                json_results['results'][str(pct)][str(seed_idx)] = None
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def parse_percentages(percentages_str: str) -> List[float]:
    """
    Parse a comma-separated string of percentages into a list of floats.
    Example input: "0.5,1,1.5,2"
    """
    return [float(p.strip()) for p in percentages_str.split(',') if p.strip()]


def replace_path_for_retina(data_split, project_root):
    """
    Convert paths to absolute paths based on project_root.
    
    Handles both absolute paths (that need replacement) and relative paths
    (that need to be converted to absolute).
    """
    if not data_split or 'path' not in data_split:
        return data_split
    
    new_paths = []
    for path in data_split['path']:
        if not path:  # Skip empty paths
            new_paths.append(path)
            continue
            
        # If it's already an absolute path, check if it needs replacement
        if os.path.isabs(path):
            # Replace /mnt/repo/ with project root
            if path.startswith('/mnt/repo/'):
                rel_part = path.replace('/mnt/repo/', '')
                path = os.path.join(project_root, rel_part.lstrip('/'))
            # Replace /home/rrao/development/gnn_predicting_effects_of_traffic_policies/ with project root
            elif path.startswith('/home/rrao/development/gnn_predicting_effects_of_traffic_policies/'):
                rel_part = path.replace('/home/rrao/development/gnn_predicting_effects_of_traffic_policies/', '')
                path = os.path.join(project_root, rel_part.lstrip('/'))
            # Replace any other /home/rrao/... paths that contain the project name
            elif path.startswith('/home/rrao/') and 'gnn_predicting_effects_of_traffic_policies/' in path:
                rel_part = path.split('gnn_predicting_effects_of_traffic_policies/')[1]
                path = os.path.join(project_root, rel_part)
        else:
            # It's a relative path - convert to absolute based on project_root
            # Handle paths like '../../data/...' or 'data/...'
            if path.startswith('../../'):
                # Remove '../../' and join with project_root
                rel_part = path.replace('../../', '')
                path = os.path.join(project_root, rel_part)
            elif path.startswith('../'):
                # Remove '../' and join with project_root
                rel_part = path.replace('../', '')
                path = os.path.join(project_root, rel_part)
            else:
                # Already relative to project root
                path = os.path.join(project_root, path)
        
        # Normalize the path (resolve .. and .)
        path = os.path.normpath(path)
        new_paths.append(path)
    
    data_split['path'] = new_paths
    return data_split


def get_predictions(seed_idx, city, train_count, val_count, test_count, seed, 
                   test_set_type, results_dir, project_root, device, splits_dir):
    """
    Load predictions for a given seed index.
    
    This function is identical to the one in spatial_maps.ipynb.
    """
    split_file_path = os.path.join(
        splits_dir, city, f'rs_{seed_idx}', 
        f't{train_count}_v{val_count}',
        f'{city}_rs{seed_idx}_t{train_count}_v{val_count}_seed{seed+seed_idx-1}_train{train_count}_val{val_count}_test{test_count}_{test_set_type}.json'
    )
    scratch_run_name = f"{city}_scratch_rs_{seed_idx}_t{train_count}_v{val_count}"
    finetune_run_name = f"{city}_finetune_rs_{seed_idx}_t{train_count}_v{val_count}"
    x_scaler_path = results_dir + finetune_run_name + "/data_created_during_finetuning/train_x_scaler.pkl"

    scratch_gnn = TransEncoder()
    finetune_gnn = TransEncoder()

    # Load the model state dictionary
    scratch_gnn.load_state_dict(torch.load(results_dir + scratch_run_name + '/finetuned_model/model.pth'))
    finetune_gnn.load_state_dict(torch.load(results_dir + finetune_run_name + '/finetuned_model/model.pth'))

    scratch_gnn.to(device)
    finetune_gnn.to(device)

    with open(split_file_path, "r") as f:
        split_data = json.load(f)

    test_data = replace_path_for_retina(split_data.get("test_data"), project_root)
    train_data = replace_path_for_retina(split_data.get("train_data"), project_root)
    val_data = replace_path_for_retina(split_data.get("val_data"), project_root)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        # Ensure the directory exists and use proper path joining with trailing slash
        # (required because prepare_data_with_graph_features uses string concatenation)
        path_to_save_dataloader = str(tmp_dir) + os.sep
        os.makedirs(path_to_save_dataloader, exist_ok=True)
        _, _, test_loader = prepare_data_with_graph_features(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            use_inductive_variant=False,
            batch_size=1,
            path_to_save_dataloader=path_to_save_dataloader,
            use_all_features=False,
            use_weighted_batches=False,
            use_nested_neighbor_loader=False,
            neighbor_sizes="7,7,7",
            subgraphs_per_graph=1,
            seed_size=1000,
            sampling_strategy="neighbor_sampling",
            min_subgraph_nodes=5000,
            max_subgraph_nodes=50000,
            aug_pos_rotation=False,
            aug_feature_noise=False,
            aug_node_masking_probability=0.0,
            use_destination_activity_param=False,
            return_test_loader=True,
            x_scaler_path=x_scaler_path
        )

        scratch_gnn.eval()
        finetune_gnn.eval()
        
        scratch_predictions = []
        finetune_predictions = []
        targets = []
        redux = []
        vol_base_case = []
        highway_types_list = []

        x_scaler = joblib.load(x_scaler_path)

        with torch.inference_mode():    
            for batch in test_loader:
                batch = batch.to(device)

                conti_feat = batch.x[:,[0,1,3,4]].cpu()
                reat_feat = x_scaler.inverse_transform(conti_feat)
                vol_base_case.append(reat_feat[:,0])

                redux.append(batch.x[:,2].cpu())
                
                # Extract highway type features (indices 4-9: PRIMARY, SECONDARY, TERTIARY, RESIDENTIAL, PT, OTHER)
                highway_types_list.append(batch.x[:,4:10].cpu())
                
                out = scratch_gnn(batch.clone())
                scratch_predictions.append(out.cpu())

                out = finetune_gnn(batch)
                finetune_predictions.append(out.cpu())
                
                targets_node_predictions = select_target_tensor(batch, "abs_vol_car")
                targets.append(targets_node_predictions.cpu())

    return (torch.stack(scratch_predictions).squeeze().numpy(), 
            torch.stack(finetune_predictions).squeeze().numpy(), 
            torch.stack(targets).squeeze().numpy(), 
            torch.stack(redux).squeeze().numpy(), 
            np.stack(vol_base_case).squeeze(),
            torch.stack(highway_types_list).squeeze().numpy())


if __name__ == "__main__":
    # Parse command-line arguments first to get actual values
    parser = argparse.ArgumentParser(description='Compute scenario overlaps for varying capacity reduction thresholds')
    parser.add_argument('--train_count', type=int, default=80, 
                       help='Number of training samples (default: 80)')
    parser.add_argument('--val_count', type=int, default=20,
                       help='Number of validation samples (default: 20)')
    parser.add_argument('--test_count', type=int, default=100,
                       help='Number of test samples (default: 100)')
    parser.add_argument('--city', type=str, default='regensburg',
                       help='City name (default: regensburg)')
    parser.add_argument('--cap_red_on_all_roads', action='store_true', default=False,
                       help='Filter by percentage of ALL roads (default: False, filters by reducible roads only)')
    parser.add_argument('--percentages', type=str, default=None,
                        help='Comma-separated list of capacity reduction percentages to use (overrides defaults)')
    parser.add_argument('--k', type=int, default=20,
                       help='Number of top/bottom scenarios to consider for overlap (default: 20)')
    parser.add_argument('--test_set_type', type=str, default='random',
                       choices=['random', 'distant_iou'],
                       help='Test set type to load from splits (default: random)')
    parser.add_argument('--splits_dir', type=str, default='data/splits',
                        help='Base directory for split files (default: data/splits)')
    parser.add_argument('--overlap_normalization', type=str, default='k',
                        choices=['k', 'eligible', 'test_set'],
                        help="Overlap denominator: 'k' (default), 'eligible', or 'test_set'")
    parser.add_argument('--direction', type=str, default='bottom',
                        choices=['top', 'bottom'],
                        help="Select 'top' or 'bottom' scenarios (default: bottom)")
    args = parser.parse_args()
    
    print("="*80)
    print("COMPUTING OVERLAPS FOR VARYING CAPACITY REDUCTION THRESHOLDS")
    print("="*80)
    print("\nThis script computes top-20 overlaps (x=20) for capacity reductions")
    print("ranging from 0.5% to 5% in 0.1% increments.")
    print("\nParameters:")
    print(f"  - x = 20: Top 20 overlap")
    print(f"  - test_count = {args.test_count}")
    print(f"  - train_count = {args.train_count}, val_count = {args.val_count}")
    print(f"  - overlap_normalization = {args.overlap_normalization}")
    print(f"  - direction = {args.direction}")
    print("="*80)
    
    # Get project root directory (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    # Configuration parameters
    city = args.city
    train_count = args.train_count
    val_count = args.val_count
    seed = 42
    test_count = args.test_count
    test_set_type = args.test_set_type
    splits_dir = args.splits_dir
    if not os.path.isabs(splits_dir):
        splits_dir = os.path.join(project_root, splits_dir)
    
    results_dir = os.path.join(project_root, 'data', 'inductive_gnn_data_results', 
                               'transductive', 'Scratch_vs_Finetune') + os.sep
    
    # Set up device
    set_cuda_visible_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load predictions for all seeds
    print("\n" + "="*80)
    print("LOADING PREDICTIONS")
    print("="*80)
    scratch_predictions = {}
    finetune_predictions = {}
    targets = {}
    reduxs = {}
    vol_base_case = {}
    highway_types = {}
    
    # Configuration: Filter by all roads or only reducible roads?
    cap_red_on_all_roads = args.cap_red_on_all_roads  # Set via --cap_red_on_all_roads flag
    
    for seed_idx in tqdm(range(1, 6), desc="Loading predictions"):
        print(f"\nLoading predictions for seed {seed_idx}...")
        sp, fp, t, r, vbc, hw = get_predictions(
            seed_idx=seed_idx,
            city=city,
            train_count=train_count,
            val_count=val_count,
            test_count=test_count,
            seed=seed,
            test_set_type=test_set_type,
            results_dir=results_dir,
            project_root=project_root,
            device=device,
            splits_dir=splits_dir
        )
        scratch_predictions[seed_idx] = sp
        finetune_predictions[seed_idx] = fp
        targets[seed_idx] = t
        reduxs[seed_idx] = r
        vol_base_case[seed_idx] = vbc
        highway_types[seed_idx] = hw
        print(f"  Loaded {sp.shape[0]} scenarios with {sp.shape[1]} links")
    
    # Compute overlaps for varying capacity reduction thresholds
    print("\n" + "="*80)
    print("COMPUTING OVERLAPS")
    print("="*80)
    if cap_red_on_all_roads:
        print("Filtering by: Percentage of ALL roads affected")
        if args.percentages:
            capacity_reduction_percentages = parse_percentages(args.percentages)
        else:
            # For all roads: use percentages from 0.2% to 2% in 0.2% increments
            capacity_reduction_percentages = np.arange(0.2, 2.1, 0.2).round(1).tolist()  # 0.2%, 0.4%, ..., 2.0%
        tolerance = 0.001  # Not used in upper_bound mode, but kept for consistency
        filter_mode = 'upper_bound'  # Filter scenarios with capacity reduction <= threshold
    else:
        print("Filtering by: Percentage of REDUCIBLE roads affected")
        if args.percentages:
            capacity_reduction_percentages = parse_percentages(args.percentages)
        else:
            # For reducible roads: use larger percentages (10% to 100% in 10% increments)
            # Since there are typically only a few hundred reducible roads, small percentages don't make sense
            # Using upper_bound mode: filter scenarios with <= X% of reducible roads affected
            capacity_reduction_percentages = np.arange(10, 101, 10).round(1).tolist()  # 10%, 20%, ..., 100%
        tolerance = 0.01  # Not used in upper_bound mode, but kept for consistency
        filter_mode = 'upper_bound'  # Filter scenarios with capacity reduction <= threshold
    print("="*80)
    
    results = compute_overlaps_for_capacity_reductions(
        scratch_predictions=scratch_predictions,
        finetune_predictions=finetune_predictions,
        targets=targets,
        reduxs=reduxs,
        k=args.k,  # Top k overlap
        capacity_reduction_percentages=capacity_reduction_percentages,
        tolerance=tolerance,  # Not used in upper_bound mode, but kept for API consistency
        direction=args.direction,
        filter_mode=filter_mode,  # Use 'upper_bound' to filter scenarios <= threshold
        cap_red_on_all_roads=cap_red_on_all_roads,
        highway_types=highway_types,
        verbose=True,
        overlap_normalization=args.overlap_normalization
    )
    
    # Print summary
    print_results_summary(results)
    
    # Prepare metadata
    metadata = {
        'city': city,
        'train_count': train_count,
        'val_count': val_count,
        'test_count': test_count,
        'seed': seed,
        'test_set_type': test_set_type,
        'k': args.k,  # Top k overlap
        'capacity_reduction_percentages': capacity_reduction_percentages,
        'tolerance': tolerance,  # Adaptive tolerance (0.1% for all roads, 1% for reducible roads)
        'direction': args.direction,
        'filter_mode': filter_mode,
        'cap_red_on_all_roads': cap_red_on_all_roads,
        'overlap_normalization': args.overlap_normalization,
        'filtering_description': 'Filter by percentage of ALL roads' if cap_red_on_all_roads else 'Filter by percentage of REDUCIBLE roads (roads that can have capacity reductions)',
        'computation_timestamp': datetime.now().isoformat(),
        'description': f'Top-{args.k} overlap computation for capacity reduction threshold {capacity_reduction_percentages[0]}%' if len(capacity_reduction_percentages) == 1 else f'Top-{args.k} overlap computation for varying capacity reduction thresholds ({capacity_reduction_percentages[0]}% to {capacity_reduction_percentages[-1]}% in {capacity_reduction_percentages[1]-capacity_reduction_percentages[0]}% increments)'
    }
    
    # Save results to file in "overlaps" folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    norm_suffix = {
        'k': 'norm_to_k',
        'eligible': 'norm_to_eligible_roads',
        'test_set': 'norm_to_test_set'
    }.get(args.overlap_normalization, f"norm_{args.overlap_normalization}")
    output_filename = (
        f"overlap_results_varying_capacity_reduction_{city}_t{train_count}_v{val_count}"
        f"_test{test_count}_{norm_suffix}_k{args.k}_{timestamp}.json"
    )
    
    # Create "overlaps" folder in the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'overlaps')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, output_filename)
    
    save_results(results, output_path, metadata=metadata)
    
    print("\n" + "="*80)
    print("COMPUTATION COMPLETE")
    print("="*80)
