#!/usr/bin/env python3
"""
Compute overlaps using a RANDOM sample of scenarios as the test set.
Still filters by capacity reduction <= 1% of all edges.
"""

import sys
import os
import json
import numpy as np
import torch
import tempfile
from pathlib import Path
import joblib
from tqdm import tqdm
import random

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

from scripts.data_preprocessing.process_simulations_for_gnn import *
from scripts.gnn.models.trans_encoder import TransEncoder
from scripts.training.help_functions import prepare_data_with_graph_features, set_cuda_visible_device
from scripts.gnn.help_functions import select_target_tensor
from compute_overlaps_varying_capacity_reduction import (
    replace_path_for_retina
)

def load_random_testset(city='regensburg', train_count=80, val_count=20, 
                        test_count=1000, seed=42, seed_idx=1, results_dir=None, 
                        project_root=None, device='cpu'):
    """
    Load a random sample of scenarios as the test set, using train/val from existing split.
    """
    # Load train/val from existing split file
    split_file_path = os.path.join(
        project_root, 'data', 'splits', city, f'rs_{seed_idx}', 
        f't{train_count}_v{val_count}',
        f'{city}_rs{seed_idx}_t{train_count}_v{val_count}_seed{seed+seed_idx-1}_train{train_count}_val{val_count}_test100_random.json'
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

    # Load train/val from split file
    with open(split_file_path, "r") as f:
        split_data = json.load(f)

    train_data = replace_path_for_retina(split_data.get("train_data"), project_root)
    val_data = replace_path_for_retina(split_data.get("val_data"), project_root)
    
    # Load ALL scenarios from metadata
    metadata_file = os.path.join(
        project_root, 'data', 'bavaria', 'inductive_data', 'training_data',
        'kreisfreistadt', city, 'metadata.json'
    )
    
    with open(metadata_file, 'r') as f:
        all_data = json.load(f)
    
    # Get train/val paths to exclude them from test set
    train_paths = set(train_data.get('path', []))
    val_paths = set(val_data.get('path', []))
    
    # Filter out train/val scenarios and convert paths
    all_paths = all_data['path']
    all_policy_region = all_data.get('policy_region', [''] * len(all_paths))
    all_scenario = all_data.get('scenario', [''] * len(all_paths))
    all_city = all_data.get('city', [city] * len(all_paths))
    
    # Convert paths and filter
    available_paths = []
    available_policy_region = []
    available_scenario = []
    available_city = []
    
    for i, path in enumerate(all_paths):
        # Convert path
        if '/kreisfreistadt/' in path:
            rel_part = path.split('/kreisfreistadt/')[1]
            new_path = f'/mnt/data_storage_ssd/elena_development/gnn_data/bavaria/inductive_data/training_data/kreisfreistadt/{rel_part}'
        elif '/home/abasu/' in path:
            if 'inductive_data/training_data/kreisfreistadt/' in path:
                rel_part = path.split('kreisfreistadt/')[1]
                new_path = f'/mnt/data_storage_ssd/elena_development/gnn_data/bavaria/inductive_data/training_data/kreisfreistadt/{rel_part}'
            else:
                new_path = path
        else:
            new_path = path
        
        # Normalize path
        new_path = os.path.normpath(new_path)
        
        # Check if it's not in train/val
        if new_path not in train_paths and new_path not in val_paths:
            available_paths.append(new_path)
            available_policy_region.append(all_policy_region[i])
            available_scenario.append(all_scenario[i])
            available_city.append(all_city[i])
    
    # Randomly sample test_count scenarios
    if len(available_paths) < test_count:
        print(f"Warning: Only {len(available_paths)} scenarios available, using all of them")
        test_paths = available_paths
        test_policy_region = available_policy_region
        test_scenario = available_scenario
        test_city = available_city
    else:
        # Use seed for reproducibility
        rng = random.Random(seed + seed_idx)
        indices = rng.sample(range(len(available_paths)), test_count)
        test_paths = [available_paths[i] for i in indices]
        test_policy_region = [available_policy_region[i] for i in indices]
        test_scenario = [available_scenario[i] for i in indices]
        test_city = [available_city[i] for i in indices]
    
    # Create test_data structure
    test_data = {
        'path': test_paths,
        'policy_region': test_policy_region,
        'scenario': test_scenario,
        'city': test_city
    }
    
    # Convert paths using replace_path_for_retina
    test_data = replace_path_for_retina(test_data, project_root)
    
    print(f"Using {len(test_data['path'])} randomly sampled scenarios as test set (from {len(available_paths)} available)")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
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
        highway_types_list = []

        x_scaler = joblib.load(x_scaler_path)

        print(f"Processing {len(test_loader)} batches...")
        with torch.inference_mode():    
            for batch in tqdm(test_loader, desc="Processing scenarios"):
                batch = batch.to(device)

                redux.append(batch.x[:,2].cpu())
                
                # Extract highway type features
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
            torch.stack(highway_types_list).squeeze().numpy())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute overlaps using random test set')
    parser.add_argument('--train_count', type=int, default=80, 
                       help='Number of training samples')
    parser.add_argument('--val_count', type=int, default=20,
                       help='Number of validation samples')
    parser.add_argument('--test_count', type=int, default=1000,
                       help='Number of test scenarios to randomly sample')
    parser.add_argument('--city', type=str, default='regensburg',
                       help='City name')
    parser.add_argument('--k', type=int, default=20,
                       help='Number of top/bottom scenarios to consider')
    parser.add_argument('--percentages', type=str, default='1.0',
                        help='Comma-separated list of capacity reduction percentages')
    args = parser.parse_args()
    
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    results_dir = os.path.join(project_root, 'data', 'inductive_gnn_data_results', 
                               'transductive', 'Scratch_vs_Finetune') + os.sep
    
    set_cuda_visible_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed = 42
    
    # Parse percentages
    capacity_reduction_percentages = [float(x.strip()) for x in args.percentages.split(',')]
    
    print("="*80)
    print("COMPUTING OVERLAPS WITH RANDOM TEST SET")
    print("="*80)
    print(f"City: {args.city}")
    print(f"Train: {args.train_count}, Val: {args.val_count}")
    print(f"Test: {args.test_count} randomly sampled scenarios")
    print(f"k: {args.k}")
    print(f"Capacity reduction threshold: {capacity_reduction_percentages}")
    print("="*80)
    
    # Load predictions for all seeds
    scratch_predictions = {}
    finetune_predictions = {}
    targets = {}
    reduxs = {}
    highway_types = {}
    
    for seed_idx in tqdm(range(1, 6), desc="Loading predictions"):
        print(f"\nLoading predictions for seed {seed_idx}...")
        sp, fp, t, r, hw = load_random_testset(
            city=args.city,
            train_count=args.train_count,
            val_count=args.val_count,
            test_count=args.test_count,
            seed=seed,
            seed_idx=seed_idx,
            results_dir=results_dir,
            project_root=project_root,
            device=device
        )
        scratch_predictions[seed_idx] = sp
        finetune_predictions[seed_idx] = fp
        targets[seed_idx] = t
        reduxs[seed_idx] = r
        highway_types[seed_idx] = hw
        print(f"  Loaded {sp.shape[0]} scenarios with {sp.shape[1]} links")
    
    # Compute overlaps
    from compute_overlaps_varying_capacity_reduction import compute_overlaps_for_capacity_reductions
    
    results = compute_overlaps_for_capacity_reductions(
        scratch_predictions=scratch_predictions,
        finetune_predictions=finetune_predictions,
        targets=targets,
        reduxs=reduxs,
        k=args.k,
        capacity_reduction_percentages=capacity_reduction_percentages,
        tolerance=0.001,
        direction='bottom',
        filter_mode='upper_bound',
        cap_red_on_all_roads=True,
        highway_types=highway_types,
        verbose=True
    )
    
    # Save results
    from compute_overlaps_varying_capacity_reduction import save_results
    from datetime import datetime
    
    metadata = {
        'city': args.city,
        'train_count': args.train_count,
        'val_count': args.val_count,
        'test_count': args.test_count,
        'test_set_type': 'random_sample',
        'k': args.k,
        'capacity_reduction_percentages': capacity_reduction_percentages,
        'tolerance': 0.001,
        'direction': 'bottom',
        'filter_mode': 'upper_bound',
        'cap_red_on_all_roads': True,
        'filtering_description': 'Filter by percentage of ALL roads',
        'computation_timestamp': datetime.now().isoformat(),
        'description': f'Top-{args.k} overlap computation using {args.test_count} randomly sampled scenarios as test set'
    }
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(script_dir, 'overlaps')
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"overlap_results_random_testset_{args.city}_t{args.train_count}_v{args.val_count}_test{args.test_count}_k{args.k}_{timestamp}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    save_results(results, output_path, metadata=metadata)
    
    print("\n" + "="*80)
    print("COMPUTATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_path}")
