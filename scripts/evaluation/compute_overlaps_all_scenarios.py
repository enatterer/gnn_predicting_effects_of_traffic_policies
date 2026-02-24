#!/usr/bin/env python3
"""
Compute overlaps using ALL available scenarios as the test set.
Still filters by capacity reduction <= 1% of all edges.

The script aggregates over 5 "random-split runs" (seed_idx 1..5). Each run has:
  - A different random train/val split (split seed = 42, 43, 44, 45, 46 for seed_idx 1..5).
  - A scratch and a finetune model trained on that split (e.g. regensburg_scratch_rs_1_..., regensburg_finetune_rs_1_...).
So the 5 things are not just random seeds: they are five independent (split seed → trained models) runs;
overlap metrics are computed per run and then aggregated (e.g. mean/std across runs).
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

def load_all_scenarios_as_test(city='regensburg', train_count=80, val_count=20, 
                                seed=42, seed_idx=1, results_dir=None, 
                                project_root=None, device='cpu'):
    """
    Load all available scenarios as the test set, using train/val from existing split.
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
    
    # Load ALL scenarios from metadata as test set
    metadata_file = os.path.join(
        project_root, 'data', 'bavaria', 'inductive_data', 'training_data',
        'kreisfreistadt', city, 'metadata.json'
    )
    
    with open(metadata_file, 'r') as f:
        all_data = json.load(f)
    
    # Create test_data structure from all scenarios
    test_data = {
        'path': all_data['path'],
        'policy_region': all_data.get('policy_region', [''] * len(all_data['path'])),
        'scenario': all_data.get('scenario', [''] * len(all_data['path'])),
        'city': all_data.get('city', [city] * len(all_data['path']))
    }
    
    # Convert paths - need to handle paths that point to different user directories
    # The metadata has paths like /home/abasu/... but we need /mnt/data_storage_ssd/...
    new_paths = []
    for path in test_data['path']:
        # Extract the relative part (e.g., "kreisfreistadt/regensburg/000001.pt")
        if '/kreisfreistadt/' in path:
            rel_part = path.split('/kreisfreistadt/')[1]
            # Use the actual data path
            new_path = f'/mnt/data_storage_ssd/elena_development/gnn_data/bavaria/inductive_data/training_data/kreisfreistadt/{rel_part}'
            new_paths.append(new_path)
        elif '/home/abasu/' in path:
            # Handle /home/abasu/ paths - convert to /mnt/data_storage_ssd/...
            if 'inductive_data/training_data/kreisfreistadt/' in path:
                rel_part = path.split('kreisfreistadt/')[1]
                new_path = f'/mnt/data_storage_ssd/elena_development/gnn_data/bavaria/inductive_data/training_data/kreisfreistadt/{rel_part}'
                new_paths.append(new_path)
            else:
                new_paths.append(path)
        else:
            new_paths.append(path)
    
    test_data['path'] = new_paths
    
    # Convert paths using replace_path_for_retina (will handle any remaining conversions)
    test_data = replace_path_for_retina(test_data, project_root)
    
    print(f"Using {len(test_data['path'])} total scenarios as test set")

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
    
    parser = argparse.ArgumentParser(description='Compute overlaps using ALL scenarios as test set')
    parser.add_argument('--train_count', type=int, default=80, 
                       help='Number of training samples')
    parser.add_argument('--val_count', type=int, default=20,
                       help='Number of validation samples')
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
    print("COMPUTING OVERLAPS WITH ALL SCENARIOS AS TEST SET")
    print("="*80)
    print(f"City: {args.city}")
    print(f"Train: {args.train_count}, Val: {args.val_count}")
    print(f"Test: ALL available scenarios (~2946)")
    print(f"k: {args.k}")
    print(f"Capacity reduction threshold: {capacity_reduction_percentages}")
    print("="*80)
    
    # Load predictions for all 5 random-split runs (seed_idx 1..5: different train/val splits and model checkpoints)
    scratch_predictions = {}
    finetune_predictions = {}
    targets = {}
    reduxs = {}
    highway_types = {}
    
    for seed_idx in tqdm(range(1, 6), desc="Loading predictions"):
        print(f"\nLoading predictions for run {seed_idx}/5 (split seed {seed + seed_idx - 1})...")
        sp, fp, t, r, hw = load_all_scenarios_as_test(
            city=args.city,
            train_count=args.train_count,
            val_count=args.val_count,
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
        'test_count': 'all',  # All available scenarios
        'k': args.k,
        'capacity_reduction_percentages': capacity_reduction_percentages,
        'tolerance': 0.001,
        'direction': 'bottom',
        'filter_mode': 'upper_bound',
        'cap_red_on_all_roads': True,
        'filtering_description': 'Filter by percentage of ALL roads',
        'computation_timestamp': datetime.now().isoformat(),
        'description': f'Top-{args.k} overlap computation using ALL scenarios as test set'
    }
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(script_dir, 'overlaps')
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"overlap_results_all_scenarios_{args.city}_t{args.train_count}_v{args.val_count}_k{args.k}_{timestamp}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    save_results(results, output_path, metadata=metadata)
    
    print("\n" + "="*80)
    print("COMPUTATION COMPLETE")
    print("="*80)
