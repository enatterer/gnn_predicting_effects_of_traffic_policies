#!/usr/bin/env python3
"""
Evaluate a pretrained model on specific cities with a fixed number of graphs per city.

This script:
1. Loads a pretrained checkpoint
2. For each specified city, loads a fixed number of graphs (default: 200)
3. Evaluates the model and computes all metrics (loss, R², Spearman, Pearson, hit rates)
4. Saves results to CSV/JSON

Usage:
    python scripts/evaluation/evaluate_pretrained_on_cities.py \
        --pretrain_run_name general_surrogate_v0 \
        --project_name Bavaria_Test \
        --cities bamberg,erlangen,muenchen,neuulm \
        --graphs_per_city 200 \
        --dataset_path data/bavaria/inductive_data/training_data/kreisfreistadt \
        --output_dir data/analysis_results \
        --device_nr 0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

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
    balanced_subset_by_city,
    create_gnn_model,
    get_available_gpus,
    load_metadata_from_disk,
    prepare_data_with_graph_features,
    select_best_gpu,
    set_cuda_visible_device,
    str_to_bool,
)


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
            pos_dim = 6  # Default assumption
            use_pos = True  # Default assumption
            use_lap_pe = False
            lap_pe_dim = 0
            
            base_in_channels_with_pos = effective_in_channels - pos_dim  # Assuming use_pos=True and pos_dim=6
            
            # Set feature configuration based on effective_in_channels
            if effective_in_channels == 7:
                config['use_all_features'] = False
                config['use_pos'] = True
                config['pos_dim'] = 2
                config['in_channels'] = 5
                print(f"  Inferred config: in_channels=5, use_all_features=False, use_pos=True, pos_dim=2 (effective_in_channels={effective_in_channels})")
            elif base_in_channels_with_pos == 5:
                config['use_all_features'] = False
                config['use_pos'] = True
                config['pos_dim'] = pos_dim
                config['in_channels'] = 5
                print(f"  Inferred config: in_channels=5, use_all_features=False, use_pos=True, pos_dim={pos_dim}")
            elif base_in_channels_with_pos == 20:
                config['use_all_features'] = True
                config['use_destination_activity'] = False
                config['use_pos'] = True
                config['pos_dim'] = pos_dim
                config['in_channels'] = 20
                print(f"  Inferred config: in_channels=20, use_all_features=True, use_destination_activity=False, use_pos=True, pos_dim={pos_dim}")
            elif base_in_channels_with_pos == 21:
                config['use_all_features'] = True
                config['use_destination_activity'] = True
                config['use_pos'] = True
                config['pos_dim'] = pos_dim
                config['in_channels'] = 21
                print(f"  Inferred config: in_channels=21, use_all_features=True, use_destination_activity=True, use_pos=True, pos_dim={pos_dim}")
            else:
                if effective_in_channels in [5, 7, 20, 21, 26, 27]:
                    config['use_pos'] = False
                    config['pos_dim'] = 0
                    config['in_channels'] = effective_in_channels
                    print(f"  WARNING: Unknown pattern. Trying use_pos=False: in_channels={effective_in_channels}, use_pos=False")
                else:
                    config['in_channels'] = base_in_channels_with_pos
                    config['use_pos'] = use_pos
                    config['pos_dim'] = pos_dim
                    print(f"  WARNING: Unknown base_in_channels={base_in_channels_with_pos} (from effective_in_channels={effective_in_channels} - pos_dim={pos_dim})")
                    print(f"  Using: in_channels={base_in_channels_with_pos}, use_pos={use_pos}, pos_dim={pos_dim}")
            
            config['ff_dim'] = ff_dim
    else:
        # For other architectures, try to infer in_channels from first layer
        if 'in_channels' not in config:
            for key in state_dict.keys():
                if 'lin' in key or 'conv' in key or 'embed' in key:
                    if 'weight' in key:
                        weight_shape = state_dict[key].shape
                        if len(weight_shape) == 2:
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
    config.setdefault('ff_dim', 256)
    config.setdefault('use_pos', True)
    config.setdefault('pos_dim', 6)
    
    # Prepare model kwargs only for architectures that support them.
    model_kwargs = {'in_channels': config.get('in_channels', 5)}
    if gnn_arch == "trans_encoder":
        model_kwargs.update({
            'ff_dim': config.get('ff_dim', 256),
            'use_pos': config.get('use_pos', True),
            'pos_dim': config.get('pos_dim', 6),
        })
    
    # Create model with inferred parameters
    model = create_gnn_model(
        gnn_arch=gnn_arch,
        config=type('Config', (), config)(),
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


def find_checkpoint(run_dir: Path) -> Optional[Path]:
    """Find the best checkpoint in a run directory."""
    # Check for model.pth first (the best model file)
    model_paths = [
        run_dir / 'trained_model' / 'model.pth',
    ]
    for model_path in model_paths:
        if model_path.exists():
            return model_path
    
    # If no model.pth, look for checkpoints
    checkpoint_dirs = [
        run_dir / 'trained_model' / 'checkpoints',
    ]
    for chk_dir in checkpoint_dirs:
        if not chk_dir.exists():
            continue
        
        checkpoint_files = list(chk_dir.glob('checkpoint_epoch_*.pt'))
        if checkpoint_files:
            import re
            def extract_epoch(filename):
                match = re.search(r'checkpoint_epoch_(\d+)\.pt', filename.name)
                return int(match.group(1)) if match else -1
            
            latest_checkpoint = max(checkpoint_files, key=extract_epoch)
            return latest_checkpoint
    
    return None


def evaluate_city(
    model: torch.nn.Module,
    config: Dict,
    city: str,
    dataset_path: Path,
    graphs_per_city: int,
    device: torch.device,
    pretrained_scaler_path: Path,
    seed: int = 42
) -> Dict:
    """
    Evaluate model on a single city with a fixed number of graphs.
    
    Returns:
        Dictionary with evaluation metrics including hit rates.
    """
    print(f"\n{'='*80}")
    print(f"Evaluating city: {city} with {graphs_per_city} graphs")
    print(f"{'='*80}")
    
    try:
        # Load metadata for the city
        city_data = {'path': list(), 'policy_region': list(), 'scenario': list(), 'city': list()}
        city_metadata_path = dataset_path / city / 'metadata.json'
        
        if not city_metadata_path.exists():
            print(f"  ERROR: Metadata file not found: {city_metadata_path}")
            return None
        
        load_metadata_from_disk(city_data, str(city_metadata_path))
        
        # Limit to graphs_per_city
        if len(city_data['path']) > graphs_per_city:
            print(f"  Limiting from {len(city_data['path'])} to {graphs_per_city} graphs")
            city_data = balanced_subset_by_city(city_data, graphs_per_city, seed=seed)
        
        if len(city_data['path']) < graphs_per_city:
            print(f"  WARNING: Only {len(city_data['path'])} graphs available (requested {graphs_per_city})")
        
        num_eval_graphs = len(city_data['path'])
        print(f"  Using {num_eval_graphs} graphs for evaluation")
        
        # Load the pretrained scaler (CRITICAL: use the same scaler as training)
        import joblib
        if not pretrained_scaler_path.exists():
            raise ValueError(f"Pretrained scaler not found: {pretrained_scaler_path}")
        
        print(f"  Loading pretrained scaler from: {pretrained_scaler_path}")
        pretrained_scaler = joblib.load(pretrained_scaler_path)
        scalers_dict = {'x_scaler': pretrained_scaler}
        
        # Verify scaler was loaded correctly
        print(f"  Scaler type: {type(pretrained_scaler)}")
        if hasattr(pretrained_scaler, 'mean_') and hasattr(pretrained_scaler, 'scale_'):
            print(f"  Scaler mean shape: {pretrained_scaler.mean_.shape}, scale shape: {pretrained_scaler.scale_.shape}")
        
        # Prepare data: load graphs directly and normalize using pretrained scaler
        # No train/val/test splits - just evaluate on all graphs
        from gnn.gnn_io import GraphDataset
        from torch.utils.data import Subset
        from training.help_functions import (
            normalize_dataset_with_scaler,
            create_split_dataloader,
        )
        from functools import partial
        from gnn.gnn_io import collate_fn
        from data_preprocessing.process_simulations_for_gnn import EdgeFeatures
        
        # Determine which features to use (must match training config!)
        use_all_features = config.get('use_all_features', True)
        use_destination_activity = config.get('use_destination_activity', False)
        
        print(f"  Feature config: use_all_features={use_all_features}, use_destination_activity={use_destination_activity}")
        
        if use_all_features:
            node_features = []
            for feat in EdgeFeatures:
                name = feat.name
                if not use_destination_activity and feat.value >= 20:
                    continue
                node_features.append(name)
        else:
            node_features = [
                "VOL_BASE_CASE",
                "CAPACITY_BASE_CASE",
                "CAPACITY_REDUCTION",
                "FREESPEED",
                "LENGTH"
            ]
        
        print(f"  Using {len(node_features)} features: {node_features[:5]}..." if len(node_features) > 5 else f"  Using {len(node_features)} features: {node_features}")
        
        # Create feature mapping
        filtered_feature_mapping = {}
        current_idx = 0
        for feature_name in node_features:
            filtered_feature_mapping[EdgeFeatures[feature_name].value] = current_idx
            current_idx += 1
        
        node_feature_filter = [EdgeFeatures[feature].value for feature in node_features]
        
        # Create collate function (no augmentation for evaluation)
        collate_fn_eval = partial(
            collate_fn,
            node_feature_filter=node_feature_filter,
            filtered_feature_mapping=filtered_feature_mapping,
            is_training=False
        )
        
        # Create dataset directly from city data (no splitting!)
        labels = [f"{city}_{policy_region}" for city, policy_region in zip(city_data['city'], city_data['policy_region'])]
        dataset = GraphDataset(city_data['path'], labels)
        
        # Create a subset with all graphs (for normalization function compatibility)
        all_indices = list(range(len(dataset)))
        test_subset = Subset(dataset, all_indices)
        
        # Normalize all graphs using pretrained scaler
        # IMPORTANT: Normalization happens on ORIGINAL feature indices [0, 1, 3, 10]
        # (VOL_BASE_CASE, CAPACITY_BASE_CASE, FREESPEED, LENGTH)
        # Feature filtering happens later in collate_fn
        print("  Normalizing data with pretrained scaler...")
        print("  Normalizing features: VOL_BASE_CASE(0), CAPACITY_BASE_CASE(1), FREESPEED(3), LENGTH(10)")
        
        # Sample a graph to check feature dimensions before normalization
        sample_graph = dataset[0]
        print(f"  Sample graph before normalization: {sample_graph.x.shape[1]} features, {sample_graph.x.shape[0]} nodes")
        
        normalized_data_list = normalize_dataset_with_scaler(
            dataset_input=test_subset,
            scalers=scalers_dict
        )
        
        # Verify normalization worked
        sample_normalized = normalized_data_list[0]
        print(f"  Sample graph after normalization: {sample_normalized.x.shape[1]} features, {sample_normalized.x.shape[0]} nodes")
        
        # Check that normalized features have reasonable values (should be ~0 mean, ~1 std)
        from data_preprocessing.process_simulations_for_gnn import EdgeFeatures
        continuous_feat = [EdgeFeatures.VOL_BASE_CASE.value, EdgeFeatures.CAPACITY_BASE_CASE.value, 
                          EdgeFeatures.FREESPEED.value, EdgeFeatures.LENGTH.value]
        normalized_features = sample_normalized.x[:, continuous_feat]
        print(f"  Normalized feature stats: mean={normalized_features.mean().item():.4f}, std={normalized_features.std().item():.4f}")
        
        print("  Data normalized")
        
        # Create a simple dataset wrapper for the normalized list
        # This ensures DataLoader compatibility
        class NormalizedDataset(Dataset):
            def __init__(self, data_list):
                self.data_list = data_list
            
            def __len__(self):
                return len(self.data_list)
            
            def __getitem__(self, idx):
                return self.data_list[idx]
        
        normalized_dataset = NormalizedDataset(normalized_data_list)
        
        # Create dataloader for evaluation
        from torch.utils.data import DataLoader
        from training.help_functions import seed_worker
        
        # Adaptive batch size: reduce for large graphs (like Munich with 54k+ nodes)
        # Sample a graph to estimate size
        sample_graph = normalized_dataset[0]
        num_nodes = sample_graph.x.shape[0]
        
        # Reduce batch size for very large graphs to avoid memory issues
        if num_nodes > 40000:
            batch_size = 2  # Very large graphs (e.g., Munich)
            print(f"  Large graph detected ({num_nodes} nodes), using batch_size={batch_size}")
        elif num_nodes > 20000:
            batch_size = 4  # Large graphs (e.g., Neuulm)
            print(f"  Large graph detected ({num_nodes} nodes), using batch_size={batch_size}")
        else:
            batch_size = 8  # Normal graphs (e.g., Bamberg)
            print(f"  Using default batch_size={batch_size} for {num_nodes} nodes")
        
        val_dl = DataLoader(
            dataset=normalized_dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffling for evaluation
            num_workers=4,
            prefetch_factor=2,
            pin_memory=(device.type == 'cuda'),  # Only pin memory if using GPU
            collate_fn=collate_fn_eval,
            worker_init_fn=seed_worker,
            drop_last=False
        )
        
        if val_dl is None or len(normalized_dataset) == 0:
            raise ValueError(f"Failed to create dataloader. Got {len(normalized_dataset)} graphs.")
        
        print(f"  Created dataloader with {len(normalized_dataset)} graphs ({len(val_dl)} batches)")
        
        # Ensure model is in eval mode (critical for evaluation)
        model.eval()
        
        # Evaluate model
        from gnn.help_functions import GNN_Loss
        loss_func = GNN_Loss(
            loss_fct='mse',
            device=device,
            weighted=False
        )
        
        # Use validate_model_during_training to get all metrics
        # This function uses torch.inference_mode() internally and computes:
        # - val_loss, r_squared, spearman_corr, pearson_corr, hit_rates
        config_obj = type('Config', (), config)()
        with torch.inference_mode():  # Ensure no gradients are computed
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
            'city': city,
            'graphs_used': num_eval_graphs,  # Number of graphs evaluated
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
        print(f"  ERROR evaluating city {city}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on specific cities with a fixed number of graphs per city."
    )
    parser.add_argument("--pretrain_run_name", type=str, required=True,
                       help="Name of the pretrained run (e.g., 'general_surrogate_v0')")
    parser.add_argument("--project_name", type=str, default="Bavaria_Test",
                       help="Project name for the pretrained model (default: Bavaria_Test)")
    parser.add_argument("--cities", type=str, required=True,
                       help="Comma-separated list of cities to evaluate (e.g., 'bamberg,erlangen,muenchen,neuulm')")
    parser.add_argument("--graphs_per_city", type=int, default=200,
                       help="Number of graphs to use per city (default: 200)")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="data/analysis_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--gnn_arch", type=str, default="trans_encoder",
                       help="GNN architecture (default: trans_encoder)")
    parser.add_argument("--device_nr", type=int, default=0,
                       help="Device number (0 or 1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for graph selection (default: 42)")
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).resolve().parents[2]
    
    # Results directory (where pretrained model is stored)
    base_dir = project_root / 'inductive_gnn_data_results' / 'transductive'
    results_dir = base_dir / args.project_name / args.pretrain_run_name
    results_dir = results_dir.resolve()
    
    if not results_dir.exists():
        raise ValueError(f"Pretrained run directory does not exist: {results_dir}")
    
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
    
    # Parse cities
    cities = [c.strip() for c in args.cities.split(',') if c.strip()]
    
    # GPU setup
    gpus = get_available_gpus()
    if args.device_nr < len(gpus):
        set_cuda_visible_device(gpus[args.device_nr]['index'])  # Fix: pass index, not the whole dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("WARNING: CUDA not available, using CPU. This will be very slow for large graphs!")
    
    # Find checkpoint
    print("=" * 80)
    print("Loading pretrained model...")
    print(f"  Run directory: {results_dir}")
    print("=" * 80)
    
    checkpoint_path = find_checkpoint(results_dir)
    if checkpoint_path is None:
        raise ValueError(f"No checkpoint found in {results_dir}")
    
    print(f"  Found checkpoint: {checkpoint_path}")
    
    # Load model
    model, config = load_model_from_checkpoint(
        checkpoint_path, results_dir, args.gnn_arch, device
    )
    model = model.to(device)
    model.eval()  # Ensure model is in evaluation mode
    print(f"  Model loaded successfully")
    print(f"  Model is in eval mode: {not model.training}")
    print(f"  Config: use_all_features={config.get('use_all_features')}, "
          f"in_channels={config.get('in_channels')}, "
          f"target_type={config.get('target_type')}")
    
    # Load pretrained scaler
    scaler_path = results_dir / 'data_created_during_training' / 'train_x_scaler.pkl'
    if not scaler_path.exists():
        raise ValueError(f"Pretrained scaler not found: {scaler_path}")
    print(f"  Pretrained scaler found: {scaler_path}")
    
    # Evaluate each city
    print("\n" + "=" * 80)
    print("Evaluating cities...")
    print("=" * 80)
    
    results = []
    for i, city in enumerate(cities, 1):
        print(f"\n[{i}/{len(cities)}]")
        result = evaluate_city(
            model=model,
            config=config,
            city=city,
            dataset_path=dataset_path,
            graphs_per_city=args.graphs_per_city,
            device=device,
            pretrained_scaler_path=scaler_path,
            seed=args.seed
        )
        
        if result is not None:
            # Add pretrain run info
            result['pretrain_run_name'] = args.pretrain_run_name
            result['project_name'] = args.project_name
            results.append(result)
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)
    
    if results:
        # Save individual results per city
        for result in results:
            city = result['city']
            city_df = pd.DataFrame([result])
            
            # Save individual CSV per city
            city_csv_path = output_dir / f'evaluation_{args.pretrain_run_name}_{city}.csv'
            city_df.to_csv(city_csv_path, index=False)
            print(f"Saved {city} CSV to: {city_csv_path}")
            
            # Save individual JSON per city
            city_json_path = output_dir / f'evaluation_{args.pretrain_run_name}_{city}.json'
            with open(city_json_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Saved {city} JSON to: {city_json_path}")
        
        # Also save combined results
        df = pd.DataFrame(results)
        
        # Save combined CSV
        csv_path = output_dir / f'evaluation_{args.pretrain_run_name}_all_cities.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved combined CSV to: {csv_path}")
        
        # Save combined JSON
        json_path = output_dir / f'evaluation_{args.pretrain_run_name}_all_cities.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved combined JSON to: {json_path}")
        
        print(f"\nEvaluated {len(results)}/{len(cities)} cities successfully")
        print(f"Results include columns: {list(df.columns)}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("Summary:")
        print("=" * 80)
        for result in results:
            print(f"{result['city']}: R²={result['r_squared']:.4f}, "
                  f"Spearman={result['spearman']:.4f}, "
                  f"Graphs={result['graphs_used']}")
    else:
        print("ERROR: No results to save!")


if __name__ == "__main__":
    main()

