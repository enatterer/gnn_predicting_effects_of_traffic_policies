'''
Finetune a pre-trained model on new cities.

This script loads a checkpoint from a previous training run and continues training
(finetuning) on a new set of cities. The finetuned model is saved to a separate
directory to avoid overwriting the original model.

The idea is that the finetuning is done transductively, i.e. training and validating on the same cities. However, the cities used are different from the training cities for the original run.
For example, if in run_models the test_cities is schweinfurt, then in finetune_models the cities is schweinfurt.

Example usage:
python finetune_models.py --run_name base_run --gnn_arch trans_encoder --cities schweinfurt --project_name GNN_Inductive --start_from_scratch False
'''

import os
import sys
import json
import argparse
import re
import torch
from pathlib import Path
import random as _rnd
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # This is to avoid memory issues in Retina. Comment it out in LRZ AI

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from training.help_functions import *
from gnn.help_functions import GNN_Loss, CityBalancedGNNLoss

# Repo root: repo/scripts/training/finetune_models.py → go two levels up
project_root = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", project_root / "data")).resolve()

# Please adjust as needed
base_dir = os.path.join(project_root, 'inductive_gnn_data_results', 'transductive') # for saving results

def main():
    parser = argparse.ArgumentParser(description="Finetune a pre-trained GNN model on new cities.")
    
    # Required arguments
    parser.add_argument("--run_name", type=str, required=True,
                        help="Name of the original training run (e.g., 'base_run').")
    parser.add_argument("--gnn_arch", type=str, required=True,
                        help="The GNN architecture to use (must match the original model).",
                        choices=["point_net_transf_gat", "gat", "gatv2", "gatv3", "gcn", "gcn2", "trans_conv", "pnc", "fc_nn", "graphSAGE", "eign", "xgboost", "trans_encoder"])
    parser.add_argument("--cities", type=str, required=True,
                        help="Comma-separated list of cities to use for finetuning (e.g., 'wuerzburg,rosenheim,regensburg').")
    
    # Project name (defaults to GNN_Inductive based on the path structure)
    parser.add_argument("--project_name", type=str, default=None,
                        help="Name of the original project directory. Defaults to GNN_Inductive for inductive finetuning and GNN_Transductive for transductive finetuning.")
    parser.add_argument("--pretraining_inductive", type=str_to_bool, default=True,
                        help="Whether the finetune should start from an inductive (True) or transductive (False) pretraining run.")
    
    # Hyperparameters (all optional, with defaults matching run_models.py)
    parser.add_argument("--in_channels", type=int, default=5, help="The number of input channels.")
    parser.add_argument("--use_all_features", type=str_to_bool, default=True, help="Whether to use all features(True) or a subset of features(False).")
    parser.add_argument("--out_channels", type=int, default=1, help="The number of output channels.")
    parser.add_argument("--model_kwargs", type=str, default=None,
                        help='Additional model parameters (as defined in the class) in JSON format (path to the file).')
    parser.add_argument("--loss_fct", type=str, default="mse", help="The loss function to use. Supported: mse, l1.")
    parser.add_argument("--use_weighted_loss", type=str_to_bool, default=False, help="Whether to use weighted loss (based on vol_base_case) or not.")
    parser.add_argument("--target_normalization", type=str_to_bool, default=False, help="Whether targets are normalized during preprocessing.")
    parser.add_argument("--predict_mode_stats", type=str_to_bool, default=False, help="Whether to predict mode stats or not.")
    parser.add_argument("--target_type", type=str, default="abs_vol_car", help="Which target to use for training.", 
                        choices=["abs_vol_car", "abs_vol_car_percentage", "vol_car_signed_log", "vol_car_percentage_signed_log", "vol_car_mean_std", "vol_car_percentage_mean_std", "vol_car_min_max", "vol_car_percentage_min_max"])
    parser.add_argument("--use_bootstrapping", type=str_to_bool, default=False, help="Whether to use bootstrapping for train-validation split.")
    parser.add_argument("--use_weighted_sampling", type=str_to_bool, default=False, help="Whether to use weighted random sampling for training.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    
    # Learning rate scheduler parameters
    parser.add_argument("--peak_lr", type=float, default=0.0006, help="The peak learning rate (after warmup) from which decay will occur.")
    parser.add_argument("--initial_lr", type=float, default=0.0002, help="The initial learning rate from which training will start (used during warmup).")
    parser.add_argument("--warmup_fraction", type=float, default=0.1, help="Fraction of total training steps to use for linear warmup (0.0 to 1.0, e.g., 0.15 = 15%%).")
    parser.add_argument("--cosine_decay_rate", type=float, default=0.5, help="The rate at which the learning rate decays after warmup.")
    parser.add_argument("--min_lr_fraction", type=float, default=0.01, help="The minimum learning rate fraction of the initial learning rate to which the learning rate decays after warmup.")
    parser.add_argument("--early_stopping_patience", type=int, default=25, help="The early stopping patience.")
    
    # Dropout parameters
    parser.add_argument("--use_dropout", type=str_to_bool, default=False, help="Whether to use dropout.")
    parser.add_argument("--dropout", type=float, default=0.3, help="The dropout rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3, help="After how many steps the gradient should be updated.")
    parser.add_argument("--use_gradient_clipping", type=str_to_bool, default=True, help="Whether to use gradient clipping.")
    parser.add_argument("--device_nr", type=int, default=0, help="The device number (0 or 1 for Retina Roaster's two GPUs).")
    
    # GraphSAGE parameters
    parser.add_argument("--use_nested_neighbor_loader", type=str_to_bool, default=False, help="Whether to use nested neighbor loader.")
    parser.add_argument("--neighbor_sizes", type=str, default="5,5,5", help="The neighbor sizes for the nested neighbor loader (comma-separated).")
    parser.add_argument("--subgraphs_per_graph", type=int, default=2, help="The number of subgraphs to sample per graph.")
    parser.add_argument("--seed_size", type=int, default=10, help="The number of seed nodes in each subgraph.")
    parser.add_argument("--sampling_strategy", type=str, default="neighbor_sampling", help="The sampling strategy to use for the nested neighbor loader.",
                        choices=["neighbor_sampling", "random_walk"])
    parser.add_argument("--min_subgraph_nodes", type=int, default=500, help="The minimum number of nodes in a subgraph.")
    parser.add_argument("--max_subgraph_nodes", type=int, default=50000, help="The maximum number of nodes in a subgraph.")
    
    # Data augmentation parameters
    parser.add_argument("--use_data_augmentation", type=str_to_bool, default=False, help="Whether to use data augmentation.")
    parser.add_argument("--use_message_dropout_probability", type=float, default=0.0, help="The probability of message dropout (random dropout on message passing) during training. 0.0 means no dropout.")
    parser.add_argument("--augment_feature_noise_prob", type=str_to_bool, default=False, help="Whether to use Gaussian noise addition to node features as data augmentation.")
    parser.add_argument("--use_node_masking_probability", type=float, default=0.0, help="The probability of masking all features of a node to 0 during training. 0.0 means no node masking.")
    
    # Fast-iteration: optionally cap dataset sizes per split
    parser.add_argument("--limit_train_graphs", type=int, default=0, help="If >0, randomly keep only this many training graphs after reading metadata.")
    parser.add_argument("--limit_val_graphs", type=int, default=0, help="If >0, randomly keep only this many validation graphs after reading metadata.")
    parser.add_argument("--limit_test_graphs", type=int, default=0, help="If >0, randomly keep only this many test graphs after reading metadata.")
    
    # Unique model description for finetuning (optional, defaults to run_name + _finetuned)
    parser.add_argument("--unique_model_description", type=str, default=None, help="Unique description for the finetuned run (default: {run_name}_finetuned).")

    parser.add_argument("--start_from_scratch", type=str_to_bool, default=False,help="If True, initialize model weights randomly instead of loading a checkpoint.")

    args = vars(parser.parse_args())
    
    # Parse city lists from comma-separated strings
    cities = [city.strip() for city in args['cities'].split(',') if city.strip()]
    val_cities = cities.copy()
    train_cities = cities.copy()
    test_cities = cities.copy()
    
    # Parse neighbor_sizes from string to list
    if isinstance(args.get('neighbor_sizes', '5,5,5'), str):
        args['neighbor_sizes'] = [int(x.strip()) for x in args['neighbor_sizes'].split(',')]
    elif 'neighbor_sizes' not in args:
        args['neighbor_sizes'] = [5, 5, 5]  # Default value
    
    variant_label = args['unique_model_description']
    args['unique_model_description'] = build_unique_model_description(
        run_name=args['run_name'],
        cities=cities,
        start_from_scratch=args['start_from_scratch'],
        run_variant=variant_label,
    )
    
    set_random_seeds()
    
    # -------------------------------------------------------------------
    # Dataset and results directory selection
    # -------------------------------------------------------------------
    try:
        dataset_path = os.path.join(project_root, 'data','bavaria','inductive_data','training_data','kreisfreistadt')
        base_dir = os.path.join(project_root, 'inductive_gnn_data_results', 'transductive')
        if args['project_name'] is None:
            args['project_name'] = 'GNN_Inductive' if args['pretraining_inductive'] else 'GNN_Transductive'
    except Exception as e:
        raise ValueError(f"Error selecting dataset and results directory: {e}")
    
    try:
        # GPU setup
        gpus = get_available_gpus()
        best_gpu = select_best_gpu(gpus)
        set_cuda_visible_device(best_gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        latest_checkpoint_path = None

        original_run_dir = os.path.join(base_dir, args['project_name'], args['run_name'])

        if not args['start_from_scratch']:
            if not os.path.exists(original_run_dir):
                candidate_projects = []
                for candidate in (args['project_name'], 'GNN_Inductive', 'GNN_Transductive'):
                    if candidate not in candidate_projects:
                        candidate_projects.append(candidate)

                for candidate in candidate_projects:
                    candidate_dir = os.path.join(base_dir, candidate, args['run_name'])
                    if os.path.exists(candidate_dir):
                        if candidate != args['project_name']:
                            print(f"Original run directory not found under project '{args['project_name']}'. Using '{candidate}' instead.")
                        args['project_name'] = candidate
                        original_run_dir = candidate_dir
                        break

            if not os.path.exists(original_run_dir):
                raise ValueError(f"Original run directory does not exist under any known project name for run '{args['run_name']}'. Checked {candidate_projects}.")

        original_checkpoint_dir = os.path.join(original_run_dir, 'trained_model', 'checkpoints')

        if not args['start_from_scratch']:
            if not os.path.exists(original_run_dir):
                raise ValueError(f"Original run directory does not exist: {original_run_dir}")

            latest_checkpoint_path = find_latest_checkpoint(original_checkpoint_dir)

            checkpoint_info = torch.load(latest_checkpoint_path, map_location='cpu')
            print(f"Found checkpoint from epoch {checkpoint_info.get('epoch', 'unknown')}")
            print(f"Checkpoint validation loss: {checkpoint_info.get('val_loss', 'unknown')}")
            del checkpoint_info
        else:
            if not os.path.exists(original_run_dir):
                print(f"Warning: Original run directory {original_run_dir} not found. Continuing from scratch without checkpoint.")
            print("Starting finetuning from scratch (no checkpoint will be loaded).")

        # Set up paths for finetuned model
        model_save_path, path_to_save_dataloader, finetuned_checkpoint_dir = get_finetuned_model_paths(
            base_dir=base_dir,
            project_name=args['project_name'],
            run_name=args['run_name']
        )
        
        print(f"Finetuned model will be saved to: {model_save_path}")
        print(f"Finetuned checkpoints will be saved to: {finetuned_checkpoint_dir}")
                                
        # Start data preparation
        # If train, val, and test cities are all the same, we need to split the data to avoid overlap
        all_same_cities = (set(train_cities) == set(val_cities) == set(test_cities))
        
        if all_same_cities:
            print(f"Training, validation, and test all use the same cities: {train_cities}")
            print("Splitting data to ensure non-overlapping train/val/test sets...")
            
            # Load all data from the shared cities
            all_data = {'path': list(), 'policy_region': list(), 'scenario': list(), 'city': list()}
            for city in sorted(train_cities):
                load_metadata_from_disk(all_data, os.path.join(dataset_path, city, 'metadata.json'))
            
            print(f"Loaded {len(all_data['path'])} total graphs from {train_cities}")
            
            # Shuffle and split into train, val, and test
            _rnd.seed(42)  # For reproducibility
            indices = list(range(len(all_data['path'])))
            _rnd.shuffle(indices)
            
            # Calculate split sizes
            limit_train = args.get('limit_train_graphs', 0) if args.get('limit_train_graphs', 0) > 0 else len(indices)
            limit_val = args.get('limit_val_graphs', 0) if args.get('limit_val_graphs', 0) > 0 else len(indices)
            limit_test = args.get('limit_test_graphs', 0) if args.get('limit_test_graphs', 0) > 0 else 0
            
            # Ensure we don't exceed available data
            total_needed = limit_train + limit_val + limit_test
            if total_needed > len(indices):
                cities_str = ', '.join(train_cities)
                raise ValueError(
                    f"Insufficient data for cities {cities_str}: "
                    f"Requested {limit_train} train + {limit_val} val + {limit_test} test = {total_needed} graphs, "
                    f"but only {len(indices)} available. Skipping this city."
                )
            
            # Split indices - these are the shuffled positions
            train_indices = indices[:limit_train]  # First N indices for training
            val_indices = indices[limit_train:limit_train + limit_val]  # Next M indices for validation
            test_indices = indices[limit_train + limit_val:limit_train + limit_val + limit_test] if limit_test > 0 else []
            
            # Create train, val, and test data using the shuffled indices
            train_data = {'path': list(), 'policy_region': list(), 'scenario': list(), 'city': list()}
            val_data = {'path': list(), 'policy_region': list(), 'scenario': list(), 'city': list()}
            # Always create test_data as a dict (even if empty) to avoid None issues in load_data_and_split_into_subsets
            test_data = {'path': list(), 'policy_region': list(), 'scenario': list(), 'city': list()}
            
            # Use the shuffled indices to extract data
            for idx in train_indices:
                for k in ['path', 'policy_region', 'scenario', 'city']:
                    train_data[k].append(all_data[k][idx])
            
            for idx in val_indices:
                for k in ['path', 'policy_region', 'scenario', 'city']:
                    val_data[k].append(all_data[k][idx])
            
            # Add test data if we have test indices
            if limit_test > 0 and test_indices:
                for idx in test_indices:
                    for k in ['path', 'policy_region', 'scenario', 'city']:
                        test_data[k].append(all_data[k][idx])
            
            print(f"Split data: {len(train_data['path'])} training graphs, {len(val_data['path'])} validation graphs, {len(test_data['path'])} test graphs")
            
        else:
            raise ValueError(f"Different cities for train and val are not supported for finetuning.")

        print(f"→ Using {'INDUCTIVE' if args['pretraining_inductive'] else 'TRANSDUCTIVE'}-style data preparation for finetuning")
        train_dl, valid_dl, scalers_train = prepare_data_with_graph_features(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            use_inductive_variant=False, # Finetuning is always transductive
            batch_size=args['batch_size'],
            path_to_save_dataloader=path_to_save_dataloader,
            use_all_features=args['use_all_features'],
            use_weighted_batches=args.get('use_weighted_sampling', False),
            use_nested_neighbor_loader=args['use_nested_neighbor_loader'],
            neighbor_sizes=args['neighbor_sizes'],
            subgraphs_per_graph=args['subgraphs_per_graph'],
            seed_size=args['seed_size'],
            sampling_strategy=args['sampling_strategy'],
            min_subgraph_nodes=args['min_subgraph_nodes'],
            max_subgraph_nodes=args['max_subgraph_nodes'],
            aug_pos_rotation=args['use_data_augmentation'],
            aug_feature_noise=args['augment_feature_noise_prob'],
            aug_node_masking_probability=args['use_node_masking_probability']
        )

        config = setup_wandb(args)
        if args["model_kwargs"] is not None:
            with open(args["model_kwargs"], 'r') as f:
                model_kwargs = json.load(f)
        else:
            model_kwargs = {}
        
        # Create model instance
        gnn_instance = create_gnn_model(gnn_arch=config.gnn_arch,
                                        config=config,
                                        model_kwargs=model_kwargs,
                                        device=device).to(device)

        if not args['start_from_scratch'] and latest_checkpoint_path is not None:
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            missing_keys, unexpected_keys = gnn_instance.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys:
                print(f"Warning: Missing keys when loading checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys when loading checkpoint: {unexpected_keys}")
            # if config.use_target_standardization and 'target_mean' in checkpoint and 'target_std' in checkpoint:
            #     # Restore target statistics if they were saved
            #     gnn_instance.target_mean = checkpoint['target_mean'].to(device)
            #     gnn_instance.target_std = checkpoint['target_std'].to(device)
            #     print("Restored target statistics from checkpoint for finetuning")
        else:
            print("Initialized model weights randomly for scratch finetuning.")

        # Set up loss function
        if args.get('use_city_balanced_loss', False):
            loss_fct = CityBalancedGNNLoss(loss_fct=config.loss_fct, 
                                           device=device, 
                                           weighted=config.use_weighted_loss,
                                           num_nodes=train_dl.dataset[0].x.shape[0])
            print(f"Using city-balanced loss function - TRANSDUCTIVE VARIANT")
        else:
            loss_fct = GNN_Loss(loss_fct=config.loss_fct, 
                                num_nodes=train_dl.dataset[0].x.shape[0],
                                device=device, 
                                weighted=config.use_weighted_loss)
            print(f"Using standard loss function - TRANSDUCTIVE VARIANT")
        
        early_stopping = EarlyStopping(patience=config.early_stopping_patience, verbose=True)

        # Finetuning: start a fresh training schedule while using pretrained weights
        config.continue_training = False
        config.base_checkpoint_path = None
        
        # Note: Checkpoints will be saved to finetuned_model/checkpoints/ automatically
        # because model_save_path is set to finetuned_model/model.pth        
        
        print("→ Using gnn_instance.train_model (TRANSDUCTIVE method)")
        best_val_loss, best_epoch = gnn_instance.train_model(config=config,
                                                            loss_fct=loss_fct,
                                                            optimizer=torch.optim.AdamW(gnn_instance.parameters(), lr=config.peak_lr, weight_decay=1e-3) if config.gnn_arch != "xgboost" else None,
                                                            train_dl=train_dl,
                                                            valid_dl=valid_dl,
                                                            device=device,
                                                            early_stopping=early_stopping,
                                                            model_save_path=model_save_path)
        
        print(f'Finetuned model saved to {model_save_path} with validation loss: {best_val_loss} at epoch {best_epoch}')
        print_model_info(gnn_instance)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ""

def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the checkpoint directory.
    
    Args:
        checkpoint_dir: Path to the directory containing checkpoints
        
    Returns:
        Path to the latest checkpoint file, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Extract epoch numbers and find the maximum
    def extract_epoch(filename):
        match = re.search(r'checkpoint_epoch_(\d+)\.pt', filename)
        return int(match.group(1)) if match else -1
    
    latest_checkpoint = max(checkpoint_files, key=extract_epoch)
    latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    print(f"Found latest checkpoint: {latest_checkpoint_path} (epoch {extract_epoch(latest_checkpoint)})")
    return latest_checkpoint_path

def get_finetuned_model_paths(base_dir, project_name, run_name):
    """
    Get paths for saving finetuned model.
    Creates a 'finetuned_model' directory at the same level as 'trained_model'.
    
    Args:
        base_dir: Base directory for results
        project_name: Project name (e.g., 'GNN_Inductive')
        run_name: Name of the original run
        
    Returns:
        Tuple of (model_save_path, path_to_save_dataloader, checkpoint_dir)
    """
    unique_run_dir = os.path.join(base_dir, project_name, run_name)
    os.makedirs(unique_run_dir, exist_ok=True)
    
    # Create finetuned_model directory at the same level as trained_model
    finetuned_model_dir = os.path.join(unique_run_dir, 'finetuned_model')
    os.makedirs(finetuned_model_dir, exist_ok=True)
    
    model_save_path = os.path.join(finetuned_model_dir, 'model.pth')
    checkpoint_dir = os.path.join(finetuned_model_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    path_to_save_dataloader = os.path.join(unique_run_dir, 'data_created_during_finetuning')
    os.makedirs(path_to_save_dataloader, exist_ok=True)
    
    return model_save_path, path_to_save_dataloader, checkpoint_dir

if __name__ == '__main__':
    main()