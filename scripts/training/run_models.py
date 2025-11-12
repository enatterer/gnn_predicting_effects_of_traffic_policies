'''
Run GNN model training with configurable architecture and hyperparameters.

'dataset_path' and 'base_dir' need to be adjusted to the correct paths.
All the other parameters can be passed as command line arguments. Run `python run_models.py --help` to see the list of available arguments.

Example usage with default architecture, dropout, and most significant features found using ablation tests:
`python run_models.py --in_channels 5 --use_all_features False --num_epochs 500 --peak_lr 0.003 --early_stopping_patience 25 --use_dropout True --dropout 0.3`

Our use case:
python run_models.py --gnn_arch gatv2 --unique_model_description trial13 --in_channels 5 --use_all_features True --num_epochs 2 --peak_lr 0.003 --early_stopping_patience 25
python run_models.py --gnn_arch graphSAGE --unique_model_description graphSAGE_5_features_16_cities_retina --in_channels 5 --use_all_features True --num_epochs 2 --lr 0.003 --early_stopping_patience 25 --use_dropout True --dropout 0.3 --use_nested_neighbor_loader True --neighbor_sizes 5,5,5,5,5 
python run_models.py --gnn_arch eign --unique_model_description EIGN_trial_unsigned --in_channels 5 --use_all_features True --num_epochs 2 --peak_lr 0.003 --early_stopping_patience 25 
python run_models.py --gnn_arch trans_encoder --unique_model_description encoder_trial --in_channels 5 --use_all_features True --num_epochs 20 --peak_lr 0.0003 --early_stopping_patience 25
'''

from html import parser
import os
import sys
import json
import argparse

import torch
from pathlib import Path

# TODO: Check if this helps
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # This is to avoid memory issues in Retina. Comment it out in LRZ AI

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from training.help_functions import *
from gnn.help_functions import GNN_Loss, CityBalancedGNNLoss

# Repo root: repo/scripts/training/run_models.py → go two levels up
project_root = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", project_root / "data")).resolve()

# Use universal un-normalized data, any normalization will be handled during training
dataset_path = os.path.join(project_root, 'data','inductive_data','training_data','kreisfreistadt')

# Please adjust as needed
base_dir = os.path.join(project_root, 'inductive_gnn_data_results', 'transductive') # for saving results

# ['wuerzburg','aschaffenburg','regensburg','landshut','bayreuth','erlangen','fuerth','kempten','neuulm','muenchen','augsburg','rosenheim','schweinfurt','bamberg','nuernberg', 'ingolstadt']
train_cities = ['aschaffenburg','landshut','wuerzburg','regensburg','bayreuth','fuerth','kempten','neuulm','augsburg','rosenheim','nuernberg', 'ingolstadt']
val_cities =[] # Non empty implies inductive learning
test_cities = ['muenchen', 'neuulm', 'erlangen', 'bamberg'] # Non empty implies inductive learning
    
def main():
    parser = argparse.ArgumentParser(description="Run GNN model training with configurable parameters.")
    parser.add_argument("--gnn_arch", type=str, default="trans_encoder",
                        help="The GNN architecture to use.",
                        choices=["gatv2", "trans_conv", "graphSAGE", "trans_encoder"])  # Add more as you implement them
    parser.add_argument("--use_inductive_variant", type=str_to_bool, default=True,
                        help="Whether to perform inductive or transductive training.")
    parser.add_argument("--project_name", type=str, default=None,
                        help="Override for project directory/WandB project. Defaults automatically based on use_inductive_variant.")
    parser.add_argument("--unique_model_description", type=str, default="trans_encoder_5_features_15_cities",
                        help="A unique description for the run.")
    parser.add_argument("--in_channels", type=int, default=5, help="The number of input channels.")
    parser.add_argument("--use_all_features", type=str_to_bool, default=True, help="Whether to use all features or 5 core features.")
    parser.add_argument("--out_channels", type=int, default=1, help="The number of output channels.")
    parser.add_argument("--model_kwargs", type=str, default=None,
                        help='Additional model parameters (as defined in the class) in JSON format (path to the file).' \
                        'If not provided, defaults params will be used.') 
    parser.add_argument("--loss_fct", type=str, default="mse", help="The loss function to use. Supported: mse, l1.")
    parser.add_argument("--use_weighted_loss", type=str_to_bool, default=False, help="Whether to use weighted loss (based on vol_base_case) or not.")
    parser.add_argument("--use_city_balanced_loss", type=str_to_bool, default=False,
                        help="Optional for inductive variant: Whether to use city-balanced loss function (based on CityBalancedGNNLoss) or not. \
                            For transductive variant use standard node-weighted loss function (based on GNN_Loss).")
    parser.add_argument("--use_target_standardization", type=str_to_bool, default=False, help="Whether to use target standardization during training.")
    parser.add_argument("--target_type", type=str, default="abs_vol_car", help="Which target to use for training.", 
                        choices=["abs_vol_car", "abs_vol_car_percentage", "vol_car_signed_log", "vol_car_percentage_signed_log", "vol_car_mean_std", "vol_car_percentage_mean_std", "vol_car_min_max", "vol_car_percentage_min_max"])
    parser.add_argument("--use_weighted_batches", type=str_to_bool, default=False, help="Whether to use weighted random sampling for training batches.")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    
    #parameters for the learning rate scheduler
    parser.add_argument("--peak_lr", type=float, default=0.003, help="The peak learning rate (after warmup) from which decay will occur.")
    parser.add_argument("--initial_lr", type=float, default=0.001, help="The initial learning rate from which training will start (used during warmup).")
    parser.add_argument("--warmup_fraction", type=float, default=0.1, help="Fraction of total training steps to use for linear warmup (0.0 to 1.0, e.g., 0.15 = 15%%).")
    parser.add_argument("--cosine_decay_rate", type=float, default=0.5, help="The rate at which the learning rate decays after warmup.")
    parser.add_argument("--min_lr_fraction", type=float, default=0.01, help="The minimum learning rate fraction of the initial learning rate to which the learning rate decays after warmup.")
    parser.add_argument("--early_stopping_patience", type=int, default=40, help="The early stopping patience.")
    
    parser.add_argument("--use_dropout", type=str_to_bool, default=False, help="Whether to use dropout.")
    parser.add_argument("--dropout", type=float, default=0.3, help="The dropout rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3, help="After how many steps the gradient should be updated.")
    parser.add_argument("--use_gradient_clipping", type=str_to_bool, default=True, help="Whether to use gradient clipping.")
    parser.add_argument("--device_nr", type=int, default=0, help="The device number (0 or 1 for Retina Roaster's two GPUs).")
    parser.add_argument("--continue_training", type=str_to_bool, default=False, help="Whether to continue training from a checkpoint.")
    parser.add_argument("--base_checkpoint_path", type=str, default=None, help="Path to the checkpoint to continue training from.")
    
    # Parameters for the GraphSAGE
    parser.add_argument("--use_nested_neighbor_loader", type=str_to_bool, default=False, help="Whether to use nested neighbor loader.")
    parser.add_argument("--neighbor_sizes", type=str, default="5,5,5", help="The neighbor sizes for the nested neighbor loader (comma-separated).")
    parser.add_argument("--subgraphs_per_graph", type=int, default=2, help="The number of subgraphs to sample per graph.")
    parser.add_argument("--seed_size", type=int, default=10, help="The number of seed nodes in each subgraph.")
    parser.add_argument("--sampling_strategy", type=str, default="neighbor_sampling", help="The sampling strategy to use for the nested neighbor loader.",
                        choices=["neighbor_sampling", "random_walk"])
    parser.add_argument("--min_subgraph_nodes", type=int, default=500, help="The minimum number of nodes in a subgraph.")
    parser.add_argument("--max_subgraph_nodes", type=int, default=50000, help="The maximum number of nodes in a subgraph.")
    
    # Parameters for Data Augmentation
    parser.add_argument("--aug_pos_rotation", type=str_to_bool, default=False, help="Whether to use Position Rotation augmentation.")
    parser.add_argument("--aug_feature_noise", type=str_to_bool, default=False, help="Whether to use Gaussian noise addition to node features as data augmentation.")
    parser.add_argument("--aug_node_masking_probability", type=float, default=0.0, help="The probability of masking all features of a node to 0 during training. 0.0 means no node masking.")

    # Fast-iteration: optionally cap dataset sizes per split (random subsample)
    parser.add_argument("--limit_available_graphs", type=int, default=0, help="If >0, randomly keep only this many available graphs after reading metadata (applies before splitting into train/val/test).")

    args = vars(parser.parse_args())
    
    # Parse neighbor_sizes from string to list
    if isinstance(args['neighbor_sizes'], str):
        args['neighbor_sizes'] = [int(x.strip()) for x in args['neighbor_sizes'].split(',')]
    
    set_random_seeds()
    
    if args.get('project_name') is None:
        args['project_name'] = "GNN_Inductive" if args['use_inductive_variant'] else "GNN_Transductive"
    
    try:
        
        # Continue with GPU setup and model training
        gpus = get_available_gpus()
        best_gpu = select_best_gpu(gpus)
        set_cuda_visible_device(best_gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create directory for the run
        unique_run_dir = os.path.join(base_dir, args['project_name'], args['unique_model_description'])
        os.makedirs(unique_run_dir, exist_ok=True)
        
        model_save_path, path_to_save_dataloader = get_paths(base_dir=os.path.join(base_dir, args['project_name']),
                                                             unique_model_description=args['unique_model_description'],
                                                             model_save_path='trained_model/model.pth')
        
        # Start data prepration
        train_data = {'path': list(), 'policy_region': list(), 'scenario': list(), 'city':list()}    
        for city in sorted(train_cities):
            load_metadata_from_disk(train_data, os.path.join(dataset_path, city, 'metadata.json'))

        # Optional: Subsample training graphs for faster iterations
        if args.get('limit_available_graphs', 0) and args['limit_available_graphs'] > 0 and len(train_data['path']) > args['limit_available_graphs']:
            import random as _rnd
            indices = list(range(len(train_data['path'])))
            _rnd.shuffle(indices)
            keep = set(indices[:args['limit_available_graphs']])
            for k in ['path','policy_region','scenario','city']:
                train_data[k] = [train_data[k][i] for i in range(len(indices)) if i in keep]

        if args['use_inductive_variant']:
            if len(val_cities) > 0:
                val_data = {'path': list(), 'policy_region': list(), 'scenario': list(), 'city':list()}
                for city in sorted(val_cities):
                    load_metadata_from_disk(val_data, os.path.join(dataset_path, city, 'metadata.json'))
                if args.get('limit_available_graphs', 0) and args['limit_available_graphs'] > 0 and len(val_data['path']) > args['limit_available_graphs']:
                    import random as _rnd
                    indices = list(range(len(val_data['path'])))
                    _rnd.shuffle(indices)
                    keep = set(indices[:args['limit_available_graphs']])
                    for k in ['path','policy_region','scenario','city']:
                        val_data[k] = [val_data[k][i] for i in range(len(indices)) if i in keep]
            else:
                val_data = None

            if len(test_cities) > 0:
                test_data = {'path': list(), 'policy_region': list(), 'scenario': list(), 'city':list()}
                for city in sorted(test_cities):
                    load_metadata_from_disk(test_data, os.path.join(dataset_path, city, 'metadata.json'))
                if args.get('limit_available_graphs', 0) and args['limit_available_graphs'] > 0 and len(test_data['path']) > args['limit_available_graphs']:
                    import random as _rnd
                    indices = list(range(len(test_data['path'])))
                    _rnd.shuffle(indices)
                    keep = set(indices[:args['limit_available_graphs']])
                    for k in ['path','policy_region','scenario','city']:
                        test_data[k] = [test_data[k][i] for i in range(len(indices)) if i in keep]
            else:
                test_data = None
        else:
            # Transductive variant: validation/test splits derived from training cities.
            val_data = None
            test_data = None

        print(f"Using {'INDUCTIVE' if args['use_inductive_variant'] else 'TRANSDUCTIVE'} data preparation!")
        train_dl, valid_dl, scalers_train = prepare_data_with_graph_features(train_data=train_data,
                                                                             val_data=val_data,
                                                                             test_data=test_data,
                                                                             use_inductive_variant=args['use_inductive_variant'], # Conditional (Transductive/Inductive)
                                                                             batch_size=args['batch_size'],
                                                                             path_to_save_dataloader=path_to_save_dataloader,
                                                                             use_all_features=args['use_all_features'],
                                                                             use_weighted_batches=args['use_weighted_batches'],
                                                                             use_nested_neighbor_loader=args['use_nested_neighbor_loader'],
                                                                             neighbor_sizes=args['neighbor_sizes'],
                                                                             subgraphs_per_graph=args['subgraphs_per_graph'],
                                                                             seed_size=args['seed_size'],
                                                                             sampling_strategy=args['sampling_strategy'],
                                                                             min_subgraph_nodes=args['min_subgraph_nodes'],
                                                                             max_subgraph_nodes=args['max_subgraph_nodes'],
                                                                             aug_pos_rotation=args['aug_pos_rotation'],
                                                                             aug_feature_noise=args['aug_feature_noise'],
                                                                             aug_node_masking_probability=args['aug_node_masking_probability'])

        # Create WandB config
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
                                        device=device)

        # LOSS FUNCTION
        if args.get('use_city_balanced_loss', False):
            loss_fct = CityBalancedGNNLoss(loss_fct=config.loss_fct, 
                                           device=device, 
                                           weighted=config.use_weighted_loss,
                                           num_nodes=train_dl.dataset[0].x.shape[0])
            print("Using city-balanced loss function- INDUCTIVE VARIANT")
        else:
            loss_fct = GNN_Loss(loss_fct=config.loss_fct,
                                device=device, 
                                weighted=config.use_weighted_loss,
                                num_nodes=train_dl.dataset[0].x.shape[0])
            print("Using standard loss function - TRANSDUCTIVE VARIANT")

        early_stopping = EarlyStopping(patience=config.early_stopping_patience, verbose=True)

        print(f"Training method: {'INDUCTIVE' if args['use_inductive_variant'] else 'TRANSDUCTIVE'}")
        best_val_loss, best_epoch = gnn_instance.train_model(config=config,
                                                             loss_fct=loss_fct,
                                                             optimizer=torch.optim.AdamW(gnn_instance.parameters(), lr=config.peak_lr, weight_decay=1e-3) if config.gnn_arch != "xgboost" else None,
                                                             train_dl=train_dl,
                                                             valid_dl=valid_dl,
                                                             device=device,
                                                             early_stopping=early_stopping,
                                                             model_save_path=model_save_path)
        
        print(f'Best model saved to {model_save_path} with validation loss: {best_val_loss} at epoch {best_epoch}')   
        print_model_info(gnn_instance)

    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ""


if __name__ == '__main__':
    main()
