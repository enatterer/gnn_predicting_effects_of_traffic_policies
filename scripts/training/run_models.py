'''
Run GNN model training with configurable architecture and hyperparameters.

'dataset_path' and 'base_dir' need to be adjusted to the correct paths.
All the other parameters can be passed as command line arguments. Run `python run_models.py --help` to see the list of available arguments.

Example usage with default architecture, dropout, and most significant features found using ablation tests:
`python run_models.py --in_channels 5 --use_all_features False --num_epochs 500 --lr 0.003 --early_stopping_patience 25 --use_dropout True --dropout 0.3`

Our use case:
python run_models.py --gnn_arch gat --unique_model_description gat_transductive_only_rosenheim --in_channels 5 --use_all_features False --num_epochs 20 --lr 0.003 --early_stopping_patience 25 --use_dropout True --dropout 0.3 --learning_variant transductive --split_variant non_uniform --sampler_variant nested_fixed_proportion --use_weighted_loss True --run_name 1
'''

import os
import sys
import json
import argparse

import torch
#from torchinfo import summary

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from training.help_functions import *
from gnn.help_functions import GNN_Loss, compute_baseline_of_mean_target, compute_baseline_of_no_policies

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Note: base_dir will be set after args parsing to include learning_variant

    
def main():
    parser = argparse.ArgumentParser(description="Run GNN model training with configurable parameters.")
    parser.add_argument("--gnn_arch", type=str, default="trans_conv",
                        help="The GNN architecture to use.",
                        choices=["point_net_transf_gat", "gat", "gatv2", "gcn", "gcn2", "trans_conv", "pnc", "fc_nn", "graphSAGE", "eign", "xgboost"])  # Add more as you implement them
    parser.add_argument("--project_name", type=str, default="Inductive GNN_16 Bavarian cities",
                        help="The name of the project, used for saving the corresponding runs, and as the WandB project name.")
    parser.add_argument("--unique_model_description", type=str, default="point_net_transf_gat_5_features_16_cities",
                        help="A unique description for the run.")
    parser.add_argument("--in_channels", type=int, default=5, help="The number of input channels.")
    parser.add_argument("--use_all_features", type=str_to_bool, default=False, help="Whether to use all features.")
    parser.add_argument("--out_channels", type=int, default=1, help="The number of output channels.")
    parser.add_argument("--model_kwargs", type=str, default=None,
                        help='Additional model parameters (as defined in the class) in JSON format (path to the file).' \
                        'If not provided, defaults params will be used.') 
    parser.add_argument("--loss_fct", type=str, default="mse", help="The loss function to use. Supported: mse, l1.")
    parser.add_argument("--use_weighted_loss", type=str_to_bool, default=False, help="Whether to use weighted loss (based on vol_base_case) or not.")
    parser.add_argument("--predict_mode_stats", type=str_to_bool, default=False, help="Whether to predict mode stats or not.")
    parser.add_argument("--use_bootstrapping", type=str_to_bool, default=False, help="Whether to use bootstrapping for train-validation split.")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="The learning rate for the model.")
    parser.add_argument("--early_stopping_patience", type=int, default=25, help="The early stopping patience.")
    parser.add_argument("--use_dropout", type=str_to_bool, default=False, help="Whether to use dropout.")
    parser.add_argument("--dropout", type=float, default=0.3, help="The dropout rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3, help="After how many steps the gradient should be updated.")
    parser.add_argument("--use_gradient_clipping", type=str_to_bool, default=True, help="Whether to use gradient clipping.")
    parser.add_argument("--device_nr", type=int, default=0, help="The device number (0 or 1 for Retina Roaster's two GPUs).")
    parser.add_argument("--continue_training", type=str_to_bool, default=False, help="Whether to continue training from a checkpoint.")
    parser.add_argument("--base_checkpoint_path", type=str, default=None, help="Path to the checkpoint to continue training from.")
    parser.add_argument("--learning_variant", type=str, default="transductive",
                   choices=["transductive", "moderate_inductive", "complete_inductive"],
                   help="Learning paradigm: transductive uses all data mixed, inductive separates seen/unseen cities")
    parser.add_argument("--split_variant", type=str, default="uniform", help="The variant of the train-validation-test data split.",
                        choices=["uniform", "non_uniform"])
    parser.add_argument("--sampler_variant", type=str, default="fixed_proportion", help="The variant of the sampler.",
                        choices=["fixed_proportion", "nested_fixed_proportion"])
    parser.add_argument('--run_name', type=str, default=None, help="The name of the run from data preprocessing.")
    args = vars(parser.parse_args())
    set_random_seeds()
    
    # Set dataset paths based on learning variant
    if args['learning_variant'] == 'transductive':
        dataset_path = os.path.join(project_root, 'inductive_gnn_data', 'training_data', 'transductive', f'run_{args["run_name"]}')
        unseen_dataset_path = None  # No separate unseen data
    else:
        # For inductive variants, load both seen and unseen
        dataset_path = os.path.join(project_root, 'inductive_gnn_data', 'training_data', 
                                   args['learning_variant'], 'seen', f'run_{args["run_name"]}')
        unseen_dataset_path = os.path.join(project_root, 'inductive_gnn_data', 'training_data', 
                                          args['learning_variant'], 'unseen', f'run_{args["run_name"]}')
    
    # Set base directory for results - organized by learning variant first
    base_dir = os.path.join(project_root, 'inductive_gnn_data_results', args['learning_variant'])
    
    try:
        # Load data based on learning variant
        if args['learning_variant'] == 'transductive':
            # Load all data into single datalist
            datalist = []
            batch_num = 1
            while True:
                batch_file = os.path.join(dataset_path, f'datalist_batch_{batch_num}.pt')
                if not os.path.exists(batch_file):
                    break
                batch_data = torch.load(batch_file, map_location='cpu')
                if isinstance(batch_data, list):
                    datalist.extend(batch_data)
                batch_num += 1
            print(f"Loaded {len(datalist)} items for transductive learning")
            
            unseen_datalist = None  # No separate unseen data
            
        else:  # inductive variants
            # Load seen data (for training/validation)
            seen_datalist = []
            batch_num = 1
            while True:
                batch_file = os.path.join(dataset_path, f'datalist_batch_{batch_num}.pt')
                if not os.path.exists(batch_file):
                    break
                batch_data = torch.load(batch_file, map_location='cpu')
                if isinstance(batch_data, list):
                    seen_datalist.extend(batch_data)
                batch_num += 1
            print(f"Loaded {len(seen_datalist)} items from seen cities")
            
            # Load unseen data (for testing)
            unseen_datalist = []
            batch_num = 1
            while True:
                batch_file = os.path.join(unseen_dataset_path, f'datalist_batch_{batch_num}.pt')
                if not os.path.exists(batch_file):
                    break
                batch_data = torch.load(batch_file, map_location='cpu')
                if isinstance(batch_data, list):
                    unseen_datalist.extend(batch_data)
                batch_num += 1
            print(f"Loaded {len(unseen_datalist)} items from unseen cities")
            
            datalist = seen_datalist  # For compatibility with existing code

        # data.num_nodes should be set correctly during preprocessing
        
        # Continue with GPU setup and model training
        gpus = get_available_gpus()
        best_gpu = select_best_gpu(gpus)
        set_cuda_visible_device(best_gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_save_path, path_to_save_dataloader = get_paths(base_dir=os.path.join(base_dir, args['project_name']), unique_model_description=args['unique_model_description'],
                                                              run_name=args['run_name'], model_save_path='trained_model/model.pth')
        
        if args['learning_variant'] == 'transductive':
            # Current behavior: train/val/test split from all data
            train_dl, valid_dl, scalers_train = prepare_data_with_graph_features(
                datalist=datalist,
                batch_size=args['batch_size'],
                path_to_save_dataloader=path_to_save_dataloader,
                use_all_features=args['use_all_features'],
                use_bootstrapping=args['use_bootstrapping'],
                is_eign=(args['gnn_arch'] == "eign"),
                split_mode="full",  # NEW: indicates train/val/test split
                split_variant=args['split_variant'], # NEW: indicates uniform or non-uniform split
                sampler_variant=args['sampler_variant'] # NEW: indicates fixed_proportion or nested_fixed_proportion
            )
            test_dl = None  # Test is handled within prepare_data_with_graph_features
            
        else:  # inductive variants
            # Train/val split on seen data only
            train_dl, valid_dl, scalers_train = prepare_data_with_graph_features(
                datalist=datalist,  # seen data only
                batch_size=args['batch_size'],
                path_to_save_dataloader=path_to_save_dataloader,
                use_all_features=args['use_all_features'],
                use_bootstrapping=args['use_bootstrapping'],
                is_eign=(args['gnn_arch'] == "eign"),
                split_mode="train_val_only",  # NEW: indicates train/val split only
                split_variant=args['split_variant'], # NEW: indicates uniform or non-uniform split
                sampler_variant=args['sampler_variant'] # NEW: indicates fixed_proportion or nested_fixed_proportion
            )
            
            # Create test dataloader from unseen data using same scalers
            test_dl = prepare_test_dataloader_from_unseen(
                unseen_datalist=unseen_datalist,
                scalers_train=scalers_train,
                batch_size=args['batch_size'],
                use_all_features=args['use_all_features'],
                path_to_save_dataloader=path_to_save_dataloader,
                is_eign=(args['gnn_arch'] == "eign")
            )
            
        verify_batch_distribution(train_dl)
        verify_batch_distribution(valid_dl)
        if args['learning_variant'] != 'transductive':
            verify_batch_distribution(test_dl)
        
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
        
        gnn_instance = gnn_instance.to(device)  
        # Check if all graphs have same size for safety
        graph_sizes = [data.x.shape[0] for data in datalist[:50]]  # Check first 50
        if len(set(graph_sizes)) > 1:
            print(f"⚠️  WARNING: Variable graph sizes detected: {set(graph_sizes)}")
            print("   Loss function relies on batch tensor being provided correctly")
        
        loss_fct = GNN_Loss(loss_fct=config.loss_fct, num_nodes=datalist[0].x.shape[0], device=device, weighted=config.use_weighted_loss)
        
        ## Not needed now, Naive MSE doesn't tell anything!
        # baseline_loss_mean_target = compute_baseline_of_mean_target(dataset=train_dl, loss_fct=loss_fct, device=device, scalers=scalers_train)
        # baseline_loss = compute_baseline_of_no_policies(dataset=train_dl, loss_fct=loss_fct, device=device, scalers=scalers_train)
        # print("baseline loss mean " + str(baseline_loss_mean_target))
        # print("baseline loss no  " + str(baseline_loss) )

        early_stopping = EarlyStopping(patience=config.early_stopping_patience, verbose=True)
        best_val_loss, best_epoch = gnn_instance.train_model(config=config,
                                                             loss_fct=loss_fct,
                                                             optimizer=torch.optim.AdamW(gnn_instance.parameters(), lr=config.lr, weight_decay=1e-4) if config.gnn_arch != "xgboost" else None,
                                                             train_dl=train_dl,
                                                             valid_dl=valid_dl,
                                                             device=device,
                                                             early_stopping=early_stopping,
                                                             model_save_path=model_save_path,
                                                             scalers_train=scalers_train
                                                            )
        
        print(f'Best model saved to {model_save_path} with validation loss: {best_val_loss} at epoch {best_epoch}')   
        print_model_info(gnn_instance)

    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ""


if __name__ == '__main__':
    main()
