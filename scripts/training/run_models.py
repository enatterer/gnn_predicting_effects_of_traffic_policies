'''
Run GNN model training with configurable architecture and hyperparameters.

'dataset_path' and 'base_dir' need to be adjusted to the correct paths.
All the other parameters can be passed as command line arguments. Run `python run_models.py --help` to see the list of available arguments.

Example usage with default architecture, dropout, and most significant features found using ablation tests:
`python run_models.py --in_channels 5 --use_all_features False --num_epochs 500 --lr 0.003 --early_stopping_patience 25 --use_dropout True --dropout 0.3`
'''

import os
import sys
import json
import argparse

import torch

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from training.help_functions import *
from gnn.help_functions import GNN_Loss, compute_baseline_of_mean_target, compute_baseline_of_no_policies

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Please adjust as needed
dataset_path = os.path.join(project_root, 'data', 'train_data', 'dist_not_connected_10k_1pct')
base_dir = os.path.join(project_root, 'data')

def main():
    try:
        datalist = []
        batch_num = 1
        while True: # Change this to "and batch_num < 10" for a faster run
            print(f"Processing batch number: {batch_num}")
            # total_memory, available_memory, used_memory = get_memory_info()
            # print(f"Total Memory: {total_memory:.2f} GB")
            # print(f"Available Memory: {available_memory:.2f} GB")
            # print(f"Used Memory: {used_memory:.2f} GB")
            batch_file = os.path.join(dataset_path, f'datalist_batch_{batch_num}.pt')
            if not os.path.exists(batch_file):
                break
            batch_data = torch.load(batch_file, map_location='cpu')
            if isinstance(batch_data, list):
                datalist.extend(batch_data)
            batch_num += 1
        print(f"Loaded {len(datalist)} items into datalist")

        # Temp fix, rerun data_preprocessing to solve.
        for i, data in enumerate(datalist):
            data.num_nodes = data.x.shape[0]

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    parser = argparse.ArgumentParser(description="Run GNN model training with configurable parameters.")
    parser.add_argument("--gnn_arch", type=str, default="point_net_transf_gat",
                        help="The GNN architecture to use.",
                        choices=["point_net_transf_gat", "gat", "gcn", "gcn2", "trans_conv", "pnc", "graphSAGE", "eign"])  # Add more as you implement them
    parser.add_argument("--project_name", type=str, default="TR-C_Benchmarks",
                        help="The name of the project, used for saving the corresponding runs, and as the WandB project name.")
    parser.add_argument("--unique_model_description", type=str, default="point_net_transf_gat_5_features",
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
    parser.add_argument("--num_epochs", type=int, default=3000, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="The learning rate for the model.")
    parser.add_argument("--early_stopping_patience", type=int, default=100, help="The early stopping patience.")
    parser.add_argument("--use_dropout", type=str_to_bool, default=False, help="Whether to use dropout.")
    parser.add_argument("--dropout", type=float, default=0.3, help="The dropout rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3, help="After how many steps the gradient should be updated.")
    parser.add_argument("--use_gradient_clipping", type=str_to_bool, default=True, help="Whether to use gradient clipping.")
    parser.add_argument("--device_nr", type=int, default=0, help="The device number (0 or 1 for Retina Roaster's two GPUs).")

    args = vars(parser.parse_args())
    set_random_seeds()
    
    try:
        gpus = get_available_gpus()
        best_gpu = select_best_gpu(gpus)
        set_cuda_visible_device(best_gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create directory for the run
        unique_run_dir = os.path.join(base_dir, args['project_name'], args['unique_model_description'])
        os.makedirs(unique_run_dir, exist_ok=True)
        
        model_save_path, path_to_save_dataloader = get_paths(base_dir=os.path.join(base_dir, args['project_name']), unique_model_description=args['unique_model_description'], model_save_path='trained_model/model.pth')
        train_dl, valid_dl, scalers_train, scalers_validation = prepare_data_with_graph_features(datalist=datalist,
                                                                                                  batch_size=args['batch_size'],
                                                                                                  path_to_save_dataloader=path_to_save_dataloader,
                                                                                                  use_all_features=args['use_all_features'],
                                                                                                  use_bootstrapping=args['use_bootstrapping'])
        
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
        loss_fct = GNN_Loss(config.loss_fct, datalist[0].x.shape[0], device, config.use_weighted_loss)
        
        baseline_loss_mean_target = compute_baseline_of_mean_target(dataset=train_dl, loss_fct=loss_fct, device=device, scalers=scalers_train)
        baseline_loss = compute_baseline_of_no_policies(dataset=train_dl, loss_fct=loss_fct, device=device, scalers=scalers_train)
        print("baseline loss mean " + str(baseline_loss_mean_target))
        print("baseline loss no  " + str(baseline_loss) )

        early_stopping = EarlyStopping(patience=config.early_stopping_patience, verbose=True)
        best_val_loss, best_epoch = gnn_instance.train_model(  
                    config=config, 
                    loss_fct=loss_fct,
                    optimizer=torch.optim.AdamW(gnn_instance.parameters(), lr=config.lr, weight_decay=1e-4),
                    train_dl=train_dl, 
                    valid_dl=valid_dl,
                    device=device, 
                    early_stopping=early_stopping,
                    model_save_path=model_save_path,
                    scalers_train=scalers_train,
                    scalers_validation=scalers_validation)
        print(f'Best model saved to {model_save_path} with validation loss: {best_val_loss} at epoch {best_epoch}')   
        
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ""


if __name__ == '__main__':
    main()
