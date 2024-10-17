import math
import numpy as np
import wandb
import random
import torch
import torch_geometric
from torch_geometric.data import Data
import sys
import os
from tqdm import tqdm
import signal
import joblib
import argparse
import json
import os
import subprocess
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import StandardScaler

import help_functions as hf

# Add the 'scripts' directory to the Python path
scripts_path = os.path.abspath(os.path.join('..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)
    
import gnn_io as gio
import gnn_architecture as garch

PARAMETER_ORDER = [
    "project_name",
    "predict_mode_stats",
    "in_channels",
    "out_channels",
    "point_net_conv_layer_structure_local_mlp",
    "point_net_conv_layer_structure_global_mlp",
    "gat_conv_layer_structure",
    "num_epochs",
    "batch_size",
    "lr",
    "early_stopping_patience",
    "use_dropout",
    "dropout",
    "gradient_accumulation_steps",
    "device_nr",
    "unique_model_description"
]

def get_parameters(args):
    params = {
        "project_name": "runs_17_10_2024",
        "predict_mode_stats": args.predict_mode_stats,
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "point_net_conv_layer_structure_local_mlp": [int(x) for x in args.point_net_conv_layer_structure_local_mlp.split(',')],
        "point_net_conv_layer_structure_global_mlp": [int(x) for x in args.point_net_conv_layer_structure_global_mlp.split(',')],
        "gat_conv_layer_structure": [int(x) for x in args.gat_conv_layer_structure.split(',')],
        "num_epochs": args.num_epochs,
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "early_stopping_patience": args.early_stopping_patience,
        "use_dropout": args.use_dropout,
        "dropout": args.dropout,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "device_nr": args.device_nr
    }
    
    params["unique_model_description"] = (
        f"pnc_local_{gio.int_list_to_string(lst=params['point_net_conv_layer_structure_local_mlp'], delimiter='_')}_"
        f"pnc_global_{gio.int_list_to_string(lst=params['point_net_conv_layer_structure_global_mlp'], delimiter='_')}_"
        f"hidden_layer_str_{gio.int_list_to_string(lst=params['gat_conv_layer_structure'], delimiter='_')}_"
        f"dropout_{params['dropout']}_"
        f"use_dropout_{params['use_dropout']}_"
        f"predict_mode_stats_{params['predict_mode_stats']}"
    )
    
    return params

def main():
    try:
        dataset_path = '../../data/train_data/sim_output_1pm_capacity_reduction_10k_15_10_2024/'
        datalist = []
        batch_num = 1
        while True and batch_num < 10:
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

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    parser = argparse.ArgumentParser(description="Run GNN model training with configurable parameters.")
    parser.add_argument("--in_channels", type=int, default=15, help="The number of input channels.")
    parser.add_argument("--out_channels", type=int, default=1, help="The number of output channels.")
    parser.add_argument("--predict_mode_stats", type=hf.str_to_bool, default=False, help="Whether to predict mode stats or not.")
    parser.add_argument("--point_net_conv_layer_structure_local_mlp", type=str, default="64,128", help="Structure of PointNet Conv local MLP (comma-separated).")
    parser.add_argument("--point_net_conv_layer_structure_global_mlp", type=str, default="256,64", help="Structure of PointNet Conv global MLP (comma-separated).")
    parser.add_argument("--gat_conv_layer_structure", type=str, default="128,256,256,128", help="Structure of GAT Conv hidden layer sizes (comma-separated).")
    parser.add_argument("--num_epochs", type=int, default=3000, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="The learning rate for the model.")
    parser.add_argument("--early_stopping_patience", type=int, default=100, help="The early stopping patience.")
    parser.add_argument("--use_dropout", type=hf.str_to_bool, default=False, help="Whether to use dropout.")
    parser.add_argument("--dropout", type=float, default=0.3, help="The dropout rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3, help="After how many steps the gradient should be updated.")
    parser.add_argument("--device_nr", type=int, default=0, help="The device number (0 or 1 for Retina Roaster's two GPUs).")

    args = parser.parse_args()
    hf.set_random_seeds()
    
    try:
        gpus = hf.get_available_gpus()
        best_gpu = hf.select_best_gpu(gpus)
        hf.set_cuda_visible_device(best_gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params = get_parameters(args)
        
        # Create base directory for the run
        base_dir = '../../data/' + params['project_name'] + '/'
        unique_run_dir = os.path.join(base_dir, params['unique_model_description'])
        os.makedirs(unique_run_dir, exist_ok=True)
        
        model_save_path, path_to_save_dataloader = hf.get_paths(base_dir=base_dir, unique_model_description= params['unique_model_description'], model_save_path= 'trained_model/model.pth')
        train_dl, valid_dl, scalers_train, scalers_validation = hf.prepare_data_with_graph_features(datalist=datalist, batch_size= params['batch_size'], path_to_save_dataloader= path_to_save_dataloader)
        
        config = hf.setup_wandb(params['project_name'], {param: params[param] for param in PARAMETER_ORDER})

        gnn_instance = garch.MyGnn(in_channels=config.in_channels, 
                        out_channels=config.out_channels, 
                        point_net_conv_layer_structure_local_mlp=config.point_net_conv_layer_structure_local_mlp,
                        point_net_conv_layer_structure_global_mlp=config.point_net_conv_layer_structure_global_mlp,
                        gat_conv_layer_structure=config.gat_conv_layer_structure,
                        dropout=config.dropout, 
                        use_dropout=config.use_dropout, 
                        predict_mode_stats=config.predict_mode_stats)
        
        print(gnn_instance)
        model = gnn_instance.to(device)
        loss_fct = torch.nn.MSELoss()
        
        baseline_loss_mean_target = gio.compute_baseline_of_mean_target(dataset=train_dl, loss_fct=loss_fct)
        baseline_loss = gio.compute_baseline_of_no_policies(dataset=train_dl, loss_fct=loss_fct)
        print("baseline loss mean " + str(baseline_loss_mean_target))
        print("baseline loss no  " +str(baseline_loss) )

        early_stopping = gio.EarlyStopping(patience=params['early_stopping_patience'], verbose=True)
        best_val_loss, best_epoch = garch.train(model=model, 
                    config=config, 
                    loss_fct=loss_fct,
                    optimizer=torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4),
                    train_dl=train_dl, 
                    valid_dl=valid_dl,
                    device=device, 
                    early_stopping=early_stopping,
                    model_save_path=model_save_path,
                    use_gradient_clipping=True,
                    lr_scheduler_warmup_steps=30000,
                    lr_scheduler_cosine_decay_rate=0.01)
        print(f'Best model saved to {model_save_path} with validation loss: {best_val_loss} at epoch {best_epoch}')   
        
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
     
if __name__ == '__main__':
    main()