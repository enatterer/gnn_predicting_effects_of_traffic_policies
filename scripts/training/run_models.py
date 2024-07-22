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

# Add the 'scripts' directory to the Python path
scripts_path = os.path.abspath(os.path.join('..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)
    
import gnn_io as gio
import gnn_architectures as garch
    
def get_available_gpus():
    command = "nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"Error executing nvidia-smi: {result.stderr.decode('utf-8')}")
    gpu_info = result.stdout.decode('utf-8').strip().split('\n')
    gpus = []
    for info in gpu_info:
        index, utilization, memory_free = info.split(', ')
        gpus.append({
            'index': int(index),
            'utilization': int(utilization),
            'memory_free': int(memory_free)
        })
    return gpus
    
def select_best_gpu(gpus):
    # Sort by free memory (descending) and then by utilization (ascending)
    gpus = sorted(gpus, key=lambda x: (-x['memory_free'], x['utilization']))
    return gpus[0]['index']

def set_cuda_visible_device(gpu_index):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    print(f"Using GPU {gpu_index} with CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

# Define the paths here
def get_paths(base_dir, unique_model_description):
    data_path = os.path.join(base_dir, unique_model_description)
    os.makedirs(data_path, exist_ok=True)
    model_save_path = os.path.join(data_path, 'trained_models/model.pth')
    path_to_save_dataloader = os.path.join(data_path, 'data_created_during_training/')
    config_save_path = os.path.join(data_path, 'trained_models/config.json')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(path_to_save_dataloader, exist_ok=True)
    data_dict_list = torch.load('../../data/train_data/dataset_1pm_0-4400.pt')
    return data_dict_list, model_save_path, config_save_path, path_to_save_dataloader

# Define parameters
def get_parameters(args):
        project_name = "experimentation"
        indices_of_datasets_to_use = [0, 1, 3, 4]
        num_epochs = 1000
        in_channels = len(indices_of_datasets_to_use) + 2
        out_channels = 1
        lr = float(args.lr)
        batch_size = int(args.batch_size)
        hidden_layers_base_for_point_net_conv = int(args.hidden_layers_base_for_point_net_conv)
        hidden_layer_structure = [int(x) for x in args.hidden_layer_structure.split(',')]
        gradient_accumulation_steps = int(args.gradient_accumulation_steps)
        early_stopping_patience = int(args.early_stopping_patience)

        unique_model_description = (
            # f"features_{gio.int_list_to_string(lst = indices_of_datasets_to_use, delimiter= '_')}_"
            # f"batch_{batch_size}_"
            f"hidden_{hidden_layers_base_for_point_net_conv}_"
            f"hidden_layer_str_{gio.int_list_to_string(lst = hidden_layer_structure, delimiter='_')}_"
            # f"gat_and_conv_structure_{gio.int_list_to_string(lst = gat_and_conv_structure, delimiter='_')}"
            # f"lr_{lr}_"
            # f"g_accumulation_steps_{gradient_accumulation_steps}"
            # f"early_stopping_{early_stopping_patience}"
            # f"in_channels_{in_channels}_"
            # f"out_channels_{out_channels}_"
        )
        return {
            "project_name": project_name,
            "indices_of_datasets_to_use": indices_of_datasets_to_use,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "hidden_layers_base_for_point_net_conv": hidden_layers_base_for_point_net_conv,
            "hidden_layer_structure": hidden_layer_structure,
            "lr": lr,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "early_stopping_patience": early_stopping_patience,
            "unique_model_description": unique_model_description
        }
        
def set_random_seeds():
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

def prepare_data(data_dict_list, indices_of_datasets_to_use, path_to_save_dataloader):
    datalist = [Data(x=d['x'], edge_index=d['edge_index'], pos=d['pos'], y=d['y']) for d in data_dict_list]
    dataset_only_relevant_dimensions = gio.cut_dimensions(dataset=datalist, indices_of_dimensions_to_keep=indices_of_datasets_to_use)
    dataset_normalized = gio.normalize_dataset(dataset_only_relevant_dimensions, y_scalar=None, x_scalar_list=None, pos_scalar=None, directory_path=path_to_save_dataloader)
    return dataset_normalized

def create_dataloaders_and_save_test_set(dataset_normalized, batch_size, path_to_save_dataloader):
    train_dl, valid_dl, test_dl = gio.create_dataloaders(batch_size=batch_size, dataset=dataset_normalized, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    gio.save_dataloader(test_dl, path_to_save_dataloader + 'test_dl.pt')
    gio.save_dataloader_params(test_dl, path_to_save_dataloader + 'test_loader_params.json')
    return train_dl, valid_dl

def setup_wandb(project_name, config):
    wandb.login()
    wandb.init(project=project_name, config=config)
    return wandb.config
        
def train_model(config, train_dl, valid_dl, device, early_stopping, checkpoint_dir, model_save_path):
    gnn_instance = garch.MyGnn(in_channels=config.in_channels, out_channels=config.out_channels, hidden_layers_base_for_point_net_conv=config.hidden_layers_base_for_point_net_conv, hidden_layer_structure=config.hidden_layer_structure)
    model = gnn_instance.to(device)
    loss_fct = torch.nn.MSELoss()
    best_val_loss, best_epoch = garch.train(model=model, 
                config=config, 
                loss_fct=loss_fct,
                optimizer=torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0),
                train_dl=train_dl, 
                valid_dl=valid_dl,
                device=device, 
                early_stopping=early_stopping,
                accumulation_steps=config.gradient_accumulation_steps,
                use_existing_checkpoint=False, 
                path_existing_checkpoints=checkpoint_dir,
                compute_r_squared=False,
                model_save_path=model_save_path)
    print(f'Best model saved to {model_save_path} with validation loss: {best_val_loss} at epoch {best_epoch}')   

def main():
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Run GNN model training with configurable parameters.")
    parser.add_argument("--hidden_layers_base_for_point_net_conv", type=int, default=64, help="Size of hidden layers.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--hidden_layer_structure", type=str, default="1,-1,0,1", help="Structure of hidden layer sizes (comma-separated).")
    parser.add_argument("--early_stopping_patience", type=str, default=20, help="The early stopping patience.")
    parser.add_argument("--gradient_accumulation_steps", type=str, default=3, help="After how many steps the gradient should be updated.")
    parser.add_argument("--lr", type=str, default=0.001, help="The learning rate for the model.")
    parser.add_argument("--device_nr", type=str, default="1", help="The device that this model should run for. The Retina Roaster has two GPUs, so the values 0 and 1 are allowed here.")
    
    args = parser.parse_args()
    
    set_random_seeds()
    
    try:
        gpus = get_available_gpus()
        best_gpu = select_best_gpu(gpus)
        set_cuda_visible_device(best_gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params = get_parameters(args=args)
        
        # Create base directory for the run
        base_dir = '../../data/runs_experiments/'
        unique_run_dir = os.path.join(base_dir, params['unique_model_description'])
        os.makedirs(unique_run_dir, exist_ok=True)
        
        data_dict_list, model_save_path, config_save_path, path_to_save_dataloader = get_paths(base_dir, params['unique_model_description'])
        dataset_normalized = prepare_data(data_dict_list, params['indices_of_datasets_to_use'], path_to_save_dataloader)
        train_dl, valid_dl = create_dataloaders_and_save_test_set(dataset_normalized, params['batch_size'], path_to_save_dataloader)
        
        config = setup_wandb(params['project_name'], {
            "epochs": params['num_epochs'],
            "batch_size": params['batch_size'],
            "lr": params['lr'],
            "gradient_accumulation_steps": params['gradient_accumulation_steps'],
            "early_stopping_patience": params['early_stopping_patience'],
            "hidden_layers_base_for_point_net_conv": params['hidden_layers_base_for_point_net_conv'],
            "hidden_layer_structure": params['hidden_layer_structure'],
            "indices_to_use": params['indices_of_datasets_to_use'],
            "dataset_length": len(dataset_normalized), 
            "in_channels": params['in_channels'],
            "out_channels": params['out_channels'],
        })
        with open(config_save_path, 'w') as f:
            json.dump(config, f)
        
        gnn_instance = garch.MyGnn(in_channels=config.in_channels, out_channels=config.out_channels, hidden_layers_base_for_point_net_conv=config.hidden_layers_base_for_point_net_conv, hidden_layer_structure=config.hidden_layer_structure)
        model = gnn_instance.to(device)
        loss_fct = torch.nn.MSELoss()
        early_stopping = gio.EarlyStopping(patience=params['early_stopping_patience'], verbose=True)
        best_val_loss, best_epoch = garch.train(model=model, 
                    config=config, 
                    loss_fct=loss_fct,
                    optimizer=torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0),
                    train_dl=train_dl, 
                    valid_dl=valid_dl,
                    device=device, 
                    early_stopping=early_stopping,
                    accumulation_steps=config.gradient_accumulation_steps,
                    compute_r_squared=False,
                    model_save_path=model_save_path)
        print(f'Best model saved to {model_save_path} with validation loss: {best_val_loss} at epoch {best_epoch}')   
        
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
     
if __name__ == '__main__':
    main()