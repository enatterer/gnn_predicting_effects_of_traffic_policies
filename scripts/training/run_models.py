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
    model_save_path = os.path.join(data_path, 'trained_model/model.pth')
    path_to_save_dataloader = os.path.join(data_path, 'data_created_during_training/')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(path_to_save_dataloader, exist_ok=True)
    data_dict_list = torch.load('../../data/train_data/dataset_1pm_0-5000.pt')
    return data_dict_list, model_save_path, path_to_save_dataloader

def set_random_seeds(seed_value=42):
    # Set environment variable for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    # Set Python built-in random module seed
    random.seed(seed_value)
    
    # Set NumPy random seed
    np.random.seed(seed_value)
    
    # Set PyTorch random seed for CPU
    torch.manual_seed(seed_value)
    
    # Set PyTorch random seed for all GPUs (if available)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # If using multi-GPU
    
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # If using torch.distributed for distributed training, set the seed
    if torch.distributed.is_initialized():
        torch.distributed.manual_seed_all(seed_value)

def prepare_data(data_dict_list, indices_of_datasets_to_use, batch_size, path_to_save_dataloader, normalize_y, normalize_pos):
    datalist = [Data(x=d['x'], edge_index=d['edge_index'], pos=d['pos'], y=d['y']) for d in data_dict_list]
    dataset_only_relevant_dimensions = gio.cut_dimensions(dataset=datalist, indices_of_dimensions_to_keep=indices_of_datasets_to_use)
    train_set, valid_set, test_set = gio.split_into_subsets(dataset=dataset_only_relevant_dimensions, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05)
    if normalize_y and normalize_pos:
        train_set_normalized, x_scaler, pos_scaler, y_scaler = gio.normalize_dataset_create_scaler(dataset_input = train_set, directory_path=path_to_save_dataloader, normalize_y=True, normalize_pos=True)
        valid_set_normalized = gio.normalize_dataset_with_given_scaler(dataset_input=valid_set, x_scalar_list=x_scaler, pos_scalar=pos_scaler, y_scalar=y_scaler, normalize_y=True, normalize_pos=True)
        test_set_normalized =  gio.normalize_dataset_with_given_scaler(dataset_input=test_set, x_scalar_list=x_scaler, pos_scalar=pos_scaler, y_scalar=y_scaler, normalize_y=True, normalize_pos=True) 
    if normalize_y and not normalize_pos:
        train_set_normalized, x_scaler, y_scaler = gio.normalize_dataset_create_scaler(dataset_input = train_set, directory_path=path_to_save_dataloader, normalize_y=True, normalize_pos=False)
        valid_set_normalized = gio.normalize_dataset_with_given_scaler(dataset_input=valid_set, x_scalar_list=x_scaler, pos_scalar=None, y_scalar=y_scaler, normalize_y=True, normalize_pos=False)
        test_set_normalized =  gio.normalize_dataset_with_given_scaler(dataset_input=test_set, x_scalar_list=x_scaler, pos_scalar=None, y_scalar=y_scaler, normalize_y=True, normalize_pos=False) 
    if not normalize_y and normalize_pos:
        train_set_normalized, x_scaler, pos_scaler = gio.normalize_dataset_create_scaler(dataset_input = train_set, directory_path=path_to_save_dataloader, normalize_y=False, normalize_pos=True)
        valid_set_normalized = gio.normalize_dataset_with_given_scaler(dataset_input=valid_set, x_scalar_list=x_scaler, pos_scalar=pos_scaler, y_scalar= None,normalize_y=False, normalize_pos=True)
        test_set_normalized =  gio.normalize_dataset_with_given_scaler(dataset_input=test_set, x_scalar_list=x_scaler, pos_scalar=pos_scaler,y_scalar=None, normalize_y=False, normalize_pos=True)
    if not normalize_y and not normalize_pos:
        train_set_normalized, x_scaler = gio.normalize_dataset_create_scaler(dataset_input = train_set, directory_path=path_to_save_dataloader, normalize_y=False, normalize_pos=False)
        valid_set_normalized = gio.normalize_dataset_with_given_scaler(dataset_input=valid_set, x_scalar_list=x_scaler, pos_scalar=None, y_scalar= None, normalize_y=False, normalize_pos=False)
        test_set_normalized =  gio.normalize_dataset_with_given_scaler(dataset_input=test_set, x_scalar_list=x_scaler, pos_scalar=None, y_scalar=None, normalize_y=False, normalize_pos=False)
        
    train_loader = DataLoader(dataset=train_set_normalized, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True, collate_fn=gio.collate_fn, worker_init_fn=seed_worker)
    val_loader = DataLoader(dataset=valid_set_normalized, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=gio.collate_fn, worker_init_fn=seed_worker)
    test_loader = DataLoader(dataset=test_set_normalized, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=gio.collate_fn, worker_init_fn=seed_worker)
    gio.save_dataloader(test_loader, path_to_save_dataloader + 'test_dl.pt')
    gio.save_dataloader_params(test_loader, path_to_save_dataloader + 'test_loader_params.json')
    return train_loader, val_loader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_wandb(project_name, config):
    wandb.login()
    wandb.init(project=project_name, config=config)
    return wandb.config

def str_to_bool(value):
    if isinstance(value, str):
        if value.lower() in ['true', '1', 'yes', 'y']:
            return True
        elif value.lower() in ['false', '0', 'no', 'n']:
            return False
    raise ValueError(f"Cannot convert {value} to a boolean.")

# Define parameters
def get_parameters(args):
        project_name = "runs_optimized"
        indices_of_datasets_to_use = [0, 1, 3, 4]
        num_epochs = 1000
        in_channels = len(indices_of_datasets_to_use) + 2
        out_channels = 1
        lr = float(args.lr)
        batch_size = int(args.batch_size)
        point_net_conv_layer_structure_local_mlp = [int(x) for x in args.pnc_local.split(',')]
        point_net_conv_layer_structure_global_mlp = [int(x) for x in args.pnc_global.split(',')]
        gat_conv_layer_structure = [int(x) for x in args.gat_conv_layer_structure.split(',')]
        gradient_accumulation_steps = int(args.gradient_accumulation_steps)
        early_stopping_patience = int(args.early_stopping_patience)
        dropout = float(args.dropout)
        use_dropout = str_to_bool(args.use_dropout)

        unique_model_description = (
            # f"features_{gio.int_list_to_string(lst = indices_of_datasets_to_use, delimiter= '_')}_"
            # f"batch_{batch_size}_"
            f"pnc_local_{gio.int_list_to_string(lst = point_net_conv_layer_structure_local_mlp, delimiter='_')}_"
            f"pnc_global_{gio.int_list_to_string(lst = point_net_conv_layer_structure_global_mlp, delimiter='_')}_"
            f"hidden_layer_str_{gio.int_list_to_string(lst = gat_conv_layer_structure, delimiter='_')}_"
            f"dropout_{dropout}_"
            f"use_dropout_{use_dropout}"
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
            "point_net_conv_layer_structure_local_mlp": point_net_conv_layer_structure_local_mlp,
            "point_net_conv_layer_structure_global_mlp": point_net_conv_layer_structure_global_mlp,
            "gat_conv_layer_structure": gat_conv_layer_structure,
            "lr": lr,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "early_stopping_patience": early_stopping_patience,
            "unique_model_description": unique_model_description,
            "dropout": dropout,
            "use_dropout": use_dropout
        } 

def main():
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Run GNN model training with configurable parameters.")
    parser.add_argument("--pnc_local", type=str, default="[64]", help="Size of hidden layers.")
    parser.add_argument("--pnc_global", type=str, default="[64]", help="Size of hidden layers.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--gat_conv_layer_structure", type=str, default="[64]", help="Structure of GAT Conv hidden layer sizes (comma-separated).")
    parser.add_argument("--early_stopping_patience", type=str, default=20, help="The early stopping patience.")
    parser.add_argument("--gradient_accumulation_steps", type=str, default=3, help="After how many steps the gradient should be updated.")
    parser.add_argument("--lr", type=str, default=0.001, help="The learning rate for the model.")
    parser.add_argument("--device_nr", type=str, default="1", help="The device that this model should run for. The Retina Roaster has two GPUs, so the values 0 and 1 are allowed here.")
    parser.add_argument("--dropout", type=str, default=0.3, help="The dropout rate.")
    parser.add_argument("--use_dropout", type=str, default="false", help="Whether to use or not use dropout.")
    args = parser.parse_args()
    
    set_random_seeds()
    
    try:
        gpus = get_available_gpus()
        best_gpu = select_best_gpu(gpus)
        set_cuda_visible_device(best_gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params = get_parameters(args=args)
        
        # Create base directory for the run
        base_dir = '../../data/' + params['project_name'] + '/'
        unique_run_dir = os.path.join(base_dir, params['unique_model_description'])
        os.makedirs(unique_run_dir, exist_ok=True)
        
        data_dict_list, model_save_path, path_to_save_dataloader = get_paths(base_dir, params['unique_model_description'])
        train_dl, valid_dl = prepare_data(data_dict_list=data_dict_list, indices_of_datasets_to_use=params['indices_of_datasets_to_use'], batch_size= params['batch_size'], path_to_save_dataloader= path_to_save_dataloader, normalize_y=False, normalize_pos=True)
        
        config = setup_wandb(params['project_name'], {
            "epochs": params['num_epochs'],
            "batch_size": params['batch_size'],
            "lr": params['lr'],
            "gradient_accumulation_steps": params['gradient_accumulation_steps'],
            "early_stopping_patience": params['early_stopping_patience'],
            "point_net_conv_local_mlp": params['point_net_conv_layer_structure_local_mlp'],
            "point_net_conv_global_mlp": params['point_net_conv_layer_structure_global_mlp'],
            "gat_conv_layer_structure": params['gat_conv_layer_structure'],
            "indices_to_use": params['indices_of_datasets_to_use'],
            "in_channels": params['in_channels'],
            "out_channels": params['out_channels'],
            "dropout": params['dropout'],
            "use_dropout": params['use_dropout']
        })

        gnn_instance = garch.MyGnn(in_channels=config.in_channels, out_channels=config.out_channels, point_net_conv_layer_structure_local_mlp=config.point_net_conv_local_mlp,
                                   point_net_conv_layer_structure_global_mlp=config.point_net_conv_global_mlp,
                                   gat_conv_layer_structure=config.gat_conv_layer_structure,
                                   dropout=config.dropout, use_dropout=config.use_dropout)
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
                    accumulation_steps=config.gradient_accumulation_steps,
                    compute_r_squared=True,
                    model_save_path=model_save_path,
                    use_gradient_clipping=True,
                    lr_scheduler_warmup_steps=20000,
                    lr_scheduler_cosine_decay_rate=0.5)
        print(f'Best model saved to {model_save_path} with validation loss: {best_val_loss} at epoch {best_epoch}')   
        
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
     
if __name__ == '__main__':
    main()