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

# Define the paths here
def get_paths(base_dir: str, unique_model_description: str, model_save_path: str = 'trained_model/model.pth', dataset_path: str = '../../data/train_data/dataset_1pm_0-5000.pt'):
    data_path = os.path.join(base_dir, unique_model_description)
    os.makedirs(data_path, exist_ok=True)
    model_save_to = os.path.join(data_path, model_save_path)
    path_to_save_dataloader = os.path.join(data_path, 'data_created_during_training/')
    os.makedirs(os.path.dirname(model_save_to), exist_ok=True)
    os.makedirs(path_to_save_dataloader, exist_ok=True)
    data_dict_list = torch.load(dataset_path)
    # data_dict_list = torch.load('../../data/train_data/dataset_1pm_0-5000.pt')
    return data_dict_list, model_save_to, path_to_save_dataloader