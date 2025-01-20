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

scripts_path = os.path.abspath(os.path.join('..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

import gnn_io as gio
from data_preprocessing.process_simulations_for_gnn import EdgeFeatures

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
        
        
def get_paths(base_dir: str, unique_model_description: str, model_save_path: str = 'trained_model/model.pth'):
    data_path = os.path.join(base_dir, unique_model_description)
    os.makedirs(data_path, exist_ok=True)
    model_save_to = os.path.join(data_path, model_save_path)
    path_to_save_dataloader = os.path.join(data_path, 'data_created_during_training/')
    os.makedirs(os.path.dirname(model_save_to), exist_ok=True)
    os.makedirs(path_to_save_dataloader, exist_ok=True)
    return model_save_to, path_to_save_dataloader

def get_memory_info():
    """Get memory information using psutil."""
    import psutil
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # Convert to GB
    used_memory = total_memory - available_memory
    return total_memory, available_memory, used_memory

def prepare_data_with_graph_features(datalist, batch_size, path_to_save_dataloader, use_all_features):
    print(f"Starting prepare_data_with_graph_features with {len(datalist)} items")
    
    try:

        print("Splitting into subsets...")
        train_set, valid_set, test_set = gio.split_into_subsets(dataset=datalist, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05)
        print(f"Split complete. Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")
        
        print("Saving test set...")
        test_set_path = os.path.join(path_to_save_dataloader, 'test_set.pt')
        torch.save(test_set, test_set_path)
        print(f"Test set saved to {test_set_path}")

        node_features = [feat.name for feat in EdgeFeatures] if use_all_features else ["VOL_BASE_CASE",
                                                                                       "CAPACITY_BASE_CASE",
                                                                                       "CAPACITY_REDUCTION",
                                                                                       "FREESPEED",
                                                                                       "LENGTH"]        
        
        print("Normalizing train set...")
        train_set_normalized, scalers_train = normalize_dataset(dataset_input=train_set, node_features=node_features, directory_path=path_to_save_dataloader + "train_")
        print("Train set normalized")      
        
        print("Normalizing validation set...")
        valid_set_normalized, scalers_validation = normalize_dataset(dataset_input=valid_set, node_features=node_features, directory_path=path_to_save_dataloader + "valid_")
        print("Validation set normalized")
        
        print("Normalizing test set...")
        test_set_normalized, scalers_test = normalize_dataset(dataset_input=test_set, node_features=node_features, directory_path=path_to_save_dataloader + "test_")
        print("Test set normalized")
        
        print("Creating train loader...")
        train_loader = DataLoader(dataset=train_set_normalized, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True, collate_fn=gio.collate_fn, worker_init_fn=seed_worker)
        print("Train loader created")
        
        print("Creating validation loader...")
        val_loader = DataLoader(dataset=valid_set_normalized, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=gio.collate_fn, worker_init_fn=seed_worker)
        print("Validation loader created")
        
        print("Creating test loader...")
        test_loader = DataLoader(dataset=test_set_normalized, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=gio.collate_fn, worker_init_fn=seed_worker)
        print("Test loader created")
        
        joblib.dump(scalers_train['x_scaler'], os.path.join(path_to_save_dataloader, 'train_x_scaler.pkl'))
        joblib.dump(scalers_train['pos_scaler'], os.path.join(path_to_save_dataloader, 'train_pos_scaler.pkl'))
        # joblib.dump(scalers_train['modestats_scaler'], os.path.join(path_to_save_dataloader, 'train_mode_stats_scaler.pkl'))

        joblib.dump(scalers_validation['x_scaler'], os.path.join(path_to_save_dataloader, 'validation_x_scaler.pkl'))
        joblib.dump(scalers_validation['pos_scaler'], os.path.join(path_to_save_dataloader, 'validation_pos_scaler.pkl'))
        # joblib.dump(scalers_validation['modestats_scaler'], os.path.join(path_to_save_dataloader, 'validation_mode_stats_scaler.pkl'))

        joblib.dump(scalers_test['x_scaler'], os.path.join(path_to_save_dataloader, 'test_x_scaler.pkl'))
        joblib.dump(scalers_test['pos_scaler'], os.path.join(path_to_save_dataloader, 'test_pos_scaler.pkl'))
        # joblib.dump(scalers_test['modestats_scaler'], os.path.join(path_to_save_dataloader, 'test_mode_stats_scaler.pkl'))  
        
        gio.save_dataloader(test_loader, path_to_save_dataloader + 'test_dl.pt')
        gio.save_dataloader_params(test_loader, path_to_save_dataloader + 'test_loader_params.json')
        print("Dataloaders and scalers saved")
        
        return train_loader, val_loader
    
    except Exception as e:
        print(f"Error in prepare_data_with_graph_features: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
        
def normalize_dataset(dataset_input, node_features, directory_path):
    data_list = [dataset_input.dataset[idx] for idx in dataset_input.indices]

    print("Fitting and normalizing x features...")
    normalized_data_list, x_scaler = normalize_x_features_batched(data_list, node_features)
    print("x features normalized")
    
    print("Fitting and normalizing pos features...")
    normalized_data_list, pos_scaler = normalize_pos_features_batched(normalized_data_list)
    print("Pos features normalized")
    
    # print("Fitting and normalizing modestats features...")
    # normalized_data_list, modestats_scaler = normalize_modestats_features_batched(normalized_data_list)
    # print("Modestats features normalized")
    
    scalers_dict = {
        "x_scaler": x_scaler,
        "pos_scaler": pos_scaler,
        # "modestats_scaler": modestats_scaler
    }
    return normalized_data_list, scalers_dict

def normalize_x_features_batched(data_list, node_features, batch_size=100):
    """
    Normalize the continuous node features (0 mean and unit variance).
    Categorical features (Allowed Modes) are left as booleans (0 or 1).
    'HIGHWAY' feature is one-hot encoded.

    Finally, features are filtered to only include the ones specified in node_features. 
    """
    scaler = StandardScaler()

    # VOL_BASE_CASE, CAPACITY_BASE_CASE, CAPACITIES_NEW, CAPACITY_REDUCTION, FREESPEED, LENGTH
    continuous_feat = [0, 1, 2, 3, 4, 6]
    
    # First pass: Fit the scaler
    for i in tqdm(range(0, len(data_list), batch_size), desc="Fitting scaler"):
        batch = data_list[i:i+batch_size]
        batch_x = np.vstack([data.x[:,continuous_feat].numpy() for data in batch])
        scaler.partial_fit(batch_x)
    
    # Second pass: Transform the data
    for i in tqdm(range(0, len(data_list), batch_size), desc="Normalizing x features"):
        batch = data_list[i:i+batch_size]
        batch_x = np.vstack([data.x[:,continuous_feat].numpy() for data in batch])
        batch_x_normalized = scaler.transform(batch_x)
        for j, data in enumerate(batch):
            data.x[:,continuous_feat] = torch.tensor(batch_x_normalized[j*31140:(j+1)*31140], dtype=data.x.dtype)

    # Filter features
    node_feature_filter = [EdgeFeatures[feature].value for feature in node_features]
    for data in data_list:
        data.x = data.x[:, node_feature_filter]

    # One-hot encode highway
    if "HIGHWAY" in node_features:
        one_hot_highway(data_list, idx=node_features.index("HIGHWAY"))
    
    return data_list, scaler

def normalize_pos_features_batched(data_list, batch_size=1000):
    scaler = StandardScaler()
    
    # First pass: Fit the scaler
    for i in tqdm(range(0, len(data_list), batch_size), desc="Fitting scaler"):
        batch = data_list[i:i+batch_size]
        batch_pos = np.vstack([data.pos.numpy().reshape(-1, 6) for data in batch])
        scaler.partial_fit(batch_pos)
    
    # Second pass: Transform the data
    for i in tqdm(range(0, len(data_list), batch_size), desc="Normalizing pos features"):
        batch = data_list[i:i+batch_size]
        for data in batch:
            pos_reshaped = data.pos.numpy().reshape(-1, 6)
            pos_normalized = scaler.transform(pos_reshaped)
            data.pos = torch.tensor(pos_normalized.reshape(31140, 3, 2), dtype=data.pos.dtype)
    
    return data_list, scaler

# def normalize_modestats_features_batched(data_list, batch_size=1000):
#     scaler = StandardScaler()
    
#     # First pass: Fit the scaler
#     for i in tqdm(range(0, len(data_list), batch_size), desc="Fitting scaler"):
#         batch = data_list[i:i+batch_size]
#         batch_modestats = np.vstack([data.mode_stats.numpy().reshape(1, -1) for data in batch])
#         scaler.partial_fit(batch_modestats)
    
#     # Second pass: Transform the data
#     for i in tqdm(range(0, len(data_list), batch_size), desc="Normalizing modestats features"):
#         batch = data_list[i:i+batch_size]
#         for data in batch:
#             modestats_reshaped = data.mode_stats.numpy().reshape(1, -1)
#             modestats_normalized = scaler.transform(modestats_reshaped)
#             data.mode_stats = torch.tensor(modestats_normalized.reshape(6, 2), dtype=torch.float32)
    
#     return data_list, scaler


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_wandb(config):
    wandb.login()
    wandb.init(project=config['project_name'], name=config['unique_model_description'], config=config)
    return wandb.config

def str_to_bool(value):
    if isinstance(value, str):
        if value.lower() in ['true', '1', 'yes', 'y']:
            return True
        elif value.lower() in ['false', '0', 'no', 'n']:
            return False
    raise ValueError(f"Cannot convert {value} to a boolean.")

# Define the paths here
# def get_paths(base_dir: str, unique_model_description: str, model_save_path: str = 'trained_model/model.pth', dataset_path: str = '../../data/train_data/dataset_1pm_0-5000.pt'):
#     data_path = os.path.join(base_dir, unique_model_description)
#     os.makedirs(data_path, exist_ok=True)
#     model_save_to = os.path.join(data_path, model_save_path)
#     path_to_save_dataloader = os.path.join(data_path, 'data_created_during_training/')
#     os.makedirs(os.path.dirname(model_save_to), exist_ok=True)
#     os.makedirs(path_to_save_dataloader, exist_ok=True)
#     data_dict_list = torch.load(dataset_path)
#     # data_dict_list = torch.load('../../data/train_data/dataset_1pm_0-5000.pt')
#     return data_dict_list, model_save_to, path_to_save_dataloader

# def prepare_data(data_dict_list, indices_of_datasets_to_use, batch_size, path_to_save_dataloader, normalize_y, normalize_pos):
#     datalist = [Data(x=d['x'], edge_index=d['edge_index'], pos=d['pos'], y=d['y']) for d in data_dict_list]
#     dataset_only_relevant_dimensions = gio.cut_dimensions(dataset=datalist, indices_of_dimensions_to_keep=indices_of_datasets_to_use)
#     train_set, valid_set, test_set = gio.split_into_subsets(dataset=dataset_only_relevant_dimensions, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05)
#     if normalize_y and normalize_pos:
#         train_set_normalized, x_scaler, pos_scaler, y_scaler = gio.normalize_dataset_create_scaler(dataset_input = train_set, directory_path=path_to_save_dataloader, normalize_y=True, normalize_pos=True)
#         valid_set_normalized = gio.normalize_dataset_with_given_scaler(dataset_input=valid_set, x_scalar_list=x_scaler, pos_scalar=pos_scaler, y_scalar=y_scaler, normalize_y=True, normalize_pos=True)
#         test_set_normalized =  gio.normalize_dataset_with_given_scaler(dataset_input=test_set, x_scalar_list=x_scaler, pos_scalar=pos_scaler, y_scalar=y_scaler, normalize_y=True, normalize_pos=True) 
#     if normalize_y and not normalize_pos:
#         train_set_normalized, x_scaler, y_scaler = gio.normalize_dataset_create_scaler(dataset_input = train_set, directory_path=path_to_save_dataloader, normalize_y=True, normalize_pos=False)
#         valid_set_normalized = gio.normalize_dataset_with_given_scaler(dataset_input=valid_set, x_scalar_list=x_scaler, pos_scalar=None, y_scalar=y_scaler, normalize_y=True, normalize_pos=False)
#         test_set_normalized =  gio.normalize_dataset_with_given_scaler(dataset_input=test_set, x_scalar_list=x_scaler, pos_scalar=None, y_scalar=y_scaler, normalize_y=True, normalize_pos=False) 
#     if not normalize_y and normalize_pos:
#         train_set_normalized, x_scaler, pos_scaler = gio.normalize_dataset_create_scaler(dataset_input = train_set, directory_path=path_to_save_dataloader, normalize_y=False, normalize_pos=True)
#         valid_set_normalized = gio.normalize_dataset_with_given_scaler(dataset_input=valid_set, x_scalar_list=x_scaler, pos_scalar=pos_scaler, y_scalar= None,normalize_y=False, normalize_pos=True)
#         test_set_normalized =  gio.normalize_dataset_with_given_scaler(dataset_input=test_set, x_scalar_list=x_scaler, pos_scalar=pos_scaler,y_scalar=None, normalize_y=False, normalize_pos=True)
#     if not normalize_y and not normalize_pos:
#         train_set_normalized, x_scaler = gio.normalize_dataset_create_scaler(dataset_input = train_set, directory_path=path_to_save_dataloader, normalize_y=False, normalize_pos=False)
#         valid_set_normalized = gio.normalize_dataset_with_given_scaler(dataset_input=valid_set, x_scalar_list=x_scaler, pos_scalar=None, y_scalar= None, normalize_y=False, normalize_pos=False)
#         test_set_normalized =  gio.normalize_dataset_with_given_scaler(dataset_input=test_set, x_scalar_list=x_scaler, pos_scalar=None, y_scalar=None, normalize_y=False, normalize_pos=False)
        
#     train_loader = DataLoader(dataset=train_set_normalized, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True, collate_fn=gio.collate_fn, worker_init_fn=seed_worker)
#     val_loader = DataLoader(dataset=valid_set_normalized, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=gio.collate_fn, worker_init_fn=seed_worker)
#     test_loader = DataLoader(dataset=test_set_normalized, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=gio.collate_fn, worker_init_fn=seed_worker)
#     gio.save_dataloader(test_loader, path_to_save_dataloader + 'test_dl.pt')
#     gio.save_dataloader_params(test_loader, path_to_save_dataloader + 'test_loader_params.json')
#     return train_loader, val_loader

def one_hot_highway(datalist, idx):

    """
    One-hot encodes the 'HIGHWAY' feature and removes the original one.
    Cluster into 6 major classes to reduce dimensionality. (defined with n_types and mapping, originaly 10 classes)
    """
    
    n_types = 6
    mapping = {
        -1: 4, # pt
        0: 5, # other
        1: 0, # primary
        2: 1, # secondary
        3: 2, # tertiary
        4: 3, # residential
        5: 5,
        6: 5,
        7: 5,
        8: 5,
        9: 5
    }

    for data in datalist:
        
        highway = data.x[:, idx].numpy()
        mapped_highway = np.vectorize(mapping.get)(highway)
        one_hot = np.eye(n_types)[mapped_highway]

        data.x = torch.cat((data.x[:, :idx], torch.tensor(one_hot, dtype=data.x.dtype), data.x[:, idx+1:]), dim=1)