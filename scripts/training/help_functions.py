import os
import sys
import copy
import random
import json
import joblib
import subprocess
from pathlib import Path
from functools import partial

import wandb
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.gnn_io import *
from gnn.models.gatv2 import GATv2
from gnn.models.trans_conv import TransConv
from gnn.models.graphSAGE import GraphSAGE
from gnn.models.trans_encoder import TransEncoder
from data_preprocessing.process_simulations_for_gnn import EdgeFeatures

########## Control Center #########
use_allowed_modes = False
use_destination_activity = False
use_highway = False
###################################


################################################# ↓ GPU + Randomness ↓ #################################################

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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


################################################# ↓ Misc Setup Helpers ↓ #################################################

def str_to_bool(value):
    if isinstance(value, str):
        if value.lower() in ['true', '1', 'yes', 'y']:
            return True
        elif value.lower() in ['false', '0', 'no', 'n']:
            return False
    raise ValueError(f"Cannot convert {value} to a boolean.")

def get_paths(base_dir: str, unique_model_description: str, model_save_path: str = 'trained_model/model.pth'):
    data_path = os.path.join(base_dir, unique_model_description)
    os.makedirs(data_path, exist_ok=True)
    model_save_to = os.path.join(data_path, model_save_path)
    path_to_save_dataloader = os.path.join(data_path, 'data_created_during_training/')
    os.makedirs(os.path.dirname(model_save_to), exist_ok=True)
    os.makedirs(path_to_save_dataloader, exist_ok=True)
    return model_save_to, path_to_save_dataloader

# TODO: Validate Pass by Reference 
def load_metadata_from_disk(data, metadata_path):

    with open(metadata_path, 'r') as f:
        city_data = json.load(f)

    metadata_dir = Path(metadata_path).parent

    resolved_paths = []
    for p in city_data['path']:
        if os.path.exists(p):
            resolved_paths.append(p)
        else:
            resolved_paths.append(str(metadata_dir / Path(p).name))

    data['path'].extend(resolved_paths)
    data['policy_region'].extend(city_data['policy_region'])
    data['scenario'].extend(city_data['scenario'])
    data['city'].extend(city_data['city'])

def setup_wandb(args):
    wandb.login()
    wandb.init(project=args['project_name'], name=args['unique_model_description'],
               config={k: v for k, v in args.items() if k not in ['project_name', 'unique_model_description', 'model_kwargs']})
    return wandb.config

def setup_wandb_metrics():
    wandb.define_metric("epoch") # Custom X-axis
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")
    wandb.define_metric("r^2", step_metric="epoch")
    wandb.define_metric("spearman", step_metric="epoch")
    wandb.define_metric("pearson", step_metric="epoch")

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


################################################# ↓ Data Normalization ↓ #################################################

def normalize_dataset(train_data_list, combined_data_list):
    """
    train_data_list: Subset on which to apply normalization (train set)
    combined_data_list: Subset for fitting scaler (train + val)
    """
    train_data = [copy.deepcopy(train_data_list.dataset[idx]) for idx in train_data_list.indices]
    # combined_data = [copy.deepcopy(combined_data_list.dataset[idx]) for idx in combined_data_list.indices]

    print("Fitting and normalizing x features...")
    normalized_data_list, x_scaler = normalize_x_features_batched(train_data, combined_data_list) 
    print("x features normalized")
        
    scalers_dict = {"x_scaler": x_scaler}
    return normalized_data_list, scalers_dict 

def normalize_x_features_batched(train_data_list, combined_data_list, batch_size=100):
    """
    Normalize the continuous node features (0 mean and unit variance).
    Categorical features (Allowed Modes, Highway etc.) are left as booleans (0 or 1).
    Fit scaler on combined train+val set, but apply normalization only to train set.
    Returns normalized train set and fitted scaler.
    """
    scaler = StandardScaler()

    # Continuous features to normalize
    continuous_feat = [EdgeFeatures.VOL_BASE_CASE,
                       EdgeFeatures.CAPACITY_BASE_CASE,
                       #EdgeFeatures.CAPACITY_REDUCTION, Since this is binary, no normalization
                       EdgeFeatures.FREESPEED,
                       EdgeFeatures.LENGTH]
    
    # First pass: Fit the scaler incrementally, graph by graph
    for i in tqdm(range(0, len(combined_data_list), batch_size), desc="Fitting scaler"):
        batch_indices = combined_data_list.indices[i:i+batch_size]
        batch = [combined_data_list.dataset[idx] for idx in batch_indices]
        for data in batch:
            # Fit scaler on each graph's node features separately
            scaler.partial_fit(data.x[:, continuous_feat].numpy())

    # Second pass: Transform the data in batches
    for i in tqdm(range(0, len(train_data_list), batch_size), desc="Normalizing x features"):
        batch = train_data_list[i:i+batch_size]
        
        # TODO: WHY?
        # Process each graph in the batch individually (no vstack!)
        for data in batch:
            data_x = data.x[:, continuous_feat].numpy()
            data_x_normalized = scaler.transform(data_x)
            data.x[:, continuous_feat] = torch.tensor(data_x_normalized, dtype=data.x.dtype)

    return train_data_list, scaler

def normalize_dataset_with_scaler(dataset_input, scalers):
    """
    Normalize dataset using pre-fitted scalers (for validation/test sets).
    """
    data_list = [copy.deepcopy(dataset_input.dataset[idx]) for idx in dataset_input.indices]

    print("Normalizing x features with existing scaler...")
    normalized_data_list = normalize_x_features_with_scaler(data_list, scalers['x_scaler'])
    print("x features normalized")
    
    return normalized_data_list

def normalize_x_features_with_scaler(data_list, x_scaler, batch_size=100):
    """
    Normalize the continuous node features with a given scaler.
    Categorical features (Allowed Modes, Highway etc.) are left as booleans (0 or 1).
    """

    # Continuous features to normalize
    continuous_feat = [EdgeFeatures.VOL_BASE_CASE,
                       EdgeFeatures.CAPACITY_BASE_CASE,
                       #EdgeFeatures.CAPACITY_REDUCTION,
                       EdgeFeatures.FREESPEED,
                       EdgeFeatures.LENGTH]
    
    # FIX: Handle variable node counts correctly
    for i in tqdm(range(0, len(data_list), batch_size), desc="Normalizing x features"):
        batch = data_list[i:i+batch_size]
        batch_x = np.vstack([data.x[:,continuous_feat].numpy() for data in batch])
        batch_x_normalized = x_scaler.transform(batch_x)
        
        # CORRECT: Use proper indexing for variable node counts
        start = 0
        for data in batch:
            num_nodes = data.x.shape[0]
            data.x[:,continuous_feat] = torch.tensor(
                batch_x_normalized[start:start+num_nodes], 
                dtype=data.x.dtype)
            start += num_nodes
    
    return data_list


################################################# ↓ Graph DATA + Model Setup ↓ #################################################

# For training batches
# Control freak, so avoided
def get_sampling_weights(dataset):
    
    # Add any other BANGER logic here
    # labels (city + policy_region) can be accessed as dataset.labels

    # Uniform weights for all samples
    return [1.0 / len(dataset)] * len(dataset)

# TODO: Adapt based on split type?
# Weighted sampling not necessarily needed for validation/test?
def create_split_dataloader(data_subset, batch_size, use_weighted_batches, collate_fn_type):
    return DataLoader(
        dataset=data_subset, 
        batch_size=batch_size, 
        shuffle=True if not use_weighted_batches else None,
        sampler=WeightedRandomSampler(get_sampling_weights(data_subset), len(data_subset)) if use_weighted_batches else None,
        num_workers=4, # Check
        prefetch_factor=2, # Check
        pin_memory=False, # Check
        collate_fn=collate_fn_type,  
        worker_init_fn=seed_worker,
        drop_last=False) # Check

# test_data implies use of complete inductive testing
def prepare_data_with_graph_features(train_data, val_data, test_data, use_inductive_variant,
                                     batch_size, path_to_save_dataloader,
                                     use_all_features, use_weighted_batches,
                                     use_nested_neighbor_loader, neighbor_sizes, subgraphs_per_graph, seed_size,
                                     min_subgraph_nodes, max_subgraph_nodes, sampling_strategy,
                                     aug_pos_rotation, aug_feature_noise, aug_node_masking_probability=0.0): 
    
    print(f"Preparing data with {len(train_data['path']) + (len(val_data['path']) if val_data is not None else 0) + (len(test_data['path']) if test_data is not None else 0)} items")
    
    print("Splitting into subsets...")

    _, train_set, valid_set, test_set = load_data_and_split_into_subsets(train_data=train_data, val_data=val_data, test_data=test_data,
                                                                         train_ratio=0.8, val_ratio=0.15, test_ratio=0.05)
    
    combined_indices = train_set.indices + valid_set.indices
    combined_train_val_set = torch.utils.data.Subset(train_set.dataset, combined_indices)
    
    print(f"Split complete. Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")

    if use_all_features:
        node_features = []
        for feat in EdgeFeatures:
            name = feat.name
            if not use_allowed_modes and name.startswith("ALLOWED_MODE"):
                continue
            if not use_destination_activity and name in {
                "HOME", "WORK", "EDUCATION", "LEISURE", "SHOP", "OTHER", "OUTSIDE" ,'IS_IN_EQASIM_TRIPS'
            }:
                continue
            if not use_highway and name.startswith("HIGHWAY"):
                continue
            node_features.append(name)
    else:
        node_features = [
            "VOL_BASE_CASE",
            "CAPACITY_BASE_CASE",
            "CAPACITY_REDUCTION",
            "FREESPEED",
            "LENGTH"]
    
    print(node_features)

    # CREATE FEATURE MAPPING ONCE - BEFORE NORMALIZATION
    filtered_feature_mapping = {}
    current_idx = 0
    
    for feature_name in node_features:
        filtered_feature_mapping[EdgeFeatures[feature_name].value] = current_idx
        current_idx += 1
    
    print(f"Global feature mapping: {filtered_feature_mapping}")

    node_feature_filter = [EdgeFeatures[feature].value for feature in node_features]

    # With augmentation, ideal for training (when not using nested loader)
    collate_fn_with_aug = partial(collate_fn,
                                  node_feature_filter=node_feature_filter,
                                  filtered_feature_mapping=filtered_feature_mapping,
                                  augment_pos_rotation=aug_pos_rotation,
                                  augment_feature_noise=aug_feature_noise,
                                  augment_node_masking_prob=aug_node_masking_probability,
                                  is_training=True)


    # Without augmentation, ideal for evaluation
    collate_fn_with_no_aug = partial(collate_fn,
                                     node_feature_filter=node_feature_filter,
                                     filtered_feature_mapping=filtered_feature_mapping,
                                     is_training=False)
    
    print('Data Augmentation Settings:')
    print('Pos Rotation Augmentation:', aug_pos_rotation, 'Feature Noise Augmentation:', aug_feature_noise, 'Node Masking Probability:', aug_node_masking_probability)
    print('Use Nested Neighbor Loader:', use_nested_neighbor_loader)

    # IF TRANSDUCTIVE
    if use_inductive_variant == False:

        print("Creating base train loader...")
        base_train_loader = create_split_dataloader(data_subset=train_set,
                                                    batch_size=batch_size,
                                                    use_weighted_batches=use_weighted_batches,
                                                    # Uses conditional augmentation
                                                    collate_fn_type=collate_fn_with_no_aug if use_nested_neighbor_loader else collate_fn_with_aug)

        if len(valid_set) > 0:
            print("Creating validation loader...")
            val_loader = create_split_dataloader(data_subset=valid_set,
                                                 batch_size=batch_size,
                                                 use_weighted_batches=use_weighted_batches,
                                                 collate_fn_type=collate_fn_with_no_aug)  # Always without augmentation
        else:
            print("Validation subset empty; skipping validation loader.")
            val_loader = None

        if len(test_set) > 0:
            print("Creating test loader...")
            test_loader = create_split_dataloader(data_subset=test_set,
                                                  batch_size=batch_size,
                                                  use_weighted_batches=use_weighted_batches,
                                                  collate_fn_type=collate_fn_with_no_aug)  # Always without augmentation
            save_dataloader(test_loader, path_to_save_dataloader + 'test_dl.pt')
            save_dataloader_params(test_loader, path_to_save_dataloader + 'test_loader_params.json')
            print("Test Dataloader saved since Transductive Variant. No scalers needed.")
        else:
            print("Test subset empty; skipping test loader.")

        print("Loaders created")
        
    else:
        
        # TODO: WHY?
        # ✅ FIX: Use training+val scaler for all splits
        print("Normalizing train set...")
        train_set_normalized, scalers_train = normalize_dataset(train_data_list=train_set, combined_data_list=combined_train_val_set)
        print("Train set normalized")      
        
        print("Normalizing validation set with TRAINING scaler...")
        valid_set_normalized = normalize_dataset_with_scaler(dataset_input=valid_set, scalers=scalers_train)
        print("Validation set normalized")
        
        print("Normalizing test set with TRAINING scaler...")
        test_set_normalized = normalize_dataset_with_scaler(dataset_input=test_set, scalers=scalers_train)
        print("Test set normalized")
        
        print("Creating train loader...")
        base_train_loader = create_split_dataloader(
            data_subset=train_set_normalized,
            batch_size=batch_size,
            use_weighted_batches=use_weighted_batches,
            # Uses conditional augmentation
            collate_fn_type=collate_fn_with_no_aug if use_nested_neighbor_loader else collate_fn_with_aug)
        print("Train loader created")
        
        print("Creating validation loader...")
        val_loader = create_split_dataloader(
            data_subset=valid_set_normalized,
            batch_size=batch_size,
            use_weighted_batches=use_weighted_batches,
            collate_fn_type=collate_fn_with_no_aug)  # Always without augmentation
        print("Validation loader created")
        
        print("Creating test loader...")
        test_loader = create_split_dataloader(
            data_subset=test_set_normalized,
            batch_size=batch_size,
            use_weighted_batches=use_weighted_batches,
            collate_fn_type=collate_fn_with_no_aug)  # Always without augmentation
        print("Test loader created")
        
        # ONLY SAVE TRAINING SCALERS (validation/test use same scalers)
        joblib.dump(scalers_train['x_scaler'], os.path.join(path_to_save_dataloader, 'train_x_scaler.pkl'))
        
        # save_dataloader(test_loader, path_to_save_dataloader + 'test_dl.pt')
        # save_dataloader_params(test_loader, path_to_save_dataloader + 'test_loader_params.json')
        print("Test dataloader NOT saved since Inductive Variant. Scalers are saved.")
    
    if use_nested_neighbor_loader:
        train_loader = nested_dataloader(
            base_train_loader, 
            neighbor_sizes=neighbor_sizes, 
            subgraphs_per_graph=subgraphs_per_graph, 
            seed_size=seed_size,  
            final_batch_size=batch_size*subgraphs_per_graph, 
            sampling_strategy=sampling_strategy, 
            min_subgraph_nodes=min_subgraph_nodes,
            max_subgraph_nodes=max_subgraph_nodes,
            node_feature_filter=node_feature_filter,
            filtered_feature_mapping=filtered_feature_mapping,
            augment_pos_rotation=aug_pos_rotation,
            augment_feature_noise=aug_feature_noise,
            augment_node_masking_prob=aug_node_masking_probability,
            is_training=True
        )
    else:
        train_loader = base_train_loader
     
    # Test the nested loader
    if use_nested_neighbor_loader:
        print("\n=== Testing Nested Loader ===")
        for batch in train_loader:
            print(f"Total subgraphs in batch: {batch.num_graphs}")
            
            # Check individual subgraphs
            individual_graphs = batch.to_data_list()
            if len(individual_graphs) > 0:
                graph = individual_graphs[0]
                print(f"First subgraph - Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
                if hasattr(graph, 'sampling_strategy'):
                    print(f"Sampling strategy: {graph.sampling_strategy}")
            break
    
    #save_dataloader(test_loader, path_to_save_dataloader + 'test_dl.pt')
    #save_dataloader_params(test_loader, path_to_save_dataloader + 'test_loader_params.json')
    # print("Test dataloader saved.")
    
    # MODIFIED: Return None for scalers since we don't need them
    if use_inductive_variant == False:
        return train_loader, val_loader, None
    else:
        # ✅ ONLY RETURN TRAINING SCALERS
        return train_loader, val_loader, scalers_train

def create_gnn_model(gnn_arch: str, config: object, model_kwargs: dict, device: torch.device):
    """
    Factory function to create the specified model architecture.
    
    Args:
    - gnn_arch (str): The architecture of the GNN model to create.
    - config (object): WandB config with run arguments.
    - device (torch.device): The device to which the model should be moved (CPU or GPU).
    - model_kwargs (dict): Additional keyword arguments specific to the model.
    
    Returns:
    - Initialized model on the specified device
    """

    common_kwargs = {
        "in_channels": config.in_channels,
        "out_channels": config.out_channels,
        "use_dropout": config.use_dropout,
        "dropout": config.dropout,
        "dtype": torch.float32,
        "log_to_wandb": True,
        "use_target_standardization": getattr(config, 'use_target_standardization', False)
    }
    
    if gnn_arch == "graphSAGE":
        return GraphSAGE(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "gatv2":
        return GATv2(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "trans_conv":
        return TransConv(**common_kwargs, **model_kwargs).to(device)
        
    elif gnn_arch == "trans_encoder":
        return TransEncoder(**common_kwargs, **model_kwargs).to(device)
        
    else:
        raise ValueError(f"Unknown architecture: {gnn_arch}")
    