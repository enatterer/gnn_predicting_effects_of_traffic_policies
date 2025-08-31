import os
import sys
import copy
import random
import joblib
import subprocess
from functools import partial
from collections import Counter

import numpy as np
from tqdm import tqdm
import wandb
from sklearn.preprocessing import StandardScaler
import torch
import psutil
import gc

from torch.utils.data import IterableDataset, Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.data import Batch, Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.gnn_io import *
from gnn.models.point_net_transf_gat import PointNetTransfGAT
from gnn.models.gcn import GCN, GCN2
from gnn.models.gat import GAT
from gnn.models.gatv2 import GATv2
from gnn.models.gatv3 import GATv3
from gnn.models.trans_conv import TransConv
from gnn.models.pnc import PNC
from gnn.models.fc_nn import FC_NN
from gnn.models.eign import EIGNLaplacianConv
from gnn.models.graphSAGE import GraphSAGE
from gnn.models.xgboost import XGBoostModel
from gnn.models.trans_encoder import TransEncoder
from data_preprocessing.process_simulations_for_gnn import EdgeFeatures

#####control center parameters#####
use_allowed_modes = False
use_destination_activity = False
use_highway = False
###################################

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

def str_to_bool(value):
    if isinstance(value, str):
        if value.lower() in ['true', '1', 'yes', 'y']:
            return True
        elif value.lower() in ['false', '0', 'no', 'n']:
            return False
    raise ValueError(f"Cannot convert {value} to a boolean.")
    
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

# test_data implies use of complete inductive testing
def prepare_data_with_graph_features(train_data, val_data, test_data,variant,
                                     batch_size, path_to_save_dataloader,
                                     use_all_features, use_bootstrapping, use_weighted_sampling,
                                     use_nested_neighbor_loader, neighbor_sizes, subgraphs_per_graph, seed_size,
                                     min_subgraph_nodes, max_subgraph_nodes, sampling_strategy, is_eign,
                                     use_data_augmentation):
    
    print(f"Preparing data with {len(train_data['path']) + (len(test_data['path']) if test_data is not None else 0)} items")
    
    print("Splitting into subsets...")

    # TODO: Fix Later
    if use_bootstrapping:
        train_set, valid_set, test_set = split_into_subsets_with_bootstrapping(dataset=train_data, test_ratio=0.1, bootstrap_seed=4) # TODO: fix later
    else:
        dataset, train_set, valid_set, test_set = load_data_and_split_into_subsets(train_data=train_data, val_data=val_data, test_data=test_data,
                                                                                    train_ratio=0.8, val_ratio=0.15, test_ratio=0.05)
    
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
        # Manual feature selection (e.g., from ablation)
        node_features = [
            "VOL_BASE_CASE",
            "CAPACITY_BASE_CASE",
            "CAPACITY_REDUCTION",
            "FREESPEED",
            "LENGTH",
            # Add/remove features as desired
            # "HOME",
            # "WORK",
            # "SHOP",
        ]
    print(node_features)
    # COMMENTED OUT: Features already normalized during preprocessing
    # # Fit GLOBAL Scaler!
    # # Assume no exclusive test scaler, #TODO:fix later if needed
    # scalers_train, continuous_feat = fit_global_scaler(dataset, batch_size=128)
    # scalers_test = copy.deepcopy(scalers_train)

    node_feature_filter = [EdgeFeatures[feature].value for feature in node_features]

    if variant == 'GNN_Transductive':
        # MODIFIED: Use simple collate function without scaling
        #collate_with_scaler = partial(scale_and_collate, scaler=scalers_train['x_scaler'], continuous_feat=continuous_feat, node_feature_filter=node_feature_filter)
        collate_without_scaling_fn = partial(collate_without_scaling, node_feature_filter=node_feature_filter, augment_pos_rotation=use_data_augmentation)
        print('Data Augmentation:', use_data_augmentation)

        print("Creating base train loader...")
        base_train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                                    shuffle=True if not use_weighted_sampling else None,
                                    sampler=WeightedRandomSampler(get_sampling_weights(train_set), len(train_set)) if use_weighted_sampling else None,
                                    num_workers=2, prefetch_factor=2, pin_memory=True, 
                                    collate_fn=collate_without_scaling_fn,
                                    worker_init_fn=seed_worker,
                                    drop_last=False)
        #print(f"Memory_per_graph: {estimate_average_graph_memory(base_train_loader, num_samples=100)} MB")
        print("Creating validation loader...")
        val_loader = DataLoader(dataset=valid_set, batch_size=batch_size,
                            shuffle=True if not use_weighted_sampling else None,
                            sampler=WeightedRandomSampler(get_sampling_weights(valid_set), len(valid_set)) if use_weighted_sampling else None,
                            num_workers=2, pin_memory=True, 
                            collate_fn=collate_without_scaling_fn,
                            worker_init_fn=seed_worker,
                            drop_last=False)
    
        print("Creating test loader...")
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size,
                                shuffle=True if not use_weighted_sampling else None,
                                sampler=WeightedRandomSampler(get_sampling_weights(test_set), len(test_set)) if use_weighted_sampling else None,
                                num_workers=2, 
                                collate_fn=collate_without_scaling_fn,
                                worker_init_fn=seed_worker,
                                drop_last=False)
    else:
        collate_fn_aug = partial(collate_fn, augment_pos_rotation=use_data_augmentation)
        print('Data Augmentation:', use_data_augmentation)
        print("Normalizing train set...")
        train_set_normalized, scalers_train = normalize_dataset(dataset_input=train_set, node_features=node_features, is_eign=is_eign)
        print("Train set normalized")      
        
        print("Normalizing validation set...")
        valid_set_normalized, scalers_validation = normalize_dataset(dataset_input=valid_set, node_features=node_features, is_eign=is_eign)
        print("Validation set normalized")
        
        print("Normalizing test set...")
        test_set_normalized, scalers_test = normalize_dataset(dataset_input=test_set, node_features=node_features, is_eign=is_eign)
        print("Test set normalized")
        
        print("Creating train loader...")
        base_train_loader = DataLoader(dataset=train_set_normalized, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True, collate_fn=collate_fn_aug, worker_init_fn=seed_worker)
        print("Train loader created")
        
        print("Creating validation loader...")
        val_loader = DataLoader(dataset=valid_set_normalized, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn_aug, worker_init_fn=seed_worker)
        print("Validation loader created")
        
        print("Creating test loader...")
        test_loader = DataLoader(dataset=test_set_normalized, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn_aug, worker_init_fn=seed_worker)
        print("Test loader created")
        
        joblib.dump(scalers_train['x_scaler'], os.path.join(path_to_save_dataloader, 'train_x_scaler.pkl'))
        #if not is_eign:
            #joblib.dump(scalers_train['pos_scaler'], os.path.join(path_to_save_dataloader, 'train_pos_scaler.pkl'))
        # joblib.dump(scalers_train['modestats_scaler'], os.path.join(path_to_save_dataloader, 'train_mode_stats_scaler.pkl'))

        joblib.dump(scalers_validation['x_scaler'], os.path.join(path_to_save_dataloader, 'validation_x_scaler.pkl'))
        #if not is_eign:
            #joblib.dump(scalers_validation['pos_scaler'], os.path.join(path_to_save_dataloader, 'validation_pos_scaler.pkl'))
        # joblib.dump(scalers_validation['modestats_scaler'], os.path.join(path_to_save_dataloader, 'validation_mode_stats_scaler.pkl'))

        joblib.dump(scalers_test['x_scaler'], os.path.join(path_to_save_dataloader, 'test_x_scaler.pkl'))
        #if not is_eign:
            #joblib.dump(scalers_test['pos_scaler'], os.path.join(path_to_save_dataloader, 'test_pos_scaler.pkl'))
        # joblib.dump(scalers_test['modestats_scaler'], os.path.join(path_to_save_dataloader, 'test_mode_stats_scaler.pkl'))  
        
        save_dataloader(test_loader, path_to_save_dataloader + 'test_dl.pt')
        save_dataloader_params(test_loader, path_to_save_dataloader + 'test_loader_params.json')
        print("Dataloaders and scalers saved")
    
    if use_nested_neighbor_loader:
        train_loader = nested_dataloader(base_train_loader, neighbor_sizes=neighbor_sizes, 
                                                         subgraphs_per_graph=subgraphs_per_graph, seed_size=seed_size,  
                                                         final_batch_size=batch_size*subgraphs_per_graph, sampling_strategy=sampling_strategy, 
                                                         min_subgraph_nodes=min_subgraph_nodes,
                                                         max_subgraph_nodes=max_subgraph_nodes,
                                                         node_feature_filter=node_feature_filter)
    else:
        train_loader = base_train_loader
    
    
    
    # COMMENTED OUT: No need to save scalers for features
    # joblib.dump(scalers_train['x_scaler'], os.path.join(path_to_save_dataloader, 'train_x_scaler.pkl'))
    # joblib.dump(scalers_train['modestats_scaler'], os.path.join(path_to_save_dataloader, 'train_mode_stats_scaler.pkl'))
    # joblib.dump(scalers_test['x_scaler'], os.path.join(path_to_save_dataloader, 'test_x_scaler.pkl'))
    # joblib.dump(scalers_test['modestats_scaler'], os.path.join(path_to_save_dataloader, 'test_mode_stats_scaler.pkl'))  
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
    
    save_dataloader(test_loader, path_to_save_dataloader + 'test_dl.pt')
    save_dataloader_params(test_loader, path_to_save_dataloader + 'test_loader_params.json')
    print("Test dataloader saved.")
    
    # MODIFIED: Return None for scalers since we don't need them
    if variant=='GNN_Transductive':
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader, scalers_train, scalers_validation

def nested_dataloader(base_train_loader: DataLoader,
                                    neighbor_sizes: list[int] = [15, 10, 5],
                                    subgraphs_per_graph: int = 3,
                                    seed_size: int = 1,  # Single seed per subgraph   
                                    final_batch_size: int = 24,  # batch_size * subgraphs_per_graph
                                    sampling_strategy: str = 'neighbor_sampling',  # Single strategy
                                    min_subgraph_nodes: int = 10,
                                    max_subgraph_nodes: int = 100,
                                    node_feature_filter: list = None) -> DataLoader:
    """
    Create enhanced nested dataloader with on-the-fly subgraph generation.
    
    Args:
        base_train_loader: Your existing base graph loader
        neighbor_sizes: Neighbor sampling sizes per hop
        subgraphs_per_graph: Number of subgraphs per original graph
        seed_size: Batch size for seed nodes in neighbor sampling 
        final_batch_size: Final batch size for the nested loader
        sampling_strategy: Single sampling strategy to use for all subgraphs
        min_subgraph_nodes: Minimum nodes in subgraph
        max_subgraph_nodes: Maximum nodes in subgraph 
    
    Returns:
        DataLoader that yields batched subgraphs (generated on-the-fly)
    """
    
    nested_dataset = NestedNeighborDataset(
        graph_loader=base_train_loader,
        neighbor_sizes=neighbor_sizes,
        subgraphs_per_graph=subgraphs_per_graph,
        seed_size=seed_size,    
        sampling_strategy=sampling_strategy,
        min_subgraph_nodes=min_subgraph_nodes,
        max_subgraph_nodes=max_subgraph_nodes,
        shuffle_mapping=False,  # Set to True for two layer randomization in combination with outer dataloader shuffle
        node_feature_filter=node_feature_filter
    )
    
    # Create final dataloader
    nested_loader = DataLoader(
        nested_dataset,
        batch_size=final_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=Batch.from_data_list,
        drop_last=False,
    )
    
    print(f"Created nested dataloader with {len(nested_dataset)} subgraphs")
    print(f"Final batches will contain {final_batch_size} subgraphs each")
    
    return nested_loader
def fit_global_scaler(dataset, batch_size=128):
    
    scaler = StandardScaler()

    # Continuous features to normalize
    continuous_feat = [EdgeFeatures.VOL_BASE_CASE,
                       EdgeFeatures.CAPACITY_BASE_CASE,
                       EdgeFeatures.CAPACITY_REDUCTION,
                       EdgeFeatures.FREESPEED,
                       EdgeFeatures.LENGTH]
    
    # First pass: Fit the scaler
    for i in tqdm(range(0, len(dataset), batch_size), desc="Fitting global scaler ..."):
        batch = dataset[i:i+batch_size]
        batch_x = np.vstack([data.x[:,continuous_feat].numpy() for data in batch])
        scaler.partial_fit(batch_x)
    
    return {"x_scaler":scaler}, continuous_feat

def get_sampling_weights(dataset):
    
    # Add any other BANGER logic here
    # labels (city + policy_region) can be accessed as dataset.labels

    # Uniform weights for all samples
    return [1.0 / len(dataset)] * len(dataset)  

        
def normalize_dataset(dataset_input, node_features, is_eign=False):
    data_list = [copy.deepcopy(dataset_input.dataset[idx]) for idx in dataset_input.indices]

    print("Fitting and normalizing x features...")
    normalized_data_list, x_scaler = normalize_x_features_batched(data_list, node_features)
    print("x features normalized")
    
    if is_eign:
        print("Fitting and normalizing x_signed features...")
        normalized_data_list, x_signed_scaler = normalize_x_signed_features_batched(
            normalized_data_list
        )
        print("x_signed features normalized")
        
    # print("Fitting and normalizing modestats features...")
    # normalized_data_list, modestats_scaler = normalize_modestats_features_batched(normalized_data_list)
    # print("Modestats features normalized")
    
    scalers_dict = {
        "x_scaler": x_scaler,
        "x_signed_scaler": x_signed_scaler,
    } if is_eign else {
        "x_scaler": x_scaler,
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

    # Continuous features to normalize
    continuous_feat = [EdgeFeatures.VOL_BASE_CASE,
                       EdgeFeatures.CAPACITY_BASE_CASE,
                       #EdgeFeatures.CAPACITY_REDUCTION, since this is binary, no normalization
                       EdgeFeatures.FREESPEED,
                       EdgeFeatures.LENGTH]
    
    # Get number of nodes in the graph
    num_nodes = data_list[0].x.shape[0]
    
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
            data.x[:,continuous_feat] = torch.tensor(batch_x_normalized[j*num_nodes:(j+1)*num_nodes], dtype=data.x.dtype)

    # Filter features
    node_feature_filter = [EdgeFeatures[feature].value for feature in node_features]
    for data in data_list:
        data.x = data.x[:, node_feature_filter]

    # One-hot encode highway
    if "HIGHWAY" in node_features:
        one_hot_highway(data_list, idx=node_features.index("HIGHWAY"))
    
    return data_list, scaler

def normalize_x_features_with_scaler(data_list, node_features, x_scaler, batch_size=100):
    """
    Normalize the continuous node features with a given scaler.
    Categorical features (Allowed Modes) are left as booleans (0 or 1).
    'HIGHWAY' feature is one-hot encoded.

    Finally, features are filtered to only include the ones specified in node_features. 
    """

    # Continuous features to normalize
    continuous_feat = [EdgeFeatures.VOL_BASE_CASE,
                       EdgeFeatures.CAPACITY_BASE_CASE,
                       #EdgeFeatures.CAPACITY_REDUCTION,
                       EdgeFeatures.FREESPEED,
                       EdgeFeatures.LENGTH]
    
    # Get number of nodes in the graph
    num_nodes = data_list[0].x.shape[0]
    
    # Second pass: Transform the data
    for i in tqdm(range(0, len(data_list), batch_size), desc="Normalizing x features"):
        batch = data_list[i:i+batch_size]
        batch_x = np.vstack([data.x[:,continuous_feat].numpy() for data in batch])
        batch_x_normalized = x_scaler.transform(batch_x)
        for j, data in enumerate(batch):
            data.x[:,continuous_feat] = torch.tensor(batch_x_normalized[j*num_nodes:(j+1)*num_nodes], dtype=data.x.dtype)

    # Filter features
    node_feature_filter = [EdgeFeatures[feature].value for feature in node_features]
    for data in data_list:
        data.x = data.x[:, node_feature_filter]

    # One-hot encode highway
    if "HIGHWAY" in node_features:
        one_hot_highway(data_list, idx=node_features.index("HIGHWAY"))
    
    return data_list

def normalize_x_signed_features_batched(data_list, batch_size=1000):
    """
    Normalize the x_signed features (0 mean and unit variance).
    x_signed typically has shape (num_nodes, 1) for EIGN models.
    """
    scaler = StandardScaler()

    # Get number of nodes in the graph
    num_nodes = (
        data_list[0].x_signed.shape[0]
        if hasattr(data_list[0], "x_signed") and data_list[0].x_signed is not None
        else 0
    )

    # Skip if no x_signed features
    if num_nodes == 0:
        return data_list, None

    # First pass: Fit the scaler
    for i in tqdm(range(0, len(data_list), batch_size), desc="Fitting x_signed scaler"):
        batch = data_list[i : i + batch_size]
        batch_x_signed = np.vstack(
            [
                data.x_signed.numpy().reshape(-1, 1)
                for data in batch
                if hasattr(data, "x_signed") and data.x_signed is not None
            ]
        )
        if batch_x_signed.size > 0:
            scaler.partial_fit(batch_x_signed)

    # Second pass: Transform the data
    for i in tqdm(
        range(0, len(data_list), batch_size), desc="Normalizing x_signed features"
    ):
        batch = data_list[i : i + batch_size]
        for data in batch:
            if hasattr(data, "x_signed") and data.x_signed is not None:
                x_signed_reshaped = data.x_signed.numpy().reshape(-1, 1)
                x_signed_normalized = scaler.transform(x_signed_reshaped)
                data.x_signed = torch.tensor(
                    x_signed_normalized.reshape(data.x_signed.shape),
                    dtype=data.x_signed.dtype,
                )

    return data_list, scaler

def normalize_x_signed_features_with_scaler(
    data_list, x_signed_scaler, batch_size=1000
):
    """
    Normalize the x_signed features with a given scaler.
    x_signed typically has shape (num_nodes, 1) for EIGN models.
    """
    # Skip if no scaler provided or no x_signed features
    if x_signed_scaler is None:
        return data_list

    # Check if any data has x_signed features
    has_x_signed = any(
        hasattr(data, "x_signed") and data.x_signed is not None for data in data_list
    )

    if not has_x_signed:
        return data_list

    # Transform the data using the provided scaler
    for i in tqdm(
        range(0, len(data_list), batch_size), desc="Normalizing x_signed features"
    ):
        batch = data_list[i : i + batch_size]
        for data in batch:
            if hasattr(data, "x_signed") and data.x_signed is not None:
                x_signed_reshaped = data.x_signed.numpy().reshape(-1, 1)
                x_signed_normalized = x_signed_scaler.transform(x_signed_reshaped)
                data.x_signed = torch.tensor(
                    x_signed_normalized.reshape(data.x_signed.shape),
                    dtype=data.x_signed.dtype,
                )
    return data_list

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


def setup_wandb(args):
    wandb.login()
    wandb.init(project=args['project_name'], name=args['unique_model_description'],
               config={k: v for k, v in args.items() if k not in ['project_name', 'unique_model_description', 'model_kwargs']})
    return wandb.config

def setup_wandb_metrics(predict_mode_stats=False):

    wandb.define_metric("epoch") # Custom X-axis
    wandb.define_metric("batch_step") # Custom X-axis
    
    wandb.define_metric("batch_train_loss", step_metric="batch_step")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")
    wandb.define_metric("r^2", step_metric="epoch")
    wandb.define_metric("spearman", step_metric="epoch")
    wandb.define_metric("pearson", step_metric="epoch")

    if predict_mode_stats:
        wandb.define_metric("batch_train_loss-node_predictions", step_metric="batch_step")
        wandb.define_metric("batch_train_loss-mode_stats", step_metric="batch_step")
        wandb.define_metric("train_loss-node_predictions", step_metric="epoch")
        wandb.define_metric("train_loss-mode_stats", step_metric="epoch")
        wandb.define_metric("val_loss-node_predictions", step_metric="epoch")
        wandb.define_metric("val_loss-mode_stats", step_metric="epoch")

def create_gnn_model(gnn_arch: str, config: object, model_kwargs: dict, device: torch.device):
    """
    Factory function to create the specified model architecture.
    
    Args:
    - gnn_arch (str): The architecture of the GNN model to create.
    - config (object): WandB config with run arguments.
    - device (torch.device): The device to which the model should be moved (CPU or GPU).
    
    Returns:
    - Initialized model on the specified device
    """

    common_kwargs = {
        "in_channels": config.in_channels,
        "out_channels": config.out_channels,
        "use_dropout": config.use_dropout,
        "dropout": config.dropout,
        "predict_mode_stats": config.predict_mode_stats,
        "dtype": torch.float32,
        "log_to_wandb": True} # During training, yes

    if gnn_arch == "point_net_transf_gat":
        return PointNetTransfGAT(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "graphSAGE":
        model = GraphSAGE(**common_kwargs, **model_kwargs).to(device)
        return model
    
    elif gnn_arch == "gcn":
        return GCN(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "gcn2":
        return GCN2(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "gat":
        return GAT(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "gatv2":
        return GATv2(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "gatv3":
        return GATv3(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "trans_conv":
        return TransConv(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "pnc":
        return PNC(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "fc_nn":
        return FC_NN(**common_kwargs, **model_kwargs).to(device)

    elif gnn_arch == "eign":
        return EIGNLaplacianConv(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "xgboost":
        return XGBoostModel(**common_kwargs, **model_kwargs)
    elif gnn_arch == "trans_encoder":
        return TransEncoder(**common_kwargs, **model_kwargs).to(device)
    else:
        raise ValueError(f"Unknown architecture: {gnn_arch}")


class NestedNeighborDataset(Dataset):
    '''
    On-the-fly subgraph generation with Dataset interface.
    Generates fresh subgraphs every epoch using neighbor sampling,
    providing better variety and following modern GNN practices.
    '''
    def __init__(self,
                 graph_loader: DataLoader,
                 neighbor_sizes: list,
                 subgraphs_per_graph: int,
                 seed_size: int,  
                 sampling_strategy: str,
                 min_subgraph_nodes: int,
                 max_subgraph_nodes: int,
                 shuffle_mapping: bool = False,
                 node_feature_filter: list = None):
        """
        Args:
            graph_loader: Your existing base graph loader
            neighbor_sizes: Neighbor sampling sizes per hop [15,10,5]
            subgraphs_per_graph: Number of subgraphs per original graph (3)
            seed_size: Batch size for seed nodes
            sampling_strategy: Single sampling strategy to use for all subgraphs.
            min_subgraph_nodes: Minimum nodes in subgraph
            max_subgraph_nodes: Maximum nodes in subgraph 
        """
        self.neighbor_sizes = neighbor_sizes
        self.subgraphs_per_graph = subgraphs_per_graph
        self.seed_size = seed_size  
        self.sampling_strategy = sampling_strategy
        self.min_subgraph_nodes = min_subgraph_nodes
        self.max_subgraph_nodes = max_subgraph_nodes
        self.shuffle_mapping = shuffle_mapping
        self.node_feature_filter = node_feature_filter
        
        print(f"Initializing On-The-Fly Nested Dataset with {subgraphs_per_graph} subgraphs per graph")
        print(f"Using single sampling strategy: {self.sampling_strategy}")
        print(f"Shuffle mapping: {shuffle_mapping}")
        print(f"Feature filter: {node_feature_filter}")
        
        # Store original dataset for on-demand loading
        self.original_dataset = graph_loader.dataset
        
        # Pre-compute the mapping from subgraph index to original graph
        self._compute_subgraph_mapping()
        
        print(f"Total subgraphs to generate on-the-fly: {len(self.subgraph_mapping)}")
    
    def _compute_subgraph_mapping(self):
        """Pre-compute which original graph each subgraph index corresponds to."""
        self.subgraph_mapping = []
        
        # Direct iteration over dataset (much simpler!)
        for graph_idx in range(len(self.original_dataset)):
            for subgraph_idx in range(self.subgraphs_per_graph):
                self.subgraph_mapping.append({
                    'original_graph_idx': graph_idx,
                    'subgraph_idx': subgraph_idx
                })
        
        # Shuffle mapping if requested (for better randomization)
        if self.shuffle_mapping:
            import random
            random.shuffle(self.subgraph_mapping)
            print("Shuffled subgraph mapping for better randomization")
        
        print(f"Processed {len(self.original_dataset)} original graphs")
    
    def _sample_subgraphs_from_single_graph(self, graph: Data, subgraph_idx: int) -> Data:
        """Sample a single subgraph from a graph."""
        
        # Skip if graph is too small
        if graph.num_nodes < self.min_subgraph_nodes:
            # For small graphs, just replicate the original graph
            subgraph = graph.clone()
            subgraph.sampling_strategy = self.sampling_strategy
            subgraph.subgraph_idx = subgraph_idx
            return subgraph
        
        try:
            if self.sampling_strategy == "neighbor_sampling":
                subgraph = self._neighbor_sampling_subgraph(graph)
            elif self.sampling_strategy == "random_walk":
                subgraph = self._random_walk_subgraph(graph)
            else:
                raise ValueError(f"Invalid sampling strategy: {self.sampling_strategy}")
            
            # Add metadata to subgraph
            subgraph.sampling_strategy = self.sampling_strategy
            subgraph.subgraph_idx = subgraph_idx
            
            # Validate subgraph size
            if subgraph.num_nodes > self.max_subgraph_nodes:
                subgraph = self._truncate_subgraph(subgraph, self.max_subgraph_nodes)
            
            return subgraph
            
        except Exception as e:
            print(f"Error sampling subgraph {subgraph_idx}: {e}")
            # Fallback to original graph
            subgraph = graph.clone()
            subgraph.sampling_strategy = self.sampling_strategy
            subgraph.subgraph_idx = subgraph_idx
            return subgraph
    
    def _neighbor_sampling_subgraph(self, graph: Data) -> Data:
        """
        Use NeighborLoader for subgraph sampling.
        
        Each subgraph contains:
        - 1 seed node (seed_size=1)   
        - Neighbors sampled according to neighbor_sizes [15, 10, 5]
        - Total nodes: ~1 + 15 + 15*10 + 15*10*5 = ~800 nodes max
        """
        all_nodes = torch.arange(graph.num_nodes)
        
        # Create neighbor loader for this specific graph
        neigh_loader = NeighborLoader(
            data=graph,
            num_neighbors=self.neighbor_sizes,
            input_nodes=all_nodes,
            batch_size=self.seed_size,  # 1 seed per subgraph
            shuffle=True,
            # Remove return_e_id parameter as it's not available in this version
        )
        
        # Get first subgraph from the loader
        for subgraph in neigh_loader:
            # NeighborLoader already properly subsets node features, but double-check pos
            if hasattr(subgraph, 'pos') and hasattr(subgraph, 'n_id'):
                # n_id contains mapping from subgraph nodes to original nodes
                subgraph.pos = graph.pos[subgraph.n_id]
                theta = random.uniform(0, 2 * math.pi)
                subgraph.pos = rotate_pos(subgraph.pos, theta)
            return subgraph
    
    def _random_walk_subgraph(self, graph: Data, walk_length: int = 13, num_walks: int = 4) -> Data:
        """Sample subgraph using random walks."""
        if graph.num_nodes == 0:
            return graph.clone()
        
        # Start from random nodes
        start_nodes = torch.randperm(graph.num_nodes)[:min(num_walks, graph.num_nodes)]
        visited_nodes = set()
        
        for start_node in start_nodes:
            current_node = start_node.item()
            visited_nodes.add(current_node)
            
            for _ in range(walk_length):
                # Get neighbors
                edge_mask = (graph.edge_index[0] == current_node)
                neighbors = graph.edge_index[1][edge_mask]
                
                if len(neighbors) == 0:
                    break
                
                # Random walk step
                next_node = neighbors[torch.randint(len(neighbors), (1,))].item()
                visited_nodes.add(next_node)
                current_node = next_node
                
                # Stop if we have enough nodes
                if len(visited_nodes) >= self.max_subgraph_nodes:
                    break
            
            if len(visited_nodes) >= self.max_subgraph_nodes:
                break
        
        # Create subgraph from visited nodes
        if len(visited_nodes) < self.min_subgraph_nodes:
            return self._fallback_subgraph(graph)
        
        subset = torch.tensor(list(visited_nodes), dtype=torch.long)
        return self._create_subgraph_from_nodes(graph, subset)
    
    def _create_subgraph_from_nodes(self, graph: Data, subset: torch.Tensor) -> Data:
        """Create subgraph from node subset."""
        if len(subset) == 0:
            return self._fallback_subgraph(graph)
        
        # Get subgraph edges
        edge_index, edge_attr = subgraph(
            subset, graph.edge_index, 
            edge_attr=graph.edge_attr if hasattr(graph, 'edge_attr') else None,
            relabel_nodes=True,
            num_nodes=graph.num_nodes
        )
        
        # Create new data object with properly subsetted features
        subgraph_data = Data(
            x=graph.x[subset] if graph.x is not None else None,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        
        # Handle positional features - subset them properly
        if hasattr(graph, 'pos') and graph.pos is not None:
            subgraph_data.pos = graph.pos[subset]  # Only keep positions of selected nodes
        
        # Handle target values - subset them properly 
        if hasattr(graph, 'y') and graph.y is not None:
            subgraph_data.y = graph.y[subset]  # Only keep targets of selected nodes
        
        # Copy other attributes selectively (avoid copying node-level attributes)
        node_level_attrs = {'x', 'edge_index', 'edge_attr', 'y', 'pos', 'num_nodes', 'num_edges'}
        for key, value in graph.items():
            if key not in node_level_attrs:
                # Only copy graph-level attributes, not node-level ones
                if not torch.is_tensor(value) or value.size(0) != graph.num_nodes:
                    setattr(subgraph_data, key, value)
        
        subgraph_data.orig_subset = subset  # Store original node indices for edge_weights mapping
        return subgraph_data
    
    def _truncate_subgraph(self, subgraph: Data, max_nodes: int) -> Data:
        """Truncate subgraph to maximum number of nodes."""
        if subgraph.num_nodes <= max_nodes:
            return subgraph
        
        # Randomly select nodes to keep
        keep_nodes = torch.randperm(subgraph.num_nodes)[:max_nodes]
        return self._create_subgraph_from_nodes(subgraph, keep_nodes)
    
    def _extract_subgraph_edge_weights(self, subgraph: Data, original_graph: Data) -> torch.Tensor:
        """Extract node weights (road segment weights) for subgraph nodes."""
        # Check if original graph has edge_weights
        if not hasattr(original_graph, 'edge_weights') or original_graph.edge_weights is None:
            raise ValueError(f"Original graph missing 'edge_weights' attribute. Available attributes: {list(original_graph.keys())}")

        # Neighbor sampling: use n_id
        if hasattr(subgraph, 'n_id') and subgraph.n_id is not None:
            return original_graph.edge_weights[subgraph.n_id]
        # Random walk: use mapping from subgraph nodes to original nodes
        elif hasattr(subgraph, 'orig_subset') and subgraph.orig_subset is not None:
            return original_graph.edge_weights[subgraph.orig_subset]
        # Fallback: if subgraph nodes are a subset of original nodes in order
        elif hasattr(subgraph, 'x') and subgraph.x.shape[0] <= original_graph.edge_weights.shape[0]:
            # Try to use the subset indices if stored
            if hasattr(subgraph, 'subset_indices'):
                return original_graph.edge_weights[subgraph.subset_indices]
            else:
                # If not, just take the first N weights (not always correct!)
                return original_graph.edge_weights[:subgraph.x.shape[0]]
        else:
            raise ValueError(f"Cannot map subgraph nodes to original graph for edge_weights. Subgraph attributes: {list(subgraph.keys())}")
        
    def _fallback_subgraph(self, graph: Data) -> Data:
        """Fallback subgraph for small graphs."""
        return graph.clone()
    
    def __len__(self):
        return len(self.subgraph_mapping)
    
    def __getitem__(self, idx):
        """Generate subgraph on-the-fly using neighbor sampling."""
        mapping = self.subgraph_mapping[idx]
        
        # Load the original graph on-demand (direct indexing)
        original_graph = self.original_dataset[mapping['original_graph_idx']]
        
        # Generate the subgraph dynamically (fresh every epoch)
        subgraph = self._sample_subgraphs_from_single_graph(
            graph=original_graph,
            subgraph_idx=mapping['subgraph_idx']
        )
        
        # Extract edge weights for subgraph (no re-normalization)
        subgraph.edge_weights = self._extract_subgraph_edge_weights(subgraph, original_graph)
        
        # Apply feature filtering to the subgraph
        if hasattr(self, 'node_feature_filter') and self.node_feature_filter is not None:
            # Only filter if number of features is greater than max index in filter
            if subgraph.x.shape[1] > max(self.node_feature_filter):
                subgraph.x = subgraph.x[:, self.node_feature_filter]
        
        return subgraph
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

def load_metadata_from_disk(data, metadata_path):

    city_data = json.load(open(metadata_path, 'r'))
    
    data['path'].extend(city_data['path'])
    data['policy_region'].extend(city_data['policy_region'])
    data['scenario'].extend(city_data['scenario'])
    data['city'].extend(city_data['city'])


def estimate_average_graph_memory(graph_loader, num_samples=5):
    gc.collect()
    torch.cuda.empty_cache()

    process = psutil.Process()
    mem_before = process.memory_info().rss

    memory_usages = []
    iterator = iter(graph_loader)

    for _ in range(num_samples):
        _ = next(iterator)  # Load one graph
        mem_after = process.memory_info().rss
        memory_usages.append(mem_after - mem_before)
        mem_before = mem_after

    avg_memory = sum(memory_usages) / len(memory_usages)
    print(f"Average memory per graph: {avg_memory / (1024 ** 2):.2f} MB")
    return avg_memory