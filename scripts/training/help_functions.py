import os
import sys
import copy
import random
import joblib
import subprocess

import numpy as np
from tqdm import tqdm
import wandb
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from collections import Counter


# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.gnn_io import *
from gnn.models.point_net_transf_gat import PointNetTransfGAT
from gnn.models.gcn import GCN, GCN2
from gnn.models.gat import GAT
from gnn.models.trans_conv import TransConv
from gnn.models.pnc import PNC
from gnn.models.fc_nn import FC_NN
from gnn.models.eign import EIGNLaplacianConv
from gnn.models.graphSAGE import GraphSAGE
from gnn.models.xgboost import XGBoostModel
from data_preprocessing.process_simulations_for_gnn import EdgeFeatures, use_allowed_modes

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

def create_hex_size_balanced_sampler(dataset):
    """
    Create a sampler that ensures balanced hex_size distribution in batches.
    """
    # Extract hex_sizes from all data points
    hex_sizes = [data.hex_size for data in dataset]
    hex_size_counts = Counter(hex_sizes)
    
    print(f"Hex size distribution: {dict(hex_size_counts)}")
    
    # Calculate weights to balance hex_sizes
    total_samples = len(dataset)
    num_classes = len(hex_size_counts)
    
    # Weight calculation: inverse frequency
    weights = []
    for data in dataset:
        hex_size = data.hex_size
        weight = total_samples / (num_classes * hex_size_counts[hex_size])
        weights.append(weight)
    
    # Create weighted random sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler

def prepare_data_with_graph_features(datalist, batch_size, path_to_save_dataloader, use_all_features, use_bootstrapping, is_eign=False, split_mode="full"):
    print(f"Starting prepare_data_with_graph_features with {len(datalist)} items")
    
    try:

        print("Splitting into subsets...")

        if split_mode == "full":
            # Full train/val/test split for transductive learning
            if use_bootstrapping:
                train_set, valid_set, test_set = split_into_subsets_with_bootstrapping(dataset=datalist, test_ratio=0.1, bootstrap_seed=4)
            else:
                train_set, valid_set, test_set = split_into_subsets(dataset=datalist, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05)
            print(f"Split complete. Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")
        else:  # split_mode == "train_val_only"
            # Train/val split only for inductive learning
            if use_bootstrapping:
                train_set, valid_set = split_into_subsets_with_bootstrapping_train_val_only(dataset=datalist, bootstrap_seed=4)
            else:
                train_set, valid_set = split_into_subsets_train_val_only(dataset=datalist, train_ratio=0.85, val_ratio=0.15)
            test_set = None
            print(f"Split complete. Train: {len(train_set)}, Valid: {len(valid_set)}, Test: None (using unseen data)")

        if use_all_features:
            node_features = [feat.name for feat in EdgeFeatures]
            if not use_allowed_modes:
                node_features = [feat for feat in node_features if "ALLOWED_MODE" not in feat]
        else:
            # Most important features (from ablation study)
            node_features = ["VOL_BASE_CASE",
                             "CAPACITY_BASE_CASE",
                             "CAPACITY_REDUCTION",
                             "FREESPEED",
                             "LENGTH"]
        
        print("Normalizing train set and fitting global scalers...")
        train_set_normalized, scalers_train = normalize_dataset(dataset_input=train_set, node_features=node_features)
        print("Train set normalized")      
        
        print("Applying global scalers to validation set...")
        valid_set_normalized= apply_global_scaler(data_list=valid_set, scaler=scalers_train['x_scaler'], node_features=node_features)
        print("Validation set normalized")
        
        if split_mode == "full":
            print("Applying global scalers to test set...")
            test_set_normalized = apply_global_scaler(data_list=test_set, scaler=scalers_train['x_scaler'], node_features=node_features)
            print("Test set normalized")
        
        print("Creating train loader with balanced hex_size sampling...")
        train_sampler = create_hex_size_balanced_sampler(train_set_normalized)
        train_loader = DataLoader(dataset=train_set_normalized, batch_size=batch_size, sampler=train_sampler, num_workers=4, prefetch_factor=2, pin_memory=True, collate_fn=collate_fn, worker_init_fn=seed_worker)
        print("Train loader created")
        
        print("Creating validation loader...")
        val_loader = DataLoader(dataset=valid_set_normalized, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn, worker_init_fn=seed_worker)
        print("Validation loader created")
        
        if split_mode == "full":
            print("Creating test loader...")
            test_loader = DataLoader(dataset=test_set_normalized, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn, worker_init_fn=seed_worker)
            print("Test loader created")
            
            save_dataloader(test_loader, path_to_save_dataloader + 'test_dl.pt')
            save_dataloader_params(test_loader, path_to_save_dataloader + 'test_loader_params.json')
        
        joblib.dump(scalers_train['x_scaler'], os.path.join(path_to_save_dataloader, 'train_x_scaler.pkl'))
        if not is_eign:
            joblib.dump(scalers_train['pos_scaler'], os.path.join(path_to_save_dataloader, 'train_pos_scaler.pkl'))
        # joblib.dump(scalers_train['modestats_scaler'], os.path.join(path_to_save_dataloader, 'train_mode_stats_scaler.pkl'))

        #joblib.dump(scalers_validation['x_scaler'], os.path.join(path_to_save_dataloader, 'validation_x_scaler.pkl'))
        #if not is_eign:
            #joblib.dump(scalers_validation['pos_scaler'], os.path.join(path_to_save_dataloader, 'validation_pos_scaler.pkl'))
        # joblib.dump(scalers_validation['modestats_scaler'], os.path.join(path_to_save_dataloader, 'validation_mode_stats_scaler.pkl'))

        #joblib.dump(scalers_test['x_scaler'], os.path.join(path_to_save_dataloader, 'test_x_scaler.pkl'))
        #if not is_eign:
            #joblib.dump(scalers_test['pos_scaler'], os.path.join(path_to_save_dataloader, 'test_pos_scaler.pkl'))
        # joblib.dump(scalers_test['modestats_scaler'], os.path.join(path_to_save_dataloader, 'test_mode_stats_scaler.pkl'))  
        
        print("Dataloaders and scalers saved")
        
        return train_loader, val_loader, scalers_train
    
    except Exception as e:
        print(f"Error in prepare_data_with_graph_features: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def prepare_test_dataloader_from_unseen(unseen_datalist, scalers_train, batch_size, use_all_features, path_to_save_dataloader, is_eign=False):
    """
    Prepare test dataloader from unseen data using pre-fitted scalers from training data.
    
    Args:
        unseen_datalist: List of data objects from unseen cities
        scalers_train: Dictionary of scalers fitted on training data
        batch_size: Batch size for DataLoader
        use_all_features: Whether to use all features or subset
        path_to_save_dataloader: Path to save dataloader and parameters
        is_eign: Whether using EIGN model architecture
    
    Returns:
        test_loader: DataLoader for unseen test data
    """
    print(f"Preparing test dataloader from {len(unseen_datalist)} unseen items")
    
    try:
        if use_all_features:
            node_features = [feat.name for feat in EdgeFeatures]
            if not use_allowed_modes:
                node_features = [feat for feat in node_features if "ALLOWED_MODE" not in feat]
        else:
            # Most important features (from ablation study)
            node_features = ["VOL_BASE_CASE",
                             "CAPACITY_BASE_CASE", 
                             "CAPACITY_REDUCTION",
                             "FREESPEED",
                             "LENGTH"]
        
        print("Applying training scalers to unseen test data...")
        test_set_normalized = apply_global_scaler(data_list=unseen_datalist, 
                                                  scaler=scalers_train['x_scaler'], 
                                                  node_features=node_features)
        print("Unseen test data normalized")
        
        print("Creating test loader for unseen data...")
        test_loader = DataLoader(dataset=test_set_normalized, 
                                batch_size=batch_size, 
                                shuffle=False,  # No shuffling for test evaluation
                                num_workers=4, 
                                pin_memory=True,
                                collate_fn=collate_fn, 
                                worker_init_fn=seed_worker)
        print("Test loader for unseen data created")
        
        # Save test dataloader and parameters (consistent with transductive case)
        save_dataloader(test_loader, path_to_save_dataloader + 'test_dl_unseen.pt')
        save_dataloader_params(test_loader, path_to_save_dataloader + 'test_loader_unseen_params.json')
        print("Unseen test dataloader and parameters saved")
        
        return test_loader
    
    except Exception as e:
        print(f"Error in prepare_test_dataloader_from_unseen: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
        
def normalize_dataset(dataset_input, node_features, is_eign=False):
    data_list = [copy.deepcopy(dataset_input.dataset[idx]) for idx in dataset_input.indices]

    print("Fitting and normalizing x features...")
    normalized_data_list, x_scaler = normalize_x_features_global_inductive(data_list, node_features)
    print("x features normalized")
    
    if is_eign:
        print("Fitting and normalizing x_signed features...")
        normalized_data_list, x_signed_scaler = normalize_x_signed_features_batched(
            normalized_data_list
        )
        print("x_signed features normalized")
    else:
        print("Fitting and normalizing pos features...")
        normalized_data_list, pos_scaler = normalize_pos_features_batched(normalized_data_list)
        print("Pos features normalized")
        
    # print("Fitting and normalizing modestats features...")
    # normalized_data_list, modestats_scaler = normalize_modestats_features_batched(normalized_data_list)
    # print("Modestats features normalized")
    
    scalers_dict = {
        "x_scaler": x_scaler,
        "x_signed_scaler": x_signed_scaler,
    } if is_eign else {
        "x_scaler": x_scaler,
        "pos_scaler": pos_scaler,
        # "modestats_scaler": modestats_scaler
    }
    return normalized_data_list, scalers_dict


def normalize_pos_features_batched(data_list, batch_size=1000):
    scaler = StandardScaler()
    
    # Get number of nodes in the graph
    num_nodes = data_list[0].x.shape[0]

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
            data.pos = torch.tensor(pos_normalized.reshape(num_nodes, 3, 2), dtype=data.pos.dtype)
    
    return data_list, scaler

def normalize_modestats_features_batched(data_list, batch_size=1000):
    scaler = StandardScaler()
    
    # First pass: Fit the scaler
    for i in tqdm(range(0, len(data_list), batch_size), desc="Fitting scaler"):
        batch = data_list[i:i+batch_size]
        batch_modestats = np.vstack([data.mode_stats.numpy().reshape(1, -1) for data in batch])
        scaler.partial_fit(batch_modestats)
    
    # Second pass: Transform the data
    for i in tqdm(range(0, len(data_list), batch_size), desc="Normalizing modestats features"):
        batch = data_list[i:i+batch_size]
        for data in batch:
            modestats_reshaped = data.mode_stats.numpy().reshape(1, -1)
            modestats_normalized = scaler.transform(modestats_reshaped)
            data.mode_stats = torch.tensor(modestats_normalized.reshape(6, 2), dtype=torch.float32)
    
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
                       EdgeFeatures.CAPACITY_REDUCTION,
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
        return GraphSAGE(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "gcn":
        return GCN(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "gcn2":
        return GCN2(**common_kwargs, **model_kwargs).to(device)
    
    elif gnn_arch == "gat":
        return GAT(**common_kwargs, **model_kwargs).to(device)
    
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
    else:
        raise ValueError(f"Unknown architecture: {gnn_arch}")

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

def normalize_x_features_global_inductive(data_list, node_features, batch_size=100):
    """
    Global normalization across ALL cities for inductive learning with batching.
    Model learns to handle the natural variation between cities.
    """
    scaler = StandardScaler()
    
    # Continuous features to normalize
    continuous_feat = [EdgeFeatures.VOL_BASE_CASE,
                       EdgeFeatures.CAPACITY_BASE_CASE,
                       EdgeFeatures.CAPACITY_REDUCTION,
                       EdgeFeatures.FREESPEED,
                       EdgeFeatures.LENGTH]
    
    # First pass: Fit the scaler on ALL cities using batching
    print("Fitting global scaler across all cities...")
    for i in tqdm(range(0, len(data_list), batch_size), desc="Fitting scaler"):
        batch = data_list[i:i+batch_size]
        batch_x = np.vstack([data.x[:,continuous_feat].numpy() for data in batch])
        scaler.partial_fit(batch_x)
    
    print(f"Global scaler fitted on features from all cities")
    print(f"Feature means: {scaler.mean_}")
    print(f"Feature scales: {scaler.scale_}")
    
    # Second pass: Transform the data using batching
    for i in tqdm(range(0, len(data_list), batch_size), desc="Normalizing x features"):
        batch = data_list[i:i+batch_size]
        batch_x = np.vstack([data.x[:,continuous_feat].numpy() for data in batch])
        batch_x_normalized = scaler.transform(batch_x)
        
        # Put normalized features back with correct indexing for different graph sizes
        start_idx = 0
        for data in batch:
            num_nodes = data.x.shape[0]  # Actual size of THIS graph
            end_idx = start_idx + num_nodes
            data.x[:,continuous_feat] = torch.tensor(
                batch_x_normalized[start_idx:end_idx], dtype=data.x.dtype)
            start_idx = end_idx  # Move to next graph's position
    
    # Filter features to keep only selected ones
    node_feature_filter = [EdgeFeatures[feature].value for feature in node_features]
    for data in data_list:
        data.x = data.x[:, node_feature_filter]
    
    # One-hot encode highway if needed
    if "HIGHWAY" in node_features:
        one_hot_highway(data_list, idx=node_features.index("HIGHWAY"))
    
    return data_list, scaler

def normalize_x_features_with_scaler_global_inductive(data_list, node_features, x_scaler, batch_size=100):
    """
    Normalize the continuous node features with a given scaler.
    Categorical features (Allowed Modes) are left as booleans (0 or 1).
    'HIGHWAY' feature is one-hot encoded.

    Finally, features are filtered to only include the ones specified in node_features.
    
    FIXED: Properly handles graphs with different numbers of nodes.
    """

    # Continuous features to normalize
    continuous_feat = [EdgeFeatures.VOL_BASE_CASE,
                       EdgeFeatures.CAPACITY_BASE_CASE,
                       EdgeFeatures.CAPACITY_REDUCTION,
                       EdgeFeatures.FREESPEED,
                       EdgeFeatures.LENGTH]
    
    # Transform the data using batching with proper indexing
    for i in tqdm(range(0, len(data_list), batch_size), desc="Normalizing x features"):
        batch = data_list[i:i+batch_size]
        batch_x = np.vstack([data.x[:,continuous_feat].numpy() for data in batch])
        batch_x_normalized = x_scaler.transform(batch_x)
        
        # Put normalized features back with correct indexing for different graph sizes
        start_idx = 0
        for data in batch:
            num_nodes = data.x.shape[0]  # Actual size of THIS graph
            end_idx = start_idx + num_nodes
            data.x[:,continuous_feat] = torch.tensor(
                batch_x_normalized[start_idx:end_idx], dtype=data.x.dtype)
            start_idx = end_idx  # Move to next graph's position

    # Filter features to keep only selected ones
    node_feature_filter = [EdgeFeatures[feature].value for feature in node_features]
    for data in data_list:
        data.x = data.x[:, node_feature_filter]

    # One-hot encode highway if needed
    if "HIGHWAY" in node_features:
        one_hot_highway(data_list, idx=node_features.index("HIGHWAY"))
    
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

def apply_global_scaler(data_list, scaler, node_features):
    """
    Apply the global scaler to the data list.
    Uses the same logic as normalize_x_features_with_scaler_global_inductive.
    Handles both regular lists and Subset objects.
    """
    # Convert Subset to list if needed
    if hasattr(data_list, 'indices'):  # It's a Subset object
        data_list_actual = [data_list[i] for i in range(len(data_list))]
    else:
        data_list_actual = data_list
    
    return normalize_x_features_with_scaler_global_inductive(data_list_actual, node_features, scaler)


def verify_batch_distribution(dataloader, num_batches_to_check=3):
    """
    Verify that batches have balanced hex_size distribution.
    """
    print("\n=== Verifying Batch Distribution ===")
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches_to_check:
            break
            
        hex_sizes = batch.hex_size.tolist()
        hex_distribution = Counter(hex_sizes)
        
        print(f"Batch {i+1}: {dict(hex_distribution)}")
        
        # Calculate percentages
        total = len(hex_sizes)
        percentages = {k: f"{(v/total)*100:.1f}%" for k, v in hex_distribution.items()}
        print(f"  Percentages: {percentages}")
    
    print("=" * 40)