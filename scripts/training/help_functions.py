import os
import sys
import copy
import random
import json
import joblib
import subprocess
from collections import defaultdict
from pathlib import Path
from functools import partial
from typing import Optional

import wandb
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import Subset

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
# Module-level defaults for feature selection (NOT command-line flags)
# Only --use_destination_activity is a command-line flag (in finetune_models.py, etc.)
use_allowed_modes = True  # Module constant: Include ALLOWED_MODE features (11-19)
use_destination_activity = False  # Module default: Excludes features 20-27. Overridden by --use_destination_activity flag.
use_highway = True  # Module constant: Include HIGHWAY features (4-9)
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


def _sanitize_string_fragment(value: str) -> str:
    """Return a filesystem and logging friendly fragment."""
    return (
        str(value)
        .replace(" ", "-")
        .replace("/", "-")
        .replace("\\", "-")
    )


def build_unique_model_description(
    run_name: str,
    cities,
    start_from_scratch: bool,
    run_variant: Optional[str] = None,
) -> str:
    """
    Compose a unique model description that records city coverage, whether the run
    reuses checkpoints, and the parent run name.
    """
    if isinstance(cities, str):
        city_list = [c.strip() for c in cities.split(",") if c.strip()]
    else:
        city_list = [str(city).strip() for city in cities if str(city).strip()]

    if not city_list:
        city_list = ["unknown-city"]

    sanitized_cities_list = [_sanitize_string_fragment(city) for city in city_list]
    sanitized_cities = "-".join(sanitized_cities_list)
    variant_fragment = _sanitize_string_fragment(
        run_variant if run_variant is not None else ("finetune" if not start_from_scratch else "scratch")
    )
    parent_fragment = _sanitize_string_fragment(run_name or "unknown-parent")

    include_city_fragment = bool(sanitized_cities)
    if include_city_fragment and variant_fragment:
        # If every city string already appears inside the variant fragment, avoid duplicating it.
        lowered_variant = variant_fragment.lower()
        if all(city and city.lower() in lowered_variant for city in sanitized_cities_list):
            include_city_fragment = False

    # If variant fragment already matches the parent fragment, skip adding parent
    # to avoid duplication (e.g., "scratch_erlangen_train25_val2" should not become "scratch_erlangen_train25_val2__parent-scratch_erlangen_train25_val2")
    include_parent_fragment = True
    if variant_fragment and parent_fragment:
        lowered_variant = variant_fragment.lower()
        lowered_parent = parent_fragment.lower()
        # Skip parent if variant equals parent (exact match only to avoid false positives)
        if lowered_variant == lowered_parent:
            include_parent_fragment = False

    fragments = [fragment for fragment in (
        variant_fragment,
        sanitized_cities if include_city_fragment else "",
        f"parent-{parent_fragment}" if include_parent_fragment else "",
    ) if fragment]

    return "__".join(fragment for fragment in fragments if fragment)

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

def balanced_subset_by_city(data_dict, limit, seed=42):
    """Return a city-balanced subset of the data_dict limited to `limit` items."""
    total_items = len(data_dict['path'])
    if limit <= 0 or total_items <= limit:
        return data_dict

    rng = random.Random(seed)
    city_to_indices = defaultdict(list)
    for idx, city in enumerate(data_dict['city']):
        city_to_indices[city].append(idx)

    for indices in city_to_indices.values():
        rng.shuffle(indices)

    selected_indices = []
    cities = list(city_to_indices.keys())
    rng.shuffle(cities)

    while len(selected_indices) < limit and cities:
        next_round_cities = []
        for city in cities:
            indices = city_to_indices[city]
            if indices:
                selected_indices.append(indices.pop())
                if len(selected_indices) >= limit:
                    break
            if indices:
                next_round_cities.append(city)
        cities = next_round_cities

    selected_indices_set = set(selected_indices)
    balanced_data = {k: [v[i] for i in range(total_items) if i in selected_indices_set]
                     for k, v in data_dict.items()}
    print(f"[DEBUG] Applied city-balanced cap to {total_items} → {len(selected_indices_set)} samples.")
    return balanced_data

def combine_and_split_data_dicts(train_data, val_data, test_data, limit, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05, seed=42):
    """
    Combine data from train, val, and test dictionaries, limit to `limit` total items,
    then split into train/val/test according to the specified ratios.
    
    Returns:
        train_data, val_data, test_data: Split data dictionaries
    """
    from sklearn.model_selection import train_test_split
    
    # Combine all data
    combined_data = {
        'path': train_data['path'].copy(),
        'policy_region': train_data['policy_region'].copy(),
        'scenario': train_data['scenario'].copy(),
        'city': train_data['city'].copy()
    }
    
    if val_data is not None:
        combined_data['path'].extend(val_data['path'])
        combined_data['policy_region'].extend(val_data['policy_region'])
        combined_data['scenario'].extend(val_data['scenario'])
        combined_data['city'].extend(val_data['city'])
    
    if test_data is not None:
        combined_data['path'].extend(test_data['path'])
        combined_data['policy_region'].extend(test_data['policy_region'])
        combined_data['scenario'].extend(test_data['scenario'])
        combined_data['city'].extend(test_data['city'])
    
    total_items = len(combined_data['path'])
    print(f"[DEBUG] Combined data from all cities: {total_items} total items")
    
    # Apply limit if needed
    if limit > 0 and total_items > limit:
        combined_data = balanced_subset_by_city(combined_data, limit, seed=seed)
        total_items = len(combined_data['path'])
        print(f"[DEBUG] After limiting: {total_items} items")
    
    # Create labels for stratification
    labels = [f"{city}_{policy_region}" for city, policy_region in zip(combined_data['city'], combined_data['policy_region'])]
    indices = list(range(total_items))
    
    # First split: train vs (val+test)
    try:
        train_indices, temp_indices = train_test_split(
            indices,
            test_size=(val_ratio + test_ratio),
            random_state=seed,
            stratify=labels
        )
    except ValueError:
        print("Warning: Stratified split not possible. Falling back to unstratified split.")
        train_indices, temp_indices = train_test_split(
            indices,
            test_size=(val_ratio + test_ratio),
            random_state=seed,
            stratify=None
        )
    
    # Second split: val vs test
    temp_labels = [labels[i] for i in temp_indices]
    val_size = val_ratio / (val_ratio + test_ratio)
    try:
        val_indices_temp, test_indices_temp = train_test_split(
            temp_indices,
            test_size=(1 - val_size),
            random_state=seed,
            stratify=temp_labels
        )
    except ValueError:
        print("Warning: Stratified val/test split not possible. Falling back to unstratified split.")
        val_indices_temp, test_indices_temp = train_test_split(
            temp_indices,
            test_size=(1 - val_size),
            random_state=seed,
            stratify=None
        )
    
    # Create split data dictionaries
    train_data_split = {k: [v[i] for i in train_indices] for k, v in combined_data.items()}
    val_data_split = {k: [v[i] for i in val_indices_temp] for k, v in combined_data.items()}
    test_data_split = {k: [v[i] for i in test_indices_temp] for k, v in combined_data.items()}
    
    print(f"[DEBUG] Split into train: {len(train_data_split['path'])}, val: {len(val_data_split['path'])}, test: {len(test_data_split['path'])}")
    
    return train_data_split, val_data_split, test_data_split

def setup_wandb(args):
    wandb.login()
    wandb.init(project=args['project_name'], name=args['unique_model_description'],
               config={k: v for k, v in args.items() if k not in ['project_name', 'unique_model_description', 'model_kwargs']})
    # Define metrics after wandb.init() to ensure hit rates are tracked
    setup_wandb_metrics()
    return wandb.config

def setup_wandb_metrics():
    wandb.define_metric("epoch") # Custom X-axis
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")
    wandb.define_metric("r^2", step_metric="epoch")
    wandb.define_metric("spearman", step_metric="epoch")
    wandb.define_metric("pearson", step_metric="epoch")
    # Hit rate metrics
    wandb.define_metric("top_1_hit_rate", step_metric="epoch")
    wandb.define_metric("closest_to_zero_1_hit_rate", step_metric="epoch")
    wandb.define_metric("minus_top_1_hit_rate", step_metric="epoch")
    wandb.define_metric("top_5_hit_rate", step_metric="epoch")
    wandb.define_metric("closest_to_zero_5_hit_rate", step_metric="epoch")
    wandb.define_metric("minus_top_5_hit_rate", step_metric="epoch")

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
    combined_data_list: Subset for fitting scaler (should be train only for inductive, train+val for transductive)
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
    Fit scaler on combined_data_list (train only for inductive, train+val for transductive),
    but apply normalization only to train set.
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
                                     aug_pos_rotation, aug_feature_noise, aug_node_masking_probability=0.0,
                                     use_destination_activity_param=None):
    """
    Prepare data with graph features.
    
    Args:
        use_destination_activity_param: If None, uses module-level use_destination_activity. If provided, overrides it.
                                       When False, excludes features 20-27 (destination/activity features with NaNs).
    """
    # Use parameter if provided, otherwise use module-level default
    _use_destination_activity = use_destination_activity_param if use_destination_activity_param is not None else use_destination_activity 
    
    print(f"Preparing data with {len(train_data['path']) + (len(val_data['path']) if val_data is not None else 0) + (len(test_data['path']) if test_data is not None else 0)} items")
    
    print("Splitting into subsets...")

    entire_set, train_set, valid_set, test_set = load_data_and_split_into_subsets(train_data=train_data, val_data=val_data, test_data=test_data,
                                                                         train_ratio=0.8, val_ratio=0.15, test_ratio=0.05)
    
    # TODO: Change if needed!
    # Transductive: TRAIN + VAL + TEST Scaler
    if use_inductive_variant == False:
        combined_indices = train_set.indices + valid_set.indices + test_set.indices
    
    # Inductive: ONLY TRAIN Scaler (to avoid data leakage - validation data should not influence training)
    else:
        combined_indices = train_set.indices  # Only use training data for scaler fitting
    
    combined_norm_set = Subset(entire_set, combined_indices)
    
    print(f"Split complete. Train: {len(train_set)}, Valid: {len(valid_set)}, Test: {len(test_set)}")

    if use_all_features:
        node_features = []
        for feat in EdgeFeatures:
            name = feat.name
            # Exclude features 20-27 (activity/destination features with NaNs) if use_destination_activity=False
            # This is controlled by the --use_destination_activity flag (ONLY flag for feature selection)
            # Default: False → excludes features 20-27 → uses features 0-19 only
            if not _use_destination_activity and feat.value >= 20:  # Features 20-27: HOME, WORK, EDUCATION, LEISURE, SHOP, OTHER, OUTSIDE, IS_IN_EQASIM_TRIPS
                continue
            # use_allowed_modes and use_highway are module-level constants (NOT command-line flags)
            if not use_allowed_modes and name.startswith("ALLOWED_MODE"):
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

    print("Normalizing train set...")
    train_set_normalized, scalers_train = normalize_dataset(train_data_list=train_set, combined_data_list=combined_norm_set)
    print("Train set normalized")      
    
    print("Normalizing validation set ...")
    valid_set_normalized = normalize_dataset_with_scaler(dataset_input=valid_set, scalers=scalers_train)
    print("Validation set normalized")
    
    print("Normalizing test set ...")
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
    
    if len(valid_set_normalized) > 0:
        print("Creating validation loader...")
        val_loader = create_split_dataloader(
            data_subset=valid_set_normalized,
            batch_size=batch_size,
            use_weighted_batches=use_weighted_batches,
            collate_fn_type=collate_fn_with_no_aug)  # Always without augmentation
        print("Validation loader created")
    else:
        val_loader = None
        print("Validation subset empty; skipping validation loader.")
    
    if len(test_set_normalized) > 0:
        print("Creating test loader...")
        test_loader = create_split_dataloader(
            data_subset=test_set_normalized,
            batch_size=batch_size,
            use_weighted_batches=use_weighted_batches,
            collate_fn_type=collate_fn_with_no_aug)  # Always without augmentation
        
        if use_inductive_variant == False:
            save_dataloader(test_loader, path_to_save_dataloader + 'test_dl.pt')
            save_dataloader_params(test_loader, path_to_save_dataloader + 'test_loader_params.json')
            print("Test Dataloader saved since Transductive Variant.")
        
        print("Test loader created")
    else:
        test_loader = None
        print("Test subset empty; skipping test loader.")
    
    # SAVE SCALERS
    joblib.dump(scalers_train['x_scaler'], os.path.join(path_to_save_dataloader, 'train_x_scaler.pkl'))
    print("Scaler saved!")
    
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

    # Allow model_kwargs to override in_channels if specified
    # This is needed when actual data feature count differs from config
    in_channels = model_kwargs.get('in_channels', config.in_channels)
    
    common_kwargs = {
        "in_channels": in_channels,
        "out_channels": config.out_channels,
        "use_dropout": config.use_dropout,
        "dropout": config.dropout,
        "dtype": torch.float32,
        "log_to_wandb": True,
        "use_target_standardization": getattr(config, 'use_target_standardization', False),  # For backward compatibility
        "target_normalization": getattr(config, 'target_normalization', None)
    }
    
    # Remove in_channels from model_kwargs to avoid duplicate argument
    model_kwargs_clean = {k: v for k, v in model_kwargs.items() if k != 'in_channels'}
    
    if gnn_arch == "graphSAGE":
        return GraphSAGE(**common_kwargs, **model_kwargs_clean).to(device)
    
    elif gnn_arch == "gatv2":
        return GATv2(**common_kwargs, **model_kwargs_clean).to(device)
    
    elif gnn_arch == "trans_conv":
        return TransConv(**common_kwargs, **model_kwargs_clean).to(device)
        
    elif gnn_arch == "trans_encoder":
        return TransEncoder(**common_kwargs, **model_kwargs_clean).to(device)
        
    else:
        raise ValueError(f"Unknown architecture: {gnn_arch}")
    