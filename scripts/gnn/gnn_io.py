import json
import random

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Subset
from torch_geometric.data import Batch

def split_into_subsets(dataset, train_ratio, val_ratio, test_ratio=None, split_variant="uniform", shuffle_seed=43, split_mode="full"):

    if split_mode == "full":
        # Ensure the ratios sum to 1
        assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    elif split_mode == "train_val_only":
        assert train_ratio + val_ratio == 1, "Ratios must sum to 1"
    else:
        raise ValueError(f"Invalid split mode: {split_mode}")

    dataset_length = len(dataset)
    print(f"Total dataset length: {dataset_length}")
    
    # Set random seed for reproducibility
    random.seed(shuffle_seed)
    
    # Define grouping key based on split variant
    if split_variant == "non_uniform":
        def get_key(graph): return graph.city
    else:  # non_uniform
        def get_key(graph): return (graph.city, graph.hex_size)
    
    # Group graphs by key
    groups = {}
    for graph in dataset:
        key = get_key(graph)
        if key not in groups:
            groups[key] = []
        groups[key].append(graph)
    
    # Split each group
    train_indices, val_indices, test_indices = [], [], []
    
    for graphs in groups.values():
        random.shuffle(graphs)
        n_graphs = len(graphs)
        
        if split_mode == "full":
            train_end = int(n_graphs * train_ratio)
            val_end = train_end + int(n_graphs * val_ratio)
            train_indices.extend(graphs[:train_end])
            val_indices.extend(graphs[train_end:val_end])
            test_indices.extend(graphs[val_end:])
        else:  # train_val_only
            train_end = int(n_graphs * train_ratio)
            val_end = train_end + int(n_graphs * val_ratio)
            train_indices.extend(graphs[:train_end])
            val_indices.extend(graphs[train_end:val_end])
    
    # Convert Data objects to indices
    def get_indices(data_objects):
        indices = []
        for data_obj in data_objects:
            # Find the index of this data object in the original dataset
            for i, item in enumerate(dataset):
                if item is data_obj:
                    indices.append(i)
                    break
        return indices
    
    # Create subsets with integer indices
    train_subset = Subset(dataset, get_indices(train_indices))
    val_subset = Subset(dataset, get_indices(val_indices))
    
    print(f"Training subset length: {len(train_subset)}")
    print(f"Validation subset length: {len(val_subset)}")
    
    if split_mode == "full":
        test_subset = Subset(dataset, get_indices(test_indices))
        print(f"Test subset length: {len(test_subset)}")
        return train_subset, val_subset, test_subset
    else:
        return train_subset, val_subset

def split_into_subsets_with_bootstrapping(dataset, test_ratio=0.1, bootstrap_seed=0, shuffle_seed=42):
    
    dataset_length = len(dataset)
    print(f"Total dataset length: {dataset_length}")

    # Split the dataset into training and testing sets
    train_indices, test_indices = train_test_split(range(dataset_length), test_size=test_ratio, random_state=shuffle_seed)
    
    # Perform bootstrapping on the training set, OOB validation set
    rng = np.random.default_rng(seed=bootstrap_seed)
    train_indices_bootstrap = rng.choice(train_indices, size=len(train_indices), replace=True)
    oob_indices = list(set(train_indices) - set(train_indices_bootstrap))
    
    # Create subsets
    train_subset_bootstrap = Subset(dataset, train_indices_bootstrap)
    val_subset_oob = Subset(dataset, oob_indices)
    test_subset = Subset(dataset, test_indices)
    
    print(f"Bootstrapping unique samples: {len(set(train_indices_bootstrap))}")
    print(f"Training subset length: {len(train_subset_bootstrap)}")
    print(f"OOB Validation subset length: {len(val_subset_oob)}")
    print(f"Test subset length: {len(test_subset)}")
    
    return train_subset_bootstrap, val_subset_oob, test_subset

def split_into_subsets_with_bootstrapping_train_val_only(dataset, bootstrap_seed=0, shuffle_seed=42):
    """
    Bootstrap split for train/val only (no test set for inductive learning).
    Uses all available data for train/val when test set comes from separate unseen data.
    """
    
    dataset_length = len(dataset)
    print(f"Total dataset length for bootstrap train/val split: {dataset_length}")

    # No test split - use all data for train/val bootstrapping
    all_indices = list(range(dataset_length))
    
    # Perform bootstrapping on all available data
    rng = np.random.default_rng(seed=bootstrap_seed)
    train_indices_bootstrap = rng.choice(all_indices, size=dataset_length, replace=True)
    oob_indices = list(set(all_indices) - set(train_indices_bootstrap))
    
    # Create subsets
    train_subset_bootstrap = Subset(dataset, train_indices_bootstrap)
    val_subset_oob = Subset(dataset, oob_indices)
    
    print(f"Bootstrapping unique samples: {len(set(train_indices_bootstrap))}")
    print(f"Training subset length: {len(train_subset_bootstrap)}")
    print(f"OOB Validation subset length: {len(val_subset_oob)}")
    print("No test subset created - using separate unseen data for testing")
    
    return train_subset_bootstrap, val_subset_oob

def save_dataloader(dataloader, file_path):
    # Extract the dataset from the DataLoader
    dataset = dataloader.dataset
    # Save the dataset to the specified file path
    torch.save(dataset, file_path)

def save_dataloader_params(dataloader, file_path):
    params = {
        'batch_size': dataloader.batch_size,
        # 'shuffle': dataloader.shuffle,
        'collate_fn': dataloader.collate_fn.__name__  # Assuming collate_fn is a known function
    }
    with open(file_path, 'w') as f:
        json.dump(params, f)

def collate_fn(data_list):
    return Batch.from_data_list(data_list)

def print_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== Model Information ===")
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    print("=" * 30)