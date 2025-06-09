import json
import random

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Subset
from torch_geometric.data import Batch

def split_into_subsets(dataset, train_ratio, val_ratio, test_ratio, shuffle_seed=42):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    
    dataset_length = len(dataset)
    print(f"Total dataset length: {dataset_length}")

    # Randomly shuffle the dataset
    random.Random(shuffle_seed).shuffle(dataset)
    
    # Calculate split indices
    train_split_idx = int(dataset_length * train_ratio)
    val_split_idx = train_split_idx + int(dataset_length * val_ratio)
    
    # Create indices for each subset
    train_indices = range(0, train_split_idx)
    val_indices = range(train_split_idx, val_split_idx)
    test_indices = range(val_split_idx, dataset_length)
    
    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    print(f"Training subset length: {len(train_subset)}")
    print(f"Validation subset length: {len(val_subset)}")
    print(f"Test subset length: {len(test_subset)}")
    
    return train_subset, val_subset, test_subset

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