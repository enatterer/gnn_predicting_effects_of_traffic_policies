import json
from functools import lru_cache

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Subset, Dataset
from torch_geometric.data import Batch

class GraphDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    @lru_cache(maxsize=8192)
    def get(self, idx):
        data = torch.load(self.paths[idx])
        return data
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slice indexing by creating a list of indices
            start, stop, step = idx.indices(len(self))
            return [self.get(i) for i in range(start, stop, step)]
        else:
            return self.get(idx)

# Use test_data for an exclusive test set, train and validation sets from train_data.
# Otherwise split into train, validation, and test sets from train_data.
def load_data_and_split_into_subsets(train_data, test_data, train_ratio, val_ratio, test_ratio=0, seed=42):
    
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    
    dataset_length = len(train_data['path']) + (len(test_data['path']) if test_data is not None else 0)
    print(f"Total dataset length: {dataset_length}")

    paths = train_data['path']
    labels = [f"{city}_{policy_region}" for city, policy_region in zip(train_data['city'], train_data['policy_region'])]

    if test_data is None:

        indices = list(range(len(paths)))
        train_indices, test_indices = train_test_split(indices, test_size=test_ratio, random_state=seed, stratify=labels)
        train_indices, val_indices = train_test_split(train_indices, test_size=val_ratio/(train_ratio + val_ratio), random_state=seed, stratify=[labels[i] for i in train_indices])

    else:

        indices = list(range(len(paths)))
        train_indices, val_indices = train_test_split(indices, test_size=val_ratio, random_state=seed, stratify=labels)
        
        paths = paths + test_data['path']
        test_labels = [f"{city}_{policy_region}" for city, policy_region in zip(test_data['city'], test_data['policy_region'])]
        labels = labels + test_labels
        test_indices = list(range(len(paths)))[len(test_labels):]

    # Create the dataset
    dataset = GraphDataset(paths, labels)
    
    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    print(f"Training subset length: {len(train_subset)}")
    print(f"Validation subset length: {len(val_subset)}")
    print(f"Test subset length: {len(test_subset)}")
    
    return dataset, train_subset, val_subset, test_subset

# TODO: Fix Later
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
        'collate_fn': dataloader.collate_fn.__name__ if hasattr(dataloader.collate_fn, '__name__') else str(type(dataloader.collate_fn))
    }
    with open(file_path, 'w') as f:
        json.dump(params, f)

def collate_fn(data_list):
    return Batch.from_data_list(data_list)

def scale_and_collate(batch, scaler, continuous_feat, node_feature_filter):
    
    # Large disconnected graph
    batch = Batch.from_data_list(batch)
    
    # Scale continuous x features
    x = batch.x[:, continuous_feat].numpy()
    x_normalized = scaler.transform(x)
    batch.x[:, continuous_feat] = torch.tensor(x_normalized, dtype=batch.x.dtype)

    # Filter node features
    batch.x = batch.x[:, node_feature_filter]
    
    return batch

def collate_without_scaling(batch, node_feature_filter):
    """
    Collate function that filters features but doesn't apply scaling
    since features are already normalized during preprocessing.
    """
    batch = Batch.from_data_list(batch)
    
    # Filter node features
    if node_feature_filter is not None:
        batch.x = batch.x[:, node_feature_filter]
    
    return batch

def print_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== Model Information ===")
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    print("=" * 30)