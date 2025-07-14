import json
import random
import os

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Batch

class LazyGraphDataset(Dataset):
    """
    Dataset that loads graph files on-demand instead of keeping them in memory.
    Perfect for large datasets with 120k+ graphs.
    """
    def __init__(self, filepaths):
        self.filepaths = filepaths
        self._cache = {}  # Optional: cache recently used files
        
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        
        # Optional caching (remove if memory is still tight)
        if filepath in self._cache:
            return self._cache[filepath]
        
        # Load the batch file
        batch_data = torch.load(filepath, map_location='cpu')
        
        # If it's a batch of graphs, randomly select one
        if isinstance(batch_data, list):
            import random
            graph = random.choice(batch_data)
        else:
            graph = batch_data
            
        # Optional caching (limit cache size)
        if len(self._cache) < 100:  # Cache max 100 files
            self._cache[filepath] = graph
            
        return graph

def parse_filepath_metadata(filepath):
    """
    Parse city and hex_size from filepath string.
    Example: 'augsburg/datalist_batch_1.pt' -> city='augsburg'
    For hex_size, we'd need to load one graph from the batch to get this info.
    """
    city = os.path.basename(os.path.dirname(filepath))
    hex_size = os.path.basename(filepath).split('_')[2]
    return city,hex_size

def split_into_subsets_from_filepaths(filepaths, train_ratio, val_ratio, test_ratio=None, split_variant="uniform", shuffle_seed=43, split_mode="full"):
    """
    Split dataset when it contains filepaths instead of loaded graph objects.
    """
    if split_mode == "full":
        assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    elif split_mode == "train_val_only":
        assert train_ratio + val_ratio == 1, "Ratios must sum to 1"
    else:
        raise ValueError(f"Invalid split mode: {split_mode}")

    dataset_length = len(filepaths)
    print(f"Total dataset length: {dataset_length}")
    
    # Set random seed for reproducibility
    random.seed(shuffle_seed)
    
    if split_variant == "non_uniform":
        def get_key(filepath): return parse_filepath_metadata(filepath)[0]
    else:
        def get_key(filepath): return parse_filepath_metadata(filepath)
    
    # Group filepaths by city
    groups = {}
    for filepath in filepaths:
        key = get_key(filepath)
        if key not in groups:
            groups[key] = []
        groups[key].append(filepath)
    
    # Split each group
    train_paths, val_paths, test_paths = [], [], []
    
    for paths in groups.values():
        random.shuffle(paths)
        n_paths = len(paths)
        
        if split_mode == "full":
            train_end = int(n_paths * train_ratio)
            val_end = train_end + int(n_paths * val_ratio)
            train_paths.extend(paths[:train_end])
            val_paths.extend(paths[train_end:val_end])
            test_paths.extend(paths[val_end:])
        else:  # train_val_only
            train_end = int(n_paths * train_ratio)
            val_end = train_end + int(n_paths * val_ratio)
            train_paths.extend(paths[:train_end])
            val_paths.extend(paths[train_end:val_end])
    
    print(f"Training filepaths: {len(train_paths)}")
    print(f"Validation filepaths: {len(val_paths)}")
    
    if split_mode == "full":
        print(f"Test filepaths: {len(test_paths)}")
        return train_paths, val_paths, test_paths
    else:
        return train_paths, val_paths

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