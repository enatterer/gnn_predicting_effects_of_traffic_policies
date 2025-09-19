import json
from functools import lru_cache

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Subset, Dataset
from torch_geometric.data import Batch
import math
import random
from functools import partial
from data_preprocessing.process_simulations_for_gnn import EdgeFeatures

def rotate_pos(pos, theta):
    """Rotate 2D positions by theta radians, with 30% chance of random reflection."""
    if pos is None or pos.shape[1] < 2:
        return pos
    rot_matrix = torch.tensor([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ], dtype=pos.dtype, device=pos.device)
    pos_xy = pos[:, :2] @ rot_matrix.T

    # With 30% probability, apply a random reflection
    if random.random() < 0.3:
        reflection_type = random.choice(['horizontal', 'vertical', 'diagonal1', 'diagonal2'])
        if reflection_type == 'horizontal':
            pos_xy[:, 1] = -pos_xy[:, 1]
        elif reflection_type == 'vertical':
            pos_xy[:, 0] = -pos_xy[:, 0]
        elif reflection_type == 'diagonal1':  # y = x
            pos_xy = pos_xy.flip(1)
        elif reflection_type == 'diagonal2':  # y = -x
            pos_xy = -pos_xy.flip(1)

    if pos.shape[1] > 2:
        pos_aug = torch.cat([pos_xy, pos[:, 2:]], dim=1)
    else:
        pos_aug = pos_xy
    return pos_aug

class GraphDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    @lru_cache(maxsize=4500)
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
def load_data_and_split_into_subsets(train_data, val_data, test_data, train_ratio, val_ratio, test_ratio=0, seed=42):
    
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    
    dataset_length = len(train_data['path']) + (len(val_data['path']) if val_data is not None else 0) + (len(test_data['path']) if test_data is not None else 0)
    print(f"Total dataset length: {dataset_length}")

    if test_data is None and val_data is None:
        paths = train_data['path']
        labels = [f"{city}_{policy_region}" for city, policy_region in zip(train_data['city'], train_data['policy_region'])]

        indices = list(range(len(paths)))
        train_indices, test_indices = train_test_split(indices, test_size=test_ratio, random_state=seed, stratify=labels)
        train_indices, val_indices = train_test_split(train_indices, test_size=val_ratio/(train_ratio + val_ratio), random_state=seed, stratify=[labels[i] for i in train_indices])

    else:
        
        train_indices = list(range(len(train_data['path'])))
        val_indices = list(range(len(train_data['path']), len(train_data['path']) + len(val_data['path'])))
        test_indices = list(range(len(train_data['path']) + len(val_data['path']), len(train_data['path']) + len(val_data['path']) + len(test_data['path'])))
        
        paths = train_data['path'] + val_data['path'] + test_data['path']
        train_labels = [f"{city}_{policy_region}" for city, policy_region in zip(train_data['city'], train_data['policy_region'])]
        val_labels = [f"{city}_{policy_region}" for city, policy_region in zip(val_data['city'], val_data['policy_region'])]
        test_labels = [f"{city}_{policy_region}" for city, policy_region in zip(test_data['city'], test_data['policy_region'])]
        labels = train_labels + val_labels + test_labels
        
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

def collate_fn(data_list, augment_pos_rotation=False, augment_edge_perturbation_prob=0.0, is_training=True, augment_feature_noise_prob=False, filtered_feature_mapping=None):
    # Extract only the middle position from pos [N, 3, 2] -> [N, 2]
    for data in data_list:
        if hasattr(data, 'pos') and data.pos is not None:
            if len(data.pos.shape) == 3 and data.pos.shape[1] == 3:
                # Extract middle position (index 1)
                data.pos = data.pos[:, 1, :].contiguous()
    
    # On-the-fly rotation augmentation
    if is_training and augment_pos_rotation:
        for data in data_list:
            if hasattr(data, 'pos') and data.pos is not None:
                theta = random.uniform(0, 2 * math.pi)
                data.pos = rotate_pos(data.pos, theta)
    
    # On-the-fly feature noise augmentation
    if is_training and augment_feature_noise_prob:
        for data in data_list:
            if hasattr(data, 'x') and data.x is not None:
                data.x = apply_gaussian_noise_to_features(data.x, filtered_feature_mapping) 
    
    # On-the-fly edge perturbation (random dropout on line graph edges)
    if is_training and augment_edge_perturbation_prob > 0:
        for data in data_list:
            if hasattr(data, 'edge_index') and data.edge_index is not None:
                edge_index = data.edge_index.clone()
                num_edges = edge_index.size(1)
                mask = torch.rand(num_edges, device=edge_index.device) > augment_edge_perturbation_prob
                
                # Ensure minimum connectivity - simplified approach
                min_edges = max(2, int(num_edges * 0.1))  # Keep at least 10% of edges
                if mask.sum() < min_edges:
                    # Randomly keep additional edges to meet minimum
                    false_indices = (~mask).nonzero(as_tuple=False).squeeze(1)
                    if len(false_indices) > 0:
                        additional_edges_needed = min_edges - mask.sum().item()
                        additional_edges_needed = min(additional_edges_needed, len(false_indices))
                        if additional_edges_needed > 0:
                            keep_indices = false_indices[torch.randperm(len(false_indices))[:additional_edges_needed]]
                            mask[keep_indices] = True
                
                data.edge_index = edge_index[:, mask]
                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    data.edge_attr = data.edge_attr[mask]
    
    return Batch.from_data_list(data_list)

def apply_gaussian_noise_to_features(x: torch.Tensor, filtered_feature_mapping: dict = None) -> torch.Tensor:
    """
    Apply Gaussian noise to normalized features using filtered feature mapping.
    If no mapping provided, skip noise application.
    """
    if filtered_feature_mapping is None:
        print("Warning: No filtered_feature_mapping provided. Skipping feature noise.")
        return x
    
    noise = torch.zeros_like(x)
    
    # Volume features
    if EdgeFeatures.VOL_BASE_CASE.value in filtered_feature_mapping:
        idx = filtered_feature_mapping[EdgeFeatures.VOL_BASE_CASE.value]
        if isinstance(idx, int) and idx < x.shape[1]:
            noise[:, idx] = torch.randn(x[:, idx].size(), device=x.device) * 0.1  # ±10%
    
    # Capacity features
    if EdgeFeatures.CAPACITY_BASE_CASE.value in filtered_feature_mapping:
        idx = filtered_feature_mapping[EdgeFeatures.CAPACITY_BASE_CASE.value]
        if isinstance(idx, int) and idx < x.shape[1]:
            noise[:, idx] = torch.randn(x[:, idx].size(), device=x.device) * 0.1  # ±10%
    
    # Length features
    if EdgeFeatures.LENGTH.value in filtered_feature_mapping:
        idx = filtered_feature_mapping[EdgeFeatures.LENGTH.value]
        if isinstance(idx, int) and idx < x.shape[1]:
            noise[:, idx] = torch.randn(x[:, idx].size(), device=x.device) * 0.05  # ±5%
    
    # Speed features
    if EdgeFeatures.FREESPEED.value in filtered_feature_mapping:
        idx = filtered_feature_mapping[EdgeFeatures.FREESPEED.value]
        if isinstance(idx, int) and idx < x.shape[1]:
            noise[:, idx] = torch.randn(x[:, idx].size(), device=x.device) * 0.05  # ±5%
    
    # Conditional noise for capacity reduction
    cap_red_idx = filtered_feature_mapping.get(EdgeFeatures.CAPACITY_REDUCTION.value)
    cap_base_idx = filtered_feature_mapping.get(EdgeFeatures.CAPACITY_BASE_CASE.value)
    
    if (cap_red_idx is not None and isinstance(cap_red_idx, int) and cap_red_idx < x.shape[1] and 
        cap_base_idx is not None and isinstance(cap_base_idx, int) and cap_base_idx < x.shape[1]):
        
        mask = x[:, cap_red_idx] == 1
        if mask.any():
            noise[mask, cap_base_idx] += torch.randn(mask.sum(), device=x.device) * 0.05  # Extra ±5% if policy applied
    
    # Apply noise and clamp to reasonable bounds
    x_noisy = x + noise
    
    return x_noisy

def collate_without_scaling(batch, node_feature_filter, augment_pos_rotation=False): #for transductive learning
    # Extract only the middle position from pos [N, 3, 2] -> [N, 2]
    for data in batch:
        if hasattr(data, 'pos') and data.pos is not None:
            if len(data.pos.shape) == 3 and data.pos.shape[1] == 3:
                # Extract middle position (index 1)
                data.pos = data.pos[:, 1, :].contiguous()
    
    # On-the-fly rotation augmentation
    if augment_pos_rotation:
        for data in batch:
            if hasattr(data, 'pos') and data.pos is not None:
                theta = random.uniform(0, 2 * math.pi)
                data.pos = rotate_pos(data.pos, theta)
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