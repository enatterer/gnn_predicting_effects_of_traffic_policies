import math
import random
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse
from torch.utils.data import DataLoader, Dataset, Subset
from torch import Tensor
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import LineGraph
from shapely.geometry import LineString
import tqdm
import wandb
import copy

import os
import sys
import json
import joblib  # For saving the scaler

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_preprocessing.process_simulations_for_gnn import EdgeFeatures

class MyGeometricDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

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

class GNN_Loss:
    """
    Custom loss function for GNN that supports weighted loss computation.
    The road with highesst vol_base_case gets a weight of 1, and the rest are scaled accordingly (sample-wise).
    """
    
    def __init__(self, loss_fct, num_nodes, device, weighted=False):

        if loss_fct == 'mse':
            self.loss_fct = torch.nn.MSELoss(reduction='none' if weighted else 'mean').to(dtype=torch.float32).to(device)
        elif self.config.loss_fct == 'l1':
            self.loss_fct = torch.nn.L1Loss(reduction='none' if weighted else 'mean').to(dtype=torch.float32).to(device)
        else:
            raise ValueError(f"Loss function {loss_fct} not supported.")
        
        self.num_nodes = num_nodes
        self.device = device
        self.weighted = weighted

    def __call__(self, y_pred:Tensor, y_true:Tensor, x: np.ndarray = None) -> Tensor: # x is before normalization (unscaled)
        
        if self.weighted:

            loss = self.loss_fct(y_pred, y_true)
            weights = x[:, EdgeFeatures.VOL_BASE_CASE]

            # Normalize by the maximum value in each sample
            for i in range(weights.shape[0] // self.num_nodes):
                weights[i * self.num_nodes:(i + 1) * self.num_nodes] /= np.max(weights[i * self.num_nodes:(i + 1) * self.num_nodes])

            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
            return torch.mean(loss * weights.unsqueeze(1))

        else:
            return self.loss_fct(y_pred, y_true)
            
def int_list_to_string(lst: list, delimiter: str = ', ') -> str:
    """
    Converts a list of integers to a string representation with the specified delimiter.

    Parameters:
    lst (list[int]): The list of integers.
    delimiter (str): The delimiter used to separate the integers in the string. Default is ', '.

    Returns:
    str: The string representation of the list.
    """
    # Join the list elements into a string with the specified delimiter
    return f"[{delimiter.join(map(str, lst))}]"


# This function should be replaced by below create_dataloaders
def create_dataloader(is_train, batch_size, dataset, train_ratio, is_test=False):
    dataset_length = len(dataset)
    print(f"Total dataset length: {dataset_length}")

    if is_test:
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Calculate split index for training and validation
    split_idx = int(dataset_length * train_ratio)
    
    # Calculate the maximum number of samples that fit into complete batches for training and validation
    train_samples = (split_idx // batch_size) * batch_size
    valid_samples = ((dataset_length - split_idx) // batch_size) * batch_size
    if is_train:
        indices = range(0, train_samples)
    else:
        indices = range(split_idx, split_idx + valid_samples)
    sub_dataset = Subset(dataset, indices)
    print(f"{'Training' if is_train else 'Validation'} subset length: {len(sub_dataset)}")
    return DataLoader(dataset=sub_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

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

# Function to copy a Subset
def copy_subset(subset):
    return Subset(copy.deepcopy(subset.dataset), copy.deepcopy(subset.indices))

# Call this function during training without the scalars and with the directory path, and during the testing with the saved scalars and without a directory path to save.
def normalize_dataset_create_scaler(dataset_input, directory_path, normalize_y, normalize_pos):
    dataset = copy_subset(dataset_input)
    dataset, x_scaler = normalize_x_values_create_scalers(dataset, directory_path)
    if normalize_pos:
        dataset, pos_scaler = normalize_positional_features_create_scaler(dataset, directory_path)
    if normalize_y:
        dataset, y_scaler = normalize_y_values_create_scaler(dataset, directory_path)
        if normalize_pos:
            return dataset, x_scaler, pos_scaler, y_scaler 
        else:
            return dataset, x_scaler, y_scaler
    else:
        if normalize_pos:
            return dataset, x_scaler, pos_scaler 
        else:
            return dataset, x_scaler

def normalize_dataset_with_given_scaler(dataset_input, x_scalar_list = None, pos_scalar=None, y_scalar=None, normalize_y=False, normalize_pos=False):
    dataset = copy_subset(dataset_input)
    dataset = normalize_x_values_given_scaler(dataset, x_scalar_list)
    if normalize_pos:
        dataset = normalize_positional_features_given_scaler(dataset, pos_scalar)
    if normalize_y:
        dataset = normalize_y_values_given_scaler(dataset, y_scalar)
    return dataset

def normalize_x_values_given_scaler(dataset, x_scaler_list):
    shape_of_x = dataset[0].x.shape[1]
    for i in range(shape_of_x):
        scaler = x_scaler_list[i]
        for data in dataset:
            data_x_dim = replace_invalid_values(data.x[:, i].reshape(-1, 1))
            normalized_x_dim = torch.tensor(scaler.transform(data_x_dim.numpy()), dtype=torch.float)
            if i == 0:
                data.normalized_x = normalized_x_dim
            else:
                data.normalized_x = torch.cat((data.normalized_x, normalized_x_dim), dim=1)
    for data in dataset:
        data.x = data.normalized_x
        del data.normalized_x
    return dataset

def normalize_positional_features_given_scaler(dataset, pos_scalar=None):
    for data in dataset:
        data.pos = torch.tensor(pos_scalar.transform(data.pos.numpy()), dtype=torch.float)
    return dataset

def normalize_y_values_given_scaler(dataset, y_scalar=None):
    for data in dataset:
        data.y = torch.tensor(y_scalar.transform(data.y.numpy()), dtype=torch.float)
    return dataset

def normalize_x_values_create_scalers(dataset, directory_path):
    shape_of_x = dataset[0].x.shape[1]
    list_of_scalers_to_save = []
    x_values = torch.cat([data.x for data in dataset], dim=0)

    for i in range(shape_of_x):
        all_node_features = replace_invalid_values(x_values[:, i].reshape(-1, 1)).numpy()
        
        scaler = StandardScaler()
        print(f"Scaler created for x values at index {i}: {scaler}")
        scaler.fit(all_node_features)
        list_of_scalers_to_save.append(scaler)

        for data in dataset:
            data_x_dim = replace_invalid_values(data.x[:, i].reshape(-1, 1))
            normalized_x_dim = torch.tensor(scaler.transform(data_x_dim.numpy()), dtype=torch.float)
            if i == 0:
                data.normalized_x = normalized_x_dim
            else:
                data.normalized_x = torch.cat((data.normalized_x, normalized_x_dim), dim=1)

    joblib.dump(list_of_scalers_to_save, (directory_path + 'x_scaler.pkl'))

    for data in dataset:
        data.x = data.normalized_x
        del data.normalized_x
    return dataset, list_of_scalers_to_save

def normalize_positional_features_create_scaler(dataset, directory_path):
    all_pos_features = torch.cat([data.pos for data in dataset], dim=0)
    all_pos_features = replace_invalid_values(all_pos_features).numpy()
    scaler = StandardScaler()
    print(f"Scaler created for pos features: {scaler}")
    scaler.fit(all_pos_features)
    joblib.dump(scaler, os.path.join(directory_path, 'pos_scaler.pkl'))
    for data in dataset:
        data.pos = torch.tensor(scaler.transform(data.pos.numpy()), dtype=torch.float)
    return dataset, scaler


def normalize_y_values_create_scaler(dataset, directory_path):
    all_y_values = torch.cat([data.y for data in dataset], dim=0).reshape(-1, 1)
    all_y_values = replace_invalid_values(all_y_values).numpy()

    scaler = RobustScaler()
    print(f"Scaler created for y values: {scaler}")
    scaler.fit(all_y_values)
    joblib.dump(scaler, os.path.join(directory_path, 'y_scaler.pkl'))

    for data in dataset:
        data.y = torch.tensor(scaler.transform(data.y.reshape(-1, 1).numpy()), dtype=torch.float)
    return dataset, scaler


def cut_dimensions(dataset, indices_of_dimensions_to_keep: list):
    dataset_with_fewer_dimensions = dataset.copy()
    for data in dataset_with_fewer_dimensions:
        if indices_of_dimensions_to_keep == [0]:
            data.x = data.x[:, 0].view(-1, 1)  # Keep only the first column and reshape
        else: 
            data.x = data.x[:, indices_of_dimensions_to_keep]
    return dataset_with_fewer_dimensions

def replace_invalid_values(tensor):
    tensor[tensor != tensor] = 0  # replace NaNs with 0
    tensor[tensor == float('inf')] = 0  # replace inf with 0
    tensor[tensor == float('-inf')] = 0  # replace -inf with 0
    return tensor

# Define a dictionary to map each mode to an integer
mode_mapping = {
    'bus': 0,
    'car': 1,
    'car_passenger': 2,
    'pt': 3,
    'bus,car,car_passenger': 4,
    'bus,car,car_passenger,pt': 5,
    'car,car_passenger': 6,
    'pt,rail,train': 7,
    'bus,pt': 8,
    'rail': 9,
    'pt,subway': 10,
    'artificial,bus': 11,
    'artificial,rail': 12,
    'artificial,stopFacilityLink,subway': 13,
    'artificial,subway': 14,
    'artificial,stopFacilityLink,tram': 15,
    'artificial,tram': 16,
    'artificial,bus,stopFacilityLink': 17,
    'artificial,funicular,stopFacilityLink': 18,
    'artificial,funicular': 19
}

# Function to encode modes into integer format
def encode_modes(modes):
    return mode_mapping.get(modes, -1)  # Use -1 for any unknown modes


def compute_baseline_of_mean_target(dataset, loss_fct, device, scalers):
    """
    Computes the baseline Mean Squared Error (MSE) for normalized y values in the dataset.

    Parameters:
    - dataset: A dataset containing normalized y values.
    - scalers: The scalers used to normalize the x and pos values.

    Returns:
    - mse_value: The baseline MSE value.
    """
    # Concatenate the normalized y values from the dataset
    y_values_normalized = np.concatenate([data.y for data in dataset])

    # Compute the mean of the normalized y values
    mean_y_normalized = np.mean(y_values_normalized)

    # Original x values
    x = np.concatenate([scalers["x_scaler"].inverse_transform(data.x) for data in dataset])
    
    # median_y_normalized = np.median(y_values_normalized)   

    # Convert numpy arrays to torch tensors
    y_values_normalized_tensor = torch.tensor(y_values_normalized, dtype=torch.float32).to(device)
    mean_y_normalized_tensor = torch.tensor(mean_y_normalized, dtype=torch.float32).to(device)
    
    # Create the target tensor with the same shape as y_values_normalized_tensor
    target_tensor = mean_y_normalized_tensor.expand_as(y_values_normalized_tensor)

    # Compute the MSE
    loss = loss_fct(y_values_normalized_tensor, target_tensor, x)
    return loss.item() 


def compute_baseline_of_no_policies(dataset, loss_fct, device, scalers):
    """
    Computes the baseline Mean Squared Error (MSE) for normalized y values in the dataset.

    Parameters:
    - dataset: A dataset containing y values: The actual difference of the volume of cars.
    - scalers: The scalers used to normalize the x and pos values.

    Returns:
    - mse_value: The baseline MSE value.
    """
    # Concatenate the normalized y values from the dataset
    actual_difference_vol_car = np.concatenate([data.y for data in dataset])

    target_tensor = np.zeros(actual_difference_vol_car.shape) # presume no difference in vol car due to policy

    # Original x values
    x = np.concatenate([scalers["x_scaler"].inverse_transform(data.x) for data in dataset])
    
    target_tensor = torch.tensor(target_tensor, dtype=torch.float32).to(device)
    actual_difference_vol_car = torch.tensor(actual_difference_vol_car, dtype=torch.float32).to(device)

    # Compute the loss
    loss = loss_fct(actual_difference_vol_car, target_tensor, x)
    return loss.item()

# def visualize_data(policy_features, flow_features, title):
#     edges = edge_index.T.tolist()

#     # Create a networkx graph from the edge list
#     G = nx.Graph(edges)

#     # Create a colormap for flow features
#     flow_cmap = plt.cm.Reds     # colormap for flow features

#     # Extract flow features from tensor
#     flow_values = flow_features.tolist()      # flow graph has only one feature atm

#     # Normalize features for colormap mapping
#     flow_min = 0
#     flow_max = 100
#     norm = Normalize(vmin=flow_min, vmax=flow_max)

#     # Draw the graph with separate lines for flow features on each edge and annotations for policy features
#     plt.figure(figsize=(8, 6))
#     pos = nx.spring_layout(G, seed=42)  # Layout for better visualization

#     # Set to store processed edges
#     processed_edges = set()

#     # Draw edges for flow features and annotate with policy features
#     for i, (u, v) in enumerate(edges):
#         # Check if the edge has already been processed
#         if (u, v) not in processed_edges and (v, u) not in processed_edges:
#             flow_color = flow_cmap((flow_values[i] - flow_min) / (flow_max - flow_min))
#             nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=flow_color, width=2, alpha=0.7)

#             # Annotate with policy feature values
#             policy_values_str = ", ".join([f"{int(val)}" for val in policy_features[i]])
#             plt.text((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2, f"({policy_values_str})", fontsize=8, color="black", ha="center", va="center")

#             # Add the edge to the set of processed edges
#             processed_edges.add((u, v))

#     # Draw nodes
#     nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue", alpha=0.8)

#     # Draw labels
#     nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

#     # Add colorbar for flow features
#     flow_sm = plt.cm.ScalarMappable(norm=norm, cmap=flow_cmap)
#     flow_sm.set_array([])
#     plt.colorbar(flow_sm, label="Flow")
#     plt.title(title)
#     # if "Train" in title:
#     #     plt.savefig(f"visualisation/train_data/{title}.png", dpi = 500)
#     # else:
#     #     plt.savefig(f"visualisation/test_data/{title}.png", dpi = 500)
