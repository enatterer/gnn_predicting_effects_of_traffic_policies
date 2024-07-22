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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse
from torch.utils.data import DataLoader, Dataset, Subset
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import LineGraph
from shapely.geometry import LineString
import tqdm
import wandb
import os 

import json

import joblib  # For saving the scaler


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

def create_dataloaders(batch_size, dataset, train_ratio, val_ratio, test_ratio):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    
    dataset_length = len(dataset)
    print(f"Total dataset length: {dataset_length}")
    
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
    
    # Create data loaders
    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_subset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader

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

# Call this function during training without the scalars and with the directory path, and during the testing with the saved scalars and without a directory path to save.
def normalize_dataset(dataset, y_scalar=None, pos_scalar=None, x_scalar_list = None, directory_path=None):
    if input_feature_normalisation == None:
        input_feature_normalisation = "standardScaler"
    print("Input normalisation: " + str(input_feature_normalisation))
    dataset = normalize_x_values(dataset, input_feature_normalisation, x_scalar_list, directory_path)
    dataset = normalize_positional_features(dataset, input_feature_normalisation, pos_scalar, directory_path)
    dataset = normalize_y_values(dataset, y_scalar, directory_path)
    return dataset

def normalize_x_values(dataset, x_scaler_list, directory_path=None):
    shape_of_x = dataset[0].x.shape[1]
    list_of_scalers_to_save = []
    create_scaler = x_scaler_list is None or len(x_scaler_list) == 0
    x_values = torch.cat([data.x for data in dataset], dim=0)

    for i in range(shape_of_x):
        all_node_features = replace_invalid_values(x_values[:, i].reshape(-1, 1)).numpy()

        if create_scaler:
            scaler = StandardScaler()
            print(f"Scaler created for x values: {scaler}")
            scaler.fit(all_node_features)
            list_of_scalers_to_save.append(scaler)
        else:
            scaler = x_scaler_list[i]

        for data in dataset:
            data_x_dim = replace_invalid_values(data.x[:, i].reshape(-1, 1))
            normalized_x_dim = torch.tensor(scaler.transform(data_x_dim.numpy()), dtype=torch.float)
            if i == 0:
                data.normalized_x = normalized_x_dim
            else:
                data.normalized_x = torch.cat((data.normalized_x, normalized_x_dim), dim=1)

    if create_scaler:
        joblib.dump(list_of_scalers_to_save, os.path.join(directory_path, 'x_scaler.pkl'))

    for data in dataset:
        data.x = data.normalized_x
        del data.normalized_x
    return dataset

def normalize_positional_features(dataset, pos_scalar=None, directory_path=None):
    all_pos_features = torch.cat([data.pos for data in dataset], dim=0)
    all_pos_features = replace_invalid_values(all_pos_features).numpy()

    if pos_scalar is None:
        scaler = StandardScaler()
        print(f"Scaler created for pos features: {scaler}")
        scaler.fit(all_pos_features)
        joblib.dump(scaler, os.path.join(directory_path, 'pos_scaler.pkl'))
    else:
        scaler = pos_scalar

    for data in dataset:
        data.pos = torch.tensor(scaler.transform(data.pos.numpy()), dtype=torch.float)
    return dataset

def normalize_y_values(dataset, y_scalar=None, directory_path=None):
    all_y_values = torch.cat([data.y for data in dataset], dim=0).reshape(-1, 1)
    all_y_values = replace_invalid_values(all_y_values).numpy()

    if y_scalar is None:
        scaler = StandardScaler()
        print(f"Scaler created for y values: {scaler}")
        scaler.fit(all_y_values)
        joblib.dump(scaler, os.path.join(directory_path, 'y_scaler.pkl'))
    else:
        scaler = y_scalar

    for data in dataset:
        data.y = torch.tensor(scaler.transform(data.y.reshape(-1, 1).numpy()), dtype=torch.float)
    return dataset

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


def compute_baseline_of_mean_target(dataset, loss_fct):
    """
    Computes the baseline Mean Squared Error (MSE) for normalized y values in the dataset.

    Parameters:
    - dataset: A dataset containing normalized y values.

    Returns:
    - mse_value: The baseline MSE value.
    """
    # Concatenate the normalized y values from the dataset
    y_values_normalized = np.concatenate([data.y for data in dataset])

    # Compute the mean of the normalized y values
    mean_y_normalized = np.mean(y_values_normalized)
    # print("mean_y_normalized: ")
    # print(mean_y_normalized)
    
    # median_y_normalized = np.median(y_values_normalized)   
    # print("median_y_normalized: ")
    # print(median_y_normalized)

    # Convert numpy arrays to torch tensors
    y_values_normalized_tensor = torch.tensor(y_values_normalized, dtype=torch.float32)
    mean_y_normalized_tensor = torch.tensor(mean_y_normalized, dtype=torch.float32)

    # Create the target tensor with the same shape as y_values_normalized_tensor
    target_tensor = mean_y_normalized_tensor.expand_as(y_values_normalized_tensor)

    # Instantiate the MSELoss function
    # mse_loss = torch.nn.MSELoss()

    # Compute the MSE
    loss = loss_fct(y_values_normalized_tensor, target_tensor)
    return loss.item() 


def compute_baseline_of_no_policies(dataset, loss_fct):
    """
    Computes the baseline Mean Squared Error (MSE) for normalized y values in the dataset.

    Parameters:
    - dataset: A dataset containing y values: The actual difference of the volume of cars.

    Returns:
    - mse_value: The baseline MSE value.
    """
    # Concatenate the normalized y values from the dataset
    actual_difference_vol_car = np.concatenate([data.y for data in dataset])

    target_tensor = np.zeros(actual_difference_vol_car.shape) # presume no difference in vol car due to policy
    
    target_tensor = torch.tensor(target_tensor, dtype=torch.float32)
    actual_difference_vol_car = torch.tensor(actual_difference_vol_car, dtype=torch.float32)
    
    # Compute the loss
    loss = loss_fct(actual_difference_vol_car, target_tensor)
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
