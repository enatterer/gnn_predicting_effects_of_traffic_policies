import random

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse
from torch_geometric.nn import GCNConv

import torch
import geopandas as gpd
from shapely.geometry import LineString

import pickle

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import math
import random
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd
import tqdm
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset

import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import LineGraph

from shapely.geometry import LineString

def validate_model(model, valid_dl, loss_func, device):
    model.eval()
    val_loss = 0
    num_batches = 0
    with torch.inference_mode():
        for idx, data in enumerate(valid_dl):
            input_node_features, targets = data.normalized_x.to(device), data.normalized_y.to(device)
            predicted = model(data)
            val_loss += loss_func(predicted, targets).item()
            num_batches += 1
    return val_loss / num_batches if num_batches > 0 else 0

def create_dataloader(is_train, batch_size, dataset, train_ratio):
    dataset_length = len(dataset)
    print(f"Total dataset length: {dataset_length}")

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
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

    
def collate_fn(data_list):
    return Batch.from_data_list(data_list)

def normalize_data(dataset):
    # Collect all node features
    all_node_features = []
    for data in dataset:
        all_node_features.append(data.x)

    # Stack all node features into a single tensor
    all_node_features = torch.cat(all_node_features, dim=0)

    # Fit the min-max scaler on the node features
    scaler = MinMaxScaler()
    scaler.fit(all_node_features)

    # Apply the scaler to each data instance and store as a new feature
    for data in dataset:
        data.normalized_x = torch.tensor(scaler.transform(data.x), dtype=torch.float)

    return dataset

def normalize_positional_features(dataset):
    # Collect all positional features
    all_pos_features = []
    for data in dataset:
        all_pos_features.append(data.pos)

    # Stack all positional features into a single tensor
    all_pos_features = torch.cat(all_pos_features, dim=0)

    # Fit the min-max scaler on the positional features
    scaler = MinMaxScaler()
    scaler.fit(all_pos_features)

    # Apply the scaler to each data instance and store as a new feature
    for data in dataset:
        data.normalized_pos = torch.tensor(scaler.transform(data.pos), dtype=torch.float)
    return dataset

def normalize_y_values(dataset):
    # Collect all y values
    all_y_values = []
    for data in dataset:
        all_y_values.append(data.y)

    # Stack all y values into a single tensor
    all_y_values = torch.cat(all_y_values, dim=0)

    # Fit the min-max scaler on the y values
    scaler = MinMaxScaler()
    scaler.fit(all_y_values)

    # Apply the scaler to each data instance and store as a new feature
    for data in dataset:
        data.normalized_y = torch.tensor(scaler.transform(data.y), dtype=torch.float)  # Keep the 2D shape

    return dataset

def normalize_dataset(dataset):
    # Normalize node features
    dataset = normalize_data(dataset)
    # Normalize positional features (if any)
    dataset = normalize_positional_features(dataset)
    # Normalize y values
    dataset = normalize_y_values(dataset)
    return dataset


# def check_and_replace_inf(policy_data):
#     has_inf = False
#     inf_sources = {"capacity": 0, "freespeed": 0, "mode": 0}
    
#     for i, row in enumerate(policy_data):
#         capacity, freespeed, modes = row[0], row[1], row[2]
        
#         # Check freespeed for inf values
#         if freespeed == float('inf') or freespeed == float('-inf'):
#             # print(f"Inf value found in freespeed at row {i}: {freespeed}")
#             has_inf = True
#             inf_sources["freespeed"] += 1
#             row[1] = 1e6 if freespeed == float('inf') else -1e6
#     return policy_data, has_inf, inf_sources

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
