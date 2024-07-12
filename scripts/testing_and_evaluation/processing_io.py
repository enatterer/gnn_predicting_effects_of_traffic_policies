import math
import numpy as np
import wandb
import pickle
import os
import shapely.wkt as wkt
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from torch_geometric.transforms import LineGraph
import torch
from torch_geometric.data import Data

# Custom mapping for highway types
highway_mapping = {
    'trunk': 0, 'trunk_link': 0, 'motorway_link': 0,
    'primary': 1, 'primary_link': 1,
    'secondary': 2, 'secondary_link': 2,
    'tertiary': 3, 'tertiary_link': 3,
    'residential': 4, 'living_street': 5,
    'pedestrian': 6, 'service': 7,
    'construction': 8, 'unclassified': 9,
    'np.nan': -1
}


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


def create_test_data_object(base_case, test_data):
    linegraph_transformation = LineGraph()
    test_data.crs = "EPSG:2154"  # Assuming the original CRS is EPSG:2154
    test_data.to_crs("EPSG:4326", inplace=True)
    vol_base_case = base_case['vol_car'].values

    # Create dictionaries for nodes and edges
    nodes = pd.concat([test_data['from_node'], test_data['to_node']]).unique()
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    test_data['from_idx'] = test_data['from_node'].map(node_to_idx)
    test_data['to_idx'] = test_data['to_node'].map(node_to_idx)

    edges = test_data[['from_idx', 'to_idx']].values
    edge_car_volumes = test_data['vol_car'].values
    capacities = test_data['capacity'].values
    freespeeds = test_data['freespeed'].values  
    lengths = test_data['length'].values  
    modes = test_data['modes'].values
    modes_encoded = np.vectorize(encode_modes)(modes)
    highway = test_data['highway'].apply(lambda x: highway_mapping.get(x, -1)).values

    edge_positions = np.array([((geom.coords[0][0] + geom.coords[-1][0]) / 2, 
                                (geom.coords[0][1] + geom.coords[-1][1]) / 2) 
                                for geom in test_data.geometry])

    # Convert lists to tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_positions_tensor = torch.tensor(edge_positions, dtype=torch.float)
    x = torch.zeros((len(nodes), 1), dtype=torch.float)

    # Create Data object
    target_values = torch.tensor(edge_car_volumes, dtype=torch.float).unsqueeze(1)
    data = Data(edge_index=edge_index, x=x, pos=edge_positions_tensor)

    # Transform to line graph
    linegraph_data = linegraph_transformation(data)

    # Prepare the x for line graph: index and capacity
    linegraph_x = torch.tensor(np.column_stack((capacities, vol_base_case, highway, freespeeds, lengths, modes_encoded)), dtype=torch.float)

    linegraph_data.x = linegraph_x

    # # Target tensor for car volumes
    linegraph_data.y = target_values

    if linegraph_data.validate(raise_on_error=True):
        return linegraph_data
    else:
        print("Invalid line graph data")