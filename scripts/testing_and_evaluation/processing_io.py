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
    capacity_base_case = base_case['capacity'].values

    # Create dictionaries for nodes and edges
    nodes = pd.concat([test_data['from_node'], test_data['to_node']]).unique()
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    test_data['from_idx'] = test_data['from_node'].map(node_to_idx)
    test_data['to_idx'] = test_data['to_node'].map(node_to_idx)

    edges = test_data[['from_idx', 'to_idx']].values
    edge_car_volumes = test_data['vol_car'].values
    capacities = test_data['capacity'].values
    # freespeeds = test_data['freespeed'].values  
    # lengths = test_data['length'].values  
    # modes = test_data['modes'].values
    # modes_encoded = np.vectorize(encode_modes)(modes)
    highway = test_data['highway'].apply(lambda x: highway_mapping.get(x, -1)).values

    edge_positions = np.array([((geom.coords[0][0] + geom.coords[-1][0]) / 2, 
                                (geom.coords[0][1] + geom.coords[-1][1]) / 2) 
                                for geom in test_data.geometry])

    # Convert lists to tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_positions_tensor = torch.tensor(edge_positions, dtype=torch.float)
    x = torch.zeros((len(nodes), 1), dtype=torch.float)

    edge_car_volume_difference = edge_car_volumes - vol_base_case

    # Create Data object
    target_values = torch.tensor(edge_car_volume_difference, dtype=torch.float).unsqueeze(1)
    data = Data(edge_index=edge_index, x=x, pos=edge_positions_tensor)

    # Transform to line graph
    linegraph_data = linegraph_transformation(data)

    # Prepare the x for line graph: index and capacity
    linegraph_x = torch.tensor(np.column_stack((vol_base_case, capacity_base_case, capacities, highway)), dtype=torch.float)

    linegraph_data.x = linegraph_x

    # # Target tensor for car volumes
    linegraph_data.y = target_values

    if linegraph_data.validate(raise_on_error=True):
        return linegraph_data
    else:
        print("Invalid line graph data")
        
        
def list_to_string(integers, delimiter=', '):
    """
    Converts a list of integers into a string, with each integer separated by the specified delimiter.

    Parameters:
    integers (list of int): The list of integers to convert.
    delimiter (str): The delimiter to use between integers in the string.

    Returns:
    str: A string representation of the list of integers.
    """
    return delimiter.join(map(str, integers))

def data_to_geodataframe(data, original_gdf):
    # Extract the edge index and node features
    node_features = data.x.numpy()
    target_values = data.y.numpy()

    # Create edge data
    edge_data = {
        'from_node': original_gdf["from_node"].values,
        'to_node': original_gdf["to_node"].values,
        'vol_base_case': node_features[:, 0],  # Assuming capacity is the first feature, and so on
        'capacity_base_case': node_features[:, 1],  
        'capacity_new': node_features[:, 2],  
        'highway': node_features[:, 3],  
        'vol_car': target_values.squeeze()  # Assuming target values are car volumes
    }
    # Convert to DataFrame
    edge_df = pd.DataFrame(edge_data)
    # Create LineString geometry
    edge_df['geometry'] = original_gdf["geometry"].values
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(edge_df, geometry='geometry')
    return gdf