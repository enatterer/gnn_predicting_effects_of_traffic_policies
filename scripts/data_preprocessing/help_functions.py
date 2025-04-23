import os

import numpy as np
import pandas as pd
import geopandas as gpd

import torch

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
districts_path = os.path.join(project_root, 'data', 'visualisation', 'districts_paris.geojson')
districts = gpd.read_file(districts_path)

# Custom mapping for highway types
highway_mapping = {
    'trunk': 0, 'trunk_link': 0, 'motorway_link': 0,
    'primary': 1, 'primary_link': 1,
    'secondary': 2, 'secondary_link': 2,
    'tertiary': 3, 'tertiary_link': 3,
    'residential': 4, 'living_street': 5,
    'pedestrian': 6, 'service': 7,
    'construction': 8, 'unclassified': 9,
    'pt': -1, 
}
    
def create_policy_key(folder_name):
    # Extract the relevant part of the folder name
    base_name = os.path.basename(folder_name)  # Get the base name of the file or folder
    parts = base_name.split('_')[1:]  # Ignore the first part ('network')
    district_info = '_'.join(parts)
    districts = district_info.split('_')
    return f"Policy introduced in Arrondissement(s) {', '.join(districts)}"
    
# Function to read and convert CSV.GZ to GeoDataFrame
def read_output_links(folder):
    file_path = os.path.join(folder, 'output_links.csv.gz')
    if os.path.exists(file_path):
        try:
            # Read the CSV file with the correct delimiter
            df = pd.read_csv(file_path, delimiter=';')
            return df
        except Exception:
            print("empty data error" + file_path)
            return None
    else:
        return None

def extend_geodataframe(gdf_base, gdf_to_extend, column_to_extend: str, new_column_name: str):
    """
    Extend a GeoDataFrame by adding a column from another GeoDataFrame.
    
    Parameters:
    gdf_base (GeoDataFrame): The GeoDataFrame containing the column to add
    gdf_to_extend (GeoDataFrame): The GeoDataFrame to be extended
    column_name (str): The column name to add to gdf_to_extend
    new_column_name (str): The new column name to use in gdf_to_extend

    
    Returns:
    GeoDataFrame: A new GeoDataFrame with the column added
    """
    # Ensure the column exists in the base GeoDataFrame
    if column_to_extend not in gdf_base.columns:
        raise ValueError(f"Column '{column_to_extend}' does not exist in the base GeoDataFrame")
    
    # Create a copy of the GeoDataFrame to be extended
    extended_gdf = gdf_to_extend.copy()
    
    # Add the column from the base GeoDataFrame
    extended_gdf[new_column_name] = gdf_base[column_to_extend]
    
    return extended_gdf
    
def calculate_avg_mode_stats(single_mode_stats_list:list):
    mode_stats_list = []
    for df in single_mode_stats_list:
        mode_stats = df.groupby('mode').agg({
            'travel_time': ['mean', 'count'],
            'routed_distance': 'mean'
        }).reset_index()
        mode_stats.columns = ['mode', 'avg_travel_time', 'trip_count', 'avg_routed_distance']
        mode_stats_list.append(mode_stats)
    all_mode_stats = pd.concat(mode_stats_list, ignore_index=True)

    # Calculate the average across all seeds
    average_mode_stats = all_mode_stats.groupby('mode').agg({
        'avg_travel_time': 'mean',
        'avg_routed_distance': 'mean',
        'trip_count': 'mean'
    }).reset_index()
    average_mode_stats.columns = ['mode', 'avg_total_travel_time', 'avg_total_routed_distance', 'avg_trip_count']
    df_average_mode_stats = pd.DataFrame(average_mode_stats)
    return df_average_mode_stats

def encode_modes(gdf):
    """Encode the 'modes' attribute based on specific strings."""
    modes_conditions = {
        'car': gdf['modes'].str.contains('car', case=False, na=False).astype(int),
        'bus': gdf['modes'].str.contains('bus', case=False, na=False).astype(int),
        'pt': gdf['modes'].str.contains('pt', case=False, na=False).astype(int),
        'train': gdf['modes'].str.contains('train', case=False, na=False).astype(int),
        'rail': gdf['modes'].str.contains('rail', case=False, na=False).astype(int),
        'subway': gdf['modes'].str.contains('subway', case=False, na=False).astype(int)
    }
    modes_encoded = pd.DataFrame(modes_conditions)
    tensor_list = [torch.tensor(modes_encoded[col].values, dtype=torch.float) for col in modes_encoded.columns]
    return tensor_list

def read_eqasim_trips(folder):
    file_path = os.path.join(folder, 'eqasim_trips.csv')
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, delimiter=';')
            return df
        except Exception:
            print("empty data error" + file_path)
            return None
    else:
        return None

def compute_target_tensor_only_edge_features(vol_base_case, gdf):
    edge_car_volume_difference = gdf['vol_car'].values - vol_base_case
    return torch.tensor(edge_car_volume_difference, dtype=torch.float).unsqueeze(1)

def get_basic_edge_attributes(capacity_base_case, gdf):
    capacities_new = np.where(gdf['modes'].str.contains('car'), gdf['capacity'], 0)
    capacity_reduction = capacities_new - capacity_base_case
    highway = gdf['highway'].apply(lambda x: highway_mapping.get(x, -1)).values
    freespeed = np.where(gdf['modes'].str.contains('car'), gdf['freespeed'], 0)
    return capacities_new,capacity_reduction,highway,freespeed

def prepare_gdf(df, gdf_input):
    gdf = gdf_input[['link', 'geometry']].merge(df, on='link', how='left')
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
    gdf.crs = gdf_input.crs
    return gdf

def get_link_geometries(links_gdf_input):
    edge_midpoints = np.array([((geom.coords[0][0] + geom.coords[-1][0]) / 2, 
                                    (geom.coords[0][1] + geom.coords[-1][1]) / 2) 
                                for geom in links_gdf_input.geometry])

    nodes = pd.concat([links_gdf_input['from_node'], links_gdf_input['to_node']]).unique()
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    links_gdf_input['from_idx'] = links_gdf_input['from_node'].map(node_to_idx)
    links_gdf_input['to_idx'] = links_gdf_input['to_node'].map(node_to_idx)
    edges_base = links_gdf_input[['from_idx', 'to_idx']].values
    edge_midpoint_tensor = torch.tensor(edge_midpoints, dtype=torch.float)

    start_points = np.array([geom.coords[0] for geom in links_gdf_input.geometry])
    end_points = np.array([geom.coords[-1] for geom in links_gdf_input.geometry])

    edge_start_point_tensor = torch.tensor(start_points, dtype=torch.float)
    edge_end_point_tensor = torch.tensor(end_points, dtype=torch.float)

    stacked_edge_geometries_tensor = torch.stack([edge_start_point_tensor, edge_end_point_tensor, edge_midpoint_tensor], dim=1)

    return edge_start_point_tensor, stacked_edge_geometries_tensor, edges_base, nodes