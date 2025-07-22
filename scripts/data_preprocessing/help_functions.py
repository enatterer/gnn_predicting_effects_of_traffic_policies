import os
import tarfile
import tempfile
import shutil

import numpy as np
import pandas as pd
import geopandas as gpd

import torch

# Get the absolute path to the project root
#project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
#districts_path = os.path.join(project_root, 'data', 'visualisation', 'districts_paris.geojson')
#districts = gpd.read_file(districts_path)

# Custom mapping for highway types
highway_mapping = {
    'trunk': 0, 'trunk_link': 0, 'motorway_link': 0,'motorway': 0,
    'primary': 1, 'primary_link': 1,
    'secondary': 2, 'secondary_link': 2,
    'tertiary': 3, 'tertiary_link': 3,
    'residential': 4, 'living_street': 5,
    'pedestrian': 6, 'service': 7,
    'construction': 8, 'unclassified': 9,
    'busway': -1, 'platform': -1, 'track': -1, 'bus_stop': -1,
    'path': -1
}
    
def create_policy_key(folder_name):
    # Extract the relevant part of the folder name
    base_name = os.path.basename(folder_name)  # Get the base name of the file or folder
    parts = base_name.split('_')[-1]  # Ignore the first part ('network')
    hex_size_name = os.path.basename(os.path.dirname(folder_name)) # Get the hexagon size name
    hex_size_name = hex_size_name.split('_')[3]  
    return (hex_size_name, parts)
    
# Function to read and convert CSV.GZ to GeoDataFrame
def read_output_links(folder):
    file_path = os.path.join(folder, 'output_links.csv.gz')
    if os.path.exists(file_path):
        try:
            # Read the CSV file with the correct delimiter
            df = pd.read_csv(file_path, delimiter=';',low_memory=False)
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
        'car_passenger': gdf['modes'].str.contains('car_passenger', case=False, na=False).astype(int),
        'bus': gdf['modes'].str.contains('bus', case=False, na=False).astype(int),
        'pt': gdf['modes'].str.contains('pt', case=False, na=False).astype(int),
        'train': gdf['modes'].str.contains('train', case=False, na=False).astype(int),
        'tram': gdf['modes'].str.contains('tram', case=False, na=False).astype(int),
        'rail': gdf['modes'].str.contains('rail', case=False, na=False).astype(int),
        'subway': gdf['modes'].str.contains('subway', case=False, na=False).astype(int),
        'funicular': gdf['modes'].str.contains('funicular', case=False, na=False).astype(int)
    }
    modes_encoded = pd.DataFrame(modes_conditions)
    tensor_list = [torch.tensor(modes_encoded[col].values, dtype=torch.float) for col in modes_encoded.columns]
    return tensor_list

def read_eqasim_trips(folder):
    file_path = os.path.join(folder, 'eqasim_trips.csv')
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, delimiter=';',low_memory=False)
            return df
        except Exception:
            print("empty data error" + file_path)
            return None
    else:
        return None

def add_destinations_to_gdf(gdf, df_eqasim_trips):
    """
    Add land use features to GeoDataFrame based on eqasim_trips data.
    Analyzes trip destinations to infer what types of activities are located near each link.
    """
    gdf_extended = gdf.copy()
    
    if df_eqasim_trips is not None and not df_eqasim_trips.empty:
        # Check if required columns exist
        required_cols = ['destination_link_id', 'following_purpose']
        if all(col in df_eqasim_trips.columns for col in required_cols):
            
            # Land use analysis: Count destinations only (where activities are located)
            # Get all unique activity types from destinations (what's actually built there)
            all_activities = set(df_eqasim_trips['following_purpose'].unique())
            
            df_land_use_activities = pd.DataFrame()
            
            for activity in all_activities:
                # Count unique people going TO each link for this activity
                # This represents land use: "how many people use this link for this activity type"
                activity_destinations = df_eqasim_trips[
                    df_eqasim_trips['following_purpose'] == activity
                ].groupby(['destination_link_id', 'person_id']).size().groupby('destination_link_id').size()
                
                df_land_use_activities[activity] = activity_destinations
            
            df_land_use_activities = df_land_use_activities.fillna(0)
            
            # 3. Create identifier for links present in eqasim_trips
            all_trip_links = set(df_eqasim_trips['destination_link_id'].astype(str))
            
            # Convert link column to same type before merging
            gdf_extended['link'] = gdf_extended['link'].astype(str)
            df_land_use_activities.index = df_land_use_activities.index.astype(str)
            
            # 4. Add trip data identifier (1 if link appears in trips, 0 otherwise)
            gdf_extended['is_in_eqasim_trips'] = gdf_extended['link'].isin(all_trip_links).astype(int)
            
            # 5. Merge land use activity features
            gdf_extended = gdf_extended.merge(df_land_use_activities, left_on='link', right_index=True, how='left').fillna(0)
    
    return gdf_extended

def compute_target_tensor_only_edge_features(vol_base_case, gdf, column_name: str, normalization_type: str):
    edge_car_volume_difference = gdf['vol_car'].values - vol_base_case
    
    if column_name == 'vol_car':
        # Keep continuous values for training - round only at inference
        # Sign preserved: +500.3 cars vs -200.7 cars
        data_to_normalize = edge_car_volume_difference
        
    elif column_name == 'vol_car_percentage':
        # Division by base case to get percentage change  
        # Sign preserved: +50% vs -30%
        epsilon = 1e-6 #adjust as needed
        base_case_with_epsilon = vol_base_case.copy()
        zero_mask = vol_base_case == 0
        base_case_with_epsilon[zero_mask] = epsilon
        data_to_normalize = edge_car_volume_difference / base_case_with_epsilon
    
    # Use signed_log_normalized for both - preserves sign, compresses range
    normalized_data = normalization_of_edge_features(
        data_to_normalize, normalization_type
    )
    return torch.tensor(normalized_data, dtype=torch.float).unsqueeze(1)

def normalization_of_edge_features(data, normalization_type: str):
    """
    Normalize numpy array data using specified method.
    Args:
        data: numpy array with data to normalize
        normalization_type: 'mean_std', 'min_max' or 'signed_log_normalization'
        base_case_data: numpy array for log-based normalizations (epsilon already handled if needed)
    """
    if normalization_type == 'mean_std':
        std = np.std(data)
        if std == 0:
            return np.zeros_like(data)  # If all values are same, return zeros
        return (data - np.mean(data)) / std
    elif normalization_type == 'min_max':
        data_min, data_max = np.min(data), np.max(data)
        if data_max == data_min:
            return np.zeros_like(data)  # If all values are same, return zeros
        return (data - data_min) / (data_max - data_min)
    elif normalization_type == 'signed_log_normalization': #TODO: try signed min-max normalization instead of standardization
        # Step 1: Signed log transformation
        signed_log = np.sign(data) * np.log1p(np.abs(data))
        # Step 2: Standard normalization with zero-std protection
        std = np.std(signed_log)
        if std == 0:
            return np.zeros_like(signed_log)  # If all values are same, return zeros
        return (signed_log - np.mean(signed_log)) / std
    else:
        raise ValueError(f"Invalid normalization type: {normalization_type}")


def get_basic_edge_attributes(capacity_base_case, gdf, required_modes_on_links):
    # Create a mask for each required mode and combine with OR logic
    mode_masks = [gdf['modes'].str.contains(mode) for mode in required_modes_on_links]
    combined_mask = mode_masks[0]
    for mask in mode_masks[1:]:
        combined_mask = combined_mask | mask
    
    capacities_new = np.where(combined_mask, gdf['capacity'], 0) # capacity is 0 for links that are not used by the required modes
    capacity_reduction = capacities_new - capacity_base_case #check the sign of this
    highway = gdf['highway'].apply(lambda x: highway_mapping.get(x, -1)).values
    freespeed = np.where(combined_mask, gdf['freespeed'], 0) # freespeed is 0 for links that are not used by the required modes
    return capacities_new, capacity_reduction, highway, freespeed

def prepare_gdf(df, gdf_input):
    gdf = gdf_input[['link', 'geometry']].merge(df, on='link', how='left')
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
    gdf.crs = gdf_input.crs
    return gdf

def get_link_geometries(links_gdf_input, apply_scaling=True):
    """
    Extract link geometries and optionally apply centering and scaling.
    Apply scaling anc centering when using multiple cities
    Args:
        links_gdf_input: GeoDataFrame with link geometry information
        apply_scaling: Whether to apply centering and scaling to coordinates
        
    Returns:
        edge_start_point_tensor: Tensor of start points
        stacked_edge_geometries_tensor: Stacked tensor of start, end, and midpoints
        edges_base: Array of edge connections
        nodes: Unique nodes
        scaling_params: Dictionary with mean and std for scaling (if apply_scaling=True)
    """
    edge_midpoints = np.array([((geom.coords[0][0] + geom.coords[-1][0]) / 2, 
                                    (geom.coords[0][1] + geom.coords[-1][1]) / 2) 
                                for geom in links_gdf_input.geometry])

    nodes = pd.concat([links_gdf_input['from_node'], links_gdf_input['to_node']]).unique()
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    links_gdf_input['from_idx'] = links_gdf_input['from_node'].map(node_to_idx)
    links_gdf_input['to_idx'] = links_gdf_input['to_node'].map(node_to_idx)
    edges_base = links_gdf_input[['from_idx', 'to_idx']].values

    start_points = np.array([geom.coords[0] for geom in links_gdf_input.geometry])
    end_points = np.array([geom.coords[-1] for geom in links_gdf_input.geometry])

    scaling_params = None
    
    if apply_scaling:
        # Combine all coordinates for computing global scaling parameters
        all_coords = np.vstack([start_points, end_points, edge_midpoints])
        
        # Calculate mean and std for centering and scaling
        coord_mean = np.mean(all_coords, axis=0)  # Shape: [2] (x_mean, y_mean)
        coord_std = np.std(all_coords, axis=0)    # Shape: [2] (x_std, y_std)
        
        # Avoid division by zero
        coord_std = np.where(coord_std == 0, 1.0, coord_std)
        
        # Apply centering and scaling
        start_points = (start_points - coord_mean) / coord_std
        end_points = (end_points - coord_mean) / coord_std
        edge_midpoints = (edge_midpoints - coord_mean) / coord_std
        
        scaling_params = {
            'mean': coord_mean,
            'std': coord_std
        }

    edge_start_point_tensor = torch.tensor(start_points, dtype=torch.float)
    edge_end_point_tensor = torch.tensor(end_points, dtype=torch.float)
    edge_midpoint_tensor = torch.tensor(edge_midpoints, dtype=torch.float)

    stacked_edge_geometries_tensor = torch.stack([edge_start_point_tensor, edge_end_point_tensor, edge_midpoint_tensor], dim=1)

    if apply_scaling:
        return edge_start_point_tensor, stacked_edge_geometries_tensor, edges_base, nodes, scaling_params
    else:
        return edge_start_point_tensor, stacked_edge_geometries_tensor, edges_base, nodes
    
def extract_and_get_networks(tar_files):
    """
    Extract all tar.gz files from compressed directories and return network paths.
    Each tar.gz contains files directly, so we need to create the proper directory structure.
    """
    networks = []
    temp_dirs = []  # Keep track for cleanup
    
    for tar_file in tar_files:
        
        tar_path = tar_file
                
        # Extract network name from tar filename (remove .tar.gz)
        network_name = os.path.basename(tar_file).replace('.tar.gz', '')
        
        # Get the compressed directory name to preserve hex size info
        compressed_dir_name = os.path.basename(os.path.dirname(tar_path))
                
        # Create temporary directory for this tar file
        temp_dir = tempfile.mkdtemp()
        temp_dirs.append(temp_dir)
        
        # Create the directory structure that preserves hex size info
        preserve_structure_dir = os.path.join(temp_dir, compressed_dir_name)
        os.makedirs(preserve_structure_dir, exist_ok=True)
        
        # Create network subdirectory (what the processing code expects)
        network_dir = os.path.join(preserve_structure_dir, network_name)
        os.makedirs(network_dir, exist_ok=True)
        
        # Extract tar.gz file directly into the network directory
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(network_dir)
        
        # Add the network directory to our list
        networks.append(network_dir)
    
    return networks, temp_dirs