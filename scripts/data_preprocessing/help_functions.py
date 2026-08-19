import os
import json
import glob
import tarfile
import tempfile

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian as csgraph_laplacian

import torch

################################################# ↓ HIGHWAY MAPPINGS ↓ #################################################

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

# Map numeric classes to one-hot classes (6 total)
highway_cluster_mapping = {
    -1: 4,  # PT / non-road
    0: 5,   # Other
    1: 0,   # PRIMARY
    2: 1,   # SECONDARY
    3: 2,   # TERTIARY
    4: 3,   # RESIDENTIAL
    5: 5, 6: 5, 7: 5, 8: 5, 9: 5  # Others
}

################################################# ↓ Raw Data IO ↓ #################################################

# Extract the relevant parts
# Specific to Bavarian Simulations for now!
def create_policy_key(file_path):
    filename = os.path.basename(file_path)  # Get the filename
    scenario = filename.split('_')[-1]  # Get scenario
    compress_dir_name = os.path.basename(os.path.dirname(file_path)) # Get the compressed directory name
    hex_size = compress_dir_name.split('_')[3] # Get hex size 
    return (hex_size, scenario)
    
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

def get_reduced_capacity_links(city, policy_region, scenario, project_root):
    """
    Load reduced capacity links from geojson file for given policy_region and scenario.
    
    Args:
        city: City name (e.g., 'rosenheim')
        policy_region: Policy region (e.g., '500') 
        scenario: Scenario (e.g., 's1')
        project_root: Project root path
        
    Returns:
        set: Set of link IDs that have reduced capacity
    """
    # Construct the directory path
    reduced_links_dir = os.path.join(
        project_root, 'data', 'inductive_data', 'reduced_links', 
        city, f'{city}_hex_{policy_region}'
    )
    
    # Find the geojson file matching the pattern
    pattern = f"network_seed3_{city}_primary_*_{scenario}_reduced_capacity_edges.geojson"
    search_pattern = os.path.join(reduced_links_dir, pattern)
    
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        print(f"WARNING: No reduced capacity geojson file found for {city}, policy_region={policy_region}, scenario={scenario}")
        print(f"  Searched in: {reduced_links_dir}")
        print(f"  Pattern: {pattern}")
        return set()
    
    if len(matching_files) > 1:
        print(f"WARNING: Multiple reduced capacity files found for {city}, policy_region={policy_region}, scenario={scenario}")
        print(f"  Files: {matching_files}")
        print(f"  Using first file: {matching_files[0]}")
    
    geojson_file = matching_files[0]
    print(f"Loading reduced capacity links from: {os.path.basename(geojson_file)}")
    
    try:
        # Load geojson and extract link IDs
        gdf_reduced = gpd.read_file(geojson_file)
        reduced_link_ids = set(gdf_reduced['link'].astype(str))
        print(f"  Found {len(reduced_link_ids)} links with reduced capacity")
        return reduced_link_ids
        
    except Exception as e:
        print(f"ERROR: Failed to load reduced capacity file {geojson_file}: {e}")
        return set()

################################################# ↓ GDF IO ↓ #################################################

def prepare_gdf(df, gdf_input):
    gdf = gdf_input[['link', 'geometry']].merge(df, on='link', how='left')
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
    gdf.crs = gdf_input.crs
    return gdf

def get_basic_edge_attributes(capacity_base_case, gdf, required_modes_on_links):
    
    # Create a mask for each required mode and combine with OR logic
    mode_masks = [gdf['modes'].str.contains(mode) for mode in required_modes_on_links]
    combined_mask = mode_masks[0]
    for mask in mode_masks[1:]:
        combined_mask = combined_mask | mask
    
    capacities_new = np.where(combined_mask, gdf['capacity'], 0) # capacity is 0 for links that are not used by the required modes
    capacity_reduction = capacities_new - capacity_base_case # check the sign of this
    highway_raw = gdf['highway'].apply(lambda x: highway_mapping.get(x, -1)).values
    highway_clustered = np.vectorize(highway_cluster_mapping.get)(highway_raw)
    
    # One-hot encode into 6 classes
    highway_onehot = np.eye(6)[highway_clustered]  # shape: (N, 6)
    return capacities_new, capacity_reduction, highway_onehot

def get_capacity_and_freespeed_base_case(links_base_case, required_modes_on_links):
    mode_masks = [links_base_case['modes'].str.contains(mode) for mode in required_modes_on_links]
    combined_mask = mode_masks[0]
    for mask in mode_masks[1:]:
        combined_mask = combined_mask | mask
    capacity_base_case = np.where(combined_mask, links_base_case['capacity'], 0)
    freespeed_base_case = np.where(combined_mask, links_base_case['freespeed'], 0)
    return capacity_base_case, freespeed_base_case

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

################################################# ↓ Compute Edge Features ↓ #################################################

def normalization_of_edge_features(data, normalization_type: str):
    """
    Normalize numpy array data using specified method.
    Args:
        data: numpy array with data to normalize
        normalization_type: 'mean_std', 'min_max', 'robust_normalization', 'none'
    """
    if normalization_type == 'mean_std':
        std = np.std(data)
        if std == 0:
            print('all values are same')
            return np.zeros_like(data)  # If all values are same, return zeros
        return (data - np.mean(data)) / std
    
    elif normalization_type == 'min_max':
        data_min, data_max = np.min(data), np.max(data)
        if data_max == data_min:
            print('all values are same')
            return np.zeros_like(data)  # If all values are same, return zeros
        return (data - data_min) / (data_max - data_min)
    
    elif normalization_type == 'robust_normalization':
        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        if iqr == 0:
            print('all values are same')
            return np.zeros_like(data)
        return (data - median) / iqr
    
    elif normalization_type == 'none':
        return data
    else:
        raise ValueError(f"Invalid normalization type: {normalization_type}")

def compute_target_tensor_only_edge_features(vol_base_case, gdf, column_name: str, normalization_type: str):
    
    edge_car_volume_difference = gdf['vol_car'].values - vol_base_case # vol_base_case is already rounded to integer
    
    if column_name == 'vol_car':
        # Keep continuous values for training - round only at inference
        # Sign preserved: +500.3 cars vs -200.7 cars
        data_to_normalize = edge_car_volume_difference
        
    elif column_name == 'vol_car_percentage':
        # Division by base case to get percentage change  
        # Sign preserved: +50% vs -30%
        epsilon = 1 # adjust as needed
        base_case_with_epsilon = vol_base_case.copy()
        zero_mask = vol_base_case == 0
        base_case_with_epsilon[zero_mask] = epsilon
        data_to_normalize = edge_car_volume_difference / base_case_with_epsilon
    
    # Use signed_log_normalized for both - preserves sign, compresses range
    normalized_data = normalization_of_edge_features(
        data_to_normalize, normalization_type
    )
    return torch.tensor(normalized_data, dtype=torch.float).unsqueeze(1)

def add_destinations_to_gdf(gdf, df_eqasim_trips, normalization_type, normalize_activities=True):
    """
    Add land use features to GeoDataFrame based on eqasim_trips data.
    Analyzes trip destinations to infer what types of activities are located near each link.
    
    Args:
        gdf: GeoDataFrame with road network data
        df_eqasim_trips: DataFrame with trip data
        normalize_activities: Whether to normalize activity features
        normalization_type: Type of normalization ('mean_std', 'min_max', 'signed_log_normalization')
    """
    gdf_extended = gdf.copy()
    
    if df_eqasim_trips is not None and not df_eqasim_trips.empty:
        # Check if required columns exist
        required_cols = ['destination_link_id', 'following_purpose']
        if all(col in df_eqasim_trips.columns for col in required_cols):
            
            # Land use analysis: Count destinations only (where activities are located)
            # Get all unique activity types from destinations (what's actually built there)
            all_activities = sorted(list(set(df_eqasim_trips['following_purpose'].unique())))  # Sort for deterministic order
            
            df_land_use_activities = pd.DataFrame()
            
            for activity in all_activities:
                # Count unique people going TO each link for this activity
                # This represents land use: "how many people use this link for this activity type"
                activity_destinations = df_eqasim_trips[
                    df_eqasim_trips['following_purpose'] == activity
                ].groupby(['destination_link_id', 'person_id'], sort=True).size().groupby('destination_link_id', sort=True).size()
                
                df_land_use_activities[activity] = activity_destinations
            
            df_land_use_activities = df_land_use_activities.fillna(0)
            
            # Normalize activity features if requested
            if normalize_activities and not df_land_use_activities.empty:
                for activity in all_activities:
                    if activity in df_land_use_activities.columns:
                        activity_data = df_land_use_activities[activity].values
                        normalized_data = normalization_of_edge_features(activity_data, normalization_type)
                        df_land_use_activities[f'{activity}_normalized'] = normalized_data
                        # Keep original values too for reference
                        df_land_use_activities[f'{activity}_original'] = activity_data

            # 3. Create identifier for links present in eqasim_trips
            all_trip_links = set(df_eqasim_trips['destination_link_id'].astype(str))
            
            # Convert link column to same type before merging
            gdf_extended['link'] = gdf_extended['link'].astype(str)
            df_land_use_activities.index = df_land_use_activities.index.astype(str)
            
            # 4. Add trip data identifier (1 if link appears in trips, 0 otherwise)
            gdf_extended['is_in_eqasim_trips'] = gdf_extended['link'].isin(all_trip_links).astype(int)
            
            # 5. Merge land use activity features
            gdf_extended = gdf_extended.merge(df_land_use_activities, left_on='link', right_index=True, how='left').fillna(0).infer_objects(copy=False)

            activity_destination_names = [f'{activity}_normalized' for activity in all_activities] + ['is_in_eqasim_trips']
    
    return gdf_extended, activity_destination_names

def compute_edge_weights(vol_base_case):
    """
    Compute robust edge weights based on volume and a small epsilon for zeros.
    
    Args:
        vol_base_case (np.array): Base case volumes
        
    Returns:
        np.array: Edge weights in [0, 1] range
    """
    w = np.asarray(vol_base_case, dtype=np.float64).copy()
    w[~np.isfinite(w)] = 0.0      # NaN/Inf → 0
    w = np.maximum(w, 0.0)        # no negatives
    
    epsilon = 1.0  # adjust as needed
    zero_mask = w == 0
    w[zero_mask] = epsilon
    
    m = w.max()
    if m > 0:
        w = w / m
    else:
        w = np.zeros_like(w)

    return w.astype(np.float32)

def compute_laplacian_pe_once(edge_index, num_nodes, lap_pe_dim=16):
    """
    Compute Laplacian Positional Encoding once for a given graph structure.
    
    Args:
        edge_index: Edge index tensor of shape [2, num_edges]
        num_nodes: Number of nodes
        lap_pe_dim: Dimension of Laplacian PE
        
    Returns:
        torch.Tensor: Laplacian PE features of shape (num_nodes, lap_pe_dim)
    """
    try:
        print(f"DEBUG: Computing Laplacian PE with edge_index shape: {edge_index.shape}, num_nodes: {num_nodes}, lap_pe_dim: {lap_pe_dim}")
        
        # Build sparse adjacency matrix
        row, col = edge_index
        data = torch.ones_like(row, dtype=torch.float32)
        adj = csr_matrix((data.cpu().numpy(), (row.cpu().numpy(), col.cpu().numpy())),
                         shape=(num_nodes, num_nodes), dtype=np.float64)  # ✅ Use float64 for better precision

        print(f"DEBUG: Adjacency matrix created with {adj.nnz} non-zero entries.")

        # Check for disconnected graph (no edges) - raise error instead of masking
        if adj.nnz == 0:
            raise ValueError(f"Graph has no edges! Cannot compute Laplacian PE. "
                           f"Graph has {num_nodes} nodes but 0 edges. "
                           f"Check your edge_index construction.")

        # Compute Laplacian (combinatorial) using sparse methods
        L = csgraph_laplacian(adj, normed=False, return_diag=False)
        print(f"DEBUG: Laplacian matrix computed.")

        # Compute the smallest lap_pe_dim + 1 eigenvalues/eigenvectors using sparse solver
        eigenvalues, eigenvectors = eigsh(L, k=lap_pe_dim + 1, which='SM', return_eigenvectors=True)  # Add tolerance and max iterations for stability
        print(f"DEBUG: Eigenvalues: {eigenvalues}")
        print(f"DEBUG: Eigenvectors shape: {eigenvectors.shape}")
        
         # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, sorted_indices[1:lap_pe_dim + 1]]  # Skip eigenvalue 0

        # Handle case where we have fewer eigenvectors than requested
        if eigenvectors.shape[1] == 0:
            raise ValueError("No non-trivial eigenvectors found")
        if eigenvectors.shape[1] < lap_pe_dim:
            padding = np.zeros((num_nodes, lap_pe_dim - eigenvectors.shape[1]))
            eigenvectors = np.concatenate([eigenvectors, padding], axis=1)

        # Stabilize signs based on maximum absolute value index
        for i in range(eigenvectors.shape[1]):
            max_abs_idx = np.argmax(np.abs(eigenvectors[:, i]))
            if eigenvectors[max_abs_idx, i] < 0:
                eigenvectors[:, i] *= -1

        # Convert back to tensor (on CPU during preprocessing)
        lap_pe = torch.tensor(eigenvectors, dtype=torch.float32)
        print(f"DEBUG: Laplacian PE computed with shape: {lap_pe.shape}, mean: {lap_pe.mean():.6f}, std: {lap_pe.std():.6f}")

    except Exception as e:
        print(f"ERROR: Failed to compute Laplacian PE: {e}")
        print(f"Graph info: {num_nodes} nodes, edge_index shape: {edge_index.shape}")
        print(f"Edge index range: [{edge_index.min().item()}, {edge_index.max().item()}]")
        if hasattr(adj, 'nnz'):
            print(f"Adjacency matrix: {adj.nnz} non-zero entries")
        raise  # Re-raise the exception to stop execution

    return lap_pe

################################################# ↓ GEOMETRY ↓ #################################################

def multipolygon_to_polygon(geom):
    '''
    This function converts a MultiPolygon to a Polygon with the largest connected area.
    A MultiPolygon with 2 Polygons inside will return the Polygon with the largest area.(z.B. Stadt Bamberg had 2 disconnected polygons, we only consider the largest one)
    '''
    return max(geom.geoms, key=lambda p: p.area)

def modify_geodataframe(gdf):
    '''
    This function modifies the zones geodataframe to ensure it is in the correct CRS 
    and has the correct columns. 
    Also it applies the 'multipolygon_to_polygon' function to all the geometries in the geodataframe so that 
    the geometries are all Polygons and not MultiPolygons.
    '''
    if (gdf.geometry.apply(lambda x: x.geom_type == "MultiPolygon")).any():
        gdf["geometry"] = gdf.geometry.apply(multipolygon_to_polygon)
    gdf["area"] = gdf.geometry.area
    gdf["perimetre"] = gdf.geometry.length
    gdf["zone_id"] = range(1, len(gdf)+1) #zone id
    zones_gdf = gdf[["zone_id", "area", "perimetre", "geometry"]]
    
    # Ensure the data is in the correct CRS (EPSG:25832) (VERY IMPORTANT)
    if zones_gdf.crs != "EPSG:25832": # Should match with the CRS of the Network Geodataframe
        zones_gdf = zones_gdf.to_crs(epsg=25832)
    return zones_gdf

def merge_edges_and_zones(gdf_csv, zones_gdf, is_in_stadt):
    '''
    This function merges network edges with zones using spatial join.
    input:
        gdf_csv: GeoDataFrame containing network edges with your specific columns
        zones_gdf: GeoDataFrame containing zone polygons
    output:
        GeoDataFrame with edges and their intersecting zones
    '''
    # Perform spatial join
    gdf_edges_with_zones = gpd.sjoin(gdf_csv, zones_gdf, how='left', predicate='intersects')
    
    # Get all columns from the original gdf_csv except 'link' since it's the groupby column
    original_columns = [col for col in gdf_csv.columns.tolist() if col != 'link']
    
    # Create aggregation dictionary with 'first' for all original columns
    agg_dict = {col: 'first' for col in original_columns}
    
    # Add the zone_id aggregation - handle NaN values properly
    agg_dict['zone_id'] = lambda x: [int(i) for i in x.dropna()] if not x.dropna().empty else []
    
    gdf_edges_with_zones = gdf_edges_with_zones.groupby('link').agg(agg_dict).reset_index()

    # Ensure it's a GeoDataFrame
    gdf_edges_with_zones = gpd.GeoDataFrame(
        gdf_edges_with_zones, 
        geometry='geometry', 
        crs='EPSG:25832'
    )
    # Return only the geodataframe where zone_id is in the stadt (1)
    if is_in_stadt:
        # Check if any zone_id contains 1 (since zone_id contains lists/tuples)
        gdf_edges_with_zones = gdf_edges_with_zones[gdf_edges_with_zones['zone_id'].apply(lambda x: 1 in x if x else False)]

    return gdf_edges_with_zones

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

################################################## ↓ MISC ↓ #################################################

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

def save_scaler_params(city, vol_base_case, project_root):
    """
    Save per-city scaler parameters to JSON file for reproducibility (only for basecase).
    
    Args:
        city (str): City name
        vol_base_case (np.array): Base case volumes
        project_root (str): Project root path
    """
    # Calculate scaler parameters
    mean_val = np.mean(vol_base_case)
    std_val = np.std(vol_base_case)
    min_val = np.min(vol_base_case)
    max_val = np.max(vol_base_case)
    
    # Calculate robust scaler parameters (median and IQR)
    median_val = np.median(vol_base_case)
    q75, q25 = np.percentile(vol_base_case, [75, 25])
    iqr_val = q75 - q25
    
    scaler_params = {
        'city': city,
        'mean': float(mean_val),
        'std': float(std_val),
        'min': float(min_val),
        'max': float(max_val),
        'median': float(median_val),
        'q25': float(q25),
        'q75': float(q75),
        'iqr': float(iqr_val),
        'count': int(len(vol_base_case)),
        'description': 'Scaler parameters for vol_base_case normalization'
    }
    
    # Create directory if it doesn't exist
    save_dir = os.path.join(project_root, 'data', 'inductive_data', 'training_data', city)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save to JSON file
    json_path = os.path.join(save_dir, f'{city}_scaler_params.json')
    with open(json_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    
    print(f"Saved scaler parameters for {city} to: {json_path}")
    print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    print(f"  Min: {min_val:.4f}, Max: {max_val:.4f}")
    print(f"  Median: {median_val:.4f}, IQR: {iqr_val:.4f}")

# Clean edge_index for basecase links of a city
def clean_duplicates_based_on_modes(gdf):
    """
    Cleans the network by:
    1. First removing full duplicates (same from_node, to_node, geometry, modes)
    2. For remaining duplicates with same node pairs but different modes:
       - Keep the entry with modes that contain both "car" and "car_passenger"
       - If multiple or none meet this criteria, keep the one with the longest modes string
    
    Args: 
        file_path: Path to the CSV file
        
    Returns:
        The cleaned DataFrame   
    """
    # Load the CSV file
    df = gdf.copy()
    
    print(f"Total edges in original file: {len(df)}")
    
    # Step 1: First identify duplicates by the key columns
    key_columns = ["from_node", "to_node", "geometry", "modes"]
    full_duplicates_mask = df.duplicated(subset=key_columns, keep=False)
    full_duplicates = df[full_duplicates_mask]
    
    print(f"\nFull duplicates (same from_node, to_node, geometry, modes): {len(full_duplicates)}")
    
    # Show examples of full duplicates
    if len(full_duplicates) > 0:
        print("\n=== EXAMPLES OF FULL DUPLICATES ===")
        for i, (_, group) in enumerate(full_duplicates.groupby(key_columns)):
            if i < 3:  # Show first 3 examples
                print(f"\nFull Duplicate Group {i+1}:")
                important_cols = ['from_node', 'to_node', 'modes', 'vol_car']
                if 'link' in group.columns:
                    important_cols.insert(0, 'link')
                print(group[important_cols].to_string())
            else:
                remaining = len(full_duplicates.groupby(key_columns)) - 3
                print(f"\n... and {remaining} more full duplicate groups (not shown) ...")
                break
    
    # Now process these duplicates - keep the highest vol_car
    df_cleaned = df[~full_duplicates_mask].copy()
    
    for _, group in full_duplicates.groupby(key_columns):
        # Keep the row with the highest vol_car
        max_idx = group['vol_car'].idxmax()
        row_to_keep = group.loc[max_idx]
        df_cleaned = pd.concat([df_cleaned, pd.DataFrame([row_to_keep])])
    
    print(f"Edges after removing full duplicates (based on higher car volume): {len(df_cleaned)}")
    print(f"Number of full duplicates removed: {len(df) - len(df_cleaned)}")
    
    # Step 2: Now look for edges with the same from_node and to_node (leftover duplicates)
    node_columns = ["from_node", "to_node"]
    leftover_duplicates_mask = df_cleaned.duplicated(subset=node_columns, keep=False)
    leftover_duplicates = df_cleaned[leftover_duplicates_mask]
    
    print(f"\nLeftover duplicates (same from_node, to_node only): {len(leftover_duplicates)}")
    print(f"Number of unique node pairs with leftover duplicates: {len(leftover_duplicates.groupby(node_columns))}")
    
    # Show examples of leftover duplicates
    if len(leftover_duplicates) > 0:
        print("\n=== EXAMPLES OF LEFTOVER DUPLICATES ===")
        for i, ((from_node, to_node), group) in enumerate(leftover_duplicates.groupby(node_columns)):
            if i < 3:  # Show first 3 examples
                print(f"\nLeftover Duplicate Group {i+1}: from_node={from_node}, to_node={to_node}")
                important_cols = ['modes', 'vol_car']
                if 'link' in group.columns:
                    important_cols.insert(0, 'link')
                print(group[important_cols].to_string())
            else:
                remaining = len(leftover_duplicates.groupby(node_columns)) - 3
                print(f"\n... and {remaining} more leftover duplicate groups (not shown) ...")
                break
    
    # Step 3: Process leftover duplicates based on modes criteria
    df_final = df_cleaned[~leftover_duplicates_mask].copy()
    
    # Track selection statistics
    selection_stats = {
        'total_groups': 0,
        'selected_by_car_modes': 0,
        'selected_by_length': 0,
        'selected_by_vol_car': 0
    }
    
    for (from_node, to_node), group in leftover_duplicates.groupby(node_columns):
        selection_stats['total_groups'] += 1
        
        # Convert modes to string and check for "car" OR "car_passenger"
        # (ensuring robust string comparison)
        group['has_car_or_passenger'] = group['modes'].apply(
            lambda x: ('"car"' in str(x) or "'car'" in str(x) or ",car," in str(x) or "[car" in str(x)) or 
                      ('"car_passenger"' in str(x) or "'car_passenger'" in str(x) or 
                       ",car_passenger," in str(x) or "[car_passenger" in str(x))
        )
        
        # Check if any entry has car or car_passenger modes
        if group['has_car_or_passenger'].any():
            # Filter to entries with car or car_passenger modes
            car_mode_entries = group[group['has_car_or_passenger']]
            
            # Among entries with car or car_passenger modes, select by highest vol_car
            row_to_keep = car_mode_entries.loc[car_mode_entries['vol_car'].idxmax()]
            selection_stats['selected_by_car_modes'] += 1
        else:
            # No entry has car or car_passenger modes - select by highest vol_car among all
            row_to_keep = group.loc[group['vol_car'].idxmax()]
            selection_stats['selected_by_vol_car'] += 1
        
        # Add the selected row to the final DataFrame
        df_final = pd.concat([df_final, pd.DataFrame([row_to_keep.drop('has_car_or_passenger')])])
    
    print(f"\nFinal edges after processing all duplicates: {len(df_final)}")
    print(f"Total edges removed: {len(df) - len(df_final)}")
    
    # Print selection statistics
    print("\nSelection criteria statistics:")
    print(f"Total duplicate groups processed: {selection_stats['total_groups']}")
    if selection_stats['total_groups'] > 0:
        print(f"Selected by having car/car_passenger modes: {selection_stats['selected_by_car_modes']} ({selection_stats['selected_by_car_modes']/selection_stats['total_groups']*100:.1f}%)")
        print(f"Selected by highest vol_car (no car modes): {selection_stats['selected_by_vol_car']} ({selection_stats['selected_by_vol_car']/selection_stats['total_groups']*100:.1f}%)")
    else:
        print("No duplicates found - no selection criteria applied.")
    
    # Step 4: Verify no duplicates remain
    final_duplicates_mask = df_final.duplicated(subset=node_columns, keep=False)
    final_duplicates = df_final[final_duplicates_mask]
    
    if len(final_duplicates) > 0:
        print(f"\n⚠️ WARNING: {len(final_duplicates)} duplicate edges remain!")
        print(f"Number of unique node pairs with remaining duplicates: {len(final_duplicates.groupby(node_columns))}")
        
        # Analyze remaining duplicates
        print("\n=== REMAINING DUPLICATES ===")
        for i, ((from_node, to_node), group) in enumerate(final_duplicates.groupby(node_columns)):
            print(f"\nRemaining Duplicate {i+1}: from_node={from_node}, to_node={to_node}")
            print(f"Number of edges: {len(group)}")
            
            # Display the group
            important_cols = ['modes', 'vol_car', 'geometry']
            if 'link' in group.columns:
                important_cols.insert(0, 'link')
            
            print(group[important_cols].to_string())
            
            # Only show a few examples
            if i >= 4:
                remaining = len(final_duplicates.groupby(node_columns)) - (i+1)
                print(f"\n... and {remaining} more duplicate pairs (not shown) ...")
                break
    else:
        print("\n✅ SUCCESS: No duplicates remain in the final network!")
    
    return df_final