"""
Process simulation data (from MATSim) for GNNs. Load basecase and simulated graphs (with policies applied in various district combinations),
convert them to dual line graphs, and compute specified edge features. Save as PyTorch Geometric data batches for efficient loading and training.

Here we specify all features, then run_models can be called with a reduced set. Note that, for example, the flag "use_allowed_modes" is accessed from the run_models script.
"""


import os
import sys
import json
from enum import IntEnum
import shutil
import random
import glob
import geopandas as gpd


import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

import torch
from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data
from process_simulations_for_gnn import *
# Set seeds for reproducibility
np.random.seed(13)
random.seed(13)
torch.manual_seed(13)

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from data_preprocessing.help_functions import *

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

######control center#######
cities_to_process = ['muenchen','neuulm','rosenheim','schweinfurt','kempten','ingolstadt','landshut','regensburg','bamberg','bayreuth','erlangen','fuerth','wuerzburg']
all_cities = ['augsburg','nuernberg','aschaffenburg']

is_in_stadt=True #if true, include only the edges that are in the stadt, else include edges in stadt and landkreis
batch_size = 256 # Do processing in batches to avoid memory issues
seed = 3 # Seed for Bavarian Simulations
hex_sizes = [500] # Hexagon Sizes for Bavarian Simulations
required_modes_on_links = ['car', 'car_passenger'] # Capacity will be reduced on links that have at least one of these modes
use_allowed_modes = True # Flag to use allowed modes as edge features
use_destination_activity = True # Flag to use destination activity as edge features

# Test with just one city
#all_cities = ['rosenheim','muenchen','augsburg', 'nuernberg','neuulm']  # Change this to test different cities
#cities_1=['nuernberg', 'augsburg', 'muenchen','schweinfurt', 'aschaffenburg', 'wuerzburg', 'bamberg', 'bayreuth', 'erlangen', 'fuerth', 'kempten','landshut', 'ingolstadt', 'regensburg', 'neuulm',rosenheim]
#cities_rest=[]
#all_cities = ['landshut', 'ingolstadt', 'regensburg', 'rosenheim']
#target_feature = 'vol_car_percentage' #other options: 'vol_car'
#target_feature_normalization_type = 'signed_log_normalization' #other options: 'mean_std', 'min_max','none'
x_normalization_type = 'mean_std' #other options: 'min_max', 'robust_normalization', 'mean_std'
target_normalization_type = 'signed_log_normalization' #other options: 'mean_std', 'min_max', 'signed_log_normalization', 'none'
############################

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

class EdgeFeatures(IntEnum):
    VOL_BASE_CASE = 0
    CAPACITY_BASE_CASE = 1
    CAPACITY_REDUCTION = 2
    FREESPEED = 3
    HIGHWAY_PRIMARY = 4
    HIGHWAY_SECONDARY = 5
    HIGHWAY_TERTIARY = 6
    HIGHWAY_RESIDENTIAL = 7
    HIGHWAY_PT = 8
    HIGHWAY_OTHER = 9
    LENGTH = 10
    ALLOWED_MODE_CAR = 11
    ALLOWED_MODE_CAR_PASSENGER = 12
    ALLOWED_MODE_BUS = 13
    ALLOWED_MODE_PT = 14
    ALLOWED_MODE_TRAIN = 15
    ALLOWED_MODE_TRAM = 16
    ALLOWED_MODE_RAIL = 17
    ALLOWED_MODE_SUBWAY = 18
    ALLOWED_MODE_FUNICULAR = 19
    # Activity features (combined origin + destination for each activity type)
    HOME=20
    WORK=21
    EDUCATION=22
    LEISURE=23
    SHOP=24
    OTHER=25
    OUTSIDE=26
    # Trip data availability
    IS_IN_EQASIM_TRIPS=27
    #EIGN specific features
    NET_FLOW = 28


def create_aggregated_edges_for_eign(links_gdf, activity_features=None):
    """
    Create aggregated edges for EIGN where each unique node pair (u,v) has only one edge.
    For undirected pairs, we aggregate the features. For directed edges, we keep them as is.

    Args:
        links_gdf: GeoDataFrame with link data
        activity_features: List of activity feature column names to include in aggregation

    Returns:
    - edge_index: (2, num_edges) tensor with unique edges
    - edge_features: dict with aggregated features for each edge
    - edge_is_directed: boolean tensor indicating if each edge is directed
    - edge_positions: positions for each aggregated edge
    """
    # Create node mapping
    nodes = pd.concat([links_gdf["from_node"], links_gdf["to_node"]]).unique()
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    # Create edge data with indices
    links_gdf = links_gdf.copy()
    links_gdf["from_idx"] = links_gdf["from_node"].map(node_to_idx)
    links_gdf["to_idx"] = links_gdf["to_node"].map(node_to_idx)

    # Group by edge pairs to identify undirected vs directed edges
    edge_groups = {}
    edge_positions = {}

    for idx, row in links_gdf.iterrows():
        u, v = row["from_idx"], row["to_idx"]

        # Create canonical edge representation (smaller node first)
        canonical_edge = tuple(sorted([u, v]))

        if canonical_edge not in edge_groups:
            edge_groups[canonical_edge] = []
            edge_positions[canonical_edge] = []

        edge_data = {
            "direction": (u, v),
            "vol_car": row["vol_car"] if pd.notna(row["vol_car"]) else 0.0,
            "capacity": (
                row["capacity"]
                if pd.notna(row["capacity"]) and ("car" in str(row.get("modes", "")) or "car_passenger" in str(row.get("modes", "")))
                else 0.0
            ),
            "length": row["length"] if pd.notna(row["length"]) else 0.0,
            "freespeed": (
                row["freespeed"]
                if pd.notna(row["freespeed"]) and ("car" in str(row.get("modes", "")) or "car_passenger" in str(row.get("modes", "")))
                else 0.0
            ),
            "highway": row["highway"] if pd.notna(row["highway"]) else "unknown",
            "modes": row["modes"] if pd.notna(row["modes"]) else "",
        }
        
        # Add activity features if provided
        if activity_features:
            for feature in activity_features:
                if feature in row:
                    edge_data[feature] = row[feature] if pd.notna(row[feature]) else 0.0
                else:
                    edge_data[feature] = 0.0
        
        edge_groups[canonical_edge].append(edge_data)

    # Process aggregated edges
    aggregated_edges = []
    aggregated_features = {
        "vol_base_case": [],
        "capacity_base_case": [],
        "length": [],
        "freespeed": [],
        "highway_onehot": [],
        "net_flow": [],
        "allowed_modes": [],
    }
    
    # Add activity features to aggregated features if provided
    if activity_features:
        for feature in activity_features:
            aggregated_features[feature] = []
    aggregated_positions = []
    edge_is_directed = []

    for canonical_edge, edge_data_list in edge_groups.items():
        u_canonical, v_canonical = canonical_edge

        # Determine if this is a directed or undirected edge
        directions = [data["direction"] for data in edge_data_list]
        has_both_directions = (u_canonical, v_canonical) in directions and (
            v_canonical,
            u_canonical,
        ) in directions

        if has_both_directions:
            # Undirected edge - aggregate features
            edge_is_directed.append(False)

            # Find data for both directions
            forward_data = next(
                data
                for data in edge_data_list
                if data["direction"] == (u_canonical, v_canonical)
            )
            backward_data = next(
                data
                for data in edge_data_list
                if data["direction"] == (v_canonical, u_canonical)
            )

            # Aggregate features
            vol_forward = (
                forward_data["vol_car"] if forward_data["vol_car"] is not None else 0.0
            )
            vol_backward = (
                backward_data["vol_car"]
                if backward_data["vol_car"] is not None
                else 0.0
            )
            net_flow = (
                vol_forward - vol_backward
            )  # Net flow in u_canonical -> v_canonical direction

            # For other features, take average or sum as appropriate
            # Handle potential None/NaN values before division
            forward_capacity = (
                forward_data["capacity"]
                if forward_data["capacity"] is not None
                else 0.0
            )
            backward_capacity = (
                backward_data["capacity"]
                if backward_data["capacity"] is not None
                else 0.0
            )
            capacity = (forward_capacity + backward_capacity) / 2

            forward_length = (
                forward_data["length"] if forward_data["length"] is not None else 0.0
            )
            backward_length = (
                backward_data["length"] if backward_data["length"] is not None else 0.0
            )
            length = (forward_length + backward_length) / 2

            forward_freespeed = (
                forward_data["freespeed"]
                if forward_data["freespeed"] is not None
                else 0.0
            )
            backward_freespeed = (
                backward_data["freespeed"]
                if backward_data["freespeed"] is not None
                else 0.0
            )
            freespeed = (forward_freespeed + backward_freespeed) / 2

            # For highway type, take the more significant one (higher numeric value)
            highway_forward = highway_mapping.get(forward_data["highway"], -1)
            highway_backward = highway_mapping.get(backward_data["highway"], -1)
            highway_raw = max(highway_forward, highway_backward)
            # Apply highway clustering and one-hot encoding
            highway_clustered = highway_cluster_mapping.get(highway_raw, 5)  # Default to OTHER
            highway_onehot = np.eye(6)[highway_clustered]  # shape: (1, 6)

            # For modes, combine them
            modes_combined = (
                str(forward_data["modes"]) + "," + str(backward_data["modes"])
            )
            
            # Aggregate activity features for undirected edges (take average)
            activity_values = {}
            if activity_features:
                for feature in activity_features:
                    forward_val = forward_data.get(feature, 0.0)
                    backward_val = backward_data.get(feature, 0.0)
                    
                    if feature == "is_in_eqasim_trips":
                        # if in either direction, there exist an activity, it means that the link is in the eqasim trips
                        activity_values[feature] = 1.0 if (forward_val > 0 or backward_val > 0) else 0.0
                    else:
                        # if there is an activity in either direction, take the max of the two
                        activity_values[feature] = max(forward_val, backward_val) 

        else:
            # Directed edge
            edge_is_directed.append(True)
            edge_data = edge_data_list[0]  # Only one direction exists

            vol_forward = (
                edge_data["vol_car"] if edge_data["vol_car"] is not None else 0.0
            )
            net_flow = vol_forward  # No backward flow for directed edges

            # Handle potential None/NaN values for directed edges
            capacity = (
                edge_data["capacity"] if edge_data["capacity"] is not None else 0.0
            )
            length = edge_data["length"] if edge_data["length"] is not None else 0.0
            freespeed = (
                edge_data["freespeed"] if edge_data["freespeed"] is not None else 0.0
            )
            highway_raw = highway_mapping.get(edge_data["highway"], -1)
            # Apply highway clustering and one-hot encoding
            highway_clustered = highway_cluster_mapping.get(highway_raw, 5)  # Default to OTHER
            highway_onehot = np.eye(6)[highway_clustered]  # shape: (1, 6)
            modes_combined = (
                str(edge_data["modes"]) if edge_data["modes"] is not None else ""
            )
            
            # Get activity features for directed edges (use original values)
            activity_values = {}
            if activity_features:
                for feature in activity_features:
                    activity_values[feature] = edge_data.get(feature, 0.0)

            # Ensure edge orientation matches direction for directed edges
            if edge_data["direction"] != (u_canonical, v_canonical):
                # Flip the canonical edge to match the actual direction
                u_canonical, v_canonical = v_canonical, u_canonical
                net_flow = -net_flow  # Flip net flow to match new orientation

        # Add to aggregated data
        aggregated_edges.append([u_canonical, v_canonical])
        aggregated_features["vol_base_case"].append(
            abs(vol_forward) if has_both_directions else vol_forward
        )
        aggregated_features["capacity_base_case"].append(capacity)
        aggregated_features["length"].append(length)
        aggregated_features["freespeed"].append(freespeed)
        aggregated_features["highway_onehot"].append(highway_onehot)
        aggregated_features["net_flow"].append(net_flow)

        # Encode allowed modes for aggregated edge
        allowed_modes = encode_modes_from_string(modes_combined)
        aggregated_features["allowed_modes"].append(allowed_modes)
        
        # Add aggregated activity features
        if activity_features:
            for feature in activity_features:
                aggregated_features[feature].append(activity_values.get(feature, 0.0))

    # Convert to tensors
    edge_index = torch.tensor(aggregated_edges, dtype=torch.long).t().contiguous()
    edge_is_directed = torch.tensor(edge_is_directed, dtype=torch.bool)

    # Convert features to tensors
    for key in aggregated_features:
        if key == "allowed_modes":
            # Handle allowed modes separately as it's a list of lists
            aggregated_features[key] = torch.stack(
                [torch.tensor(modes) for modes in aggregated_features[key]]
            )
        else:
            # Convert list to numpy array first to avoid slow tensor creation warning
            if key == "highway_onehot":
                # Stack the one-hot arrays into a single numpy array
                aggregated_features[key] = torch.tensor(
                    np.array(aggregated_features[key]), dtype=torch.float
                )
            else:
                aggregated_features[key] = torch.tensor(
                    aggregated_features[key], dtype=torch.float
                )

    return edge_index, aggregated_features, edge_is_directed

def encode_modes_from_string(modes_str):
    """Helper function to encode allowed modes from a string"""
    allowed_modes = [0, 0, 0, 0, 0, 0,0,0,0]  # car, bus, pt, train, rail, subway

    modes_lower = modes_str.lower()
    if "car" in modes_lower:
        allowed_modes[0] = 1
    if "car_passenger" in modes_lower:
        allowed_modes[1] = 1
    if "bus" in modes_lower:
        allowed_modes[2] = 1
    if "pt" in modes_lower:
        allowed_modes[3] = 1
    if "train" in modes_lower:
        allowed_modes[4] = 1
    if "tram" in modes_lower:
        allowed_modes[5] = 1
    if "rail" in modes_lower:
        allowed_modes[6] = 1
    if "subway" in modes_lower:
        allowed_modes[7] = 1
    if "funicular" in modes_lower:
        allowed_modes[8] = 1
    return allowed_modes


def process_result_dic_eign(city,
    result_dic,
    result_dic_mode_stats,
    save_path=None,
    batch_size=1,
    links_base_case=None,
    gdf_basecase_mean_mode_stats=None,
    activity_destination_names=None,
): 
    # Create aggregated edges for EIGN - only one edge per node pair
    aggregated_data = create_aggregated_edges_for_eign(links_base_case, activity_features=activity_destination_names)
    edge_index, base_features, edge_is_directed_base = (
        aggregated_data
    )
    _, stacked_edge_geometries_tensor, _, _ ,_= get_link_geometries(links_base_case)

    os.makedirs(save_path, exist_ok=True)
    datalist = []

    for key, df in result_dic.items():
        if isinstance(df, pd.DataFrame) and key != "base_network_no_policies":
            # Create aggregated edges for the current simulation data
            sim_aggregated_data = create_aggregated_edges_for_eign(df,activity_features=None) #no need for activity features here as they are the same as in the base case
            sim_edge_index, sim_features, _ = sim_aggregated_data

            # Ensure consistent edge ordering between base case and simulation
            if not torch.equal(edge_index, sim_edge_index):
                print(f"Warning: Edge indices don't match for {key}. Skipping.")
                continue

            # Calculate signed features (net flow changes)
            net_flow_base = base_features["net_flow"] #shape: (num_edges,)
            net_flow_base_normalized = torch.tensor(
                normalization_of_edge_features(net_flow_base.numpy(), x_normalization_type), # normalization should be same as other x features
                dtype=torch.float
            )
            net_flow_sim = sim_features["net_flow"]
            net_flow_change = net_flow_sim - net_flow_base
            net_flow_change_normalized = torch.tensor(
                normalization_of_edge_features(net_flow_change.numpy(), target_normalization_type), 
                dtype=torch.float
            )

            # Calculate unsigned features changes
            vol_change = sim_features["vol_base_case"] - base_features["vol_base_case"]
            vol_change_normalized = torch.tensor(
                normalization_of_edge_features(vol_change.numpy(), target_normalization_type), 
                dtype=torch.float
            )
            capacity_change = (
                sim_features["capacity_base_case"] - base_features["capacity_base_case"]
            )
            
            # Convert capacity change to binary: 0 if no change, 1 if any change
            capacity_change_binary = (capacity_change != 0).float()
            
            # Count reduced links for cross-checking
            num_reduced_links = int(capacity_change_binary.sum().item())
            print(f"Simulation {city,key}: {num_reduced_links} aggregated edges have capacity reduction")

            # Normalize base case features before adding to edge features
            vol_base_case_normalized = normalization_of_edge_features(base_features["vol_base_case"].numpy(), x_normalization_type)
            capacity_base_case_normalized = normalization_of_edge_features(base_features["capacity_base_case"].numpy(), x_normalization_type)
            freespeed_base_case_normalized = normalization_of_edge_features(base_features["freespeed"].numpy(), x_normalization_type)
            length_normalized = normalization_of_edge_features(base_features["length"].numpy(), x_normalization_type)

            # Prepare unsigned features (invariant)
            edge_feature_dict = {
                EdgeFeatures.VOL_BASE_CASE: torch.tensor(vol_base_case_normalized, dtype=torch.float),
                EdgeFeatures.CAPACITY_BASE_CASE: torch.tensor(capacity_base_case_normalized, dtype=torch.float),
                EdgeFeatures.CAPACITY_REDUCTION: capacity_change_binary.float(),
                EdgeFeatures.FREESPEED: torch.tensor(freespeed_base_case_normalized, dtype=torch.float),
                EdgeFeatures.LENGTH: torch.tensor(length_normalized, dtype=torch.float),
            }
            
            # Add highway one-hot encoding
            highway_onehot = base_features["highway_onehot"]
            highway_feature_keys = [
                EdgeFeatures.HIGHWAY_PRIMARY,
                EdgeFeatures.HIGHWAY_SECONDARY,
                EdgeFeatures.HIGHWAY_TERTIARY,
                EdgeFeatures.HIGHWAY_RESIDENTIAL,
                EdgeFeatures.HIGHWAY_PT,
                EdgeFeatures.HIGHWAY_OTHER
            ]
            for i, highway_key in enumerate(highway_feature_keys):
                edge_feature_dict[highway_key] = highway_onehot[:, i].float()

            # Prepare signed features (equivariant)
            edge_feature_dict_signed = {
                EdgeFeatures.NET_FLOW: net_flow_base_normalized,
            }

            if use_allowed_modes:
                allowed_modes_tensor = base_features["allowed_modes"]
                edge_feature_dict.update({
                    EdgeFeatures.ALLOWED_MODE_CAR: allowed_modes_tensor[:, 0],
                    EdgeFeatures.ALLOWED_MODE_CAR_PASSENGER: allowed_modes_tensor[:, 1],
                    EdgeFeatures.ALLOWED_MODE_BUS: allowed_modes_tensor[:, 2],
                    EdgeFeatures.ALLOWED_MODE_PT: allowed_modes_tensor[:, 3],
                    EdgeFeatures.ALLOWED_MODE_TRAIN: allowed_modes_tensor[:, 4],
                    EdgeFeatures.ALLOWED_MODE_TRAM: allowed_modes_tensor[:, 5],
                    EdgeFeatures.ALLOWED_MODE_RAIL: allowed_modes_tensor[:, 6],
                    EdgeFeatures.ALLOWED_MODE_SUBWAY: allowed_modes_tensor[:, 7],
                    EdgeFeatures.ALLOWED_MODE_FUNICULAR: allowed_modes_tensor[:, 8],
            })
            
            # Add activity features if available
            if use_destination_activity:
                edge_feature_dict.update({
                    EdgeFeatures.HOME: base_features["home_normalized"],
                    EdgeFeatures.WORK: base_features["work_normalized"],
                    EdgeFeatures.EDUCATION: base_features["education_normalized"],
                    EdgeFeatures.LEISURE: base_features["leisure_normalized"],
                    EdgeFeatures.SHOP: base_features["shop_normalized"],
                    EdgeFeatures.OTHER: base_features["other_normalized"],
                    EdgeFeatures.OUTSIDE: base_features["outside_normalized"],
                    EdgeFeatures.IS_IN_EQASIM_TRIPS: base_features["is_in_eqasim_trips"],
                })
            # Create the edge tensors by iterating through the EdgeFeatures enum
            edge_tensor = [
                edge_feature_dict[feature]
                for feature in EdgeFeatures
                if feature in edge_feature_dict
            ]

            edge_tensor_signed = [
                edge_feature_dict_signed[feature]
                for feature in EdgeFeatures
                if feature in edge_feature_dict_signed
            ]

            # Stack the tensors
            edge_tensor = torch.stack(edge_tensor, dim=1)
            edge_tensor_signed = torch.stack(edge_tensor_signed, dim=1)

            # Create data object
            data = Data(edge_index=edge_index)
            data.num_nodes = edge_index.shape[1]
            data.x = edge_tensor
            data.x_signed = edge_tensor_signed #shape: (num_edges, 1)
            data.pos = stacked_edge_geometries_tensor
            data.y = vol_change_normalized.unsqueeze(1)  # TODO:Volume change as target (need to use normalization tactics here)
            data.y_signed = net_flow_change_normalized.unsqueeze( # TODO: also think about normalization techniques here
                1
            )  # Net flow change as signed target
            data.edge_is_directed = edge_is_directed_base #shape: (num_edges,)
            
            # Add separate data attributes (using original unnormalized values)
            data.edge_weights = torch.tensor(compute_edge_weights(base_features["vol_base_case"]), dtype=torch.float32)  #shape: (num_edges,)
            data.unscaled_vol_base = base_features["vol_base_case"].float() 
            
            # Add metadata attributes
            data.city = city
            # Extract policy_region and scenario from tuple key
            if isinstance(key, tuple):
                data.policy_region = f"{key[0]}"
                data.scenario = f"{key[1]}"
            else:
                data.policy_region = str(key)
                data.scenario = str(key)

            df_mode_stats = result_dic_mode_stats.get(key)
            if df_mode_stats is not None:
                pd.set_option("display.float_format", lambda x: "%.10f" % x)
                numeric_cols_base_case = gdf_basecase_mean_mode_stats.select_dtypes(
                    include=[np.number]
                ).columns
                numeric_cols = df_mode_stats.select_dtypes(include=[np.number]).columns
                mode_stats_diff = (
                    df_mode_stats[numeric_cols].values
                    - gdf_basecase_mean_mode_stats[numeric_cols_base_case].values
                )
                mode_stats_tensor = torch.tensor(mode_stats_diff, dtype=torch.float)
                data.mode_stats_diff = mode_stats_tensor
                mode_stats_diff_perc = (
                    mode_stats_tensor
                    / gdf_basecase_mean_mode_stats[numeric_cols_base_case].values
                    * 100
                )
                data.mode_stats_diff_perc = mode_stats_diff_perc

            if validate_data_eign(data):
                datalist.append(data)
                if len(datalist) == 1:
                    print(f"\n=== DEBUG: First Graph Features for {city} ===")
                    # Debug: Print which features are being included
                    print(f"\n=== Features Included in Tensor ===")
                    print(f"Total features: {len(EdgeFeatures)}")
                    print(f"Feature names: {EdgeFeatures}")
                    print("=" * 30)
                    print(f"Unsigned Graph shape: {data.x.shape}")
                    print(f"Signed Graph shape: {data.x_signed.shape}")
                    print(f"Number of nodes: {data.num_nodes}")
                    print(f"Number of features: {data.x.shape[1]}")
                    print(f"Number of edges: {data.edge_index.shape[1]}")
                    print(f'Number of directed edges: {data.edge_is_directed.sum()}')
                    print(f"Target (vol_change [normalization: {target_normalization_type}]): {data.y.shape}")
                    print(f"Target (net_flow_change [normalization: {target_normalization_type}]): {data.y_signed.shape}")
                    print(f"Policy region: {data.policy_region}")
                    print(f"Scenario: {data.scenario}")
                    
                    # Print new data attributes
                    print(f"\n--- New Data Attributes ---")
                    print(f"unscaled_vol_base: {data.unscaled_vol_base.shape}")
                    print(f"edge_weights: {data.edge_weights.shape}")
                    print(f"unscaled_vol_base stats: mean={data.unscaled_vol_base.mean():.4f}, std={data.unscaled_vol_base.std():.4f}, "
                        f"min={data.unscaled_vol_base.min():.4f}, max={data.unscaled_vol_base.max():.4f}")
                    print(f"edge_weights stats: mean={data.edge_weights.mean():.4f}, std={data.edge_weights.std():.4f}, "
                        f"min={data.edge_weights.min():.4f}, max={data.edge_weights.max():.4f}")
                    
                    # Print feature statistics
                    print(f"\n--- Feature Statistics ---")
                    for i, feat_name in enumerate(EdgeFeatures):
                        if feat_name == EdgeFeatures.NET_FLOW:
                            feat_values = data.x_signed.numpy() #shape: (num_edges, 1)
                            print(f"{feat_name}: mean={feat_values.mean():.4f}, std={feat_values.std():.4f}, "
                                f"min={feat_values.min():.4f}, max={feat_values.max():.4f}")
                        else:   
                            feat_values = data.x[:, i].numpy()
                            print(f"{feat_name}: mean={feat_values.mean():.4f}, std={feat_values.std():.4f}, "
                                f"min={feat_values.min():.4f}, max={feat_values.max():.4f}")
                    
                    # Print target statistics
                    target_values_vol_change = data.y.numpy()
                    target_values_net_flow_change = data.y_signed.numpy()
                    print(f"\n--- Target Statistics ---")
                    print(f"Target (vol_change [normalization: {target_normalization_type}]): mean={target_values_vol_change.mean():.4f}, std={target_values_vol_change.std():.4f}, "
                        f"min={target_values_vol_change.min():.4f}, max={target_values_vol_change.max():.4f}")
                    print(f"Target (net_flow_change [normalization: {target_normalization_type}]): mean={target_values_net_flow_change.mean():.4f}, std={target_values_net_flow_change.std():.4f}, "
                        f"min={target_values_net_flow_change.min():.4f}, max={target_values_net_flow_change.max():.4f}")
                    
                    
                    # Check for NaN or Inf values
                    if torch.isnan(data.x).any():   
                        print("WARNING: NaN values found in features!")
                    if torch.isinf(data.x).any():
                        print("WARNING: Inf values found in features!")
                        
                    if torch.isnan(data.x_signed).any():
                        print("WARNING: NaN values found in signed features!")
                    if torch.isinf(data.x_signed).any():
                        print("WARNING: Inf values found in signed features!")
                        
                    if torch.isnan(data.y).any():
                        print("WARNING: NaN values found in target!")
                    if torch.isinf(data.y).any():
                        print("WARNING: Inf values found in target!")
                        
                    if torch.isnan(data.y_signed).any():
                        print("WARNING: NaN values found in target!")
                    if torch.isinf(data.y_signed).any():
                        print("WARNING: Inf values found in target!")
                        
                    if torch.isnan(data.unscaled_vol_base).any():
                        print("WARNING: NaN values found in unscaled_vol_base!")
                    if torch.isinf(data.unscaled_vol_base).any():
                        print("WARNING: Inf values found in unscaled_vol_base!")
                        
                    if torch.isnan(data.edge_weights).any():
                        print("WARNING: NaN values found in edge_weights!")
                    if torch.isinf(data.edge_weights).any():
                        print("WARNING: Inf values found in edge_weights!")
                    
                    print("=" * 50)
            else:
                print("Invalid EIGNgraph data")
                return None
    # Return the processed data list
    return datalist


def add_edge_is_directed(data: Data) -> Data:
    edge_index = data.edge_index
    src, dst = edge_index
    edges = torch.stack([src, dst], dim=1)

    # Create a set of reversed edges
    reversed_edges_set = {
        tuple(edge.tolist()) for edge in torch.stack([dst, src], dim=1)
    }

    # Mark each edge as directed if its reverse is not in the set
    is_directed = torch.tensor(
        [tuple(edge.tolist()) not in reversed_edges_set for edge in edges],
        dtype=torch.bool,
        device=edge_index.device,
    )

    return is_directed



def validate_data_eign(data: Data) -> bool:
    """Validation function for EIGN data with aggregated edges"""
    num_edges = data.edge_index.shape[1]

    # Check x (unsigned features)
    expected_unsigned_features = 6 if not use_allowed_modes else 28
    if data.x.shape != (num_edges, expected_unsigned_features):
        print(
            f"Invalid shape for data.x: expected ({num_edges}, {expected_unsigned_features}), got {data.x.shape}"
        )
        return False

    # Check x_signed (signed features)
    if data.x_signed.shape != (num_edges, 1):
        print(
            f"Invalid shape for data.x_signed: expected ({num_edges}, 1), got {data.x_signed.shape}"
        )
        return False

    # Check y (unsigned target)
    if data.y.shape != (num_edges, 1):
        print(
            f"Invalid shape for data.y: expected ({num_edges}, 1), got {data.y.shape}"
        )
        return False

    # Check y_signed (signed target)
    if data.y_signed.shape != (num_edges, 1):
        print(
            f"Invalid shape for data.y_signed: expected ({num_edges}, 1), got {data.y_signed.shape}"
        )
        return False

    # Check edge_index
    if data.edge_index.shape != (2, num_edges):
        print(
            f"Invalid shape for data.edge_index: expected (2, {num_edges}), got {data.edge_index.shape}"
        )
        return False

    # Check edge_is_directed
    if data.edge_is_directed.shape != (num_edges,):
        print(
            f"Invalid shape for data.edge_is_directed: expected ({num_edges},), got {data.edge_is_directed.shape}"
        )
        return False

    return True

def process_single_city(city, project_root, result_path, use_destination_activity, use_allowed_modes):
    
    """Process a single city and save its graph data."""
    print(f"\nProcessing city: {city}\n")
    
    # sim_input_paths should contain paths for individual simulations
    if city != 'paris':
        
        sim_input_paths = list()
        
        # Paths to compressed directories
        compressed_input_paths = [os.path.join(project_root, 'data','inductive_data','raw_data',city, f'compressed_{city}_hex_{hex_size}_seed_{seed}') for hex_size in hex_sizes]

        for path in compressed_input_paths:
            if os.path.exists(path) and os.path.isdir(path):
                for f in os.listdir(path):
                    if f.endswith('.tar.gz'):
                        sim_input_paths.append(os.path.join(path, f)) 
    
    else:
        # Paris isn't compressed
        sim_input_paths = [os.path.join(project_root, 'data', 'raw_data', city)] # TODO: try for paris
    
    # Path to basecase files
    if city != 'paris':
        administrative_boundaries_path = os.path.join(project_root, 'data_new', 'links_and_stats', 'city_boundaries', city, f'{city}.json')
        basecase_links_path = os.path.join(project_root, 'data_new', 'links_and_stats', 'basecases_mean', city, 'basecase_average_output_links.geojson')
        basecase_stats_path = os.path.join(project_root, 'data_new', 'links_and_stats', 'basecases_mean', city, 'basecase_average_trips.csv')
        basecase_eqasim_trips_path = os.path.join(project_root, 'data_new', 'links_and_stats', 'basecases_mean', city, 'eqasim_trips.csv')
    else:
        basecase_links_path = os.path.join(project_root, 'data_new', 'links_and_stats', 'basecases_mean', city, 'basecase_average_output_links.geojson') #TODO: check for paris, the filename is different
        basecase_stats_path = os.path.join(project_root, 'data_new', 'links_and_stats', 'basecases_mean', city, 'basecase_average_trips.csv') #TODO: check for paris, the filename is different
        basecase_eqasim_trips_path = os.path.join(project_root, 'data_new', 'links_and_stats', 'basecases_mean', city, 'eqasim_trips.csv') #TODO: check for paris, the filename is different
    
    print(f"Found {len(sim_input_paths)} graphs for {city}.")

    # Load administrative boundaries
    if city != 'paris' and is_in_stadt:
        gdf_administrative_boundaries = gpd.read_file(administrative_boundaries_path)
        zones_gdf = modify_geodataframe(gdf_administrative_boundaries)

    df_basecase_eqasim_trips = pd.read_csv(basecase_eqasim_trips_path, delimiter=';')

    # Load basecase data
    gdf_basecase_links = gpd.read_file(basecase_links_path)
    gdf_basecase_links = gdf_basecase_links.set_crs("EPSG:25832", allow_override=True)  # TODO: fix for paris is EPSG:4326
    if city != 'paris' and is_in_stadt:
        gdf_basecase_links = merge_edges_and_zones(gdf_basecase_links, zones_gdf)
        gdf_basecase_links=clean_duplicates_based_on_modes(gdf_basecase_links)
        
    gdf_basecase_mean_mode_stats = pd.read_csv(basecase_stats_path, delimiter=',')

    # Initialize activity_destination_names
    activity_destination_names = []
    
    if use_destination_activity:    
        gdf_basecase_links, activity_destination_names = add_destinations_to_gdf(gdf_basecase_links, df_basecase_eqasim_trips, x_normalization_type, normalize_activities=True)
        print(f"\n=== Activity Features Added === *****BASECASE LINKS*****")
        print(f"Activity destination names: {activity_destination_names}")
        #print(f"Base case columns: {list(gdf_basecase_links.columns)}")
        # Check if normalized features exist
        for feat in activity_destination_names:
            if feat in gdf_basecase_links.columns:
                values = gdf_basecase_links[feat].values
                print(f"{feat}: mean={values.mean():.4f}, std={values.std():.4f}, min={values.min():.4f}, max={values.max():.4f}")
        print("=" * 30)

    # Sort for reproducibility, hopefully!
    sim_input_paths.sort()

    # Some metadata, helps later in DataLoader
    idx = 1
    metadata = {'path': list(), 'policy_region': list(), 'scenario': list(), 'city':list()}

    for i in tqdm(range(0, len(sim_input_paths), batch_size), desc="Processing in batches ...", unit="batch"):
        
        sliced_inputs = sim_input_paths[i:i+batch_size]
        
        try:        
            if city != 'paris':
                # Extract all tar.gz files from compressed directories (for Bavarian Simulations)
                networks, temp_dirs = extract_and_get_networks(sliced_inputs)
                networks = [network for network in networks if not network.endswith(".DS_Store")]
            else:
                networks = [network for network in sliced_inputs if not network.endswith(".DS_Store")]
                temp_dirs = []
            
            # Process networks and generate graph data
            result_dic_output_links, result_dic_eqasim_trips = compute_result_dic(basecase_links=gdf_basecase_links, networks=networks, use_destination_activity=use_destination_activity, activity_destination_names=activity_destination_names)
            base_gdf = result_dic_output_links["base_network_no_policies"]
            
            city_data = process_result_dic_eign(city, result_dic=result_dic_output_links, result_dic_mode_stats=result_dic_eqasim_trips,
                                            links_base_case=base_gdf, gdf_basecase_mean_mode_stats=gdf_basecase_mean_mode_stats,
                                            activity_destination_names=activity_destination_names,
                                            save_path=result_path, batch_size=50)
            
            if city_data is not None:
                for graph in city_data:
                    filename = f'{idx:06d}.pt'
                    torch.save(graph, os.path.join(result_path, filename))
                    
                    idx += 1
                    metadata['path'].append(os.path.join(result_path, filename))
                    metadata['policy_region'].append(graph.policy_region)
                    metadata['scenario'].append(graph.scenario)
                    metadata['city'].append(graph.city)
                
                del city_data
            else:
                print(f"Warning: No valid data returned for batch {i}")
            
        finally:
            # Clean up temporary directories
            for temp_dir in temp_dirs:
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    print(f"Warning: Could not clean up {temp_dir}: {e}")
            print(f"Cleaned up: {len(temp_dirs)} temp directories")

    # Save metadata
    metadata_path = os.path.join(result_path, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def main():
    # Create the result base path
    result_base_path = os.path.join(project_root, 'data', 'inductive_data', 'training_data_eign','kreisfreistadt')
    os.makedirs(result_base_path, exist_ok=True)
    
    for city in all_cities:
        
        result_path = os.path.join(result_base_path, city)
        os.makedirs(result_path, exist_ok=True)
        
        process_single_city(city, project_root, result_path, use_destination_activity, use_allowed_modes)
        
if __name__ == "__main__":
    main()
