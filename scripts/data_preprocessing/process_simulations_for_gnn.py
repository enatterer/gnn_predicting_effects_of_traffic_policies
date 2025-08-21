"""
Process simulation data (from MATSim) for GNNs. Load basecase and simulated graphs (with policies applied in various district combinations),
convert them to dual line graphs, and compute specified edge features. Save as PyTorch Geometric data batches for efficient loading and training.

Here we specify all features, then run_models can be called with a reduced set. Note that, for example, the flag "use_allowed_modes" is accessed from the run_models script.

***TO call this script, use the following command:***
python process_simulations_for_gnn.py
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

# Set seeds for reproducibility
np.random.seed(13)
random.seed(13)
torch.manual_seed(13)

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from data_preprocessing.help_functions import *

##########################
##### Control Center #####
##########################
is_in_stadt=True #if true, include only the edges that are in the stadt, else include edges in stadt and landkreis
batch_size = 256 # Do processing in batches to avoid memory issues
seed = 3 # Seed for Bavarian Simulations
hex_sizes = [500] # Hexagon Sizes for Bavarian Simulations
required_modes_on_links = ['car', 'car_passenger'] # Capacity will be reduced on links that have at least one of these modes
use_allowed_modes = True # Flag to use allowed modes as edge features
use_destination_activity = True # Flag to use destination activity as edge features
use_linegraph = True # Flag to use line graph transformation
# Test with just one city
#all_cities = ['rosenheim','muenchen','augsburg', 'nuernberg','neuulm']  # Change this to test different cities
#cities_1=['nuernberg', 'augsburg', 'muenchen','schweinfurt', 'aschaffenburg', 'wuerzburg', 'bamberg', 'bayreuth', 'erlangen', 'fuerth', 'kempten','landshut', 'ingolstadt', 'regensburg', 'neuulm',rosenheim]
#cities_rest=[]
all_cities = ['landshut', 'ingolstadt', 'regensburg' 'rosenheim']
#target_feature = 'vol_car_percentage' #other options: 'vol_car'
#target_feature_normalization_type = 'signed_log_normalization' #other options: 'mean_std', 'min_max','none'
x_normalization_type = 'mean_std' #other options: 'min_max', 'robust_normalization', 'mean_std'

##########################

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
    # Ensure the data is in the correct CRS (EPSG:25832) **********VERY IMPORTANT**********
    if zones_gdf.crs != "EPSG:25832": #should match with the CRS of the Network Geodataframe
        zones_gdf = zones_gdf.to_crs(epsg=25832)
    return zones_gdf


def multipolygon_to_polygon(geom):
    '''
    This function converts a MultiPolygon to a Polygon with the largest connected area.
    A MultiPolygon with 2 Polygons inside will return the Polygon with the largest area.(z.B. Stadt Bamberg had 2 disconnected polygons, we only consider the largest one)
    '''
    return max(geom.geoms, key=lambda p: p.area)

def merge_edges_and_zones(gdf_csv, zones_gdf):
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
    #return only the geodataframe where zone_id is in the stadt(1)
    if is_in_stadt:
        # Check if any zone_id contains 1 (since zone_id contains lists/tuples)
        gdf_edges_with_zones = gdf_edges_with_zones[gdf_edges_with_zones['zone_id'].apply(lambda x: 1 in x if x else False)]

    return gdf_edges_with_zones

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

def get_capacity_and_freespeed_base_case(links_base_case, required_modes_on_links):
    mode_masks = [links_base_case['modes'].str.contains(mode) for mode in required_modes_on_links]
    combined_mask = mode_masks[0]
    for mask in mode_masks[1:]:
        combined_mask = combined_mask | mask
    capacity_base_case = np.where(combined_mask, links_base_case['capacity'], 0)
    freespeed_base_case = np.where(combined_mask, links_base_case['freespeed'], 0)
    return capacity_base_case, freespeed_base_case

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
    # Pattern: network_seed3_{city}_primary_*_{scenario}_reduced_capacity_edges.geojson
    pattern = f"network_seed3_{city}_primary_*_{scenario}_reduced_capacity_edges.geojson"
    search_pattern = os.path.join(reduced_links_dir, f"network_seed3_{city}_primary_*_{scenario}_reduced_capacity_edges.geojson")
    
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        print(f"WARNING: No reduced capacity geojson file found for {city}, policy_region={policy_region}, scenario={scenario}")
        print(f"  Searched in: {reduced_links_dir}")
        print(f"  Pattern: network_seed3_{city}_primary_*_{scenario}_reduced_capacity_edges.geojson")
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

# Read all network data into a dictionary of GeoDataFrames
# For paris, please use the flag 'use_destination_activity' as False
def compute_result_dic(basecase_links, networks, use_destination_activity, activity_destination_names=None):
    
    result_dic_output_links = {}
    result_dic_eqasim_trips = {}
    result_dic_output_links["base_network_no_policies"] = basecase_links
    
    for network in networks:
        
        policy_key = create_policy_key(network) #TODO: fix for paris
        df_output_links = read_output_links(network)
        df_eqasim_trips = read_eqasim_trips(network)
        if (df_output_links is not None):
            df_output_links.drop(columns=['geometry'], inplace=True)
            # first include only the links that are in the cleaned basecase_links
            df_output_links = df_output_links[df_output_links['link'].isin(basecase_links['link'])]
            
            # Reorder simulation data to match base case ordering
            # Create a mapping from link ID to base case index
            link_to_index = {link: idx for idx, link in enumerate(basecase_links['link'])}
            # Sort simulation data by base case ordering
            df_output_links['base_order'] = df_output_links['link'].map(link_to_index)
            df_output_links = df_output_links.sort_values('base_order').reset_index(drop=True)
            df_output_links = df_output_links.drop(columns=['base_order'])
            
            gdf_extended = extend_geodataframe(gdf_base=basecase_links, gdf_to_extend=df_output_links, column_to_extend='highway', new_column_name='highway')
            gdf_extended = extend_geodataframe(gdf_base=basecase_links, gdf_to_extend=gdf_extended, column_to_extend='vol_car', new_column_name='vol_car_base_case')
            if use_destination_activity:
                # Extend the normalized activity features from base case instead of re-normalizing
                gdf_extended_with_destinations = gdf_extended
                # Add normalized activity features from base case
                for feature in activity_destination_names:
                    if feature in basecase_links.columns:
                        gdf_extended_with_destinations = extend_geodataframe(
                            gdf_base=basecase_links, 
                            gdf_to_extend=gdf_extended_with_destinations, 
                            column_to_extend=feature, 
                            new_column_name=feature
                        )
            else:
                gdf_extended_with_destinations = gdf_extended
            result_dic_output_links[policy_key] = gdf_extended_with_destinations
            
            # Only calculate mode stats if eqasim_trips data is available
            if df_eqasim_trips is not None:
                df_eqasim_trips_list = [df_eqasim_trips]
                mode_stats = calculate_avg_mode_stats(df_eqasim_trips_list)
                result_dic_eqasim_trips[policy_key] = mode_stats
    
    return result_dic_output_links, result_dic_eqasim_trips

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

def generate_graph_data(city, result_dic, result_dic_mode_stats, links_base_case,
                        gdf_basecase_mean_mode_stats, use_destination_activity, use_allowed_modes, 
                        x_normalization_type, required_modes_on_links, project_root):
    
    datalist = []
    linegraph_transformation = LineGraph()

    vol_base_case = np.round(links_base_case['vol_car'].values) #round to integer
    
    # Save scaler parameters for this city
    save_scaler_params(city, vol_base_case, project_root)
    
    capacity_base_case, freespeed_base_case = get_capacity_and_freespeed_base_case(links_base_case, required_modes_on_links)
    length = links_base_case['length'].values
    vol_base_case_normalized = normalization_of_edge_features(vol_base_case, x_normalization_type)
    capacity_base_case_normalized = normalization_of_edge_features(capacity_base_case, x_normalization_type)
    freespeed_base_case_normalized = normalization_of_edge_features(freespeed_base_case, x_normalization_type)
    length_normalized = normalization_of_edge_features(length, x_normalization_type)
    
    # Only compute allowed modes if the flag is True
    allowed_modes = encode_modes(links_base_case) if use_allowed_modes else None
    
    # Get link geometries and edges_base FIRST
    _, stacked_edge_geometries_tensor, edges_base, nodes, _ = get_link_geometries(links_base_case, apply_scaling=True)
    
    # THEN use edges_base to create edge_index
    edge_index = torch.tensor(edges_base, dtype=torch.long).t().contiguous()

    # Filter out base_network_no_policies before the loop
    graph_items = {k: v for k, v in result_dic.items() 
                   if isinstance(v, pd.DataFrame) and k != "base_network_no_policies"}
    
    print(f"\n=== PROCESSING {len(graph_items)} GRAPHS FOR {city.upper()} ===")
    print("Using BINARY capacity reduction from GeoJSON files")
    print("Policy regions and scenarios:")
    
    for key, df in graph_items.items():
        gdf = prepare_gdf(df, links_base_case) 
        policy_region, scenario = key
        print(f"  - Policy: {policy_region}, Scenario: {scenario}")
        # Get reduced capacity links for this policy/scenario
        reduced_links = get_reduced_capacity_links(city, policy_region, scenario, project_root)
        
        # Create binary capacity reduction: 1 if link is in reduced set, 0 otherwise
        gdf['link_str'] = gdf['link'].astype(str)  # Ensure string comparison
        capacity_reduction_binary = gdf['link_str'].isin(reduced_links).astype(int).values
        
        # Get highway attributes (unchanged)
        _, _, highway = get_basic_edge_attributes(capacity_base_case, gdf, required_modes_on_links)
        
        print(f"  Policy: {policy_region}, Scenario: {scenario}")
        print(f"  Links with reduced capacity: {capacity_reduction_binary.sum()}/{len(capacity_reduction_binary)}")
        print(f"  Reduced links found in GDF: {len(set(gdf['link_str']) & reduced_links)}")

        # Apply normalization to all features before creating tensors - WITH DEBUG
        def debug_normalize(feature_name, data, norm_type):
            print(f"DEBUG: Normalizing {feature_name}")
            print(f"  Data shape: {data.shape}")
            print(f"  Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
            print(f"  Mean: {np.mean(data):.3f}, Std: {np.std(data):.3f}")
            if np.std(data) == 0:
                print(f"  WARNING: {feature_name} has zero std - all values identical!")
            result = normalization_of_edge_features(data, norm_type)
            print(f"  Normalized range: [{np.min(result):.3f}, {np.max(result):.3f}]")
            print()
            return result
            
        edge_feature_dict = {
            EdgeFeatures.VOL_BASE_CASE: torch.tensor(vol_base_case_normalized),
            EdgeFeatures.CAPACITY_BASE_CASE: torch.tensor(capacity_base_case_normalized),
            EdgeFeatures.CAPACITY_REDUCTION: torch.tensor(capacity_reduction_binary, dtype=torch.float),  # Binary: 1 if reduced, 0 otherwise
            EdgeFeatures.FREESPEED: torch.tensor(freespeed_base_case_normalized),
            EdgeFeatures.LENGTH: torch.tensor(length_normalized)}
        # Add highway one-hot encoding
        highway_feature_keys = [
            EdgeFeatures.HIGHWAY_PRIMARY,
            EdgeFeatures.HIGHWAY_SECONDARY,
            EdgeFeatures.HIGHWAY_TERTIARY,
            EdgeFeatures.HIGHWAY_RESIDENTIAL,
            EdgeFeatures.HIGHWAY_PT,
            EdgeFeatures.HIGHWAY_OTHER]
        for i, key in enumerate(highway_feature_keys):
            edge_feature_dict[key] = torch.tensor(highway[:, i], dtype=torch.float)
        
        if use_allowed_modes:
            edge_feature_dict.update({
                EdgeFeatures.ALLOWED_MODE_CAR: allowed_modes[0],
                EdgeFeatures.ALLOWED_MODE_CAR_PASSENGER: allowed_modes[1],
                EdgeFeatures.ALLOWED_MODE_BUS: allowed_modes[2],
                EdgeFeatures.ALLOWED_MODE_PT: allowed_modes[3],
                EdgeFeatures.ALLOWED_MODE_TRAIN: allowed_modes[4],
                EdgeFeatures.ALLOWED_MODE_TRAM: allowed_modes[5],
                EdgeFeatures.ALLOWED_MODE_RAIL: allowed_modes[6],
                EdgeFeatures.ALLOWED_MODE_SUBWAY: allowed_modes[7],
                EdgeFeatures.ALLOWED_MODE_FUNICULAR: allowed_modes[8],
            })
        if use_destination_activity:
            edge_feature_dict.update({
                # Activity features (using pre-normalized values from add_destinations_to_gdf)
                EdgeFeatures.HOME: torch.tensor(gdf.get('home_normalized', pd.Series(0.0, index=gdf.index)).values),
                EdgeFeatures.WORK: torch.tensor(gdf.get('work_normalized', pd.Series(0.0, index=gdf.index)).values),
                EdgeFeatures.EDUCATION: torch.tensor(gdf.get('education_normalized', pd.Series(0.0, index=gdf.index)).values),
                EdgeFeatures.LEISURE: torch.tensor(gdf.get('leisure_normalized', pd.Series(0.0, index=gdf.index)).values),
                EdgeFeatures.SHOP: torch.tensor(gdf.get('shop_normalized', pd.Series(0.0, index=gdf.index)).values),
                EdgeFeatures.OTHER: torch.tensor(gdf.get('other_normalized', pd.Series(0.0, index=gdf.index)).values),
                EdgeFeatures.OUTSIDE: torch.tensor(gdf.get('outside_normalized', pd.Series(0.0, index=gdf.index)).values),
                
                # Trip data availability identifier
                EdgeFeatures.IS_IN_EQASIM_TRIPS: torch.tensor(gdf.get('is_in_eqasim_trips', pd.Series(0, index=gdf.index)).values, dtype=torch.float),
            })
        # Create the edge_tensor by stacking only the features that are actually in the dictionary
        feature_list = []
        feature_names = []
        for feat in EdgeFeatures:
            if feat in edge_feature_dict:
                feature_list.append(edge_feature_dict[feat])
                feature_names.append(feat.name)
        
        edge_tensor = torch.stack(feature_list, dim=1)
        
        # Compute edge weights
        edge_weights = compute_edge_weights(vol_base_case)
        
        data = Data(edge_index=edge_index)
        data.num_nodes = len(nodes)
        if use_linegraph:
            data = linegraph_transformation(data)
        data.x = edge_tensor
        data.pos = stacked_edge_geometries_tensor
        
        # Add new data attributes (using float32 to save memory)
        data.unscaled_vol_base = torch.tensor(vol_base_case, dtype=torch.float32)
        data.edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
        
        # Store multiple target options
        #data.y = compute_target_tensor_only_edge_features(vol_base_case, gdf, target_feature, target_feature_normalization_type)
        
        # Additional target variable options
        data.y_abs_vol_car=compute_target_tensor_only_edge_features(vol_base_case, gdf, 'vol_car', 'none') #no normalization
        data.y_abs_vol_car_percentage=compute_target_tensor_only_edge_features(vol_base_case, gdf, 'vol_car_percentage', 'none') #no normalization
        data.y_vol_car_signed_log = compute_target_tensor_only_edge_features(vol_base_case, gdf, 'vol_car', 'signed_log_normalization') #signed_log_normalization
        data.y_vol_car_percentage_signed_log = compute_target_tensor_only_edge_features(vol_base_case, gdf, 'vol_car_percentage', 'signed_log_normalization') #signed_log_normalization
        data.y_vol_car_mean_std=compute_target_tensor_only_edge_features(vol_base_case, gdf, 'vol_car', 'mean_std') #mean_std
        data.y_vol_car_percentage_mean_std=compute_target_tensor_only_edge_features(vol_base_case, gdf, 'vol_car_percentage', 'mean_std') #mean_std
        data.y_vol_car_min_max=compute_target_tensor_only_edge_features(vol_base_case, gdf, 'vol_car', 'min_max') #min_max
        data.y_vol_car_percentage_min_max=compute_target_tensor_only_edge_features(vol_base_case, gdf, 'vol_car_percentage', 'min_max') #min_max
       
        data.policy_region = policy_region
        data.city = city
        data.scenario = scenario
        
        # Set num_nodes AFTER transformations and feature assignment
        data.num_nodes = data.x.shape[0]

        df_mode_stats = result_dic_mode_stats.get(key)
        if df_mode_stats is not None:
            pd.set_option('display.float_format', lambda x: '%.10f' % x)
            numeric_cols_base_case = gdf_basecase_mean_mode_stats.select_dtypes(include=[np.number]).columns
            numeric_cols = df_mode_stats.select_dtypes(include=[np.number]).columns
            mode_stats_diff = df_mode_stats[numeric_cols].values - gdf_basecase_mean_mode_stats[numeric_cols_base_case].values 
            mode_stats_tensor = torch.tensor(mode_stats_diff, dtype=torch.float)
            data.mode_stats_diff = mode_stats_tensor
            mode_stats_diff_perc = mode_stats_tensor / gdf_basecase_mean_mode_stats[numeric_cols_base_case].values *100
            data.mode_stats_diff_perc = mode_stats_diff_perc

        if data.validate(raise_on_error=True):
            datalist.append(data)
            
            # Debug: Print capacity reduction statistics for each graph
            capacity_reduction_feature = data.x[:, EdgeFeatures.CAPACITY_REDUCTION].numpy()
            print(f"  Graph created: {data.x.shape[0]} nodes, {data.x.shape[1]} features")
            print(f"  Capacity reduction feature: {capacity_reduction_feature.sum():.0f} links reduced out of {len(capacity_reduction_feature)}")
            print()
            
            # Debug: Print feature information for the first graph
            if len(datalist) == 1:
                print(f"\n=== DEBUG: First Graph Features for {city} ===")
                # Debug: Print which features are being included
                print(f"\n=== Features Included in Tensor ===")
                print(f"Total features: {len(feature_list)}")
                print(f"Feature names: {feature_names}")
                print("=" * 30)
                print(f"Graph shape: {data.x.shape}")
                print(f"Number of nodes: {data.num_nodes}")
                print(f"Number of features: {data.x.shape[1]}")
                print(f"Target (abs_vol_car): {data.y_abs_vol_car.shape}")
                print(f"Target (abs_vol_car_percentage): {data.y_abs_vol_car_percentage.shape}")
                print(f"Target (vol_car_signed_log): {data.y_vol_car_signed_log.shape}")
                print(f"Target (vol_car_percentage_signed_log): {data.y_vol_car_percentage_signed_log.shape}")
                print(f"Target (vol_car_mean_std): {data.y_vol_car_mean_std.shape}")
                print(f"Target (vol_car_percentage_mean_std): {data.y_vol_car_percentage_mean_std.shape}")
                print(f"Target (vol_car_min_max): {data.y_vol_car_min_max.shape}")
                print(f"Target (vol_car_percentage_min_max): {data.y_vol_car_percentage_min_max.shape}")
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
                for i, feat_name in enumerate(feature_names):
                    feat_values = data.x[:, i].numpy()
                    print(f"{feat_name}: mean={feat_values.mean():.4f}, std={feat_values.std():.4f}, "
                          f"min={feat_values.min():.4f}, max={feat_values.max():.4f}")
                
                # Print target statistics
                target_values_abs_vol_car = data.y_abs_vol_car.numpy()
                target_values_abs_vol_car_percentage = data.y_abs_vol_car_percentage.numpy()
                target_values_vol_car_signed_log = data.y_vol_car_signed_log.numpy()
                target_values_vol_car_percentage_signed_log = data.y_vol_car_percentage_signed_log.numpy()
                target_values_vol_car_mean_std = data.y_vol_car_mean_std.numpy()
                target_values_vol_car_percentage_mean_std = data.y_vol_car_percentage_mean_std.numpy()
                target_values_vol_car_min_max = data.y_vol_car_min_max.numpy()
                target_values_vol_car_percentage_min_max = data.y_vol_car_percentage_min_max.numpy()
                print(f"\n--- Target Statistics ---")
                print(f"Target (abs_vol_car): mean={target_values_abs_vol_car.mean():.4f}, std={target_values_abs_vol_car.std():.4f}, "
                      f"min={target_values_abs_vol_car.min():.4f}, max={target_values_abs_vol_car.max():.4f}")
                print(f"Target (abs_vol_car_percentage): mean={target_values_abs_vol_car_percentage.mean():.4f}, std={target_values_abs_vol_car_percentage.std():.4f}, "
                      f"min={target_values_abs_vol_car_percentage.min():.4f}, max={target_values_abs_vol_car_percentage.max():.4f}")
                print(f"Target (vol_car_signed_log): mean={target_values_vol_car_signed_log.mean():.4f}, std={target_values_vol_car_signed_log.std():.4f}, "
                      f"min={target_values_vol_car_signed_log.min():.4f}, max={target_values_vol_car_signed_log.max():.4f}")
                print(f"Target (vol_car_percentage_signed_log): mean={target_values_vol_car_percentage_signed_log.mean():.4f}, std={target_values_vol_car_percentage_signed_log.std():.4f}, "
                      f"min={target_values_vol_car_percentage_signed_log.min():.4f}, max={target_values_vol_car_percentage_signed_log.max():.4f}")
                print(f"Target (vol_car_mean_std): mean={target_values_vol_car_mean_std.mean():.4f}, std={target_values_vol_car_mean_std.std():.4f}, "
                      f"min={target_values_vol_car_mean_std.min():.4f}, max={target_values_vol_car_mean_std.max():.4f}")
                print(f"Target (vol_car_percentage_mean_std): mean={target_values_vol_car_percentage_mean_std.mean():.4f}, std={target_values_vol_car_percentage_mean_std.std():.4f}, "
                      f"min={target_values_vol_car_percentage_mean_std.min():.4f}, max={target_values_vol_car_percentage_mean_std.max():.4f}")
                print(f"Target (vol_car_min_max): mean={target_values_vol_car_min_max.mean():.4f}, std={target_values_vol_car_min_max.std():.4f}, "
                      f"min={target_values_vol_car_min_max.min():.4f}, max={target_values_vol_car_min_max.max():.4f}")
                print(f"Target (vol_car_percentage_min_max): mean={target_values_vol_car_percentage_min_max.mean():.4f}, std={target_values_vol_car_percentage_min_max.std():.4f}, "
                      f"min={target_values_vol_car_percentage_min_max.min():.4f}, max={target_values_vol_car_percentage_min_max.max():.4f}")
                
                # Check for NaN or Inf values
                if torch.isnan(data.x).any():   
                    print("WARNING: NaN values found in features!")
                if torch.isinf(data.x).any():
                    print("WARNING: Inf values found in features!")
                if torch.isnan(data.y_abs_vol_car).any():
                    print("WARNING: NaN values found in target!")
                if torch.isnan(data.y_abs_vol_car_percentage).any():
                    print("WARNING: NaN values found in target!")
                if torch.isnan(data.y_vol_car_signed_log).any():
                    print("WARNING: NaN values found in target!")
                if torch.isnan(data.y_vol_car_percentage_signed_log).any():
                    print("WARNING: NaN values found in target!")
                if torch.isnan(data.y_vol_car_mean_std).any():
                    print("WARNING: NaN values found in target!")
                if torch.isnan(data.y_vol_car_percentage_mean_std).any():
                    print("WARNING: NaN values found in target!")
                if torch.isnan(data.y_vol_car_min_max).any():
                    print("WARNING: NaN values found in target!")
                if torch.isnan(data.y_vol_car_percentage_min_max).any():
                    print("WARNING: NaN values found in target!")
                if torch.isnan(data.unscaled_vol_base).any():
                    print("WARNING: NaN values found in unscaled_vol_base!")
                if torch.isnan(data.edge_weights).any():
                    print("WARNING: NaN values found in edge_weights!")
                print("=" * 50)
    
    return datalist

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
            
            city_data = generate_graph_data(city, result_dic=result_dic_output_links, result_dic_mode_stats=result_dic_eqasim_trips,
                                            links_base_case=base_gdf, gdf_basecase_mean_mode_stats=gdf_basecase_mean_mode_stats,
                                            use_destination_activity=use_destination_activity, use_allowed_modes=use_allowed_modes,
                                            x_normalization_type=x_normalization_type, required_modes_on_links=required_modes_on_links,
                                            project_root=project_root)
            
            for graph in city_data:
                filename = f'{idx:06d}.pt'
                torch.save(graph, os.path.join(result_path, filename))
                
                idx += 1
                metadata['path'].append(os.path.join(result_path, filename))
                metadata['policy_region'].append(graph.policy_region)
                metadata['scenario'].append(graph.scenario)
                metadata['city'].append(graph.city)
            
            del city_data
            
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
    result_base_path = os.path.join(project_root, 'data', 'inductive_data', 'training_data','kreisfreistadt')
    os.makedirs(result_base_path, exist_ok=True)
    
    for city in all_cities:
        
        result_path = os.path.join(result_base_path, city)
        os.makedirs(result_path, exist_ok=True)
        
        process_single_city(city, project_root, result_path, use_destination_activity, use_allowed_modes)

if __name__ == '__main__':
    main()
    