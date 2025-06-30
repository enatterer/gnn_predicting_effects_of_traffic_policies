"""
Process simulation data (from MATSim) for GNNs. Load basecase and simulated graphs (with policies applied in various district combinations),
convert them to dual line graphs, and compute specified edge features. Save as PyTorch Geometric data batches for efficient loading and training.

Here we specify all features, then run_models can be called with a reduced set. Note that, for example, the flag "use_allowed_modes" is accessed from the run_models script.
"""

import os
import sys
from enum import IntEnum
import tarfile
import tempfile
import shutil
import random
import argparse


import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

import torch
from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data
from case_variants import *

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from data_preprocessing.help_functions import *

#control center
seed = 2 # Seed for the simulation
hex_sizes = [500, 1000, 2000] # Hexagon sizes to process
required_modes_on_links = ['car', 'car_passenger'] # Capacity will be reduced on links that have at least one of these modes
use_linegraph = True # Flag to use line graph transformation
use_allowed_modes = False # Flag to use allowed modes or not
required_batch_size = 1 # Batch size for saving the data

class EdgeFeatures(IntEnum):
    VOL_BASE_CASE = 0
    CAPACITY_BASE_CASE = 1
    CAPACITY_REDUCTION = 2
    FREESPEED = 3
    HIGHWAY = 4
    LENGTH = 5
    ALLOWED_MODE_CAR = 6
    ALLOWED_MODE_CAR_PASSENGER = 7
    ALLOWED_MODE_BUS = 8
    ALLOWED_MODE_PT = 9
    ALLOWED_MODE_TRAIN = 10
    ALLOWED_MODE_TRAM = 11
    ALLOWED_MODE_RAIL = 12
    ALLOWED_MODE_SUBWAY = 13
    ALLOWED_MODE_FUNICULAR = 14


# Read all network data into a dictionary of GeoDataFrames
def compute_result_dic(basecase_links, networks):
    
    result_dic_output_links = {}
    result_dic_eqasim_trips = {}
    result_dic_output_links["base_network_no_policies"] = basecase_links
    
    for network in tqdm(networks, desc="Processing Networks", unit="network"):
        
        policy_key = create_policy_key(network)
        df_output_links = read_output_links(network)
        df_eqasim_trips = read_eqasim_trips(network)
        if (df_output_links is not None and df_eqasim_trips is not None):
            df_output_links.drop(columns=['geometry'], inplace=True)
            gdf_extended = extend_geodataframe(gdf_base=basecase_links, gdf_to_extend=df_output_links, column_to_extend='highway', new_column_name='highway')
            gdf_extended = extend_geodataframe(gdf_base=basecase_links, gdf_to_extend=gdf_extended, column_to_extend='vol_car', new_column_name='vol_car_base_case')
            result_dic_output_links[policy_key] = gdf_extended
            df_eqasim_trips_list = [df_eqasim_trips]
            mode_stats = calculate_avg_mode_stats(df_eqasim_trips_list)
            result_dic_eqasim_trips[policy_key] = mode_stats
    
    return result_dic_output_links, result_dic_eqasim_trips

def generate_graph_data(result_dic, result_dic_mode_stats, links_base_case, gdf_basecase_mean_mode_stats):
    datalist = []
    linegraph_transformation = LineGraph()

    vol_base_case = links_base_case['vol_car'].values
    capacity_base_case = get_capacity_base_case(links_base_case, required_modes_on_links)
    length = links_base_case['length'].values
    freespeed_base_case = links_base_case['freespeed'].values
    allowed_modes = encode_modes(links_base_case)
    
    # Get link geometries and edges_base FIRST
    _, stacked_edge_geometries_tensor, edges_base, nodes, _ = get_link_geometries(links_base_case, apply_scaling=True)
    
    # THEN use edges_base to create edge_index
    edge_index = torch.tensor(edges_base, dtype=torch.long).t().contiguous()

    # Filter out base_network_no_policies before the loop
    graph_items = {k: v for k, v in result_dic.items() 
                   if isinstance(v, pd.DataFrame) and k != "base_network_no_policies"}
    
    for key, df in tqdm(graph_items.items(), desc="Processing graphs"):
        gdf = prepare_gdf(df, links_base_case)
        _, capacity_reduction, highway, freespeed_scenario = get_basic_edge_attributes(capacity_base_case, gdf, required_modes_on_links)
        hex_size,scenario = key

        edge_feature_dict = {
            EdgeFeatures.VOL_BASE_CASE: torch.tensor(vol_base_case),
            EdgeFeatures.CAPACITY_BASE_CASE: torch.tensor(capacity_base_case),
            EdgeFeatures.CAPACITY_REDUCTION: torch.tensor(capacity_reduction),
            EdgeFeatures.FREESPEED: torch.tensor(freespeed_scenario),  # Using filtered freespeed (0 for non-required modes)
            EdgeFeatures.HIGHWAY: torch.tensor(highway),
            EdgeFeatures.LENGTH: torch.tensor(length),
        }

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

        edge_tensor = torch.stack([edge_feature_dict[feat] for feat in EdgeFeatures if feat in edge_feature_dict], dim=1)

        data = Data(edge_index=edge_index)
        if use_linegraph:
            data = linegraph_transformation(data)
        data.x = edge_tensor
        data.pos = stacked_edge_geometries_tensor
        data.y = compute_target_tensor_only_edge_features(vol_base_case, gdf)
        data.hex_size = hex_size
        
        # Set num_nodes AFTER transformations and feature assignment
        data.num_nodes = data.x.shape[0]

        df_mode_stats = result_dic_mode_stats.get(key)
        if df_mode_stats is not None:
            numeric_cols_base = gdf_basecase_mean_mode_stats.select_dtypes(include=[np.number]).columns
            numeric_cols = df_mode_stats.select_dtypes(include=[np.number]).columns
            diff = df_mode_stats[numeric_cols].values - gdf_basecase_mean_mode_stats[numeric_cols_base].values
            data.mode_stats_diff = torch.tensor(diff, dtype=torch.float)
            data.mode_stats_diff_perc = data.mode_stats_diff / gdf_basecase_mean_mode_stats[numeric_cols_base].values * 100

        if data.validate(raise_on_error=True):
            datalist.append(data)
    return datalist

def get_capacity_base_case(links_base_case, required_modes_on_links):
    mode_masks = [links_base_case['modes'].str.contains(mode) for mode in required_modes_on_links]
    combined_mask = mode_masks[0]
    for mask in mode_masks[1:]:
        combined_mask = combined_mask | mask
    capacity_base_case = np.where(combined_mask, links_base_case['capacity'], 0)
    return capacity_base_case

def process_result_dic(result_dic, result_dic_mode_stats, save_path=None, batch_size=500, links_base_case=None, gdf_basecase_mean_mode_stats=None):

    # PROCESS LINK GEOMETRIES (with scaling and centering)
    _, stacked_edge_geometries_tensor, edges_base, nodes, pos_scaling_params = get_link_geometries(links_base_case, apply_scaling=True)
    
    os.makedirs(save_path, exist_ok=True)
    datalist = []
    linegraph_transformation = LineGraph()
    
    vol_base_case = links_base_case['vol_car'].values
    capacity_base_case = get_capacity_base_case(links_base_case, required_modes_on_links)
    length = links_base_case['length'].values
    freespeed_base = links_base_case['freespeed'].values
    allowed_modes = encode_modes(links_base_case)
    edge_index = torch.tensor(edges_base, dtype=torch.long).t().contiguous()
    
    batch_counter = 0
    for key, df in tqdm(result_dic.items(), desc="Processing result_dic", unit="dataframe"):   
        if isinstance(df, pd.DataFrame) and key != "base_network_no_policies":
            gdf = prepare_gdf(df, links_base_case)
            _, capacity_reduction, highway, freespeed_scenario =  get_basic_edge_attributes(capacity_base_case, gdf, required_modes_on_links)

            edge_feature_dict = {
                EdgeFeatures.VOL_BASE_CASE: torch.tensor(vol_base_case),
                EdgeFeatures.CAPACITY_BASE_CASE: torch.tensor(capacity_base_case),
                EdgeFeatures.CAPACITY_REDUCTION: torch.tensor(capacity_reduction),
                EdgeFeatures.FREESPEED: torch.tensor(freespeed_scenario),  # Using filtered freespeed (0 for non-required modes)
                EdgeFeatures.HIGHWAY: torch.tensor(highway),
                EdgeFeatures.LENGTH: torch.tensor(length),
            }

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
                    EdgeFeatures.ALLOWED_MODE_FUNICULAR: allowed_modes[8]})

            # Create the edge_tensor by iterating through the EdgeFeatures enum
            edge_tensor = [edge_feature_dict[feature] for feature in EdgeFeatures if feature in edge_feature_dict]

            # Stack the tensors
            edge_tensor = torch.stack(edge_tensor, dim=1)
            
            data = Data(edge_index=edge_index)
            if use_linegraph:
                data = linegraph_transformation(data)
            
            data.x = edge_tensor
            data.pos = stacked_edge_geometries_tensor
            data.y = compute_target_tensor_only_edge_features(vol_base_case, gdf)
            data.hex_size = hex_size
            
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
                batch_counter += 1

                # Save intermediate result every batch_size data points
                if batch_counter % batch_size == 0:
                    batch_index = batch_counter // batch_size
                    torch.save(datalist, os.path.join(save_path, f'datalist_batch_{batch_index}.pt'))
                    datalist = []  # Reset datalist for the next batch
            else:
                print("Invalid line graph data")
    
    # Save any remaining data points
    if datalist:
        batch_index = (batch_counter // batch_size) + 1
        torch.save(datalist, os.path.join(save_path, f'datalist_batch_{batch_index}.pt'))

def extract_and_get_networks(compressed_dirs):
    """
    Extract all tar.gz files from compressed directories and return network paths.
    Each tar.gz contains files directly, so we need to create the proper directory structure.
    """
    networks = []
    temp_dirs = []  # Keep track for cleanup
    
    for compressed_dir in compressed_dirs:
        if os.path.exists(compressed_dir) and os.path.isdir(compressed_dir):
            print(f"Processing directory: {compressed_dir}")
            
            # Find all .tar.gz files in this directory
            tar_files = [f for f in os.listdir(compressed_dir) if f.endswith('.tar.gz')]
            print(f"  Found {len(tar_files)} tar.gz files")
            
            for tar_file in tar_files:
                tar_path = os.path.join(compressed_dir, tar_file)
                #print(f"  Extracting: {tar_file}")
                
                # Extract network name from tar filename (remove .tar.gz)
                network_name = tar_file.replace('.tar.gz', '')
                
                # Create temporary directory for this tar file
                temp_dir = tempfile.mkdtemp()
                temp_dirs.append(temp_dir)
                
                # Create the original directory structure to preserve hex size info
                # Extract hex size from compressed_dir name (e.g., compressed_rosenheim_hex_500_seed_2)
                compressed_dir_name = os.path.basename(compressed_dir)
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
                #print(f"    Created network: {network_name} in {compressed_dir_name}")
    
    return networks, temp_dirs

def process_single_city(city, project_root):
    """Process a single city and return its graph data."""
    print(f"Processing city: {city}")
    
    # Paths to compressed directories and basecase files
    sim_input_paths = [os.path.join(project_root, 'inductive_gnn_data', 'raw_data', city, f'compressed_{city}_hex_{hex_size}_seed_{seed}') for hex_size in hex_sizes]   
    basecase_links_path = os.path.join(project_root, 'inductive_gnn_data', 'links_and_stats', 'basecases_mean', city, f'{city}_basecase_average_output_links.geojson')
    basecase_stats_path = os.path.join(project_root, 'inductive_gnn_data', 'links_and_stats', 'basecases_mean', city, f'{city}_basecase_average_trips.csv')
    
    try:
        # Extract all tar.gz files from compressed directories
        networks, temp_dirs = extract_and_get_networks(sim_input_paths)
        networks = [network for network in networks if not network.endswith(".DS_Store")]
        networks.sort()

        # Load basecase data
        gdf_basecase_links = gpd.read_file(basecase_links_path)
        gdf_basecase_links = gdf_basecase_links.set_crs("EPSG:25832", allow_override=True)
        gdf_basecase_mean_mode_stats = pd.read_csv(basecase_stats_path, delimiter=',')

        # Process networks and generate graph data
        result_dic_output_links, result_dic_eqasim_trips = compute_result_dic(basecase_links=gdf_basecase_links, networks=networks)
        base_gdf = result_dic_output_links["base_network_no_policies"]
        city_data = generate_graph_data(result_dic=result_dic_output_links, result_dic_mode_stats=result_dic_eqasim_trips, links_base_case=base_gdf, gdf_basecase_mean_mode_stats=gdf_basecase_mean_mode_stats)
        
        print(f"Processed {city} with {len(city_data)} graphs")
        return city_data
        
    finally:
        # Clean up temporary directories
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
            #print(f"Cleaned up: {temp_dir}")
        print(f"Cleaned up: left {len(temp_dirs)} temp directories")

def process_cities(cities, project_root):
    """Process multiple cities and return flattened, shuffled data."""
    all_data = []
    
    for city in cities:
        city_data = process_single_city(city, project_root)
        all_data.append(city_data)
    
    return flatten_and_shuffle_data(all_data)

def flatten_and_shuffle_data(all_data):
    """Flatten nested data and shuffle with seed for reproducibility."""
    # Flatten all_data from list of lists to single list
    flattened_data = []
    for city_data in all_data:
        flattened_data.extend(city_data)
    
    print(f"Total graphs collected: {len(flattened_data)}")
    
    return flattened_data

def save_data_in_batches(flattened_data, result_path):
    """Save data in batches to specified path."""
    os.makedirs(result_path, exist_ok=True)
    
    for i in range(0, len(flattened_data), required_batch_size):
        batch = flattened_data[i:i + required_batch_size]
        batch_index = (i // required_batch_size) + 1 
        torch.save(batch, os.path.join(result_path, f'datalist_batch_{batch_index}.pt'))
        print(f"Saved batch {batch_index} with {len(batch)} graphs to {result_path}")

def main():
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process GNN data for different generalization scenarios.")
    parser.add_argument('--case_variant', type=str, choices=['transductive', 'moderate_inductive', 'complete_inductive'], required=True,
                        help='Choose the batch-processing variant: transductive / moderate_inductive / complete_inductive')
    args = parser.parse_args()
    case_variant = args.case_variant
    
    all_cities = ['rosenheim']
    seen_cities = []
    unseen_cities = []
    
    # Determine cities and paths based on case variant
    if case_variant == 'transductive':
        selected_cities = all_cities
        result_path_for_seen_cities = os.path.join(project_root, 'inductive_gnn_data', 'training_data', 'transductive')
    else:   #this block would be modified as per the case variant
        selected_cities = seen_cities
        result_path_for_seen_cities = os.path.join(project_root, 'inductive_gnn_data', 'training_data', args.case_variant, 'seen')
        result_path_for_unseen_cities = os.path.join(project_root, 'inductive_gnn_data', 'training_data', args.case_variant, 'unseen')
    
    # Process seen cities
    print(f"Processing {case_variant} case - seen cities: {selected_cities}")
    flattened_data_seen = process_cities(selected_cities, project_root)
    save_data_in_batches(flattened_data_seen, result_path_for_seen_cities)
    
    # Process unseen cities (for inductive cases only)
    if case_variant in ['moderate_inductive', 'complete_inductive']:
        print(f"Processing unseen cities: {unseen_cities}")
        flattened_data_unseen = process_cities(unseen_cities, project_root)
        save_data_in_batches(flattened_data_unseen, result_path_for_unseen_cities)

if __name__ == '__main__':
    main()
    