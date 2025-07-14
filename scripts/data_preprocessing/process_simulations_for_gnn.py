"""
Process simulation data (from MATSim) for GNNs. Load basecase and simulated graphs (with policies applied in various district combinations),
convert them to dual line graphs, and compute specified edge features. Save as PyTorch Geometric data batches for efficient loading and training.

Here we specify all features, then run_models can be called with a reduced set. Note that, for example, the flag "use_allowed_modes" is accessed from the run_models script.

***TO call this script, use the following command:***
python process_simulations_for_gnn.py --case_variant transductive --use_destination_activity True --use_allowed_modes False --run_number 1 --required_batch_size 1

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

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from data_preprocessing.help_functions import *

#control center
seed = 2 # Seed for Bavarian simulation
hex_sizes = [500, 1000, 2000] # Hexagon sizes to process
required_modes_on_links = ['car', 'car_passenger'] # Capacity will be reduced on links that have at least one of these modes
use_linegraph = True # Flag to use line graph transformation
use_allowed_modes = False
all_cities = ['rosenheim', 'schweinfurt', 'aschaffenburg', 'wuerzburg', 
              'bamberg', 'bayreuth', 'erlangen', 'fuerth', 'kempten', 
              'landshut', 'ingolstadt', 'regensburg']

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

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
    DESTINATION_HOME=15
    DESTINATION_WORK=16
    DESTINATION_OTHER=17
    DESTINATION_EDUCATION=18
    DESTINATION_LEISURE=19
    DESTINATION_SHOP=20
    DESTINATION_OUTSIDE=21


# Read all network data into a dictionary of GeoDataFrames
# for paris, please use the flag 'use_destination_activity' as False
def compute_result_dic(basecase_links, networks, use_destination_activity):
    
    result_dic_output_links = {}
    result_dic_eqasim_trips = {}
    result_dic_output_links["base_network_no_policies"] = basecase_links
    
    for network in networks:
        
        policy_key = create_policy_key(network) #TODO: fix for paris
        df_output_links = read_output_links(network)
        df_eqasim_trips = read_eqasim_trips(network)
        if (df_output_links is not None and df_eqasim_trips is not None):
            df_output_links.drop(columns=['geometry'], inplace=True)
            gdf_extended = extend_geodataframe(gdf_base=basecase_links, gdf_to_extend=df_output_links, column_to_extend='highway', new_column_name='highway')
            gdf_extended = extend_geodataframe(gdf_base=basecase_links, gdf_to_extend=gdf_extended, column_to_extend='vol_car', new_column_name='vol_car_base_case')
            if use_destination_activity:
                gdf_extended_with_destinations = add_destinations_to_gdf(gdf_extended, df_eqasim_trips)
            else:
                gdf_extended_with_destinations = gdf_extended
            result_dic_output_links[policy_key] = gdf_extended_with_destinations
            df_eqasim_trips_list = [df_eqasim_trips]
            mode_stats = calculate_avg_mode_stats(df_eqasim_trips_list)
            result_dic_eqasim_trips[policy_key] = mode_stats
    
    return result_dic_output_links, result_dic_eqasim_trips

def generate_graph_data_with_hexagons(city, result_dic, result_dic_mode_stats, links_base_case, gdf_basecase_mean_mode_stats, use_destination_activity, use_allowed_modes):
    datalist = []
    linegraph_transformation = LineGraph()

    vol_base_case = links_base_case['vol_car'].values
    capacity_base_case = get_capacity_base_case(links_base_case, required_modes_on_links)
    length = links_base_case['length'].values
    allowed_modes = encode_modes(links_base_case)
    
    # Get link geometries and edges_base FIRST
    _, stacked_edge_geometries_tensor, edges_base, nodes, pos_scaling_params = get_link_geometries(links_base_case, apply_scaling=True)
    
    # THEN use edges_base to create edge_index
    edge_index = torch.tensor(edges_base, dtype=torch.long).t().contiguous()

    # Filter out base_network_no_policies before the loop
    graph_items = {k: v for k, v in result_dic.items() 
                   if isinstance(v, pd.DataFrame) and k != "base_network_no_policies"}
    
    for key, df in graph_items.items():
        gdf = prepare_gdf(df, links_base_case) 
        _, capacity_reduction, highway, freespeed_scenario = get_basic_edge_attributes(capacity_base_case, gdf, required_modes_on_links)
        policy_region, scenario = key

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
        if use_destination_activity:
            edge_feature_dict.update({
                EdgeFeatures.DESTINATION_HOME: torch.tensor(gdf.get('home', pd.Series(0.0, index=gdf.index)).values),
                EdgeFeatures.DESTINATION_WORK: torch.tensor(gdf.get('work', pd.Series(0.0, index=gdf.index)).values),
                EdgeFeatures.DESTINATION_OTHER: torch.tensor(gdf.get('other', pd.Series(0.0, index=gdf.index)).values),
                EdgeFeatures.DESTINATION_EDUCATION: torch.tensor(gdf.get('education', pd.Series(0.0, index=gdf.index)).values),
                EdgeFeatures.DESTINATION_LEISURE: torch.tensor(gdf.get('leisure', pd.Series(0.0, index=gdf.index)).values),
                EdgeFeatures.DESTINATION_SHOP: torch.tensor(gdf.get('shop', pd.Series(0.0, index=gdf.index)).values),
                EdgeFeatures.DESTINATION_OUTSIDE: torch.tensor(gdf.get('outside', pd.Series(0.0, index=gdf.index)).values),
            })
        # Create the edge_tensor by iterating through the EdgeFeatures enum
        edge_tensor = torch.stack([edge_feature_dict[feat] for feat in EdgeFeatures if feat in edge_feature_dict], dim=1)
        
        data = Data(edge_index=edge_index)
        data.num_nodes = len(nodes)
        if use_linegraph:
            data = linegraph_transformation(data)
        data.x = edge_tensor
        data.pos = stacked_edge_geometries_tensor
        data.y = compute_target_tensor_only_edge_features(vol_base_case, gdf)
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
    return datalist

def get_capacity_base_case(links_base_case, required_modes_on_links):
    mode_masks = [links_base_case['modes'].str.contains(mode) for mode in required_modes_on_links]
    combined_mask = mode_masks[0]
    for mask in mode_masks[1:]:
        combined_mask = combined_mask | mask
    capacity_base_case = np.where(combined_mask, links_base_case['capacity'], 0)
    return capacity_base_case

def extract_and_get_networks(tar_files):
    """
    Extract all tar.gz files from compressed directories and return network paths.
    Each tar.gz contains files directly, so we need to create the proper directory structure.
    """
    networks = []
    temp_dirs = []  # Keep track for cleanup
    
    for tar_file in tar_files:
        tar_path = tar_file
        #print(f"  Extracting: {tar_file}")
                
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
        #print(f"    Created network: {network_name} in temp directory")
    
    return networks, temp_dirs

def process_single_city(city, project_root, result_path, use_destination_activity, use_allowed_modes, required_batch_size):
    
    """Process a single city and return its graph data."""
    print(f"\nProcessing city: {city}\n")
    
    # Paths to compressed directories and basecase files
    if city != 'paris':
        sim_input_paths = [os.path.join(project_root, 'data', 'raw_data', city, f'compressed_{city}_hex_{hex_size}_seed_{seed}') for hex_size in hex_sizes] 
    else:
        sim_input_paths = [os.path.join(project_root, 'data', 'raw_data', city, f'compressed_{city}_hex_{hex_size}_seed_{seed}') for hex_size in hex_sizes] # TODO: fix for paris
    
    basecase_links_path = os.path.join(project_root, 'data', 'links_and_stats', 'basecases_mean', city, 'basecase_average_output_links.geojson')
    basecase_stats_path = os.path.join(project_root, 'data', 'links_and_stats', 'basecases_mean', city, 'basecase_average_trips.csv')
    basecase_eqasim_trips_path = os.path.join(project_root, 'data', 'links_and_stats', 'basecases_mean', city, 'eqasim_trips.csv')
    
    """Process the city and save data in balanced batches."""
    total_batch_counter = 0
    
    # Collect tar files with hex_size info
    tar_files_with_hex = []
    for compressed_dir in sim_input_paths:
        if os.path.exists(compressed_dir) and os.path.isdir(compressed_dir):
            print(f"Processing directory: {compressed_dir}")
            
            # Extract hex_size from directory name (e.g., "compressed_schweinfurt_hex_2000_seed_2" -> 2000)
            import re
            hex_match = re.search(r'hex_(\d+)', os.path.basename(compressed_dir))
            hex_size = int(hex_match.group(1)) if hex_match else "unknown"
            
            # Find all .tar.gz files in this directory and tag with hex_size
            for f in os.listdir(compressed_dir):
                if f.endswith('.tar.gz'):
                    tar_files_with_hex.append({
                        'path': os.path.join(compressed_dir, f),
                        'hex_size': hex_size
                    })
    
    print(f"  Found {len(tar_files_with_hex)} tar.gz files")
    
    # Group by hex_size for batch counters
    hex_batch_counters = {}

    for i in tqdm(range(0, len(tar_files_with_hex), required_batch_size), desc="Processing batches", unit="batch"):
        
        sliced_inputs_with_hex = tar_files_with_hex[i:i+required_batch_size]
        sliced_inputs = [item['path'] for item in sliced_inputs_with_hex]
        
        # Get hex_size from this batch (should be the same for all files in batch)
        current_hex_size = sliced_inputs_with_hex[0]['hex_size']
        
        # Initialize batch counter for this hex_size if not exists
        if current_hex_size not in hex_batch_counters:
            hex_batch_counters[current_hex_size] = 0
        
        try:
            # Extract all tar.gz files from compressed directories
            networks, temp_dirs = extract_and_get_networks(sliced_inputs)
            networks = [network for network in networks if not network.endswith(".DS_Store")]
            networks.sort()

            # Load basecase eqasim trips
            df_basecase_eqasim_trips = pd.read_csv(basecase_eqasim_trips_path, delimiter=';')

            # Load basecase data
            gdf_basecase_links = gpd.read_file(basecase_links_path)
            gdf_basecase_links = gdf_basecase_links.set_crs("EPSG:25832", allow_override=True) # TODO: fix for paris
            
            if use_destination_activity:    
                gdf_basecase_links = add_destinations_to_gdf(gdf_basecase_links, df_basecase_eqasim_trips)
            
            gdf_basecase_mean_mode_stats = pd.read_csv(basecase_stats_path, delimiter=',')

            # Process networks and generate graph data
            result_dic_output_links, result_dic_eqasim_trips = compute_result_dic(basecase_links=gdf_basecase_links, networks=networks, use_destination_activity=use_destination_activity)
            base_gdf = result_dic_output_links["base_network_no_policies"]
            
            city_data = generate_graph_data_with_hexagons(city, result_dic=result_dic_output_links, result_dic_mode_stats=result_dic_eqasim_trips, links_base_case=base_gdf, gdf_basecase_mean_mode_stats=gdf_basecase_mean_mode_stats,
                                                            use_destination_activity=use_destination_activity, use_allowed_modes=use_allowed_modes)
            
            current_batch = []
            for graph in city_data:
                current_batch.append(graph)
            
            # Increment counter for this hex_size
            hex_batch_counters[current_hex_size] += 1
            total_batch_counter += 1
            
            # Save with hex_size in filename
            batch_filename = f'datalist_batch_{current_hex_size}_{hex_batch_counters[current_hex_size]}.pt'
            torch.save(current_batch, os.path.join(result_path, batch_filename))
            print(f"Saved batch: {batch_filename}")
            
            del city_data
            
        finally:
            # Clean up temporary directories
            for temp_dir in temp_dirs:
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        #print(f"Cleaned up: {temp_dir}")
                except Exception as e:
                    print(f"Warning: Could not clean up {temp_dir}: {e}")
            print(f"Cleaned up: {len(temp_dirs)} temp directories")
            
    print(f"Processed {city} with {len(tar_files_with_hex)} graphs")
    print(f"Batch counters by hex_size: {hex_batch_counters}")

def main():
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process GNN data for different generalization scenarios.")
    
    parser.add_argument('--use_destination_activity', type=bool, default=True,
                        help='Choose whether to use destination activity as feature for each link or not')
    
    parser.add_argument('--use_allowed_modes', type=bool, default=False,
                        help='Choose whether to use allowed modes as feature for each link or not')
    
    parser.add_argument('--required_batch_size', type=int, default=1,
                        help='how many graphs to save in each storage batch')
    
    args = parser.parse_args()
    
    use_destination_activity = args.use_destination_activity
    use_allowed_modes = args.use_allowed_modes
    required_batch_size = args.required_batch_size
    
    # Create the result base path
    result_base_path = os.path.join(project_root, 'data', 'training_data_2')
    os.makedirs(result_base_path, exist_ok=True)
    
    for city in all_cities:
        
        result_path = os.path.join(result_base_path, city)
        os.makedirs(result_path, exist_ok=True)
        
        process_single_city(city, project_root, result_path, use_destination_activity, use_allowed_modes, required_batch_size)

if __name__ == '__main__':
    main()
    