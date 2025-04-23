"""
Process simulation data (from MATSim) for GNNs. Load basecase and simulated graphs (with policies applied in various district combinations),
convert them to dual line graphs, and compute specified edge features. Save as PyTorch Geometric data batches for efficient loading and training.

Here we specify all features, then run_models can be called with a reduced set. Note that, for example, the flag "use_allowed_modes" is accessed from the run_models script.
"""

import os
import sys
from enum import IntEnum

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

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Paths to raw simulation data
sim_input_paths = [os.path.join(project_root, 'data', 'raw_data', 'exp_dist_not_connected_5k', 'output_networks_1000'),
                   os.path.join(project_root, 'data', 'raw_data', 'exp_dist_not_connected_5k', 'output_networks_2000'),
                   os.path.join(project_root, 'data', 'raw_data', 'exp_dist_not_connected_5k', 'output_networks_3000'),
                   os.path.join(project_root, 'data', 'raw_data', 'exp_dist_not_connected_5k', 'output_networks_4000'),
                   os.path.join(project_root, 'data', 'raw_data', 'exp_dist_not_connected_5k', 'output_networks_5000'),
                   os.path.join(project_root, 'data', 'raw_data', 'norm_dist_not_connected_5k', 'output_networks_1000'),
                   os.path.join(project_root, 'data', 'raw_data', 'norm_dist_not_connected_5k', 'output_networks_2000'),
                   os.path.join(project_root, 'data', 'raw_data', 'norm_dist_not_connected_5k', 'output_networks_3000'),
                   os.path.join(project_root, 'data', 'raw_data', 'norm_dist_not_connected_5k', 'output_networks_4000'),
                   os.path.join(project_root, 'data', 'raw_data', 'norm_dist_not_connected_5k', 'output_networks_5000')]

# Path to save the processed simulation data
result_path = os.path.join(project_root, 'data', 'train_data', 'dist_not_connected_10k_1pct')

# Path to the basecase links and stats
basecase_links_path = os.path.join(project_root, 'data', 'links_and_stats', 'pop_1pct_basecase_average_output_links.geojson')
basecase_stats_path = os.path.join(project_root, 'data', 'links_and_stats', 'pop_1pct_basecase_average_mode_stats.csv')

# Flag to use line graph transformation
use_linegraph = True

# Flag to use allowed modes or not
use_allowed_modes = False

class EdgeFeatures(IntEnum):
    VOL_BASE_CASE = 0
    CAPACITY_BASE_CASE = 1
    CAPACITY_REDUCTION = 2
    FREESPEED = 3
    HIGHWAY = 4
    LENGTH = 5
    ALLOWED_MODE_CAR = 6
    ALLOWED_MODE_BUS = 7
    ALLOWED_MODE_PT = 8
    ALLOWED_MODE_TRAIN = 9
    ALLOWED_MODE_RAIL = 10
    ALLOWED_MODE_SUBWAY = 11


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


def process_result_dic(result_dic, result_dic_mode_stats, save_path=None, batch_size=500, links_base_case=None, gdf_basecase_mean_mode_stats=None):

    # PROCESS LINK GEOMETRIES
    _, stacked_edge_geometries_tensor, edges_base, nodes = get_link_geometries(links_base_case)
    
    os.makedirs(save_path, exist_ok=True)
    datalist = []
    linegraph_transformation = LineGraph()
    
    vol_base_case = links_base_case['vol_car'].values
    capacity_base_case = np.where(links_base_case['modes'].str.contains('car'), links_base_case['capacity'], 0)
    length = links_base_case['length'].values
    freespeed = links_base_case['freespeed'].values
    allowed_modes = encode_modes(links_base_case)
    edge_index = torch.tensor(edges_base, dtype=torch.long).t().contiguous()
    
    batch_counter = 0
    for key, df in tqdm(result_dic.items(), desc="Processing result_dic", unit="dataframe"):   
        if isinstance(df, pd.DataFrame) and key != "base_network_no_policies":
            gdf = prepare_gdf(df, links_base_case)
            _, capacity_reduction, highway, freespeed =  get_basic_edge_attributes(capacity_base_case, gdf)

            edge_feature_dict = {
                EdgeFeatures.VOL_BASE_CASE: torch.tensor(vol_base_case),
                EdgeFeatures.CAPACITY_BASE_CASE: torch.tensor(capacity_base_case),
                EdgeFeatures.CAPACITY_REDUCTION: torch.tensor(capacity_reduction),
                EdgeFeatures.FREESPEED: torch.tensor(freespeed),
                EdgeFeatures.HIGHWAY: torch.tensor(highway),
                EdgeFeatures.LENGTH: torch.tensor(length),
            }

            if use_allowed_modes:
                edge_feature_dict.update({
                    EdgeFeatures.ALLOWED_MODE_CAR: allowed_modes[0],
                    EdgeFeatures.ALLOWED_MODE_BUS: allowed_modes[1],
                    EdgeFeatures.ALLOWED_MODE_PT: allowed_modes[2],
                    EdgeFeatures.ALLOWED_MODE_TRAIN: allowed_modes[3],
                    EdgeFeatures.ALLOWED_MODE_RAIL: allowed_modes[4],
                    EdgeFeatures.ALLOWED_MODE_SUBWAY: allowed_modes[5]})

            # Create the edge_tensor by iterating through the EdgeFeatures enum
            edge_tensor = [edge_feature_dict[feature] for feature in EdgeFeatures if feature in edge_feature_dict]

            # Stack the tensors
            edge_tensor = torch.stack(edge_tensor, dim=1)
            
            data = Data(edge_index=edge_index)
            data.num_nodes = edge_index.shape[1] if use_linegraph else len(nodes)
            if use_linegraph:
                data = linegraph_transformation(data)
            
            data.x = edge_tensor
            data.pos = stacked_edge_geometries_tensor
            data.y = compute_target_tensor_only_edge_features(vol_base_case, gdf)
                        
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


def main():

    networks = list()

    for path in sim_input_paths:
        networks += [os.path.join(path, network) for network in os.listdir(path)]
    
    networks = [network for network in networks if os.path.isdir(network) and not network.endswith(".DS_Store")]
    networks.sort()

    gdf_basecase_links = gpd.read_file(basecase_links_path)
    gdf_basecase_links = gdf_basecase_links.set_crs("EPSG:4326", allow_override=True)
    gdf_basecase_mean_mode_stats = pd.read_csv(basecase_stats_path, delimiter=',')

    result_dic_output_links, result_dic_eqasim_trips = compute_result_dic(basecase_links=gdf_basecase_links, networks=networks)
    base_gdf = result_dic_output_links["base_network_no_policies"]

    gdf_basecase_mean_mode_stats.rename(columns={'avg_total_travel_time': 'total_travel_time', 'avg_total_routed_distance': 'total_routed_distance', 'avg_trip_count': 'trip_count'}, inplace=True)
    process_result_dic(result_dic=result_dic_output_links, result_dic_mode_stats=result_dic_eqasim_trips, save_path=result_path, batch_size=50, links_base_case=base_gdf, gdf_basecase_mean_mode_stats=gdf_basecase_mean_mode_stats)


if __name__ == '__main__':
    main()