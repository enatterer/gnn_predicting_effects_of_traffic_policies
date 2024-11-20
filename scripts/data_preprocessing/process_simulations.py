import os
import glob
import math
import pickle
import argparse
from enum import IntEnum

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from collections import defaultdict

import processing_io as pio
from torch_geometric.transforms import LineGraph

from torch_geometric.data import Data, Batch
import shapely.wkt as wkt
from tqdm import tqdm
import fiona
import os

import alphashape
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from shapely.geometry import Point
import random


class EdgeFeatures(IntEnum):
    VOL_BASE_CASE = 0
    CAPACITY_BASE_CASE = 1
    CAPACITIES_NEW = 2
    CAPACITY_REDUCTION = 3
    FREESPEED = 4
    HIGHWAY = 5
    LENGTH = 6
    ALLOWED_MODE_CAR = 7
    ALLOWED_MODE_BUS = 8
    ALLOWED_MODE_PT = 9
    ALLOWED_MODE_TRAIN = 10
    ALLOWED_MODE_RAIL = 11
    ALLOWED_MODE_SUBWAY = 12


# Read all network data into a dictionary of GeoDataFrames
def compute_result_dic(basecase_links, subdirs):
    
    result_dic_output_links = {}
    result_dic_eqasim_trips = {}
    result_dic_output_links["base_network_no_policies"] = basecase_links
    # counter = 0
    for subdir in tqdm(subdirs, desc="Processing subdirs", unit="subdir"):
        # counter += 1
        # if counter > 1:
        #     break
        # print(f'Accessing folder: {subdir}')
        # print(len(os.listdir(subdir)))
        networks = [network for network in os.listdir(subdir) if not network.endswith(".DS_Store")]
        for network in networks:
            file_path = os.path.join(subdir, network)
            policy_key = pio.create_policy_key_1pm(network)
            df_output_links = pio.read_output_links(file_path)
            df_eqasim_trips = pio.read_eqasim_trips(file_path)
            if (df_output_links is not None and df_eqasim_trips is not None):
                df_output_links.drop(columns=['geometry'], inplace=True)
                gdf_extended = pio.extend_geodataframe(gdf_base=basecase_links, gdf_to_extend=df_output_links, column_to_extend='highway', new_column_name='highway')
                gdf_extended = pio.extend_geodataframe(gdf_base=basecase_links, gdf_to_extend=gdf_extended, column_to_extend='vol_car', new_column_name='vol_car_base_case')
                result_dic_output_links[policy_key] = gdf_extended
                df_eqasim_trips_list = [df_eqasim_trips]
                mode_stats = pio.calculate_avg_mode_stats(df_eqasim_trips_list)
                result_dic_eqasim_trips[policy_key] = mode_stats
    
    return result_dic_output_links, result_dic_eqasim_trips


def process_result_dic(result_dic, result_dic_mode_stats, districts, save_path=None, batch_size=500, links_base_case=None, gdf_basecase_mean_mode_stats=None):

    # PROCESS LINK GEOMETRIES
    edge_start_point_tensor, stacked_edge_geometries_tensor, district_centroids_tensor_padded, edges_base, nodes = pio.get_link_geometries(links_base_case, districts)
    
    os.makedirs(save_path, exist_ok=True)
    datalist = []
    linegraph_transformation = LineGraph()
    
    vol_base_case = links_base_case['vol_car'].values
    capacity_base_case = np.where(links_base_case['modes'].str.contains('car'), links_base_case['capacity'], 0)
    length = links_base_case['length'].values
    freespeed = links_base_case['freespeed'].values
    allowed_modes = pio.encode_modes(links_base_case)
    edge_index = torch.tensor(edges_base, dtype=torch.long).t().contiguous()
    x = torch.zeros((len(nodes), 1), dtype=torch.float)
    data = Data(edge_index=edge_index, x=x)
    
    batch_counter = 0
    for key, df in tqdm(result_dic.items(), desc="Processing result_dic", unit="dataframe"):   
        if isinstance(df, pd.DataFrame) and key != "base_network_no_policies":
            gdf = pio.prepare_gdf(df, links_base_case)
            capacities_new, capacity_reduction, highway, freespeed =  pio.get_basic_edge_attributes(capacity_base_case, gdf)

            edge_feature_dict = {
                EdgeFeatures.VOL_BASE_CASE: torch.tensor(vol_base_case),
                EdgeFeatures.CAPACITY_BASE_CASE: torch.tensor(capacity_base_case),
                EdgeFeatures.CAPACITIES_NEW: torch.tensor(capacities_new),
                EdgeFeatures.CAPACITY_REDUCTION: torch.tensor(capacity_reduction),
                EdgeFeatures.FREESPEED: torch.tensor(freespeed),
                EdgeFeatures.HIGHWAY: torch.tensor(highway),
                EdgeFeatures.LENGTH: torch.tensor(length),
                EdgeFeatures.ALLOWED_MODE_CAR: allowed_modes[0],
                EdgeFeatures.ALLOWED_MODE_BUS: allowed_modes[1],
                EdgeFeatures.ALLOWED_MODE_PT: allowed_modes[2],
                EdgeFeatures.ALLOWED_MODE_TRAIN: allowed_modes[3],
                EdgeFeatures.ALLOWED_MODE_RAIL: allowed_modes[4],
                EdgeFeatures.ALLOWED_MODE_SUBWAY: allowed_modes[5],
            }

            # Create the edge_tensor by iterating through the EdgeFeatures enum
            edge_tensor = [edge_feature_dict[feature] for feature in EdgeFeatures]

            # Stack the tensors
            edge_tensor = torch.stack(edge_tensor, dim=1)  # Shape: (31140, 14)
            
            linegraph_data = linegraph_transformation(data)
            linegraph_data.x = edge_tensor
            linegraph_data.pos = stacked_edge_geometries_tensor
            linegraph_data.y = pio.compute_target_tensor_only_edge_features(vol_base_case, gdf)
                        
            df_mode_stats = result_dic_mode_stats.get(key)
            if df_mode_stats is not None:
                pd.set_option('display.float_format', lambda x: '%.10f' % x)
                numeric_cols_base_case = gdf_basecase_mean_mode_stats.select_dtypes(include=[np.number]).columns
                numeric_cols = df_mode_stats.select_dtypes(include=[np.number]).columns
                mode_stats_diff = df_mode_stats[numeric_cols].values - gdf_basecase_mean_mode_stats[numeric_cols_base_case].values 
                mode_stats_tensor = torch.tensor(mode_stats_diff, dtype=torch.float)
                linegraph_data.mode_stats_diff = mode_stats_tensor
                mode_stats_diff_perc = mode_stats_tensor / gdf_basecase_mean_mode_stats[numeric_cols_base_case].values *100
                linegraph_data.mode_stats_diff_perc = mode_stats_diff_perc
                # print("mode_stats_diff_perc: ", mode_stats_diff_perc)

            if linegraph_data.validate(raise_on_error=True):
                datalist.append(linegraph_data)
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

    string_is_for_1pm = "pop_1pm"

    sim_input_path = "/home/enatterer/Development/matsim-ile-de-france/ile_de_france/data/"
    base_dir_sample_sim_input = sim_input_path + string_is_for_1pm + '_simulations/' + string_is_for_1pm + '_cap_reduction/'
    subdirs_pattern = os.path.join(base_dir_sample_sim_input, 'output_networks_*')
    subdirs = list(set(glob.glob(subdirs_pattern)))
    subdirs.sort()

    gdf_basecase_links = gpd.read_file('results/' + string_is_for_1pm + '_basecase_mean_links.geojson')
    gdf_basecase_links = gdf_basecase_links.set_crs("EPSG:4326", allow_override=True)

    gdf_basecase_mean_mode_stats = pd.read_csv('results/' + string_is_for_1pm + '_basecase_mean_mode_stats.csv', delimiter=',')
    districts = gpd.read_file("../../data/visualisation/districts_paris.geojson")

    result_dic_output_links, result_dic_eqasim_trips = compute_result_dic(basecase_links=gdf_basecase_links, subdirs=subdirs)
    base_gdf = result_dic_output_links["base_network_no_policies"]
    districts['district_centroid'] = districts['geometry'].centroid
    links_gdf_with_districts = gpd.sjoin(base_gdf, districts, how='left', op='intersects')

    # Group by edge and aggregate the district names
    links_gdf_with_districts = links_gdf_with_districts.groupby('link').agg({
        'from_node': 'first',
        'to_node': 'first',
        'length': 'first',
        'freespeed': 'first',
        'capacity': 'first',
        'lanes': 'first',
        'modes': 'first',
        'vol_car': 'first',
        'highway': 'first',
        'geometry': 'first',
        'c_ar': lambda x: list(x.dropna()),
        'district_centroid': lambda x: list(x.dropna()),
        'perimetre': lambda x: list(x.dropna()),
        'surface': lambda x: list(x.dropna()),
    }).reset_index()

    # # Analyze results
    # pio.analyze_geodataframes(result_dic=result_dic_output_links, consider_only_highway_edges=True)

    gdf_basecase_mean_mode_stats.rename(columns={'avg_total_travel_time': 'total_travel_time', 'avg_total_routed_distance': 'total_routed_distance', 'avg_trip_count': 'trip_count'}, inplace=True)
    # print(gdf_basecase_mean_mode_stats)

    result_df_name = 'sim_output_1pm_24_10_2024'
    result_path = '../../data/datasets_simulation_outputs/' + result_df_name
    process_result_dic(result_dic=result_dic_output_links, result_dic_mode_stats=result_dic_eqasim_trips, districts=districts, save_path=result_path, batch_size=50, links_base_case=base_gdf, gdf_basecase_mean_mode_stats=gdf_basecase_mean_mode_stats)