
from collections import defaultdict
import fiona
import geopandas as gpd
import glob
import gzip
import math
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import shapely.wkt as wkt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torchvision
import torchvision.transforms as T
import tqdm
from matplotlib.colors import LogNorm, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import nearest_points, unary_union
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import LineGraph


districts = gpd.read_file("../../data/visualisation/districts_paris.geojson")

# paris_inside_bvd_peripherique = "../../data/paris_inside_bvd_per/referentiel-comptages-edit.shp"
# gdf_paris_inside_bvd_per = gpd.read_file(paris_inside_bvd_peripherique)
# boundary_df = alphashape.alphashape(gdf_paris_inside_bvd_per, 435).exterior[0]
# linear_ring_polygon = Polygon(boundary_df)

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

def find_duplicate_edges_in_gdf(gdf):
    edge_count = defaultdict(list)
    for idx, row in gdf.iterrows():
        # Keep the edge direction by using the tuple without sorting
        edge = (row['from_node'], row['to_node'])
        edge_count[edge].append(idx)
    
    # Filter to include only edges that appear more than once
    duplicates = {edge: indices for edge, indices in edge_count.items() if len(indices) > 1}
    return duplicates

def summarize_duplicate_edges(gdf):
    if 'vol_car' not in gdf.columns:
        print("'vol_car' column does not exist in the dataframe")
        return gdf

    gdf['edge_id'] = gdf.apply(lambda row: (row['from_node'], row['to_node']), axis=1)
    grouped = gdf.groupby('edge_id')
    
    def aggregate_edges(group):
        non_zero_vol = group[group['vol_car'] != 0]
        if len(non_zero_vol) > 1:
            # If there are multiple non-zero entries, take the one with the highest vol_car
            combined = non_zero_vol.loc[non_zero_vol['vol_car'].idxmax()].copy()
        elif not non_zero_vol.empty:
            combined = non_zero_vol.iloc[0].copy()
        else:
            combined = group.iloc[0].copy()
        
        # We're no longer summing vol_car, just keeping the value from the selected row
        combined['original_directions'] = list(group[['from_node', 'to_node']].itertuples(index=False, name=None))
        return combined
    
    summarized_gdf = grouped.apply(aggregate_edges)
    summarized_gdf = summarized_gdf.reset_index(drop=True)
    summarized_gdf = summarized_gdf.drop(columns=['edge_id'])
    return summarized_gdf

def identify_summarized_entries_detailed(original_gdf, summarized_gdf):
    original_gdf['edge_id'] = original_gdf['from_node'].astype(str) + '_' + original_gdf['to_node'].astype(str)
    summarized_gdf['edge_id'] = summarized_gdf['from_node'].astype(str) + '_' + summarized_gdf['to_node'].astype(str)
    
    original_counts = original_gdf['edge_id'].value_counts()
    summarized_entries = summarized_gdf[summarized_gdf['edge_id'].isin(original_counts[original_counts > 1].index)]
    
    detailed_entries = []
    both_zero = []
    both_nonzero = []
    one_zero_one_nonzero = []
    
    for _, summarized_row in summarized_entries.iterrows():
        edge_id = summarized_row['edge_id']
        original_rows = original_gdf[original_gdf['edge_id'] == edge_id]
        
        vol_car_values = original_rows['vol_car'].values
        entry = {
            'summarized': summarized_row,
            'original': original_rows,
            'count': len(original_rows)
        }
        
        if all(vol_car == 0 for vol_car in vol_car_values):
            both_zero.append(entry)
        elif all(vol_car != 0 for vol_car in vol_car_values):
            both_nonzero.append(entry)
        else:
            one_zero_one_nonzero.append(entry)
        
        detailed_entries.append(entry)
    
    return detailed_entries, both_zero, both_nonzero, one_zero_one_nonzero


def analyze_geodataframes(result_dic: dict, consider_only_highway_edges: bool = True):
    """
    Analyse the results of the simulation and compare them to the base network data.

    Parameters:
    result_dic: The dictionary containing the results of the simulation as gdfs.
    is_1pm (bool): True if the policy is introduced at 1pm, else False.
    consider_only_highway_edges (bool): Compute the total volume and capacity for only those edges on "highstreets" if set to "True", for all edges otherwise.
    """
    base_gdf = result_dic.get("base_network_no_policies")
    if base_gdf is None:
        raise ValueError("Base network data not found in the result dictionary")
    highway_types = ["primary", "secondary", "tertiary", "primary_link", "secondary_link", "tertiary_link"]
    if consider_only_highway_edges:
        base_gdf = base_gdf[base_gdf["highway"].isin(highway_types)]
    base_vol_car = round(base_gdf['vol_car'].sum())
    base_capacity_car = round(base_gdf['capacity'].sum())
    # print(f"Base, volume: {base_vol_car}")
    # print(f"Base, capacity: {base_capacity_car}")

    for policy, gdf in result_dic.items():
        if (policy == "base_network_no_policies"):
            continue
        print(f"Policy: {policy}")
        if consider_only_highway_edges:
            gdf = gdf[gdf["highway"].isin(highway_types)]
        total_vol_car = gdf['vol_car'].sum()
        total_capacity_car = round(gdf['capacity'].sum())
        vol_car_increase = ((total_vol_car - base_vol_car) / base_vol_car) * 100
        capacity_car_increase = ((total_capacity_car - base_capacity_car) / base_capacity_car) * 100
        # print(f"With policy, volume: {total_vol_car}")
        # print(f"With policy, capacity: {total_capacity_car}")
        print(f"Total change in 'vol_car': {vol_car_increase:.2f}%")
        print(f"Total change in capacity (car edges): {capacity_car_increase:.2f}%")
        
        
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

    
def create_policy_key_1pct(folder_name):
    # Extract the relevant part of the folder name
    parts = folder_name.split('_')[1:]  # Ignore the first part ('network')
    district_info = '_'.join(parts).replace('d_', '')
    districts = district_info.split('_')
    return f"policy introduced in Arrondissement(s) {', '.join(districts)}"
    
def create_policy_key_1pm(folder_name):
    # Extract the relevant part of the folder name
    base_name = os.path.basename(folder_name)  # Get the base name of the file or folder
    parts = base_name.split('_')[1:]  # Ignore the first part ('network')
    district_info = '_'.join(parts)
    districts = district_info.split('_')
    return f"Policy introduced in Arrondissement(s) {', '.join(districts)}"

def is_single_district(filename):
    return filename.count('_') == 2

def plot_simulation_output(df, districts_of_interest: list, is_for_1pm: str, in_percentage: bool):
    # Convert DataFrame to GeoDataFrame
    
    column_to_plot = "vol_car" if in_percentage else "vol_car_percentage_difference"
    
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:2154")
    gdf = gdf.to_crs(epsg=4326)

    x_min = gdf.total_bounds[0] + 0.05
    y_min = gdf.total_bounds[1] + 0.05
    x_max = gdf.total_bounds[2]
    y_max = gdf.total_bounds[3]
    bbox = box(x_min, y_min, x_max, y_max)
    
    # Filter the network to include only the data within the bounding box
    gdf = gdf[gdf.intersects(bbox)]
    
    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # Filter edges based on the "osm:way:highway" column
    highway_types = ["primary", "secondary", "tertiary", "primary_link", "secondary_link", "tertiary_link"]
    gdf = gdf[gdf["highway"].isin(highway_types)]
    
    target_districts = districts[districts['c_ar'].isin(districts_of_interest)]
    other_districts = districts[~districts['c_ar'].isin(districts_of_interest)]

    gdf['intersects_target_districts'] = gdf.apply(lambda row: target_districts.intersects(row.geometry).any(), axis=1)

    # Use TwoSlopeNorm for custom normalization
    norm = TwoSlopeNorm(vmin=gdf[column_to_plot].min(), vcenter=gdf[column_to_plot].median(), vmax=gdf[column_to_plot].max())
    
    # Plot the edges that intersect with target districts thicker
    gdf[gdf['intersects_target_districts']].plot(column=column_to_plot, cmap='coolwarm', linewidth=4, ax=ax, legend=False,
             norm=norm, label = "Higher order roads", zorder=2)
    
    # Plot the other edges
    gdf[~gdf['intersects_target_districts']].plot(column=column_to_plot, cmap='coolwarm', linewidth=4, ax=ax, legend=False,
             norm=norm, zorder=1)
    
    # Add buffer to target districts to avoid overlapping with edges
    buffered_target_districts = target_districts.copy()
    buffered_target_districts['geometry'] = buffered_target_districts.buffer(0.0005)
    # Ensure the buffered_target_districts GeoDataFrame is in the same CRS
    if buffered_target_districts.crs != gdf.crs:
        buffered_target_districts.to_crs(gdf.crs, inplace=True)

    # Create a single outer boundary
    outer_boundary = unary_union(buffered_target_districts.geometry).boundary

    # Plot only the outer boundary
    gpd.GeoSeries(outer_boundary, crs=gdf.crs).plot(ax=ax, edgecolor='black', linewidth=1, label="Arrondissements " + list_to_string(districts_of_interest), zorder=4)

    # ax.set_aspect('equal')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Customize the plot with Times New Roman font and size 15
    plt.xlabel("Longitude", fontname='Times New Roman', fontsize=15)
    plt.ylabel("Latitude", fontname='Times New Roman', fontsize=15)

    # Customize tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(15)
    ax.legend(prop={'family': 'Times New Roman', 'size': 15})
    
    # Manually set the position of the main plot axis
    ax.set_position([0.1, 0.1, 0.75, 0.75])

    # Create an axis on the right side for the color bar
    cax = fig.add_axes([0.87, 0.22, 0.03, 0.5])  # Manually position the color bar

    # Create the color bar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, cax=cax)
    
    # Set color bar font properties
    cbar.ax.tick_params(labelsize=15)
    for t in cbar.ax.get_yticklabels():
        t.set_fontname('Times New Roman')
    cbar.ax.yaxis.label.set_fontname('Times New Roman')
    cbar.ax.yaxis.label.set_size(15)
    if in_percentage:
        cbar.set_label('Car volume: Difference to base case (%)', fontname='Times New Roman', fontsize=15)
    else:
        cbar.set_label('Car volume: Difference to base case (absolut)', fontname='Times New Roman', fontsize=15)
    plt.savefig("results/difference_to_policies_in_zones_" + list_to_string(districts_of_interest, "_") + is_for_1pm, bbox_inches='tight')
    plt.show()
    
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

def get_subdirs(full_path: str):
    subdirs_pattern = os.path.join(full_path, 'output_seed_*')
    subdirs_list = list(set(glob.glob(subdirs_pattern)))
    subdirs_list.sort()
    return subdirs_list

# Function to read and convert CSV.GZ to GeoDataFrame
def read_network_data(file_path):
    if os.path.exists(file_path):
        # Read the CSV file with the correct delimiter
        df = pd.read_csv(file_path, delimiter=';')
        # Convert the 'geometry' column to actual geometrical data
        df['geometry'] = df['geometry'].apply(wkt.loads)
        
        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        gdf.crs = "EPSG:2154"  # Assuming the original CRS is EPSG:2154
        gdf.to_crs("EPSG:4326", inplace=True)
        return gdf
    else:
        return None
    
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
    

def extract_numbers(path):
    name = path.split('/')[-1]
    # Use regular expression to find all numbers in the string
    numbers = re.findall(r'\d+', name)
    # Convert the list of numbers to a set of integers
    return set(map(int, numbers))

def create_dic_seed_2_output_links(subdir: str):
    result_dic = {}
    for s in subdir:
        random_seed = extract_numbers(s)
        
        output_links = s + "/output_links.csv.gz"
        gdf = read_network_data(output_links)
        if gdf is not None:
            result_dic[str(random_seed)] = gdf
    return result_dic

def create_dic_seed_2_eqasim_trips(subdir: str):
    result_dic = {}
    for s in subdir:
        random_seed = extract_numbers(s)
        eqasim_trips = s + "/eqasim_trips.csv"
        if os.path.exists(eqasim_trips):
            df_eqasim_trips = pd.read_csv(eqasim_trips, delimiter=';')
            if df_eqasim_trips is not None:
                result_dic[str(random_seed)] = df_eqasim_trips
    return result_dic

def compute_average_or_median_geodataframe(geodataframes, column_name, is_mean: bool = True):
    """
    Compute the average GeoDataFrame from a list of GeoDataFrames for a specified column.
    
    Parameters:
    geodataframes (list of GeoDataFrames): List containing GeoDataFrames
    column_name (str): The column name for which to compute the average
    
    Returns:
    GeoDataFrame: A new GeoDataFrame with the average values for the specified column
    """
    # Create a copy of the first GeoDataFrame to use as the base
    average_gdf = geodataframes[0].copy()
    
    # Extract the specified column values from all GeoDataFrames
    column_values = np.array([gdf[column_name].values for gdf in geodataframes])
    
    if (is_mean):
    # Calculate the average values for the specified column
        column_average = np.mean(column_values, axis=0)
    else:
        column_average = np.median(column_values, axis=0)

    # Assign the average values to the new GeoDataFrame
    average_gdf[column_name] = column_average
    
    return average_gdf


def compute_difference_geodataframe(gdf_to_substract_from, gdf_to_substract, column_name):
    """
    Compute the difference of a specified column between two GeoDataFrames.
    
    Parameters:
    gdf1 (GeoDataFrame): The first GeoDataFrame
    gdf2 (GeoDataFrame): The second GeoDataFrame
    column_name (str): The column name for which to compute the difference
    
    Returns:
    GeoDataFrame: A new GeoDataFrame with the differences for the specified column
    """
    # Ensure the two GeoDataFrames have the same shape
    if gdf_to_substract_from.shape != gdf_to_substract.shape:
        raise ValueError("GeoDataFrames must have the same shape")

    # Ensure the two GeoDataFrames have the same indices
    if not gdf_to_substract_from.index.equals(gdf_to_substract.index):
        raise ValueError("GeoDataFrames must have the same indices")
    
    # Ensure the two GeoDataFrames have the same geometries
    if not gdf_to_substract_from.geometry.equals(gdf_to_substract.geometry):
        raise ValueError("GeoDataFrames must have the same geometries")
    
    # Create a copy of the first GeoDataFrame to use as the base for the difference GeoDataFrame
    difference_gdf = gdf_to_substract_from.copy()

    # Compute the difference for the specified column
    difference_gdf[column_name] = gdf_to_substract_from[column_name] - gdf_to_substract[column_name]
    difference_gdf[column_name + "_percentage_difference"] =  (
        difference_gdf[column_name] / gdf_to_substract[column_name] * 100
    )
 
    return difference_gdf

def remove_columns(gdf_with_correct_columns, gdf_to_be_adapted):
    """
    Remove columns from gdf1 that are not present in gdf2.
    
    Parameters:
    gdf1 (GeoDataFrame): The GeoDataFrame from which columns will be removed
    gdf2 (GeoDataFrame): The GeoDataFrame that provides the column template
    
    Returns:
    GeoDataFrame: A new GeoDataFrame with only the columns present in gdf2
    """
    columns_to_keep = gdf_with_correct_columns.columns
    gdf1_filtered = gdf_to_be_adapted[columns_to_keep]
    return gdf1_filtered

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

def compute_close_homes(links_gdf_input:pd.DataFrame, information_gdf_input:pd.DataFrame, utm_crs:str, distance:int=50):
    links_gdf = links_gdf_input.copy()
    information_gdf = information_gdf_input.copy()
    close_places = []
    links_gdf_utm = links_gdf.to_crs(utm_crs)
    information_gdf_utm = information_gdf.to_crs(utm_crs)
    for i, row in tqdm(enumerate(links_gdf_utm.iterrows()), desc="Processing rows", unit="row"):
        buffer_utm = row[1].geometry.buffer(distance=distance)
        buffer = gpd.GeoSeries([buffer_utm], crs=utm_crs).to_crs(links_gdf_utm.crs)[0]
        matched_information = information_gdf_utm[information_gdf_utm.geometry.within(buffer)]
        socioprofessional_classes = matched_information['socioprofessional_class'].tolist()
        close_places.append((len(socioprofessional_classes), socioprofessional_classes))
    return close_places

def process_close_count_to_tensor(close_count_list: list):
    socio_professional_classes = [item[1] for item in close_count_list]
    unique_classes = set([2, 3, 4, 5, 6, 7, 8])
    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}

    tensor_shape = (len(close_count_list), len(unique_classes))
    close_homes_tensor = torch.zeros(tensor_shape)

    for i, classes in enumerate(socio_professional_classes):
        for cls in classes:
            if cls in class_to_index:  # Ensure the class is in the predefined set
                close_homes_tensor[i, class_to_index[cls]] += 1
    
    close_homes_tensor_sparse = close_homes_tensor.to_sparse()
    return close_homes_tensor_sparse


# def calculate_averaged_results(trips_df):
#     """Calculate average travel time and routed distance grouped by mode."""
#     return trips_df.groupby('mode').agg(
#         total_travel_time=('travel_time', 'mean'),
#         total_routed_distance=('routed_distance', 'mean')
#     ).reset_index()
    
    
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
    print(len(tensor_list))
    return tensor_list
    # return torch.tensor(modes_encoded.values, dtype=torch.float)


def encode_modes_string(mode_string):
    """Encode the 'modes' attribute based on specific strings."""
    modes_conditions = {
        'car': int("car" in mode_string),
        'bus': int("bus" in mode_string),
        'pt': int("pt" in mode_string),
        'train': int("train" in mode_string),
        'rail': int("rail" in mode_string),
        'subway': int("subway" in mode_string),
    }
    modes_encoded_tensor = torch.tensor(list(modes_conditions.values()), dtype=torch.float)
    return modes_encoded_tensor

def get_dfs(base_dir:str):
    files = os.listdir(base_dir)
    for file in files:
        file_path = os.path.join(base_dir, file)
        base_name, ext = os.path.splitext(file)
        if base_name.startswith("idf_1pm_"):
            base_name = base_name.replace("idf_1pm_", "")
        var_name = base_name  # Start with the cleaned base name
    
        if file.endswith('.csv'):
            try:
                var_name = f"{var_name}_df"  
                globals()[var_name] = pd.read_csv(file_path, sep=";")
                print(f"Loaded CSV file: {file} into variable: {var_name}")
            except Exception as e:
                print(f"Error loading CSV file {file}: {e}")
            
        elif file.endswith('.gpkg'):
            try:
                var_name = f"{var_name}_gdf"  
                layers = fiona.listlayers(file_path)
                geodataframes = {layer: gpd.read_file(file_path, layer=layer, geometry = 'geometry', crs="EPSG:2154") for layer in layers}
                for layer, gdf in geodataframes.items():
                # print(f"Layer: {layer}")
                    gdf = gdf.to_crs(epsg=4326)
                    globals()[var_name] = gdf
                    print(f"Loaded GPKG file: {file} into variable: {var_name}")
            except Exception as e:
                print(f"Error loading CSV file {file}: {e}")
    homes_gdf = globals()["homes_gdf"]
    households_df = globals()["households_df"]
    persons_df = globals()["persons_df"]
    activities_gdf = globals()["activities_gdf"]
    trips_df = globals()["trips_gdf"]
    return homes_gdf, households_df, persons_df, activities_gdf, trips_df

def extract_start_end_points(geometry):
    if len(geometry.coords) != 2:
        raise ValueError("Linestring does not have exactly 2 elements.")
    return geometry.coords[0], geometry.coords[-1]

def get_close_trips_tensor(links_gdf_input, trips_gdf_input, utm_crs, distance):
    close_trips_count = compute_close_homes(links_gdf_input = links_gdf_input, information_gdf_input = trips_gdf_input, utm_crs = utm_crs, distance=distance)
    close_trips_count_tensor = process_close_count_to_tensor(close_trips_count)
    return close_trips_count, close_trips_count_tensor

def get_start_and_end_gdf(trips_with_socio, crs):
    trips_start = trips_with_socio.copy()
    trips_end = trips_with_socio.copy()

    trips_start_gdf = gpd.GeoDataFrame(
    trips_start, 
    geometry=gpd.points_from_xy(
        trips_start['start_point'].apply(lambda p: p[0]), 
        trips_start['start_point'].apply(lambda p: p[1])
    ), 
    crs=crs
)

    trips_end_gdf = gpd.GeoDataFrame(
    trips_end, 
    geometry=gpd.points_from_xy(
        trips_end['end_point'].apply(lambda p: p[0]), 
        trips_end['end_point'].apply(lambda p: p[1])
    ), 
    crs=crs
)
    return trips_start_gdf,trips_end_gdf

def process_centroid(geom_list):
    if not geom_list:  # Empty list
        return [np.nan, np.nan, np.nan]
    elif len(geom_list) == 1:
        return [geom_list[0], np.nan, np.nan]
    elif len(geom_list) == 2:
        return [geom_list[0], geom_list[1], np.nan]
    else:
        return [geom_list[0], geom_list[1], geom_list[2]]
    
def extract_point_coordinates(geom_list):
    coordinates = []
    for geom in geom_list:
        if isinstance(geom, Point):
            coordinates.append((geom.x, geom.y))
        else:
            coordinates.append((np.nan, np.nan))
    return coordinates

def process_value_list(perimeter_list):
    if not perimeter_list:  # Empty list
        return [np.nan, np.nan, np.nan]
    elif len(perimeter_list) == 1:
        return [perimeter_list[0], np.nan, np.nan]
    elif len(perimeter_list) == 2:
        return [perimeter_list[0], perimeter_list[1], np.nan]
    else:
        return [perimeter_list[0], perimeter_list[1], perimeter_list[2]]
    
def compute_district_2_information_counts(district_information_counts, column_to_filter_for):
    district_group_2_information_counts = {}
    for district, group in district_information_counts:        
        # ignore groups with more than one district here. 
        if len(district) == 1:
            total_counts = 0
            total_distributions = []
            counts = group[column_to_filter_for].values            
            for c in counts:
                total_counts += c[0]
                if c[1] is not None and len(c[1]) > 0:
                    total_distributions.extend(c[1])
            distribution_counts = [total_distributions.count(i) for i in range(2, 9)]   
            district_group_2_information_counts[district] = distribution_counts
    return district_group_2_information_counts, distribution_counts

def compute_district_2_information_tensor(district_2_information_counts, distribution_counts, gdf_input):
    district_home_counts_tensor = torch.zeros((len(gdf_input), 3, len(distribution_counts)), dtype=torch.float)
    nan_tensor = torch.full((len(distribution_counts),), float('nan'))

    for idx, row in gdf_input.iterrows():
        district_combination = row['district']
        district_combination_tuple = tuple(district_combination)
        if len(district_combination_tuple) == 0:
            district_home_counts_tensor[idx] = torch.stack([nan_tensor, nan_tensor, nan_tensor])
        elif len(district_combination_tuple) == 1:
            district_home_counts_tensor[idx] = torch.stack([torch.tensor(district_2_information_counts[district_combination_tuple]), nan_tensor, nan_tensor])
        elif len(district_combination_tuple) == 2:
            a, b = district_combination_tuple
            district_home_counts_tensor[idx] = torch.stack([torch.tensor(district_2_information_counts[(a,)]), torch.tensor(district_2_information_counts[(b,)]), nan_tensor])
        elif len(district_combination_tuple) == 3:
            a, b, c = district_combination_tuple
            district_home_counts_tensor[idx] = torch.stack([torch.tensor(district_2_information_counts[(a,)]), torch.tensor(district_2_information_counts[(b,)]), torch.tensor(district_2_information_counts[(c,)])])
        else:
            print("NOT OK!")
            print(district_combination_tuple)
    return district_home_counts_tensor

def preprocess_links(links_gdf):
    for index, row in links_gdf.iterrows():
        if len(row['district']) >= 4:
            row['district'].pop(random.randint(0, len(row['district']) - 1))
    return links_gdf

# def read_output_links(folder):
#     file_path = os.path.join(folder, 'output_links.csv.gz')
#     if os.path.exists(file_path):
#         try:
#             # Read the CSV file with the correct delimiter
#             df = pd.read_csv(file_path, delimiter=';')
#             return df
#         except Exception:
#             print("empty data error" + file_path)
#             return None
#     else:
#         return None

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

def aggregate_district_information(links_gdf, tensors_edge_information):
    
    # Assuming tensors_edge_information is a list of tensors
    # vol_base_case = tensors_edge_information[0]  # Adjust index if needed
    capacities_base = tensors_edge_information[1]  
    capacities_new = tensors_edge_information[2] 
    capacity_reduction = tensors_edge_information[3]  
    freespeed_base = tensors_edge_information[4]
    freespeed = tensors_edge_information[5]
    # highway = tensors_edge_information[6]
    length = tensors_edge_information[7]
    cars_allowed = tensors_edge_information[8]
    bus_allowed = tensors_edge_information[9]
    pt_allowed = tensors_edge_information[10]
    train_allowed = tensors_edge_information[11]
    rail_allowed = tensors_edge_information[12]
    subway_allowed = tensors_edge_information[13]
    
    district_info = {}
            
    # modes_str = ""
    for idx, row in links_gdf.iterrows():
        districts = row['district']
        modes = row['modes']
        # modes_str += modes + ","
        for district in districts:
            if district not in district_info:
                district_info[district] = {
                    'vol_base_case': 0,
                    'capacity_base': 0,
                    'capacity_new': 0,
                    'capacity_reduction': 0,
                    'freespeed_base_sum': 0,
                    'freespeed_base_count': 0,
                    'freespeed_sum': 0,
                    'freespeed_count': 0,
                    'highway_sum': 0,
                    'highway_count': 0,
                    'length': 0,
                    'edge_count': 0,
                    'cars_allowed': 0,
                    'bus_allowed': 0,
                    'pt_allowed': 0,
                    'train_allowed': 0,
                    'rail_allowed': 0,
                    'subway_allowed': 0,
                }
            
            if "car" in modes:
                # district_info[district]['vol_base_case'] += vol_base_case[idx].item()
                district_info[district]['capacity_base'] += capacities_base[idx].item()
                district_info[district]['capacity_new'] += capacities_new[idx].item()
                district_info[district]['capacity_reduction'] += capacity_reduction[idx].item()
                district_info[district]['freespeed_sum'] += freespeed[idx].item()
                district_info[district]['freespeed_base_sum'] += freespeed_base[idx].item()
                district_info[district]['freespeed_base_count'] += 1
                district_info[district]['freespeed_count'] += 1
            
            district_info[district]['length'] += length[idx].item()

            highway_value = highway_mapping.get(row['highway'], -1)
            district_info[district]['highway_sum'] += highway_value
            district_info[district]['highway_count'] += 1
            district_info[district]['edge_count'] += 1
            
            district_info[district]['cars_allowed'] += cars_allowed[idx].item()
            district_info[district]['bus_allowed'] += bus_allowed[idx].item()
            district_info[district]['pt_allowed'] += pt_allowed[idx].item()
            district_info[district]['train_allowed'] += train_allowed[idx].item()
            district_info[district]['rail_allowed'] += rail_allowed[idx].item()
            district_info[district]['subway_allowed'] += subway_allowed[idx].item()

    # Compute averages 
    for district in district_info:
        district_info[district]['freespeed_base'] = district_info[district]['freespeed_base_sum'] / district_info[district]['freespeed_base_count']
        district_info[district]['freespeed'] = district_info[district]['freespeed_sum'] / district_info[district]['freespeed_count']
        district_info[district]['highway'] = district_info[district]['highway_sum'] / district_info[district]['highway_count']
            
    # Sort districts by their identifiers
    districts = sorted(district_info.keys())
    
    vol_base_case_tensor = torch.tensor([district_info[d]['vol_base_case'] for d in districts])
    capacity_base_tensor = torch.tensor([district_info[d]['capacity_base'] for d in districts])
    capacity_new_tensor = torch.tensor([district_info[d]['capacity_new'] for d in districts])
    capacity_reduction_tensor = torch.tensor([district_info[d]['capacity_reduction'] for d in districts])
    
    length_tensor = torch.tensor([district_info[d]['length'] for d in districts])
    edge_count_tensor = torch.tensor([district_info[d]['edge_count'] for d in districts])
    highway_tensor = torch.tensor([district_info[d]['highway'] for d in districts])
    freespeed_base_tensor = torch.tensor([district_info[d]['freespeed_base'] for d in districts])
    freespeed_tensor = torch.tensor([district_info[d]['freespeed'] for d in districts])
    
    cars_allowed_tensor = torch.tensor([district_info[d]['cars_allowed'] for d in districts])
    bus_allowed_tensor = torch.tensor([district_info[d]['bus_allowed'] for d in districts])
    pt_allowed_tensor = torch.tensor([district_info[d]['pt_allowed'] for d in districts])
    train_allowed_tensor = torch.tensor([district_info[d]['train_allowed'] for d in districts])
    rail_allowed_tensor = torch.tensor([district_info[d]['rail_allowed'] for d in districts])
    subway_allowed_tensor = torch.tensor([district_info[d]['subway_allowed'] for d in districts])

    return {
        'districts': districts,
        'vol_base_case': vol_base_case_tensor,
        'capacity_base': capacity_base_tensor,
        'capacity_new': capacity_new_tensor,
        'capacity_reduction': capacity_reduction_tensor,
        'length': length_tensor,
        'highway': highway_tensor,
        'freespeed_base': freespeed_base_tensor,
        'freespeed': freespeed_tensor,
        'cars_allowed': cars_allowed_tensor,
        'bus_allowed': bus_allowed_tensor,
        'pt_allowed': pt_allowed_tensor,
        'train_allowed': train_allowed_tensor,
        'rail_allowed': rail_allowed_tensor,
        'subway_allowed': subway_allowed_tensor,
        'edge_count': edge_count_tensor,
    }

    
def compute_combined_tensor_edge_features(vol_base_case, capacity_base_case, length, freespeed_base_case, allowed_modes, capacities_new, capacity_reduction, highway, freespeed):
    edge_tensors = [
                torch.tensor(vol_base_case), 
                torch.tensor(capacity_base_case), 
                torch.tensor(capacities_new), 
                torch.tensor(capacity_reduction), 
                torch.tensor(freespeed_base_case), 
                torch.tensor(freespeed), 
                torch.tensor(highway), 
                torch.tensor(length), 
                allowed_modes[0],
                allowed_modes[1],
                allowed_modes[2],
                allowed_modes[3],
                allowed_modes[4],
                allowed_modes[5],
            ]
    stacked_edge_tensor = torch.stack(edge_tensors, dim=1)  # Shape: (31140, 14)
    return stacked_edge_tensor

def compute_combined_tensor_district_features(gdf, district_info, vol_base_case, capacity_base_case, length, freespeed_base_case, allowed_modes, capacities_new, capacity_reduction, highway, freespeed):
    edge_tensors = compute_combined_tensor_edge_features(vol_base_case, capacity_base_case, length, freespeed_base_case, allowed_modes, capacities_new, capacity_reduction, highway, freespeed)
    district_info = aggregate_district_information(links_gdf=gdf, tensors_edge_information= edge_tensors)
    district_tensors = [
                district_info['vol_base_case'],
                district_info['capacity_base'],
                district_info['capacity_new'],
                district_info['capacity_reduction'],
                district_info['freespeed_base'],
                district_info['freespeed'],
                district_info['highway'],
                district_info['length'],
                district_info['cars_allowed'],
                district_info['bus_allowed'],
                district_info['pt_allowed'],
                district_info['train_allowed'],
                district_info['rail_allowed'],
                district_info['subway_allowed'],
        ]
    stacked_tensor2 = torch.stack(district_tensors, dim=1)  # Shape: (20, 14)
    combined_tensor = torch.cat((edge_tensors, stacked_tensor2), dim=0)  # Shape: (31,160, 14)
    return district_info, combined_tensor

def compute_node_attributes(district_info, len_edges):
    num_edge_nodes = len_edges
    num_district_nodes = len(district_info['districts'])
    node_type_feature = torch.zeros((num_edge_nodes + num_district_nodes, 1), dtype=torch.long)
    node_type_feature[num_edge_nodes:, :] = 1
    return node_type_feature

def compute_edge_attributes(district_info, linegraph_data, len_edges, gdf_input):
    district_node_offset = len_edges
    edge_to_district_edges = []
    for idx, row in gdf_input.iterrows():
        for district in row['district']:
            district_idx = district_info['districts'].index(district) + district_node_offset
            edge_to_district_edges.append([idx, district_idx])
            edge_to_district_edges.append([district_idx, idx])  # Add reverse edge for undirected graph  # TODO is one way enough ? 
    edge_to_district_index = torch.tensor(edge_to_district_edges, dtype=torch.long).t()
    linegraph_data.edge_index = torch.cat([linegraph_data.edge_index, edge_to_district_index], dim=1)
    edge_to_district_index = torch.tensor(edge_to_district_edges, dtype=torch.long).t()
    edge_to_district_attr = torch.ones((edge_to_district_index.shape[1], 1), dtype=torch.long)
    return edge_to_district_index, edge_to_district_attr

def compute_target_tensor_only_edge_features(vol_base_case, gdf):
    edge_car_volume_difference = gdf['vol_car'].values - vol_base_case
    return torch.tensor(edge_car_volume_difference, dtype=torch.float).unsqueeze(1)
    
def compute_target_tensor_with_district_features(compute_district_nodes, vol_base_case, gdf, district_info):
    edge_car_volume_difference = gdf['vol_car'].values - vol_base_case
    if compute_district_nodes:
        district_car_volume_difference = []
        for district in district_info['districts']:
            district_edges = gdf[gdf['district'].apply(lambda x: district in x)]
            district_volume_diff = district_edges['vol_car'].sum() - district_edges['vol_car_base_case'].sum()
            district_car_volume_difference.append(district_volume_diff)
        district_car_volume_difference = torch.tensor(district_car_volume_difference, dtype=torch.float).unsqueeze(1)
        target_values = torch.cat([torch.tensor(edge_car_volume_difference, dtype=torch.float).unsqueeze(1), district_car_volume_difference], dim=0)
        return target_values
    else:
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

def get_link_geometries(links_gdf_input, districts_input):
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

    district_centroids = districts_input['district_centroid'].apply(lambda point: [point.x, point.y])
    district_centroids_tensor = torch.tensor(district_centroids.tolist(), dtype=torch.float32)
    if district_centroids_tensor.size(0) != 20 or district_centroids_tensor.size(1) != 2:
        raise ValueError("The resulting tensor does not have the expected size of (20, 2)")
    district_centroids_tensor_padded = district_centroids_tensor.unsqueeze(1).expand(-1, 3, -1)
    return edge_start_point_tensor,stacked_edge_geometries_tensor, district_centroids_tensor_padded, edges_base, nodes


# def process_result_dic(result_dic):
#     datalist = []
#     linegraph_transformation = LineGraph()
#     base_network_no_policies = result_dic.get("base_network_no_policies")
#     vol_base_case = base_network_no_policies['vol_car'].values
#     capacity_base_case = base_network_no_policies['capacity'].values

#     for key, df in result_dic.items():
#         if isinstance(df, pd.DataFrame):
#             gdf = gpd.GeoDataFrame(df, geometry='geometry')
#             gdf.crs = "EPSG:2154"  # Assuming the original CRS is EPSG:2154
#             gdf.to_crs("EPSG:4326", inplace=True)
            
#             # Create dictionaries for nodes and edges
#             nodes = pd.concat([gdf['from_node'], gdf['to_node']]).unique()
#             node_to_idx = {node: idx for idx, node in enumerate(nodes)}
            
#             gdf['from_idx'] = gdf['from_node'].map(node_to_idx)
#             gdf['to_idx'] = gdf['to_node'].map(node_to_idx)
            
#             edges = gdf[['from_idx', 'to_idx']].values
#             # edge_car_volumes = gdf['vol_car'].values
#             edge_car_volume_difference = gdf['vol_car'].values - vol_base_case
#             # if vol_base_case == 0:
#             #     if edge_car_volume_difference == 0:
#             #         edge_car_volume_difference_in_percent = 0
#             #     else:
#             #         edge_car_volume_difference_in_percent = 1
#             #         print("now it is not zero where before it was zero")
                    
#             # edge_car_volume_difference_in_percent = edge_car_volume_difference / vol_base_case 
            
#             # Initialize the percentage difference array
#             # edge_car_volume_difference_in_percent = np.zeros_like(edge_car_volume_difference, dtype=float)

#             # # Handle cases where vol_base_case is zero
#             # for i in range(len(vol_base_case)):
#             #     if vol_base_case[i] == 0:
#             #         if edge_car_volume_difference[i] == 0:
#             #             edge_car_volume_difference_in_percent[i] = 0
#             #         else:
#             #             edge_car_volume_difference_in_percent[i] = 100  # or any large number to indicate infinity
#             #             print(f"Edge {i}: now it is not zero where before it was zero")
#             #     else:
#             #         edge_car_volume_difference_in_percent[i] = edge_car_volume_difference[i] / vol_base_case[i] * 100

#             # # Add these calculations as new columns to the GeoDataFrame
#             # gdf['edge_car_volume_difference'] = edge_car_volume_difference
#             # gdf['edge_car_volume_difference_in_percent'] = edge_car_volume_difference_in_percent


#             capacities = gdf['capacity'].values
#             capacity_reduction = gdf['capacity'].values - capacity_base_case
#             # freespeeds = gdf['freespeed'].values  
#             # lengths = gdf['length'].values  
#             # modes = gdf['modes'].values
#             # modes_encoded = np.vectorize(encode_modes)(modes)
#             highway = gdf['highway'].apply(lambda x: highway_mapping.get(x, -1)).values
            
#             edge_positions = np.array([((geom.coords[0][0] + geom.coords[-1][0]) / 2, 
#                                         (geom.coords[0][1] + geom.coords[-1][1]) / 2) 
#                                        for geom in gdf.geometry])

#             # Convert lists to tensors
#             edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#             edge_positions_tensor = torch.tensor(edge_positions, dtype=torch.float)
#             x = torch.zeros((len(nodes), 1), dtype=torch.float)
            
#             # Create Data object
#             target_values = torch.tensor(edge_car_volume_difference, dtype=torch.float).unsqueeze(1)
#             data = Data(edge_index=edge_index, x=x, pos=edge_positions_tensor)
            
#             # Transform to line graph
#             linegraph_data = linegraph_transformation(data)
            
#             # Prepare the x for line graph: index and capacity
#             # linegraph_x = torch.tensor(np.column_stack((capacities, vol_base_case, highway)), dtype=torch.float)
#             linegraph_x = torch.tensor(np.column_stack((capacities, capacity_reduction, vol_base_case, highway)), dtype=torch.float)

#             linegraph_data.x = linegraph_x
            
#             # Target tensor for car volumes
#             linegraph_data.y = target_values
            
#             if linegraph_data.validate(raise_on_error=True):
#                 datalist.append(linegraph_data)
#             else:
#                 print("Invalid line graph data")
                
#     # Convert dataset to a list of dictionaries
#     data_dict_list = [{'x': lg_data.x, 'edge_index': lg_data.edge_index, 'pos': lg_data.pos, 'y': lg_data.y} for lg_data in datalist]
    
#     return data_dict_list
