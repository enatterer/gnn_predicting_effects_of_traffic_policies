import math
import numpy as np
import wandb
import pickle
import os
import shapely.wkt as wkt
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from torch_geometric.transforms import LineGraph

import gzip
import xml.etree.ElementTree as ET

import torch
import torch_geometric
from torch_geometric.data import Data

import processing_io as pio
import sys
import os
import joblib
import json

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from shapely.geometry import Point, LineString, box
from matplotlib.colors import TwoSlopeNorm

from shapely.ops import unary_union
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch_geometric.data import Data, Batch
import torch
from torch_geometric.data import Data
import alphashape
from matplotlib.lines import Line2D

from shapely.geometry import Polygon

districts = gpd.read_file("../../data/visualisation/districts_paris.geojson")

# Add the 'scripts' directory to the Python path
scripts_path = os.path.abspath(os.path.join('..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

import gnn_io as gio
import gnn_architecture as garch

from enum import IntEnum

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

def data_to_geodataframe_with_og_values(data, original_gdf, predicted_values, inversed_x):
    target_values = data.y.cpu().numpy()
    predicted_values = predicted_values.cpu().numpy() if isinstance(predicted_values, torch.Tensor) else predicted_values
    
    edge_data = {
        'from_node': original_gdf["from_node"].values,
        'to_node': original_gdf["to_node"].values,
        'vol_base_case': inversed_x[:, EdgeFeatures.VOL_BASE_CASE],  
        'capacity_base_case': inversed_x[:, EdgeFeatures.CAPACITY_BASE_CASE],  
        'capacities_new': inversed_x[:, EdgeFeatures.CAPACITIES_NEW],  
        'capacity_reduction': inversed_x[:, EdgeFeatures.CAPACITY_REDUCTION],  
        'freespeed': inversed_x[:, EdgeFeatures.FREESPEED],  
        'highway': original_gdf['highway'].values,
        'length': inversed_x[:, EdgeFeatures.LENGTH],        
        'allowed_mode_car': inversed_x[:, EdgeFeatures.ALLOWED_MODE_CAR],
        'allowed_mode_bus': inversed_x[:, EdgeFeatures.ALLOWED_MODE_BUS],
        'allowed_mode_pt': inversed_x[:, EdgeFeatures.ALLOWED_MODE_PT],
        'allowed_mode_train': inversed_x[:, EdgeFeatures.ALLOWED_MODE_TRAIN],
        'allowed_mode_rail': inversed_x[:, EdgeFeatures.ALLOWED_MODE_RAIL],
        'allowed_mode_subway': inversed_x[:, EdgeFeatures.ALLOWED_MODE_SUBWAY],
        'vol_car_change_actual': target_values.squeeze(),  
        'vol_car_change_predicted': predicted_values.squeeze(),
    }
    edge_df = pd.DataFrame(edge_data)
    edge_df['geometry'] = original_gdf["geometry"].values
    gdf = gpd.GeoDataFrame(edge_df, geometry='geometry')
    return gdf

def compute_r2_torch_with_mean_targets(mean_targets, preds, targets):
    ss_tot = torch.sum((targets - mean_targets) ** 2)
    ss_res = torch.sum((targets - preds) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def validate_model_on_test_set(model, dataset, loss_func, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.inference_mode():
        if isinstance(dataset, list):
            for data in dataset:
                input_node_features, targets = data.x.to(device), data.y.to(device)
                predicted = model(data.to(device))
                loss = loss_func(predicted, targets).item()
                total_loss += loss
                all_preds.append(predicted)
                all_targets.append(targets)
        else:
            input_node_features, targets = dataset.x.to(device), dataset.y.to(device)
            predicted = model(dataset.to(device))
            loss = loss_func(predicted, targets).item()
            total_loss += loss
            all_preds.append(predicted)
            all_targets.append(targets)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    mean_targets = torch.mean(all_targets)
    r_squared = compute_r2_torch_with_mean_targets(mean_targets=mean_targets, preds=all_preds, targets=all_targets)
    baseline_loss = loss_func(all_targets, torch.full_like(all_preds, mean_targets))
    
    avg_loss = total_loss / len(dataset)
    
    return avg_loss, r_squared, all_targets, all_preds, baseline_loss

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

    return edge_start_point_tensor,stacked_edge_geometries_tensor, edges_base, nodes

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

def create_test_object(links_base_case, test_data, stacked_edge_geometries_tensor):
    linegraph_transformation = LineGraph()
    vol_base_case = links_base_case['vol_car'].values
    capacity_base_case = np.where(links_base_case['modes'].str.contains('car'), links_base_case['capacity'], 0)

    length = links_base_case['length'].values
    freespeed = links_base_case['freespeed'].values
    allowed_modes = encode_modes(links_base_case)
    edges_base = links_base_case[['from_idx', 'to_idx']].values
    nodes = pd.concat([links_base_case['from_node'], links_base_case['to_node']]).unique()

    edge_index = torch.tensor(edges_base, dtype=torch.long).t().contiguous()
    x = torch.zeros((len(nodes), 1), dtype=torch.float)
    data = Data(edge_index=edge_index, x=x)
    
    capacities_new = np.where(test_data['modes'].str.contains('car'), test_data['capacity'], 0)
    capacity_reduction = capacities_new - capacity_base_case
    highway = test_data['highway'].apply(lambda x: highway_mapping.get(x, -1)).values

    capacities_new = test_data['capacity'].values
    capacity_reduction= capacities_new - capacity_base_case
    
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
        EdgeFeatures.ALLOWED_MODE_PT:  allowed_modes[2],
        EdgeFeatures.ALLOWED_MODE_TRAIN: allowed_modes[3],
        EdgeFeatures.ALLOWED_MODE_RAIL: allowed_modes[4],
        EdgeFeatures.ALLOWED_MODE_SUBWAY: allowed_modes[5],
    }
    
    edge_tensor = [edge_feature_dict[feature] for feature in EdgeFeatures]
    edge_tensor = torch.stack(edge_tensor, dim=1)  # Shape: (31140, 14)
    
    linegraph_data = linegraph_transformation(data)
    linegraph_data.x = edge_tensor
    linegraph_data.pos = stacked_edge_geometries_tensor
    edge_car_volume_difference = test_data['vol_car'].values - vol_base_case
    linegraph_data.y = torch.tensor(edge_car_volume_difference, dtype=torch.float).unsqueeze(1)
    
    if linegraph_data.validate(raise_on_error=True):
        return linegraph_data
    else:
        print("Invalid line graph data")
        
def normalize_tensor(tensor, scaler):
    """
    Normalizes a given tensor using the provided scaler.

    Parameters:
    tensor (torch.Tensor): The tensor to normalize.
    scaler (sklearn.preprocessing.StandardScaler or similar): The scaler to use for normalization.

    Returns:
    torch.Tensor: The normalized tensor.
    """
    # Convert the tensor to a numpy array, apply the scaler, and convert back to a tensor
    tensor_np = tensor.numpy()
    normalized_np = scaler.transform(tensor_np)
    normalized_tensor = torch.tensor(normalized_np, dtype=tensor.dtype)
    return normalized_tensor

def normalize_pos_features(tensor, scaler):
    """
    Normalizes the position features of a given tensor using the provided scaler.

    Parameters:
    tensor (torch.Tensor): The tensor to normalize, expected shape (31140, 3, 2).
    scaler (sklearn.preprocessing.StandardScaler or similar): The scaler to use for normalization.

    Returns:
    torch.Tensor: The normalized tensor.
    """
    # Reshape the tensor to (31140, 6) for normalization
    tensor_reshaped = tensor.view(-1, 6).numpy()
    normalized_np = scaler.transform(tensor_reshaped)
    # Reshape back to (31140, 3, 2) after normalization
    normalized_tensor = torch.tensor(normalized_np, dtype=tensor.dtype).view(31140, 3, 2)
    return normalized_tensor


def validate_trained_model(model, valid_dl, loss_func, device):
    model.eval()
    val_loss = 0
    num_batches = 0
    
    actual_vals = []
    predictions = []
    
    with torch.inference_mode():
        for idx, data in enumerate(valid_dl):
            input_node_features, targets = data.x.to(device), data.y.to(device)
            predicted = model(data.to(device))
            actual_vals.append(targets)
            predictions.append(predicted)
            val_loss += loss_func(predicted, targets).item()
            num_batches += 1
            
    actual_vals_cat = torch.cat(actual_vals)
    predictions_cat = torch.cat(predictions)
    r_squared = compute_r2_torch(preds=predictions_cat, targets=actual_vals_cat)
    return val_loss / num_batches if num_batches > 0 else 0, r_squared, actual_vals, predictions

def validate_one_model(model, data, loss_func, device):
    model.eval()
    with torch.inference_mode():
        input_node_features, targets = data.x.to(device), data.y.to(device)
        predicted = model(data.to(device))
        val_loss = loss_func(predicted, targets).item()
    r_squared = compute_r2_torch(preds=predicted, targets=targets)
    return val_loss, r_squared, targets, predicted

def compute_r2_torch(preds, targets):
    """Compute R^2 score using PyTorch."""
    mean_targets = torch.mean(targets)
    ss_tot = torch.sum((targets - mean_targets) ** 2)
    ss_res = torch.sum((targets - preds) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

# def map_to_original_values(input_gdf: gpd.GeoDataFrame, scaler_x, scaler_y=None):
#     gdf = input_gdf.copy()
#     if scaler_y is None:
#          # y was not normalized, so we don't need to convert i back
#         gdf['og_vol_car_change_actual'] = gdf['vol_car_change_actual']
#         gdf['og_vol_car_change_predicted'] = gdf['vol_car_change_predicted']
#     else:
#        # y was normalized, now we need to compute it back
#         original_values_vol_car_change_actual = scaler_y.inverse_transform(gdf['vol_car_change_actual'].values.reshape(-1, 1))
#         original_values_vol_car_change_predicted = scaler_y.inverse_transform(gdf['vol_car_change_predicted'].values.reshape(-1, 1))
#         gdf['og_vol_car_change_actual'] = original_values_vol_car_change_actual
#         gdf['og_vol_car_change_predicted'] = original_values_vol_car_change_predicted
    
#     original_values_vol_base_case = scaler_x[0].inverse_transform(gdf['vol_base_case'].values.reshape(-1, 1))
#     original_values_capacity_base_case = scaler_x[1].inverse_transform(gdf['capacity_base_case'].values.reshape(-1, 1))
#     original_values_capacity_new = scaler_x[2].inverse_transform(gdf['capacity_reduction'].values.reshape(-1, 1))
#     original_values_highway = scaler_x[3].inverse_transform(gdf['highway'].values.reshape(-1, 1))
        
#     gdf['og_vol_base_case'] = original_values_vol_base_case
#     gdf['og_capacity_base_case'] = original_values_capacity_base_case
#     gdf['og_capacity_reduction'] = original_values_capacity_new
#     gdf['og_highway'] = original_values_highway
#     return gdf

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

def plot_districts_of_capacity_reduction(gdf_input:gpd.GeoDataFrame, font:str ='DejaVu Serif', save_it: bool=False, number_to_plot : int=0):    
    gdf = gdf_input.copy()
    x_min = gdf.total_bounds[0] + 0.05
    y_min = gdf.total_bounds[1] + 0.05
    x_max = gdf.total_bounds[2]
    y_max = gdf.total_bounds[3]
    bbox = box(x_min, y_min, x_max, y_max)
    
    # Filter the network to include only the data within the bounding box
    gdf = gdf[gdf.intersects(bbox)]
    
    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    gdf = gdf[gdf["highway"].isin([1, 2, 3])]
    
    # Round og_capacity_reduction and filter
    gdf['capacity_reduction_rounded'] = gdf['capacity_reduction'].round(decimals=3)
    tolerance = 1e-3
    edges_with_capacity_reduction = gdf[np.abs(gdf['capacity_reduction_rounded']) > tolerance]
    # edges_without_capacity_reduction = gdf[np.abs(gdf['og_capacity_reduction_rounded']) <= tolerance]

    norm = TwoSlopeNorm(vmin=gdf["capacity_reduction"].min(), vcenter=gdf["capacity_reduction"].median(), vmax=gdf["capacity_reduction"].max())
    
    # edges_without_capacity_reduction.plot(
    #     ax=ax, column=column_to_plot, cmap='coolwarm', linewidth=3, legend=False, norm=norm, zorder=1, label = "Capacity reduction")
    edges_with_capacity_reduction.plot(
        ax=ax, column='capacity_reduction', cmap='coolwarm', linewidth=5, legend=False, norm=norm, zorder=2, label = "Edges with capacity reduction")
        
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Customize the plot with Times New Roman font and size 15
    plt.xlabel("Longitude", fontname=font, fontsize=15)
    plt.ylabel("Latitude", fontname=font, fontsize=15)

    # Customize tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname(font)
        label.set_fontsize(15)
    ax.legend(prop={'family': font, 'size': 15})
    ax.set_position([0.1, 0.1, 0.75, 0.75])
    cax = fig.add_axes([0.87, 0.22, 0.03, 0.5])  # Manually position the color bar

    # Create the color bar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, cax=cax)
    
    # Set color bar font properties
    cbar.ax.tick_params(labelsize=15)
    for t in cbar.ax.get_yticklabels():
        t.set_fontname(font)
    cbar.ax.yaxis.label.set_fontname(font)
    cbar.ax.yaxis.label.set_size(15)
    cbar.set_label('Capacity reduction', fontname=font, fontsize=15)
    if save_it:
        plt.savefig("results/gdf_capacity_reduction_" + str(number_to_plot), bbox_inches='tight')
    plt.show()

def replace_invalid_values(tensor):
    tensor[tensor != tensor] = 0  # replace NaNs with 0
    tensor[tensor == float('inf')] = 0  # replace inf with 0
    tensor[tensor == float('-inf')] = 0  # replace -inf with 0
    return tensor

def plot_combined_output(gdf_input: gpd.GeoDataFrame, column_to_plot: str, font: str = 'Times New Roman', 
                         save_it: bool = False, number_to_plot: int = 0,
                         zone_to_plot:str= "this_zone",
                         is_predicted: bool = False, alpha:int=100, 
                         use_fixed_norm:bool=True, 
                         fixed_norm_max: int= 10, normalized_y:bool=False, known_districts:bool=False, buffer: float = 0.0005, districts_of_interest: list =[1, 2, 3, 4]):

    gdf = gdf_input.copy()
    gdf, x_min, y_min, x_max, y_max = filter_for_geographic_section(gdf)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))    
    if use_fixed_norm:
        norm = TwoSlopeNorm(vmin=-fixed_norm_max, vcenter=0, vmax=fixed_norm_max)
    else:
        norm = TwoSlopeNorm(vmin=gdf[column_to_plot].min(), vcenter=gdf[column_to_plot].median(), vmax=gdf[column_to_plot].max())
    
    linewidths = gdf["highway"].apply(get_linewidth)
    gdf['linewidth'] = linewidths
    large_lines = gdf[gdf['linewidth'] > 1]
    small_lines = gdf[gdf['linewidth'] == 1]
    small_lines.plot(column=column_to_plot, cmap='coolwarm', linewidth=small_lines['linewidth'], ax=ax, legend=False,
                    norm=norm, label="Street network", zorder=1)
    large_lines.plot(column=column_to_plot, cmap='coolwarm', linewidth=large_lines['linewidth'], ax=ax, legend=False,
                    norm=norm, label="Street network", zorder=2)
    
    relevant_area_to_plot = get_relevant_area_to_plot(alpha, known_districts, buffer, districts_of_interest, gdf)
    
    if isinstance(relevant_area_to_plot, set):
        for area in relevant_area_to_plot:
            if isinstance(area, Polygon):
                gdf_area = gpd.GeoDataFrame(index=[0], crs=gdf.crs, geometry=[area])
                gdf_area.plot(ax=ax, edgecolor='black', linewidth=2, facecolor='None', zorder=2)
    else:
        relevant_area_to_plot.plot(ax=ax, edgecolor='black', linewidth=2, facecolor='None', zorder=2)
        
    cbar = plotting(font, x_min, y_min, x_max, y_max, fig, ax, norm)
    cbar.set_label('Car volume: Difference to base case (%)', fontname=font, fontsize=15)
    if save_it:
        p = "predicted" if is_predicted else "actual"
        identifier = "n_" + str(number_to_plot) if number_to_plot is not None else zone_to_plot
        plt.savefig("results/" + identifier + "_" + p, bbox_inches='tight')
    plt.show()

def get_norm(column_to_plot, use_fixed_norm, fixed_norm_max, gdf):
    if use_fixed_norm:
        norm = TwoSlopeNorm(vmin=-fixed_norm_max, vcenter=0, vmax=fixed_norm_max)
    else:
        norm = TwoSlopeNorm(vmin=gdf[column_to_plot].min(), vcenter=gdf[column_to_plot].median(), vmax=gdf[column_to_plot].max())
    return norm
    
def filter_for_geographic_section(gdf):
    x_min = gdf.total_bounds[0] + 0.05
    y_min = gdf.total_bounds[1] + 0.05
    x_max = gdf.total_bounds[2]
    y_max = gdf.total_bounds[3]
    bbox = box(x_min, y_min, x_max, y_max)

    # Filter the network to include only the data within the bounding box
    gdf = gdf[gdf.intersects(bbox)]
    return gdf,x_min,y_min,x_max,y_max

def plotting(font, x_min, y_min, x_max, y_max, fig, ax, norm):
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("Longitude", fontname=font, fontsize=15)
    plt.ylabel("Latitude", fontname=font, fontsize=15)

    # Customize tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname(font)
        label.set_fontsize(15)
    
    # Create custom legend
    custom_lines = [Line2D([0], [0], color='grey', lw=4, label='Street network'),# Add more lines for other labels as needed
                    Line2D([0], [0], color='black', lw=2, label='Capacity was decreased in this section')]

    ax.legend(handles=custom_lines, prop={'family': font, 'size': 15})
    ax.set_position([0.1, 0.1, 0.75, 0.75])
    cax = fig.add_axes([0.87, 0.22, 0.03, 0.5])  # Manually position the color bar
    
    # Create the color bar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, cax=cax)

    # Set color bar font properties
    cbar.ax.tick_params(labelsize=15)
    for t in cbar.ax.get_yticklabels():
        t.set_fontname(font)
    cbar.ax.yaxis.label.set_fontname(font)
    cbar.ax.yaxis.label.set_size(15)
    return cbar

def get_relevant_area_to_plot(alpha, known_districts, buffer, districts_of_interest, gdf):
    if known_districts:
        target_districts = districts[districts['c_ar'].isin(districts_of_interest)]
        gdf['intersects_target_districts'] = gdf.apply(lambda row: target_districts.intersects(row.geometry).any(), axis=1)
        buffered_target_districts = target_districts.copy()
        buffered_target_districts['geometry'] = buffered_target_districts.buffer(buffer)
        if buffered_target_districts.crs != gdf.crs:
            buffered_target_districts.to_crs(gdf.crs, inplace=True)
        outer_boundary = unary_union(buffered_target_districts.geometry).boundary
        relevant_area_to_plot = gpd.GeoSeries(outer_boundary, crs=gdf.crs)
        return relevant_area_to_plot
    else:
        gdf['capacity_reduction_rounded'] = gdf['capacity_reduction'].round(decimals=3)
        tolerance = 1e-3
        edges_with_capacity_reduction = gdf[np.abs(gdf['capacity_reduction_rounded']) > tolerance]        
        relevant_districts = set()
        for _, edge in edges_with_capacity_reduction.iterrows():
            for _, district in districts.iterrows():
                if district.geometry.contains(edge.geometry):
                    relevant_districts.add(district.geometry)
        return relevant_districts
        # coords = [(x, y) for geom in edges_with_capacity_reduction.geometry for x, y in zip(geom.xy[0], geom.xy[1])]
        # alpha_shape = alphashape.alphashape(coords, alpha)
        # relevant_area_to_plot = gpd.GeoSeries([alpha_shape], crs=gdf.crs)

def get_linewidth(value):
        if value in [0, 1]:
            return 5
        elif value == 2:
            return 3
        elif value == 3:
            return 2
        else:
            return 1
        
def normalize_one_dataset_given_scaler(dataset_input, x_scalar_list = None, pos_scalar=None):
    dataset = normalize_x_values_given_scaler(dataset_input, x_scalar_list)
    dataset.pos = torch.tensor(pos_scalar.transform(dataset.pos.numpy()), dtype=torch.float)
    return dataset

def normalize_x_values_given_scaler(dataset, x_scaler_list):
    for i in range(4):
        scaler = x_scaler_list[i]
        data_x_dim = replace_invalid_values(dataset.x[:, i].reshape(-1, 1))
        normalized_x_dim = torch.tensor(scaler.transform(data_x_dim.numpy()), dtype=torch.float)
        dataset.x[:, i]=  normalized_x_dim.squeeze()
    return dataset

def compute_r2_torch_with_mean_targets(mean_targets, preds, targets):
    ss_tot = torch.sum((targets - mean_targets) ** 2)
    ss_res = torch.sum((targets - preds) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def validate_one_model(model, data, loss_func, device):
    model.eval()
    pred = []
    actual = []
    with torch.inference_mode():
        input_node_features, targets = data.x.to(device), data.y.to(device)
        predicted = model(data.to(device))
        # print(predicted.shape)
        pred.append(predicted)
        actual.append(targets)
        val_loss = loss_func(predicted, targets).item()
    actual_vals = torch.cat(actual)
    predicted_vals = torch.cat(pred)
    
    mean_targets = torch.mean(targets)
    r_squared = compute_r2_torch_with_mean_targets(mean_targets = mean_targets, preds=predicted_vals, targets=actual_vals)
    baseline_loss = loss_func(targets, torch.full_like(predicted_vals, mean_targets))
    return val_loss, r_squared, targets, predicted, baseline_loss

def compute_r2_torch(preds, targets):
    """Compute R^2 score using PyTorch."""
    print(targets.shape)
    mean_targets = torch.mean(targets)
    ss_tot = torch.sum((targets - mean_targets) ** 2)
    ss_res = torch.sum((targets - preds) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

# def data_to_geodataframe(data, original_gdf, predicted_values):
#     # Extract the edge index and node features
#     node_features = data.x.cpu().numpy()
#     target_values = data.y.cpu().numpy()
#     predicted_values = predicted_values.cpu().numpy() if isinstance(predicted_values, torch.Tensor) else predicted_values

#     # Create edge data
#     edge_data = {
#         'from_node': original_gdf["from_node"].values,
#         'to_node': original_gdf["to_node"].values,
#         'vol_base_case': node_features[:, 0],  # Assuming capacity is the first feature, and so on
#         'capacity_base_case': node_features[:, 1],  
#         'capacity_reduction': node_features[:, 2],  
#         'highway': node_features[:, 3],  
#         'vol_car_change_actual': target_values.squeeze(),  # Assuming target values are car volumes
#         'vol_car_change_predicted': predicted_values.squeeze()
#     }
#     # Convert to DataFrame
#     edge_df = pd.DataFrame(edge_data)
#     # Create LineString geometry
#     edge_df['geometry'] = original_gdf["geometry"].values
#     # Create GeoDataFrame
#     gdf = gpd.GeoDataFrame(edge_df, geometry='geometry')
#     return gdf


# def plot_difference_output(gdf_input: gpd.GeoDataFrame, column1: str, column2: str, diff_column: str = 'difference', font: str = 'Times New Roman', save_it: bool = False, number_to_plot: int = 0,
#                            zone_to_plot:str= "this_zone", alpha:int=100, 
#                          use_fixed_norm:bool=True, 
#                          fixed_norm_max: int= 10, normalized_y: bool=False, known_districts:bool=False, buffer: float = 0.0005, districts_of_interest: list =[1, 2, 3, 4]):
#     gdf = gdf_input.copy()
#     gdf[diff_column] = gdf[column1] - gdf[column2]
#     column_to_plot = diff_column

#     gdf, x_min, y_min, x_max, y_max = filter_for_geographic_section(gdf)

#     fig, ax = plt.subplots(1, 1, figsize=(15, 15))    
#     norm = get_norm(column_to_plot=column_to_plot, use_fixed_norm=use_fixed_norm, fixed_norm_max=fixed_norm_max, gdf=gdf)
#     relevant_area_to_plot = get_relevant_area_to_plot(alpha, known_districts, buffer, districts_of_interest, gdf, ax, column_to_plot, norm, "og_highway")
#     relevant_area_to_plot.plot(ax=ax, edgecolor='black', linewidth=2, facecolor='None', zorder=2)

#     cbar = plotting(font, x_min, y_min, x_max, y_max, fig, ax, norm)
#     cbar.set_label('Difference between predicted and actual (%)', fontname=font, fontsize=15)
#     if save_it:
#         identifier = "n_" + str(number_to_plot) if number_to_plot is not None else zone_to_plot
#         plt.savefig("results/" + identifier  + "_difference", bbox_inches='tight')
    # plt.show()