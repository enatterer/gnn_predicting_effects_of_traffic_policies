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
import gnn_architectures as garch

# Assuming 'highway_mapping' and 'encode_modes' are defined as in your context
highway_mapping = {
    'residential': 0, 'tertiary': 1, 'living_street': 2, 'secondary': 3, 
    'primary': 4, 'trunk_link': 5, 'primary_link': 6, 'motorway': 7, 
    'service': 8, 'unclassified': 9, 'secondary_link': 10, 
    'pedestrian': 11, 'trunk': 12, 'motorway_link': 13, 
    'construction': 14, 'tertiary_link': 15, np.nan: -1
}

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
    val_loss = 0
    num_batches = 0
    with torch.inference_mode():
        input_node_features, targets = data.x.to(device), data.y.to(device)
        predicted = model(data.to(device))
        val_loss += loss_func(predicted, targets).item()
        num_batches += 1
    r_squared = compute_r2_torch(preds=predicted, targets=targets)
    return val_loss / num_batches if num_batches > 0 else 0, r_squared, targets, predicted

def compute_r2_torch(preds, targets):
    """Compute R^2 score using PyTorch."""
    mean_targets = torch.mean(targets)
    ss_tot = torch.sum((targets - mean_targets) ** 2)
    ss_res = torch.sum((targets - preds) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def data_to_geodataframe(data, original_gdf, predicted_values):
    # Extract the edge index and node features
    node_features = data.x.cpu().numpy()
    target_values = data.y.cpu().numpy()
    predicted_values = predicted_values.cpu().numpy() if isinstance(predicted_values, torch.Tensor) else predicted_values

    # Create edge data
    edge_data = {
        'from_node': original_gdf["from_node"].values,
        'to_node': original_gdf["to_node"].values,
        'vol_base_case': node_features[:, 0],  # Assuming capacity is the first feature, and so on
        'capacity_base_case': node_features[:, 1],  
        'capacity_reduction': node_features[:, 2],  
        'highway': node_features[:, 3],  
        'vol_car_change_actual': target_values.squeeze(),  # Assuming target values are car volumes
        'vol_car_change_predicted': predicted_values.squeeze()
    }
    # Convert to DataFrame
    edge_df = pd.DataFrame(edge_data)
    # Create LineString geometry
    edge_df['geometry'] = original_gdf["geometry"].values
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(edge_df, geometry='geometry')
    return gdf

def map_to_original_values(input_gdf: gpd.GeoDataFrame, scaler_x, scaler_y=None):
    gdf = input_gdf.copy()
    if scaler_y is None:
         # y was not normalized, so we don't need to convert i back
        gdf['og_vol_car_change_actual'] = gdf['vol_car_change_actual']
        gdf['og_vol_car_change_predicted'] = gdf['vol_car_change_predicted']
    else:
       # y was normalized, now we need to compute it back
        original_values_vol_car_change_actual = scaler_y.inverse_transform(gdf['vol_car_change_actual'].values.reshape(-1, 1))
        original_values_vol_car_change_predicted = scaler_y.inverse_transform(gdf['vol_car_change_predicted'].values.reshape(-1, 1))
        gdf['og_vol_car_change_actual'] = original_values_vol_car_change_actual
        gdf['og_vol_car_change_predicted'] = original_values_vol_car_change_predicted
    
    original_values_vol_base_case = scaler_x[0].inverse_transform(gdf['vol_base_case'].values.reshape(-1, 1))
    original_values_capacity_base_case = scaler_x[1].inverse_transform(gdf['capacity_base_case'].values.reshape(-1, 1))
    original_values_capacity_new = scaler_x[2].inverse_transform(gdf['capacity_reduction'].values.reshape(-1, 1))
    original_values_highway = scaler_x[3].inverse_transform(gdf['highway'].values.reshape(-1, 1))
        
    gdf['og_vol_base_case'] = original_values_vol_base_case
    gdf['og_capacity_base_case'] = original_values_capacity_base_case
    gdf['og_capacity_reduction'] = original_values_capacity_new
    gdf['og_highway'] = original_values_highway
    return gdf

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

def plot_combined_output(gdf_input: gpd.GeoDataFrame, column_to_plot: str, font: str = 'Times New Roman', 
                         save_it: bool = False, number_to_plot: int = 0,
                         zone_to_plot:str= "this_zone",
                         is_predicted: bool = False, alpha:int=100, 
                         use_fixed_norm:bool=True, 
                         fixed_norm_max: int= 10, normalized_y:bool=False, known_districts:bool=False, buffer: float = 0.0005, districts_of_interest: list =[1, 2, 3, 4]):
    # call with known_districts if call with 0 or 1

    gdf = gdf_input.copy()
    gdf, x_min, y_min, x_max, y_max = filter_for_geographic_section(gdf)
    gdf = gdf[gdf["og_highway"].isin([1, 2, 3])]

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))    
    norm = get_norm(column_to_plot=column_to_plot, use_fixed_norm=use_fixed_norm, fixed_norm_max=fixed_norm_max, gdf=gdf)
    relevant_area_to_plot = get_relevant_area_to_plot(alpha, known_districts, buffer, districts_of_interest, gdf, ax, column_to_plot, norm)
    relevant_area_to_plot.plot(ax=ax, edgecolor='black', linewidth=2, facecolor='None', zorder=2)

    cbar = plotting(font, x_min, y_min, x_max, y_max, fig, ax, norm)
    
    cbar.set_label('Car volume: Difference to base case (%)', fontname=font, fontsize=15)
    if save_it:
        p = "predicted" if is_predicted else "actual"
        n = "normalized_y" if normalized_y else "not_normalized_y"
        identifier = "n_" + str(number_to_plot) if number_to_plot is not None else zone_to_plot
        plt.savefig("results/" + identifier + "_" + n + "_" + p, bbox_inches='tight')
    plt.show()

def get_norm(column_to_plot, use_fixed_norm, fixed_norm_max, gdf):
    if use_fixed_norm:
        norm = TwoSlopeNorm(vmin=-fixed_norm_max, vcenter=0, vmax=fixed_norm_max)
    else:
        norm = TwoSlopeNorm(vmin=gdf[column_to_plot].min(), vcenter=gdf[column_to_plot].median(), vmax=gdf[column_to_plot].max())
    return norm
    
def plot_difference_output(gdf_input: gpd.GeoDataFrame, column1: str, column2: str, diff_column: str = 'difference', font: str = 'Times New Roman', save_it: bool = False, number_to_plot: int = 0,
                           zone_to_plot:str= "this_zone", alpha:int=100, 
                         use_fixed_norm:bool=True, 
                         fixed_norm_max: int= 10, normalized_y: bool=False, known_districts:bool=False, buffer: float = 0.0005, districts_of_interest: list =[1, 2, 3, 4]):
    gdf = gdf_input.copy()
    gdf[diff_column] = gdf[column1] - gdf[column2]
    column_to_plot = diff_column

    gdf, x_min, y_min, x_max, y_max = filter_for_geographic_section(gdf)
    gdf = gdf[gdf["og_highway"].isin([1, 2, 3])]

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))    
    norm = get_norm(column_to_plot=column_to_plot, use_fixed_norm=use_fixed_norm, fixed_norm_max=fixed_norm_max, gdf=gdf)
    relevant_area_to_plot = get_relevant_area_to_plot(alpha, known_districts, buffer, districts_of_interest, gdf, ax, column_to_plot, norm)
    relevant_area_to_plot.plot(ax=ax, edgecolor='black', linewidth=2, facecolor='None', zorder=2)

    cbar = plotting(font, x_min, y_min, x_max, y_max, fig, ax, norm)
    cbar.set_label('Difference between predicted and actual (%)', fontname=font, fontsize=15)
    if save_it:
        n = "normalized_y" if normalized_y else "not_normalized_y"
        identifier = "n_" + str(number_to_plot) if number_to_plot is not None else zone_to_plot
        plt.savefig("results/" + identifier  + "_" +  n + "_difference", bbox_inches='tight')
    plt.show()

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
    custom_lines = [Line2D([0], [0], color='grey', lw=4, label='Higher order street network'),# Add more lines for other labels as needed
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

def get_relevant_area_to_plot(alpha, known_districts, buffer, districts_of_interest, gdf, ax, column_to_plot, norm):
    if known_districts:
        target_districts = districts[districts['c_ar'].isin(districts_of_interest)]
        gdf['intersects_target_districts'] = gdf.apply(lambda row: target_districts.intersects(row.geometry).any(), axis=1)
        gdf[gdf['intersects_target_districts']].plot(column=column_to_plot, cmap='coolwarm', linewidth=5, ax=ax, legend=False,
                norm=norm, label = "Higher order roads", zorder=2)
        gdf[~gdf['intersects_target_districts']].plot(column=column_to_plot, cmap='coolwarm', linewidth=3, ax=ax, legend=False,
                norm=norm, zorder=1)

        buffered_target_districts = target_districts.copy()
        buffered_target_districts['geometry'] = buffered_target_districts.buffer(buffer)
        if buffered_target_districts.crs != gdf.crs:
            buffered_target_districts.to_crs(gdf.crs, inplace=True)
        outer_boundary = unary_union(buffered_target_districts.geometry).boundary
        relevant_area_to_plot = gpd.GeoSeries(outer_boundary, crs=gdf.crs)
        
    else:
        gdf['og_capacity_reduction_rounded'] = gdf['og_capacity_reduction'].round(decimals=3)
        tolerance = 1e-3
        edges_with_capacity_reduction = gdf[np.abs(gdf['og_capacity_reduction_rounded']) > tolerance]
        coords = [(x, y) for geom in edges_with_capacity_reduction.geometry for x, y in zip(geom.xy[0], geom.xy[1])]
        alpha_shape = alphashape.alphashape(coords, alpha)
        relevant_area_to_plot = gpd.GeoSeries([alpha_shape], crs=gdf.crs)
    return relevant_area_to_plot
    


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
    gdf = gdf[gdf["og_highway"].isin([1, 2, 3])]
    
    # Round og_capacity_reduction and filter
    gdf['og_capacity_reduction_rounded'] = gdf['og_capacity_reduction'].round(decimals=3)
    tolerance = 1e-3
    edges_with_capacity_reduction = gdf[np.abs(gdf['og_capacity_reduction_rounded']) > tolerance]
    # edges_without_capacity_reduction = gdf[np.abs(gdf['og_capacity_reduction_rounded']) <= tolerance]

    norm = TwoSlopeNorm(vmin=gdf["og_capacity_reduction"].min(), vcenter=gdf["og_capacity_reduction"].median(), vmax=gdf["og_capacity_reduction"].max())
    
    # edges_without_capacity_reduction.plot(
    #     ax=ax, column=column_to_plot, cmap='coolwarm', linewidth=3, legend=False, norm=norm, zorder=1, label = "Capacity reduction")
    edges_with_capacity_reduction.plot(
        ax=ax, column='og_capacity_reduction', cmap='coolwarm', linewidth=5, legend=False, norm=norm, zorder=2, label = "Edges with capacity reduction")
        
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
    
# def plot_simulation_output(gdf_input:gpd.GeoDataFrame, column_to_plot: str, font:str ='DejaVu Serif', save_it: bool=False, number_to_plot : int=0, is_predicted:bool= False):    
#     gdf = gdf_input.copy()
#     x_min = gdf.total_bounds[0] + 0.05
#     y_min = gdf.total_bounds[1] + 0.05
#     x_max = gdf.total_bounds[2]
#     y_max = gdf.total_bounds[3]
#     bbox = box(x_min, y_min, x_max, y_max)
    
#     # Filter the network to include only the data within the bounding box
#     gdf = gdf[gdf.intersects(bbox)]
    
#     # Set up the plot
#     fig, ax = plt.subplots(1, 1, figsize=(15, 15))
#     gdf = gdf[gdf["og_highway"].isin([1, 2, 3])]
    
#     # Round og_capacity_reduction and filter
#     gdf['og_capacity_reduction_rounded'] = gdf['og_capacity_reduction'].round(decimals=3)
#     tolerance = 1e-3
#     edges_with_capacity_reduction = gdf[np.abs(gdf['og_capacity_reduction_rounded']) > tolerance]
#     edges_without_capacity_reduction = gdf[np.abs(gdf['og_capacity_reduction_rounded']) <= tolerance]

#     norm = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)
    
#     edges_without_capacity_reduction.plot(
#         ax=ax, column=column_to_plot, cmap='coolwarm', linewidth=3, legend=False, norm=norm, zorder=1, label = "Edges without capacity reduction")
#     edges_with_capacity_reduction.plot(
#         ax=ax, column=column_to_plot, cmap='coolwarm', linewidth=5, legend=False, norm=norm, zorder=2, label = "Edges with capacity reduction")
        
#     plt.xlim(x_min, x_max)
#     plt.ylim(y_min, y_max)
    
#     # Customize the plot with Times New Roman font and size 15
#     plt.xlabel("Longitude", fontname=font, fontsize=15)
#     plt.ylabel("Latitude", fontname=font, fontsize=15)

#     # Customize tick labels
#     ax.tick_params(axis='both', which='major', labelsize=10)
#     for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#         label.set_fontname(font)
#         label.set_fontsize(15)
#     ax.legend(prop={'family': font, 'size': 15})
#     ax.set_position([0.1, 0.1, 0.75, 0.75])
#     cax = fig.add_axes([0.87, 0.22, 0.03, 0.5])  # Manually position the color bar

#     # Create the color bar
#     sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
#     sm._A = []
#     cbar = plt.colorbar(sm, cax=cax)
    
#     # Set color bar font properties
#     cbar.ax.tick_params(labelsize=15)
#     for t in cbar.ax.get_yticklabels():
#         t.set_fontname(font)
#     cbar.ax.yaxis.label.set_fontname(font)
#     cbar.ax.yaxis.label.set_size(15)
#     cbar.set_label('Car volume: Difference to base case (%)', fontname=font, fontsize=15)
#     if save_it:
#         p = "predicted" if is_predicted else "actual"
#         plt.savefig("results/gdf_" + str(number_to_plot) + "_" + p, bbox_inches='tight')
#     plt.show()