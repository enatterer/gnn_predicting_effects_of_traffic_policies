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
from enum import IntEnum

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
from matplotlib.colors import TwoSlopeNorm, Normalize

districts = gpd.read_file("../../data/visualisation/districts_paris.geojson")

# Add the 'scripts' directory to the Python path
scripts_path = os.path.abspath(os.path.join('..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

import gnn_io as gio
import gnn_architecture as garch
from data_preprocessing.process_simulations_for_gnn import EdgeFeatures
    
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

# DATAFRAME FUNCTIONS


def data_to_geodataframe_with_og_values(data, original_gdf, predicted_values, inversed_x, use_all_features=False):
    ' use_all_features is a flag for whether to use all features or not, as shown in the ablation tests'
    target_values = data.y.cpu().numpy()
    predicted_values = predicted_values.cpu().numpy() if isinstance(predicted_values, torch.Tensor) else predicted_values
    
    if use_all_features:
        edge_data = {
        'from_node': original_gdf["from_node"].values,
        'to_node': original_gdf["to_node"].values,
        'vol_base_case': inversed_x[:, EdgeFeatures.VOL_BASE_CASE],  
        'capacity_base_case': inversed_x[:, EdgeFeatures.CAPACITY_BASE_CASE],
        'capacity_reduction': inversed_x[:, EdgeFeatures.CAPACITY_REDUCTION],  
        'freespeed': inversed_x[:, EdgeFeatures.FREESPEED],  
        'highway': original_gdf['highway'].values,
        'length': inversed_x[:, EdgeFeatures.LENGTH-1], # -1 since we didn't use Highway, fix later        
        'allowed_mode_car': inversed_x[:, EdgeFeatures.ALLOWED_MODE_CAR],
        'allowed_mode_bus': inversed_x[:, EdgeFeatures.ALLOWED_MODE_BUS],
        'allowed_mode_pt': inversed_x[:, EdgeFeatures.ALLOWED_MODE_PT],
        'allowed_mode_train': inversed_x[:, EdgeFeatures.ALLOWED_MODE_TRAIN],
        'allowed_mode_rail': inversed_x[:, EdgeFeatures.ALLOWED_MODE_RAIL],
        'allowed_mode_subway': inversed_x[:, EdgeFeatures.ALLOWED_MODE_SUBWAY],
        'vol_car_change_actual': target_values.squeeze(),  
        'vol_car_change_predicted': predicted_values.squeeze(),
        }
    else:
        edge_data = {
            'from_node': original_gdf["from_node"].values,
            'to_node': original_gdf["to_node"].values,
            'vol_base_case': inversed_x[:, EdgeFeatures.VOL_BASE_CASE],  
            'capacity_base_case': inversed_x[:, EdgeFeatures.CAPACITY_BASE_CASE],  
            'capacity_reduction': inversed_x[:, EdgeFeatures.CAPACITY_REDUCTION],  
            'freespeed': inversed_x[:, EdgeFeatures.FREESPEED],  
            'highway': original_gdf['highway'].values,
            'length': inversed_x[:, EdgeFeatures.LENGTH-1], # -1 since we didn't use Highway, fix later
            'vol_car_change_actual': target_values.squeeze(),
            'vol_car_change_predicted': predicted_values.squeeze(),
            'mean_car_vol': original_gdf['vol_car'].values,
            'variance': original_gdf['variance'].values,
            'std_dev': original_gdf['std_dev'].values,
            'std_dev_multiplied': original_gdf['std_dev_multiplied'].values,
            'cv_percent': original_gdf['cv_percent'].values,
        }
    
    edge_df = pd.DataFrame(edge_data)
    edge_df['geometry'] = original_gdf["geometry"].values
    gdf = gpd.GeoDataFrame(edge_df, geometry='geometry')
    return gdf

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
    
    # # Repeated? Why is this here?
    # capacities_new = np.where(test_data['modes'].str.contains('car'), test_data['capacity'], 0)
    # capacity_reduction = capacities_new - capacity_base_case
    
    capacities_new = test_data['capacity'].values
    capacity_reduction= capacities_new - capacity_base_case

    highway = test_data['highway'].apply(lambda x: highway_mapping.get(x, -1)).values
    
    edge_feature_dict = {
        EdgeFeatures.VOL_BASE_CASE: torch.tensor(vol_base_case),
        EdgeFeatures.CAPACITY_BASE_CASE: torch.tensor(capacity_base_case),
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
        

# NORMALIZATION FUNCTIONS

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

def compute_r2_torch(preds, targets):
    """Compute R^2 score using PyTorch."""
    mean_targets = torch.mean(targets)
    ss_tot = torch.sum((targets - mean_targets) ** 2)
    ss_res = torch.sum((targets - preds) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def compute_r2_torch_with_mean_targets(mean_targets, preds, targets):
    ss_tot = torch.sum((targets - mean_targets) ** 2)
    ss_res = torch.sum((targets - preds) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


# PLOTTING FUNCTIONS

def plot_combined_output(gdf_input: gpd.GeoDataFrame, column_to_plot: str, font: str = 'Times New Roman', 
                         save_it: bool = False, number_to_plot: int = 0,
                         zone_to_plot:str= "this_zone",
                         is_predicted: bool = False, alpha:int=100, 
                         use_fixed_norm:bool=True, 
                         fixed_norm_max: int= 10, known_districts:bool=False, buffer: float = 0.0005, 
                         districts_of_interest: list =[1, 2, 3, 4],
                         plot_contour_lines:bool=False, plot_policy_roads:bool=False,
                         is_absolute:bool=False,
                         cmap:str='coolwarm',
                         result_path:str=None):

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
    small_lines.plot(column=column_to_plot, cmap=cmap, linewidth=small_lines['linewidth'], ax=ax, legend=False,
                    norm=norm, label="Street network", zorder=1)
    large_lines.plot(column=column_to_plot, cmap=cmap, linewidth=large_lines['linewidth'], ax=ax, legend=False,
                    norm=norm, label="Street network", zorder=2)
    
    if plot_policy_roads:
        tolerance = 1e-3
        gdf['capacity_reduction_rounded'] = gdf['capacity_reduction'].round(decimals=3)
        edges_with_capacity_reduction = gdf[np.abs(gdf['capacity_reduction_rounded']) > tolerance]
        edges_with_capacity_reduction.plot(color='black', linewidth=large_lines['linewidth'], ax=ax, legend=False,
                                        norm=norm, label="Capacity was decreased on these roads", zorder=3)

    relevant_area_to_plot = get_relevant_area_to_plot(alpha, known_districts, buffer, districts_of_interest, gdf)
    if plot_contour_lines:
        if isinstance(relevant_area_to_plot, set):
            for area in relevant_area_to_plot:
                if isinstance(area, Polygon):
                    gdf_area = gpd.GeoDataFrame(index=[0], crs=gdf.crs, geometry=[area])
                    gdf_area.plot(ax=ax, edgecolor='black', linewidth=2, facecolor='None', zorder=2)
        else:
            relevant_area_to_plot.plot(ax=ax, edgecolor='black', linewidth=2, facecolor='None', zorder=2)
        
    cbar = plotting(font, x_min, y_min, x_max, y_max, fig, ax, norm, plot_contour_lines, cmap, plot_policy_roads)
    if is_absolute:
        cbar.set_label('Car volume', fontname=font, fontsize=15)
    else:
        cbar.set_label('Car volume: Difference to base case (%)', fontname=font, fontsize=15)
    if save_it:
        p = "predicted" if is_predicted else "actual"
        identifier = "n_" + str(number_to_plot) if number_to_plot is not None else zone_to_plot
        plt.savefig(result_path + identifier + "_" + p, bbox_inches='tight')
    plt.show()
    
    
def plot_prediction_difference(gdf_input: gpd.GeoDataFrame, 
                             font: str = 'Times New Roman',
                             save_it: bool = False, 
                             number_to_plot: int = 0,
                             zone_to_plot: str = "this_zone",
                             alpha: int = 100,
                             use_fixed_norm: bool = True,
                             fixed_norm_max: int = 10,
                             known_districts: bool = False,
                             buffer: float = 0.0005,
                             districts_of_interest: list = [1, 2, 3, 4],
                             plot_contour_lines: bool = True,
                             cmap: str = 'coolwarm',
                             result_path: str = None):
    """
    Plot the difference between predicted and actual values on a map.
    
    Parameters are the same as plot_combined_output, but this function specifically shows
    the prediction error (predicted - actual values).
    """
    gdf = gdf_input.copy()
    
    # Calculate the difference between predicted and actual values
    gdf['prediction_error'] = gdf['vol_car_change_predicted'] - gdf['vol_car_change_actual']
    
    # Filter geographic section
    gdf, x_min, y_min, x_max, y_max = filter_for_geographic_section(gdf)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))    
    
    if use_fixed_norm:
        norm = TwoSlopeNorm(vmin=-fixed_norm_max, vcenter=0, vmax=fixed_norm_max)
    else:
        norm = TwoSlopeNorm(vmin=gdf['prediction_error'].min(), 
                           vcenter=0, 
                           vmax=gdf['prediction_error'].max())
    
    # Plot with different line widths based on highway type
    linewidths = gdf["highway"].apply(get_linewidth)
    gdf['linewidth'] = linewidths
    large_lines = gdf[gdf['linewidth'] > 1]
    small_lines = gdf[gdf['linewidth'] == 1]
    
    small_lines.plot(column='prediction_error', cmap=cmap, linewidth=small_lines['linewidth'], 
                    ax=ax, legend=False, norm=norm, label="Street network", zorder=1)
    large_lines.plot(column='prediction_error', cmap=cmap, linewidth=large_lines['linewidth'], 
                    ax=ax, legend=False, norm=norm, label="Street network", zorder=2)
    
    relevant_area_to_plot = get_relevant_area_to_plot(alpha, known_districts, buffer, 
                                                     districts_of_interest, gdf)
    if plot_contour_lines:
        if isinstance(relevant_area_to_plot, set):
            for area in relevant_area_to_plot:
                if isinstance(area, Polygon):
                    gdf_area = gpd.GeoDataFrame(index=[0], crs=gdf.crs, geometry=[area])
                    gdf_area.plot(ax=ax, edgecolor='black', linewidth=2, facecolor='None', zorder=2)
        else:
            relevant_area_to_plot.plot(ax=ax, edgecolor='black', linewidth=2, facecolor='None', zorder=2)
    
    cbar = plotting(font, x_min, y_min, x_max, y_max, fig, ax, norm, plot_contour_lines, cmap)
    cbar.set_label('Prediction Error (Predicted - Actual)', fontname=font, fontsize=15)
    
    if save_it:
        identifier = "n_" + str(number_to_plot) if number_to_plot is not None else zone_to_plot
        plt.savefig(result_path + identifier + "_prediction_error", bbox_inches='tight')
    
    plt.show()
    
def plot_average_prediction_differences(gdf_inputs: list, 
                                     font: str = 'Times New Roman',
                                     save_it: bool = False,                                      
                                     use_fixed_norm: bool = True,
                                     fixed_norm_max: int = 10,
                                     use_absolute_value_of_difference: bool = True,
                                     use_percentage: bool = False,
                                     disagreement_threshold: float = None,
                                     result_path: str = None,
                                     loss_fct: str = "l1",
                                     scale_type: str = "continuous",
                                     discrete_thresholds: list = None):
    """
    Plot the average prediction error across multiple models.
    
    Parameters:
    -----------
    gdf_inputs : list
        List of GeoDataFrames containing model predictions
    font : str, optional
        Font to use for plotting
    save_it : bool, optional
        Whether to save the plot
    use_fixed_norm : bool, optional
        Whether to use fixed normalization values
    fixed_norm_max : int, optional
        Maximum value for normalization if use_fixed_norm is True
    use_absolute_value_of_difference : bool
        If True, uses the absolute value of the differences
    use_percentage : bool
        If True, computes differences as percentages relative to actual values
    disagreement_threshold : float
        If set, highlights areas where coefficient of variation exceeds this value
    result_path : str, optional
        Path where to save the plot if save_it is True
    loss_fct : str, optional
        Loss function to use ("l1" or "mse")
    scale_type : str, optional
        Either "continuous" or "discrete"
    discrete_thresholds : list, optional
        List of threshold values defining the boundaries between categories
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))    
    
    base_gdf = gdf_inputs[0].copy()
    
    # Calculate errors for each model
    all_errors = []
    for gdf in gdf_inputs:
        if use_percentage:
            # Avoid division by zero by adding small epsilon where actual is zero
            epsilon = 1e-10
            if use_absolute_value_of_difference:
                if loss_fct == "l1":
                    error = abs((gdf['vol_car_change_predicted'] - gdf['vol_car_change_actual']) / 
                            (abs(gdf['vol_car_change_actual'] + gdf['vol_base_case']))) * 100
                elif loss_fct == "mse":
                    error = ((gdf['vol_car_change_predicted'] - gdf['vol_car_change_actual']).pow(2) / 
                            (abs(gdf['vol_car_change_actual'] + gdf['vol_base_case']) + epsilon)) * 100
            else:
                if loss_fct == "l1":
                    error = (gdf['vol_car_change_predicted'] - gdf['vol_car_change_actual']) / (gdf['vol_car_change_actual'] + gdf['vol_base_case'])* 100
                elif loss_fct == "mse":
                    error = (gdf['vol_car_change_predicted'] - gdf['vol_car_change_actual']).pow(2) / (gdf['vol_car_change_actual'] + gdf['vol_base_case']) * 100
        else:
            if use_absolute_value_of_difference:
                error = abs(gdf['vol_car_change_predicted'] - gdf['vol_car_change_actual'])
            else:
                error = gdf['vol_car_change_predicted'] - gdf['vol_car_change_actual']
        all_errors.append(error)
    
    # Calculate statistics
    errors_df = pd.concat(all_errors, axis=1)
    base_gdf['mean_prediction_error'] = errors_df.mean(axis=1)
    base_gdf['std_prediction_error'] = errors_df.std(axis=1)
    
    # Calculate coefficient of variation for disagreement highlighting
    if disagreement_threshold is not None:
        # Use absolute values for std and mean in coefficient calculation
        abs_mean = abs(base_gdf['mean_prediction_error'])
        # Avoid division by zero
        mask = abs_mean > 1e-10
        base_gdf['coefficient_of_variation'] = float('inf')  # Default value for zero means
        base_gdf.loc[mask, 'coefficient_of_variation'] = base_gdf.loc[mask, 'std_prediction_error'] / abs_mean[mask]
        high_disagreement = base_gdf['coefficient_of_variation'] > disagreement_threshold
    
    base_gdf, x_min, y_min, x_max, y_max = filter_for_geographic_section(base_gdf)
    
    # Handle discrete vs continuous scale
    if scale_type == "discrete":
        if discrete_thresholds is None:
            discrete_thresholds = [33, 66]  # default thresholds
            
        # Create discrete bins for the data
        def categorize_value(x, thresholds):
            for i, threshold in enumerate(thresholds):
                if abs(x) <= threshold:
                    return i
            return len(thresholds)  # for values greater than the last threshold
        
        base_gdf['discrete_error'] = base_gdf['mean_prediction_error'].apply(
            lambda x: categorize_value(x, discrete_thresholds))
        
        # Create a color gradient based on number of categories
        n_categories = len(discrete_thresholds) + 1
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, n_categories))  # Using _r to invert the colormap
        cmap = plt.cm.colors.ListedColormap(colors)
        norm = plt.Normalize(vmin=-0.5, vmax=n_categories - 0.5)
        plot_column = 'discrete_error'
    else:
        if use_fixed_norm:
            norm = TwoSlopeNorm(vmin=-fixed_norm_max, vcenter=0, vmax=fixed_norm_max) if not use_absolute_value_of_difference else \
                   Normalize(vmin=0, vmax=fixed_norm_max)
        else:
            norm = TwoSlopeNorm(vmin=base_gdf['mean_prediction_error'].min(), 
                               vcenter=0, 
                               vmax=base_gdf['mean_prediction_error'].max()) if not use_absolute_value_of_difference else \
                   Normalize(vmin=0, vmax=base_gdf['mean_prediction_error'].max())
        plot_column = 'mean_prediction_error'
        cmap = 'Reds_r' if use_absolute_value_of_difference else 'coolwarm_r'  # Using _r to invert the colormaps
    
    # Plot with different line widths
    linewidths = base_gdf["highway"].apply(get_linewidth)
    base_gdf['linewidth'] = linewidths
    large_lines = base_gdf[base_gdf['linewidth'] > 1]
    small_lines = base_gdf[base_gdf['linewidth'] == 1]
    
    # Plot the data
    small_lines.plot(column=plot_column, cmap=cmap, 
                    linewidth=small_lines['linewidth'],
                    ax=ax, legend=False, norm=norm, zorder=1)
    large_lines.plot(column=plot_column, cmap=cmap, 
                    linewidth=large_lines['linewidth'],
                    ax=ax, legend=False, norm=norm, zorder=2)
    
    # Highlight areas of high disagreement if threshold is set
    if disagreement_threshold is not None:
        disagreement_lines = base_gdf[high_disagreement]
        if not disagreement_lines.empty:
            # Create a simple line for the legend
            legend_line = plt.Line2D([], [], color='black', linestyle='--', 
                                   linewidth=2, 
                                   label=f'High model disagreement (CV > {disagreement_threshold})')
            
            # Plot the disagreement lines
            disagreement_lines.plot(color='none', edgecolor='black', 
                                  linewidth=disagreement_lines['linewidth']*1.5,
                                  ax=ax, zorder=3, linestyle='--')
            
            # Add legend with the custom line
            ax.legend(handles=[legend_line], fontsize=12)
    
    # Styling
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("Longitude", fontname=font, fontsize=15)
    plt.ylabel("Latitude", fontname=font, fontsize=15)
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname(font)
        label.set_fontsize(15)

    # Colorbar setup
    ax.set_position([0.1, 0.1, 0.75, 0.75])
    cax = fig.add_axes([0.87, 0.22, 0.03, 0.5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    
    # Create and customize colorbar based on scale type
    if scale_type == "discrete":
        ticks = range(len(discrete_thresholds) + 1)
        cbar = plt.colorbar(sm, cax=cax, ticks=ticks)
        
        # Create labels based on thresholds
        labels = []
        for i in range(len(discrete_thresholds) + 1):
            if i == 0:
                labels.append(f'0-{discrete_thresholds[0]}')
            elif i == len(discrete_thresholds):
                labels.append(f'>{discrete_thresholds[-1]}')
            else:
                labels.append(f'{discrete_thresholds[i-1]}-{discrete_thresholds[i]}')
        
        cbar.ax.set_yticklabels(labels)
    else:
        cbar = plt.colorbar(sm, cax=cax)
    
    # Customize colorbar
    cbar.ax.tick_params(labelsize=15)
    for t in cbar.ax.get_yticklabels():
        t.set_fontname(font)
    cbar.ax.yaxis.label.set_fontname(font)
    cbar.ax.yaxis.label.set_size(15)
    
    error_type = "Absolute" if use_absolute_value_of_difference else "Signed"
    units = "%" if use_percentage else "vehicles"
    
    cbar.set_label(f'{error_type} Prediction Error\n'
                   f'{loss_fct} difference in {units}\n'
                   f'(Averaged across {len(gdf_inputs)} samples)',
                   fontname=font, fontsize=15)

    if save_it:
        error_type_str = "average_absolute_value_of_difference" if use_absolute_value_of_difference else "average_signed_difference"
        metric_str = "percent" if use_percentage else "abs_vehicles"
        plt.savefig(f"{result_path}/{error_type_str}_prediction_error_{metric_str}", 
                   bbox_inches='tight')
    
    plt.show()
    
    return base_gdf
    


def get_norm(column_to_plot, use_fixed_norm, fixed_norm_max, gdf):
    if use_fixed_norm:
        norm = TwoSlopeNorm(vmin=-fixed_norm_max, vcenter=0, vmax=fixed_norm_max)
    else:
        norm = TwoSlopeNorm(vmin=gdf[column_to_plot].min(), vcenter=gdf[column_to_plot].median(), vmax=gdf[column_to_plot].max())
    return norm
    

def plotting(font, x_min, y_min, x_max, y_max, fig, ax, norm, plot_contour_lines, cmap, plot_policy_roads=False):
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
    custom_lines = [Line2D([0], [0], color='grey', lw=4, label='Street network')] # Add more lines for other labels as needed

    if plot_contour_lines:
        custom_lines.append(Line2D([0], [0], color='black', lw=2, label='Capacity was decreased in this section'))

    if plot_policy_roads:
        custom_lines.append(Line2D([0], [0], color='black', lw=2, label='Capacity was decreased on these roads'))

    ax.legend(handles=custom_lines, prop={'family': font, 'size': 15})
    ax.set_position([0.1, 0.1, 0.75, 0.75])
    cax = fig.add_axes([0.87, 0.22, 0.03, 0.5])  # Manually position the color bar
    
    # Create the color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
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
        

def filter_for_geographic_section(gdf):
    x_min = gdf.total_bounds[0] + 0.05
    y_min = gdf.total_bounds[1] + 0.05
    x_max = gdf.total_bounds[2]
    y_max = gdf.total_bounds[3]
    bbox = box(x_min, y_min, x_max, y_max)

    # Filter the network to include only the data within the bounding box
    gdf = gdf[gdf.intersects(bbox)]
    return gdf,x_min,y_min,x_max,y_max
    

# def plot_districts_of_capacity_reduction(gdf_input:gpd.GeoDataFrame, font:str ='DejaVu Serif', save_it: bool=False, number_to_plot : int=0):    
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
#     gdf = gdf[gdf["highway"].isin([1, 2, 3])]
    
#     # Round og_capacity_reduction and filter
#     gdf['capacity_reduction_rounded'] = gdf['capacity_reduction'].round(decimals=3)
#     tolerance = 1e-3
#     edges_with_capacity_reduction = gdf[np.abs(gdf['capacity_reduction_rounded']) > tolerance]
#     # edges_without_capacity_reduction = gdf[np.abs(gdf['og_capacity_reduction_rounded']) <= tolerance]

#     norm = TwoSlopeNorm(vmin=gdf["capacity_reduction"].min(), vcenter=gdf["capacity_reduction"].median(), vmax=gdf["capacity_reduction"].max())
    
#     # edges_without_capacity_reduction.plot(
#     #     ax=ax, column=column_to_plot, cmap='coolwarm', linewidth=3, legend=False, norm=norm, zorder=1, label = "Capacity reduction")
#     edges_with_capacity_reduction.plot(
#         ax=ax, column='capacity_reduction', cmap='coolwarm', linewidth=5, legend=False, norm=norm, zorder=2, label = "Edges with capacity reduction")
        
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
#     cbar.set_label('Capacity reduction', fontname=font, fontsize=15)
#     if save_it:
#         plt.savefig("results/gdf_capacity_reduction_" + str(number_to_plot), bbox_inches='tight')
#     plt.show()

# def validate_one_model(model, data, loss_func, device):
#     model.eval()
#     with torch.inference_mode():
#         input_node_features, targets = data.x.to(device), data.y.to(device)
#         predicted = model(data.to(device))
#         val_loss = loss_func(predicted, targets).item()
#     r_squared = compute_r2_torch(preds=predicted, targets=targets)
#     return val_loss, r_squared, targets, predicted


# def validate_one_model(model, data, loss_func, device):
#     model.eval()
#     pred = []
#     actual = []
#     with torch.inference_mode():
#         input_node_features, targets = data.x.to(device), data.y.to(device)
#         predicted = model(data.to(device))
#         # print(predicted.shape)
#         pred.append(predicted)
#         actual.append(targets)
#         val_loss = loss_func(predicted, targets).item()
#     actual_vals = torch.cat(actual)
#     predicted_vals = torch.cat(pred)
    
#     mean_targets = torch.mean(targets)
#     r_squared = compute_r2_torch_with_mean_targets(mean_targets = mean_targets, preds=predicted_vals, targets=actual_vals)
#     baseline_loss = loss_func(targets, torch.full_like(predicted_vals, mean_targets))
#     return val_loss, r_squared, targets, predicted, baseline_loss
