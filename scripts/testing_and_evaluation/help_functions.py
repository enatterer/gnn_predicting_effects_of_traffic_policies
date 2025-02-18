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
from scipy import stats


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

# Option 1: Ocean Blues (Serene)
colors = {
    'All Roads': '#01579b',                               # Darkest ocean blue
    'Trunk Roads': '#0277bd',                             # Dark ocean blue
    'Primary Roads': '#0288d1',                           # Medium-dark blue
    'Secondary Roads': '#039be5',                         # Medium blue
    'Tertiary Roads': '#03a9f4',                          # Medium-light blue
    'Residential Streets': '#29b6f6',                     # Light blue
    'Living Streets': '#4fc3f7',                          # Lighter blue
    'P/S/T Roads with Capacity Reduction': '#81d4fa',     # Very light blue
    'P/S/T Roads with No Capacity Reduction': '#b3e5fc',   # Lightest blue
    # 'Secondary Roads with Capacity Reduction': '#81d4fa',     # Very light blue
    # 'Secondary Roads with No Capacity Reduction': '#b3e5fc',   # Lightest blue
    # 'Tertiary Roads with Capacity Reduction': '#81d4fa',     # Very light blue
    # 'Tertiary Roads with No Capacity Reduction': '#b3e5fc'   # Lightest blue
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

def compute_correlations_scipy(predictions, targets):
    """
    Compute correlations using scipy (for verification)
    """
    spearman_corr, _ = stats.spearmanr(predictions, targets)
    pearson_corr, _ = stats.pearsonr(predictions, targets)
    return spearman_corr, pearson_corr

def get_road_type_indices(gdf, tolerance=1e-3):
    """
    Get indices for different road types, including dynamic conditions like capacity reduction
    """
    tolerance = 1e-3
    
    indices = {
        # Static conditions (road types)
        "All Roads": gdf.index,
        "Trunk Roads": gdf[gdf['highway'].isin([0])].index,
        "Primary Roads": gdf[gdf['highway'].isin([1])].index,
        "Secondary Roads": gdf[gdf['highway'].isin([2])].index,
        "Tertiary Roads": gdf[gdf['highway'].isin([3])].index,
        "Residential Streets": gdf[gdf['highway'].isin([4])].index,
        "Living Streets": gdf[gdf['highway'].isin([5])].index,
        # "P/S/T Roads": gdf[gdf['highway'].isin([1, 2, 3])].index,
        # Dynamic conditions (capacity reduction)
        # "Roads with Capacity Reduction": gdf[gdf['capacity_reduction_rounded'] < -tolerance].index,
        # "Roads with No Capacity Reduction": gdf[gdf['capacity_reduction_rounded'] >= -tolerance].index,
        
        "P/S/T Roads with Capacity Reduction": gdf[(gdf['highway'].isin([1, 2, 3])) & (gdf['capacity_reduction_rounded'] < -tolerance)].index,
        "P/S/T Roads with No Capacity Reduction": gdf[(gdf['highway'].isin([1, 2, 3])) & (gdf['capacity_reduction_rounded'] >= -tolerance)].index,
        # Combined conditions
        # "Primary Roads with Capacity Reduction": gdf[
        #     (gdf['highway'].isin([1])) & 
        #     (gdf['capacity_reduction_rounded'] < -tolerance)
        # ].index,
        # "Primary Roads with No Capacity Reduction": gdf[
        #     (gdf['highway'].isin([1])) & 
        #     (gdf['capacity_reduction_rounded'] >= -tolerance)
        # ].index,
        # "Secondary Roads with Capacity Reduction": gdf[
        #     (gdf['highway'].isin([2])) & 
        #     (gdf['capacity_reduction_rounded'] < -tolerance)
        # ].index,
        # "Secondary Roads with No Capacity Reduction": gdf[
        #     (gdf['highway'].isin([2])) & 
        #     (gdf['capacity_reduction_rounded'] >= -tolerance)
        # ].index,
        # "Tertiary Roads with Capacity Reduction": gdf[
        #     (gdf['highway'].isin([3])) & 
        #     (gdf['capacity_reduction_rounded'] < -tolerance)
        # ].index,    
        # "Tertiary Roads with No Capacity Reduction": gdf[
        #     (gdf['highway'].isin([3])) & 
        #     (gdf['capacity_reduction_rounded'] >= -tolerance)
        # ].index
    }
    return indices


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
                         result_path:str=None,
                         scale_type:str="continuous",
                         discrete_thresholds: list = None,
                         with_legend: bool = False):

    gdf = gdf_input.copy()
    gdf, x_min, y_min, x_max, y_max = filter_for_geographic_section(gdf)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))    

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
        
        gdf['discrete_plot_column'] = gdf[column_to_plot].apply(
            lambda x: categorize_value(x, discrete_thresholds))
        
        # Create a color gradient based on number of categories
        n_categories = len(discrete_thresholds) + 1
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, n_categories))  # Using _r to invert the colormap
        cmap = plt.cm.colors.ListedColormap(colors)
        norm = plt.Normalize(vmin=-0.5, vmax=n_categories - 0.5)
        column_to_plot = 'discrete_plot_column'
    else:
        if use_fixed_norm:
            norm = TwoSlopeNorm(vmin=-fixed_norm_max, vcenter=0, vmax=fixed_norm_max)
        else:
            norm = TwoSlopeNorm(vmin=gdf[column_to_plot].min(), vcenter=gdf[column_to_plot].median(), vmax=gdf[column_to_plot].max())
    
    linewidths = gdf["highway"].apply(get_linewidth)
    gdf['linewidth'] = linewidths
    large_lines = gdf[gdf['linewidth'] > 1]
    small_lines = gdf[gdf['linewidth'] == 1]
    # small_lines.plot(column=column_to_plot, cmap=cmap, linewidth=small_lines['linewidth'], ax=ax, legend=False,
    #                 norm=norm, label="Street network", zorder=1)
    # large_lines.plot(column=column_to_plot, cmap=cmap, linewidth=large_lines['linewidth'], ax=ax, legend=False,
    #                 norm=norm, label="Street network", zorder=2)
    
    small_lines.plot(column=column_to_plot, cmap=cmap, linewidth=small_lines['linewidth'], ax=ax, 
                    legend=False,
                    norm=norm, 
                    label="Street network" if with_legend else None,  # Only add label if legend is wanted
                    zorder=1)
    large_lines.plot(column=column_to_plot, cmap=cmap, linewidth=large_lines['linewidth'], ax=ax, 
                    legend=False,
                    norm=norm, 
                    label="Street network" if with_legend else None,  # Only add label if legend is wanted
                    zorder=2)
    
    if plot_policy_roads:
        tolerance = 1e-3
        gdf['capacity_reduction_rounded'] = gdf['capacity_reduction'].round(decimals=3)
        edges_with_capacity_reduction = gdf[np.abs(gdf['capacity_reduction_rounded']) > tolerance]
        # edges_with_capacity_reduction.plot(color='black', linewidth=large_lines['linewidth'], ax=ax, legend=False,
        #                                 norm=norm, label="Capacity was decreased on these roads", zorder=3)
        edges_with_capacity_reduction.plot(color='black', linewidth=large_lines['linewidth'], ax=ax, 
                                        legend=False,
                                        norm=norm, 
                                        label="Capacity was decreased on these roads" if with_legend else None,  # Only add label if legend is wanted
                                        zorder=3)


    relevant_area_to_plot = get_relevant_area_to_plot(alpha, known_districts, buffer, districts_of_interest, gdf)
    if plot_contour_lines:
        if isinstance(relevant_area_to_plot, set):
            for area in relevant_area_to_plot:
                if isinstance(area, Polygon):
                    gdf_area = gpd.GeoDataFrame(index=[0], crs=gdf.crs, geometry=[area])
                    gdf_area.plot(ax=ax, edgecolor='black', linewidth=2, facecolor='None', zorder=2)
        else:
            relevant_area_to_plot.plot(ax=ax, edgecolor='black', linewidth=2, facecolor='None', zorder=2)
    
    if scale_type == "discrete":
        cax = fig.add_axes([0.87, 0.22, 0.03, 0.5])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
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
        cbar = plotting(font, x_min, y_min, x_max, y_max, fig, ax, norm, plot_contour_lines, cmap, plot_policy_roads, with_legend)
    
    if is_absolute:
        cbar.set_label('Car volume', fontname=font, fontsize=15)
    else:
        cbar.set_label('Car volume: Difference to base case (%)', fontname=font, fontsize=15)
    if save_it:
        p = "predicted" if is_predicted else "actual"
        identifier = "n_" + str(number_to_plot) if number_to_plot is not None else zone_to_plot
        plt.savefig(result_path + identifier + "_" + p, bbox_inches='tight')
    plt.show()


def plotting(font, x_min, y_min, x_max, y_max, fig, ax, norm, plot_contour_lines, cmap, plot_policy_roads=False, with_legend=False):
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    if with_legend:
        plt.xlabel("Longitude", fontname=font, fontsize=15)
        plt.ylabel("Latitude", fontname=font, fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=10)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname(font)
            label.set_fontsize(15)
        ax.legend(prop={'family': font, 'size': 15})
        ax.set_position([0.1, 0.1, 0.75, 0.75])
    else:
        # Remove absolutely everything from the axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_position([0, 0, 1, 1])  # Make plot take up entire figure space
    
    # Colorbar positioning and creation
    if with_legend:
        cax = fig.add_axes([0.87, 0.22, 0.03, 0.5])
    else:
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjusted colorbar position for no-legend case

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


def create_correlation_radar_plot_sort_by_r2(metrics_by_type, selected_metrics=None, result_path=None, save_it=False):

    """
    Create a radar plot for model performance metrics.
    
    Args:
        metrics_by_type (dict): Dictionary containing metrics for each road type
        selected_metrics (list, optional): List of metrics to display. Each metric should be a dict with:
            - 'id': identifier in metrics_by_type
            - 'label': display label
            - 'transform': function to transform the value (or None to use directly)
            - 'y_pos': y-position of the label
    """
    # Default metrics if none specified
    if selected_metrics is None:
        selected_metrics = [
            {
                'id': 'r_squared',
                'label': 'R²',
                'transform': lambda x: max(0, x * 100),
                'y_pos': -0.05
            },
            {
                'id': 'mse_ratio',
                'label': 'MSE/Naive MSE',
                'transform': lambda x, max_ratio: (1 - x/max_ratio) * 100,
                'y_pos': -0.1
            },
            {
                'id': 'pearson',
                'label': 'Pearson\nCorrelation',
                'transform': lambda x: max(0, x * 100),
                'y_pos': -0.05
            },
            {
                'id': 'spearman',
                'label': 'Spearman\nCorrelation',
                'transform': lambda x: max(0, x * 100),
                'y_pos': -0.05
            }
        ]
    
    # Select specific road types
    selected_types = [
        'All Roads',
        'Trunk Roads',
        'Primary Roads',
        'Secondary Roads',
        'Tertiary Roads',
        'Residential Streets',
        'Living Streets',
        'P/S/T Roads with Capacity Reduction',
        'P/S/T Roads with No Capacity Reduction',
    ]
    
    filtered_metrics = {rt: metrics_by_type[rt] for rt in selected_types}
    road_types = sorted(filtered_metrics.keys(), 
                       key=lambda x: filtered_metrics[x]['r_squared'],
                       reverse=True)
    
    # Calculate maximum ratios for normalization if needed
    max_ratios = {}
    for metric in selected_metrics:
        if 'ratio' in metric['id']:
            if metric['id'] == 'mse_ratio':
                max_ratios['mse'] = max(metrics_by_type[rt]['mse'] / 
                                      metrics_by_type[rt]['naive_mse'] 
                                      for rt in road_types)
            elif metric['id'] == 'l1_ratio':
                max_ratios['l1'] = max(metrics_by_type[rt]['l1'] / 
                                     metrics_by_type[rt]['naive_l1'] 
                                     for rt in road_types)
    
    # Setup plot
    num_vars = len(selected_metrics)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.grid(True, color='gray', alpha=0.3)
    
    # Plot data
    for road_type in road_types:
        values = []
        for metric in selected_metrics:
            if 'ratio' in metric['id']:
                if metric['id'] == 'mse_ratio':
                    ratio = filtered_metrics[road_type]['mse'] / filtered_metrics[road_type]['naive_mse']
                    values.append(metric['transform'](ratio, max_ratios['mse']))
                elif metric['id'] == 'l1_ratio':
                    ratio = filtered_metrics[road_type]['l1'] / filtered_metrics[road_type]['naive_l1']
                    values.append(metric['transform'](ratio, max_ratios['l1']))
            else:
                val = filtered_metrics[road_type][metric['id']]
                values.append(metric['transform'](val))
        values += values[:1]
        ax.plot(angles, values, linewidth=3, linestyle='solid',
            label=f"{road_type} (R²: {filtered_metrics[road_type]['r_squared']:.2f})",  
            color=colors[road_type])
        ax.fill(angles, values, alpha=0.1, color=colors[road_type])

    # Set chart properties
    ax.set_xticks(angles[:-1])
    
    # Set the labels with proper positioning
    ax.set_xticklabels(
        [m['label'] for m in selected_metrics],
        fontsize=15,
        y=-0.05  # Move labels outward
    )
    
    ax.set_ylim(0, 1)
    ax.set_rgrids([0, 0.2, 0.4, 0.6, 0.8, 1], angle=45, fontsize=15)
        
        # Grid lines and outer circle styling
    for line in ax.yaxis.get_gridlines() + ax.xaxis.get_gridlines():
        line.set_color('gray')
        line.set_linewidth(1.0)
        line.set_alpha(0.3)

    # Center point and circle
    ax.plot(0, 0, 'k.', markersize=10)
    
        # In your radar plot code, modify the legend part:
    ax.legend(loc='center left', 
            bbox_to_anchor=(1.1, 0.5),
            fontsize=15,          # Increase font size (default is usually 10)
            markerscale=2,        # Make the markers/lines in legend bigger
            frameon=True,         # Add a frame
            framealpha=0.9,       # Make frame slightly transparent
            edgecolor='gray',     # Add edge color to frame
            borderpad=1,          # Add padding inside legend border
            labelspacing=1.2,     # Increase spacing between legend entries
            handlelength=3)       # Make the lines in legend longer
    if save_it:
        plt.savefig(result_path + "radar_plot.png", bbox_inches='tight', dpi=300)
        
    plt.show()
    


def create_error_vs_variability_scatterplots(metrics_by_type, result_path=None, save_it=False):
    """
    Create scatter plot for MSE vs Variance with blue best-fit line and larger labels
    """
    plt.rcParams["font.family"] = "Times New Roman"
    
    # Define selected road types
    selected_types = [
        'All Roads',
        'Trunk Roads',
        'Primary Roads',
        'Secondary Roads',
        'Tertiary Roads',
        'Residential Streets',
        'Living Streets',
        # 'P/S/T Roads with Capacity Reduction',
        # 'P/S/T Roads with No Capacity Reduction'
        'Primary Roads with Capacity Reduction',
        'Primary Roads with No Capacity Reduction',
        'Secondary Roads with Capacity Reduction',
        'Secondary Roads with No Capacity Reduction',
        'Tertiary Roads with Capacity Reduction',
        'Tertiary Roads with No Capacity Reduction'
    ]
    
    # Get data
    mse_values = [metrics_by_type[rt]['mse'] for rt in selected_types]
    variance_values = [metrics_by_type[rt]['variance'] for rt in selected_types]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add best-fit line with harmonious blue color
    z = np.polyfit(variance_values, mse_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(variance_values), max(variance_values), 100)
    ax.plot(x_line, p(x_line), '--', color='#0277bd', alpha=0.8, linewidth=2)  # Using a medium-dark blue
    
    # Plot points
    for i, rt in enumerate(selected_types):
        ax.scatter(variance_values[i], mse_values[i], color=colors[rt], s=150, label=rt)
        ax.annotate(rt, (variance_values[i], mse_values[i]), 
                   xytext=(10, 10), textcoords='offset points', fontsize=16)
    
    # Labels with larger font size
    ax.set_xlabel('Variance', fontsize=16)
    ax.set_ylabel('MSE', fontsize=16)
    
    # Tick labels
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add correlation coefficient
    pearson_mse_var = stats.pearsonr(variance_values, mse_values)[0]
    ax.text(0.02, 0.98, f'Pearson r = {pearson_mse_var:.2f}', 
            transform=ax.transAxes, verticalalignment='top', fontsize=16)
    
    plt.tight_layout()
    if save_it:
        plt.savefig(result_path + "error_vs_variability_scatterplot.png", bbox_inches='tight', dpi=300)
    plt.show()
    
    
def create_error_vs_variability_scatterplots_mse_and_mae(metrics_by_type, result_path=None, save_it=False):
    """
    Create two scatter plots:
    1. MSE vs Variance
    2. L1 vs Normalized Std Dev
    With CV on secondary axis
    """
    plt.rcParams["font.family"] = "Times New Roman"
    
    # Define selected road types
    selected_types = [
        'All Roads',
        'Trunk Roads',
        'Primary Roads',
        'Secondary Roads',
        'Tertiary Roads',
        'Residential Streets',
        'Living Streets',
        'P/S/T Roads with Capacity Reduction',
        'P/S/T Roads with No Capacity Reduction'
    ]
    
    # Create two subplots
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Get data
    mse_values = [metrics_by_type[rt]['mse'] for rt in selected_types]
    variance_values = [metrics_by_type[rt]['variance'] for rt in selected_types]
    l1_values = [metrics_by_type[rt]['l1'] for rt in selected_types]
    std_dev_norm_values = [metrics_by_type[rt]['std_dev_normalized'] for rt in selected_types]
    cv_values = [metrics_by_type[rt]['cv_percent'] for rt in selected_types]
    
    # Plot 1: MSE vs Variance
    ax1.scatter(variance_values, mse_values, color='#2ecc71', s=100)
    # Add road type labels to points
    for i, txt in enumerate(selected_types):
        ax1.annotate(txt, (variance_values[i], mse_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Variance', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    
    # Secondary y-axis for CV in first plot (inverted)
    ax2 = ax1.twinx()
    ax2.scatter(variance_values, cv_values, color='#3498db', alpha=0)  # Invisible points to set scale
    ax2.set_ylabel('CV (%)', color='#3498db', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#3498db')
    ax2.invert_yaxis()  # Invert the CV axis
    
    # Plot 2: L1 vs Normalized Std Dev
    ax3.scatter(std_dev_norm_values, l1_values, color='#e74c3c', s=100)
    # Add road type labels to points
    for i, txt in enumerate(selected_types):
        ax3.annotate(txt, (std_dev_norm_values[i], l1_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax3.set_xlabel('Normalized Std Dev', fontsize=12)
    ax3.set_ylabel('L1', fontsize=12)
    
    # Secondary y-axis for CV in second plot (inverted)
    ax4 = ax3.twinx()
    ax4.scatter(std_dev_norm_values, cv_values, color='#3498db', alpha=0)  # Invisible points to set scale
    ax4.set_ylabel('CV (%)', color='#3498db', fontsize=12)
    ax4.tick_params(axis='y', labelcolor='#3498db')
    ax4.invert_yaxis()  # Invert the CV axis
    
    # Calculate correlations
    pearson_mse_var = stats.pearsonr(variance_values, mse_values)[0]
    pearson_l1_std = stats.pearsonr(std_dev_norm_values, l1_values)[0]
    
    # Add correlation coefficients
    ax1.text(0.02, 0.98, f'Pearson r = {pearson_mse_var:.2f}', 
             transform=ax1.transAxes, verticalalignment='top')
    ax3.text(0.02, 0.98, f'Pearson r = {pearson_l1_std:.2f}', 
             transform=ax3.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    if save_it:
        plt.savefig(result_path, bbox_inches='tight', dpi=300)
    plt.show()

    
# # Convert the GeoDataFrame to the appropriate coordinate reference system (CRS) for length calculation
# gdf_in_meters = gdf_with_og_values.to_crs("EPSG:32633")
# gdf_in_meters['length'] = gdf_in_meters.length
# total_length = gdf_in_meters['length'].sum() / 1000
# print(f"Total length of the street network: {total_length:.2f} km")
# gdf_with_reductions = gdf_in_meters.loc[indices_roads_with_cap_reduction]
# total_length_with_reductions = gdf_with_reductions['length'].sum() / 1000
# print(f"Total length of the street network with capacity reductions: {total_length_with_reductions:.2f} km")

# def create_correlation_radar_plot(metrics_by_type, selected_metrics=None):

#     """
#     Create a radar plot for model performance metrics.
    
#     Args:
#         metrics_by_type (dict): Dictionary containing metrics for each road type
#         selected_metrics (list, optional): List of metrics to display. Each metric should be a dict with:
#             - 'id': identifier in metrics_by_type
#             - 'label': display label
#             - 'transform': function to transform the value (or None to use directly)
#             - 'y_pos': y-position of the label
#     """
#     # Default metrics if none specified
#     if selected_metrics is None:
#         selected_metrics = [
#             {
#                 'id': 'r_squared',
#                 'label': 'R²',
#                 'transform': lambda x: max(0, x * 100),
#                 'y_pos': -0.05
#             },
#             {
#                 'id': 'mse_ratio',
#                 'label': 'MSE/Naive MSE',
#                 'transform': lambda x, max_ratio: (1 - x/max_ratio) * 100,
#                 'y_pos': -0.1
#             },
#             {
#                 'id': 'pearson',
#                 'label': 'Pearson\nCorrelation',
#                 'transform': lambda x: max(0, x * 100),
#                 'y_pos': -0.05
#             },
#             {
#                 'id': 'spearman',
#                 'label': 'Spearman\nCorrelation',
#                 'transform': lambda x: max(0, x * 100),
#                 'y_pos': -0.05
#             }
#         ]
    
#     # Select specific road types
#     selected_types = [
#         'All Roads',
#         'Trunk Roads',
#         'Primary Roads',
#         'Secondary Roads',
#         'Tertiary Roads',
#         'Residential Streets',
#         'Living Streets',
#         'Primary Roads with Capacity Reduction',
#         'Primary Roads with No Capacity Reduction',
#         'Secondary Roads with Capacity Reduction',
#         'Secondary Roads with No Capacity Reduction',
#         'Tertiary Roads with Capacity Reduction',
#         'Tertiary Roads with No Capacity Reduction'
#     ]
    
#     filtered_metrics = {rt: metrics_by_type[rt] for rt in selected_types}
#     road_types = sorted(filtered_metrics.keys(), 
#                        key=lambda x: filtered_metrics[x]['r_squared'],
#                        reverse=True)
    
#     # Compute scores
#     total_scores = {}
#     all_scores = []  # to compute average later
#     for road_type in road_types:
#         metrics = filtered_metrics[road_type]
#         # Normalize metrics to 0-1 range
#         normalized_scores = {
#             'r_squared': metrics['r_squared'],                    # Already 0-1
#             'mse_ratio': 1 - metrics['mse']/metrics['naive_mse'], # Transform to 0-1
#             'l1_ratio': 1 - metrics['l1']/metrics['naive_l1'],    # Transform to 0-1
#             'pearson': (metrics['pearson'] + 1)/2,                # Transform -1,1 to 0-1
#             'spearman': (metrics['spearman'] + 1)/2               # Transform -1,1 to 0-1
#         }
#         # Equal weighting (0.2 each)
#         score = sum(normalized_scores.values()) * 0.2 * 100
#         total_scores[road_type] = score
#         all_scores.append(score)

#     # Print all scores
#     print("\nModel Performance Scores:")
#     print("-" * 50)
#     for road_type in road_types:
#         print(f"{road_type}: {total_scores[road_type]:.1f}")
#     print("-" * 50)
#     print(f"Overall Average Score: {np.mean(all_scores):.1f}")
#     print(f"Best Score: {np.max(all_scores):.1f} ({road_types[np.argmax(all_scores)]})")
#     print(f"Worst Score: {np.min(all_scores):.1f} ({road_types[np.argmin(all_scores)]})")
#     print("-" * 50)
    
    
#     # Calculate maximum ratios for normalization if needed
#     max_ratios = {}
#     for metric in selected_metrics:
#         if 'ratio' in metric['id']:
#             if metric['id'] == 'mse_ratio':
#                 max_ratios['mse'] = max(metrics_by_type[rt]['mse'] / 
#                                       metrics_by_type[rt]['naive_mse'] 
#                                       for rt in road_types)
#             elif metric['id'] == 'l1_ratio':
#                 max_ratios['l1'] = max(metrics_by_type[rt]['l1'] / 
#                                      metrics_by_type[rt]['naive_l1'] 
#                                      for rt in road_types)
    
#     # Setup plot
#     num_vars = len(selected_metrics)
#     angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
#     angles += angles[:1]
    
#     fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
#     ax.grid(True, color='gray', alpha=0.3)
    
    
#     # Option 1: Ocean Blues (Serene)
#     colors = {
#         'All Roads': '#01579b',                               # Darkest ocean blue
#         'Trunk Roads': '#0277bd',                             # Dark ocean blue
#         'Primary Roads': '#0288d1',                           # Medium-dark blue
#         'Secondary Roads': '#039be5',                         # Medium blue
#         'Tertiary Roads': '#03a9f4',                          # Medium-light blue
#         'Residential Streets': '#29b6f6',                     # Light blue
#         'Living Streets': '#4fc3f7',                          # Lighter blue
#         'Primary Roads with Capacity Reduction': '#81d4fa',     # Very light blue
#         'Primary Roads with No Capacity Reduction': '#b3e5fc',   # Lightest blue
#         'Secondary Roads with Capacity Reduction': '#81d4fa',     # Very light blue
#         'Secondary Roads with No Capacity Reduction': '#b3e5fc',   # Lightest blue
#         'Tertiary Roads with Capacity Reduction': '#81d4fa',     # Very light blue
#         'Tertiary Roads with No Capacity Reduction': '#b3e5fc'   # Lightest blue
#     }
    
#     # Plot data
#     for road_type in road_types:
#         values = []
#         for metric in selected_metrics:
#             if 'ratio' in metric['id']:
#                 if metric['id'] == 'mse_ratio':
#                     ratio = filtered_metrics[road_type]['mse'] / filtered_metrics[road_type]['naive_mse']
#                     values.append(metric['transform'](ratio, max_ratios['mse']))
#                 elif metric['id'] == 'l1_ratio':
#                     ratio = filtered_metrics[road_type]['l1'] / filtered_metrics[road_type]['naive_l1']
#                     values.append(metric['transform'](ratio, max_ratios['l1']))
#             else:
#                 val = filtered_metrics[road_type][metric['id']]
#                 values.append(metric['transform'](val))
#         values += values[:1]
#         ax.plot(angles, values, linewidth=3, linestyle='solid',
#             label=f"{road_type} (Score: {total_scores[road_type]:.1f}%)",  # Added % symbol
#             color=colors[road_type])
#         ax.fill(angles, values, alpha=0.1, color=colors[road_type])

#     # Set chart properties
#     ax.set_xticks(angles[:-1])
    
#     # Set the labels with proper positioning
#     ax.set_xticklabels(
#         [m['label'] for m in selected_metrics],
#         fontsize=15,
#         y=-0.05  # Move labels outward
#     )
    
#     ax.set_ylim(0, 1)
#     ax.set_rgrids([0, 0.2, 0.4, 0.6, 0.8, 1], angle=45, fontsize=15)
        
#         # Grid lines and outer circle styling
#     for line in ax.yaxis.get_gridlines() + ax.xaxis.get_gridlines():
#         line.set_color('gray')
#         line.set_linewidth(1.0)
#         line.set_alpha(0.3)

#     # Add grey outer circle with same style as grid
#     # circle = plt.Circle((0, 0), 1, fill=False, 
#     #                 color='gray',      # Same grey as grid
#     #                 linewidth=1.0,     # Same width as grid
#     #                 alpha=0.3)         # Same transparency as grid
#     # ax.add_artist(circle)
    
#     # Center point and circle
#     ax.plot(0, 0, 'k.', markersize=10)
    
#         # In your radar plot code, modify the legend part:
#     ax.legend(loc='center left', 
#             bbox_to_anchor=(1.1, 0.5),
#             fontsize=15,          # Increase font size (default is usually 10)
#             markerscale=2,        # Make the markers/lines in legend bigger
#             frameon=True,         # Add a frame
#             framealpha=0.9,       # Make frame slightly transparent
#             edgecolor='gray',     # Add edge color to frame
#             borderpad=1,          # Add padding inside legend border
#             labelspacing=1.2,     # Increase spacing between legend entries
#             handlelength=3)       # Make the lines in legend longer
    
#     plt.savefig("correlation_radar_plot.png", bbox_inches='tight', dpi=300)
#     plt.show()
    

# # Use custom metrics
# custom_metrics_3 = [
#     {
#         'id': 'r_squared',
#         'label': 'R²',
#         'transform': lambda x: max(0, x * 100),
#         'y_pos': -0.05
#     },
#     {
#         'id': 'pearson',
#         'label': 'Pearson\nCorrelation',
#         'transform': lambda x: max(0, x * 100),
#         'y_pos': -0.05
#     },
#     {
#         'id': 'spearman',
#         'label': 'Spearman\nCorrelation',
#         'transform': lambda x: max(0, x * 100),
#         'y_pos': -0.05
#     }
# ]

# selected_metrics_5 = [
#             {
#             'id': 'spearman',
#             'label': 'Spearman\nCorrelation',
#             'transform': lambda x: max(0, x),
#             'y_pos': -0.05
#             },
#             {
#                 'id': 'r_squared',
#                 'label': 'R²',
#                 'transform': lambda x: max(0, x),
#                 'y_pos': -0.05
#             },
#             # {
#             #     'id': 'mse_ratio',
#             #     'label': '1 - MSE/Naive MSE',
#             #     'transform': lambda x, max_ratio: (1 - x/max_ratio),
#             #     'y_pos': -0.1
#             # },
#             {
#                 'id': 'l1_ratio',
#                 'label': '1 - MAE/Naive MAE',
#                 'transform': lambda x, max_ratio: (1 - x/max_ratio),
#                 'y_pos': -0.1
#             },
#             {
#                 'id': 'pearson',
#                 'label': 'Pearson\nCorrelation',
#                 'transform': lambda x: max(0, x),
#                 'y_pos': -0.05
#             }
#         ]
# create_correlation_radar_plot(metrics_by_type, selected_metrics_5)
    

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
