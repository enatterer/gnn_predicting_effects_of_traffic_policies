import geopandas as gpd

import os
import glob
import gzip
import math
import random
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import shapely.wkt as wkt
from shapely.geometry import Point, LineString, box
from shapely.ops import nearest_points
import tqdm
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import LineGraph
import re
from matplotlib.colors import TwoSlopeNorm

from shapely.ops import unary_union
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_simulation_output(df, districts_of_interest: list, is_for_1pm: str, in_percentage: bool, districts: gpd.GeoDataFrame, do_save: bool=False, buffer = 0.0005):
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
    buffered_target_districts['geometry'] = buffered_target_districts.buffer(buffer)
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
    if do_save:
        plt.savefig("results/" + is_for_1pm + "_difference_to_policies_in_zones_" + list_to_string(districts_of_interest, "_"), bbox_inches='tight')
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

def plot_model_output(gdf, districts_of_interest: list, column_to_plot = "og_vol_car_change_predicted", districts=gpd.GeoDataFrame()): 
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
    
    target_districts = districts[districts['c_ar'].isin(districts_of_interest)]
    gdf['intersects_target_districts'] = gdf.apply(intersects_target, axis=1, target_districts=target_districts)
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
    ax.set_position([0.1, 0.1, 0.75, 0.75])
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
    # if in_percentage:
    cbar.set_label('Car volume: Difference to base case (%)', fontname='Times New Roman', fontsize=15)
    # else:
    #     cbar.set_label('Car volume: Difference to base case (absolut)', fontname='Times New Roman', fontsize=15)
    # plt.savefig("results/difference_to_policies_in_zones_" + list_to_string(districts_of_interest, "_") + is_for_1pm, bbox_inches='tight')
    plt.show()

def plot_simulation_output_basecase(df, districts_of_interest: list, is_for_1pm: str, districts: gpd.GeoDataFrame, do_save:bool=False):
    # Convert DataFrame to GeoDataFrame
    
    column_to_plot = "vol_car"
    
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

    norm = plt.Normalize()
    
    # Plot the edges that intersect with target districts thicker
    gdf.plot(column=column_to_plot, cmap='coolwarm', linewidth=4, ax=ax, legend=False,
             norm=norm, label = "Higher order roads", zorder=2)
    
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
    
    cbar.set_label('Car volume (absolut)', fontname='Times New Roman', fontsize=15)
    if do_save:
        plt.savefig("results/" + is_for_1pm + "_base_case", bbox_inches='tight')
    plt.show()
    
    
def plot_just_zone(df, districts_of_interest: list, is_for_1pm: str, districts:gpd.GeoDataFrame, do_save:bool=False, buffer = 0.0005):
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:2154")
    gdf = gdf.to_crs(epsg=4326)

    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    target_districts = districts[districts['c_ar'].isin(districts_of_interest)]

    # Add buffer to target districts to avoid overlapping with edges
    buffered_target_districts = target_districts.copy()
    buffered_target_districts['geometry'] = buffered_target_districts.buffer(buffer)
    # Ensure the buffered_target_districts GeoDataFrame is in the same CRS
    if buffered_target_districts.crs != gdf.crs:
        buffered_target_districts.to_crs(gdf.crs, inplace=True)

    # Create a single outer boundary
    outer_boundary = unary_union(buffered_target_districts.geometry).boundary

    # Plot only the outer boundary
    gpd.GeoSeries(outer_boundary, crs=gdf.crs).plot(ax=ax, edgecolor='black', linewidth=1, label="Arrondissements " + list_to_string(districts_of_interest), zorder=4)
    
    # Customize the plot with Times New Roman font and size 15
    plt.xlabel("Longitude", fontname='Times New Roman', fontsize=15)
    plt.ylabel("Latitude", fontname='Times New Roman', fontsize=15)

    # Customize tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontsize(15)
    ax.legend(prop={'family': 'Times New Roman', 'size': 15})
    
    if do_save:
        plt.savefig("results/zones_" + list_to_string(districts_of_interest, "_"), bbox_inches='tight')
    plt.show()
    
    
# Define the intersects function
def intersects_target(row, target_districts):
    return target_districts.geometry.apply(lambda geom: geom.intersects(row.geometry)).any()
    
