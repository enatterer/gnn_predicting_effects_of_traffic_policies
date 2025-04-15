
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.lines import Line2D
from shapely.geometry import box, Polygon
from shapely.ops import unary_union

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

districts = gpd.read_file(os.path.join(project_root, "data", "visualisation", "districts_paris.geojson"))

def plot_combined_output(gdf_input: gpd.GeoDataFrame, column_to_plot: str, font: str = 'Times New Roman', 
                         save_it: bool = False, number_to_plot: int = 0,
                         zone_to_plot:str= "this_zone",
                         is_predicted: bool = False, alpha:int=100, 
                         use_fixed_norm:bool=True, 
                         fixed_norm_max: int= 10, known_districts:bool=False, buffer: float = 0.0005, 
                         districts_of_interest: list =[1, 2, 3, 4],
                         plot_contour_lines:bool=False, 
                         plot_policy_roads:bool=False,
                         is_absolute:bool=False,
                         cmap:str='RdBu',  # Changed default from 'coolwarm' to 'RdBu'
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
                                     discrete_thresholds: list = None, 
                                     error_threshold: float = 100,
                                     cmap: str = 'RdYlGn_r'):  # Changed default to RdYlGn_r
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
    cmap : str, optional
        Colormap to use. Recommended options:
        - 'RdYlGn_r' (default, green->yellow->red)
        - 'viridis_r' (blue/green->yellow)
        - Custom colormap can be created for specific needs
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))    
    
    # If using absolute values, we might want to create a custom colormap
    if use_absolute_value_of_difference and cmap == 'RdYlGn_r':
        # Create custom colormap: green -> yellow -> red
        colors = ['#1a9850',  # green
                 '#91cf60',   # light green
                 '#d9ef8b',   # yellow-green
                 '#fee08b',   # light yellow
                 '#fc8d59',   # orange
                 '#d73027']   # red
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom_div', colors)

    base_gdf = gdf_inputs[0].copy()
    
    all_errors = []
    for gdf in gdf_inputs:
        if use_percentage:
            epsilon = 1e-10
            if use_absolute_value_of_difference:
                if loss_fct == "l1":
                    error = abs((gdf['vol_car_change_predicted'] - gdf['vol_car_change_actual']) / 
                            (abs(gdf['vol_car_change_actual'] + gdf['vol_base_case'] + epsilon))) * 100
                elif loss_fct == "mse":
                    error = ((gdf['vol_car_change_predicted'] - gdf['vol_car_change_actual']).pow(2) / 
                            (abs(gdf['vol_car_change_actual'] + gdf['vol_base_case']) + epsilon)) * 100
            else:
                if loss_fct == "l1":
                    error = (gdf['vol_car_change_predicted'] - gdf['vol_car_change_actual']) / (gdf['vol_car_change_actual'] + gdf['vol_base_case'] + epsilon)* 100
                elif loss_fct == "mse":
                    error = (gdf['vol_car_change_predicted'] - gdf['vol_car_change_actual']).pow(2) / (gdf['vol_car_change_actual'] + gdf['vol_base_case'] + epsilon) * 100
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
            
        def categorize_value(x, thresholds):
            for i, threshold in enumerate(thresholds):
                if abs(x) <= threshold:
                    return i
            return len(thresholds)  # for values greater than the last threshold
        
        base_gdf['discrete_error'] = base_gdf['mean_prediction_error'].apply(
            lambda x: categorize_value(x, discrete_thresholds))
        
        # Create a color gradient based on number of categories
        n_categories = len(discrete_thresholds) + 1
        
        # Get colors from the provided colormap
        base_cmap = plt.cm.get_cmap(cmap)
        if use_absolute_value_of_difference:
            # For absolute values, use evenly spaced colors from the colormap
            colors = base_cmap(np.linspace(0, 1, n_categories))
        else:
            # For signed differences, center around 0.5
            center_idx = n_categories // 2
            if n_categories % 2 == 0:
                # Even number of categories
                spacing = 1.0 / (n_categories - 1)
                color_positions = np.linspace(0, 1, n_categories)
            else:
                # Odd number of categories, center middle category at 0.5
                spacing = 1.0 / (n_categories)
                color_positions = np.linspace(0, 1, n_categories)
            colors = base_cmap(color_positions)
        
        custom_cmap = plt.cm.colors.ListedColormap(colors)
        norm = plt.Normalize(vmin=-0.5, vmax=n_categories - 0.5)
        plot_column = 'discrete_error'
        cmap = custom_cmap
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
        
        # For continuous scale, use appropriate colormap
        if use_absolute_value_of_difference and cmap == 'RdYlGn_r':
            colors = [
                '#2ecc71',  # green
                '#a8e6cf',  # light green
                '#f1c40f',  # yellow
                '#e67e22',  # orange
                '#e74c3c'   # red
            ]
            cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom_div', colors)
    
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
    
    error_type ="Relative" if use_percentage else "Absolute"
    # error_type = "Absolute" if use_absolute_value_of_difference else "Signed"
    units =  "%" if use_percentage else "vehicles"
    
    cbar.set_label(f'{error_type} Prediction Error ({units})' ,
                #    f'{loss_fct} difference in {units}\n',
                #    f'(Averaged across {len(gdf_inputs)} samples)',
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
    
def create_correlation_radar_plot_sort_by_r2(metrics_by_type, selected_metrics=None, result_path=None, save_it=False, selected_types=None, colors=None  ):
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
                'transform': lambda x: max(0, x),
                'y_pos': -0.05
            },
            {
                'id': 'l1_ratio',
                'label': '1 - MAE/Naive MAE',
                'transform': lambda x, max_ratio: (1 - x/max_ratio),
                'y_pos': -0.1
            },
            {
                'id': 'pearson',
                'label': 'Pearson\nCorrelation',
                'transform': lambda x: max(0, x),
                'y_pos': -0.05
            },
            {
                'id': 'spearman',
                'label': 'Spearman\nCorrelation',
                'transform': lambda x: max(0, x),
                'y_pos': -0.05
            },
            {
                'id': 'error_distribution',
                'label': 'Error\nDistribution',
                'transform': lambda x: max(0, (x - 68) / (100 - 68)),
                'y_pos': -0.05
            }
        ]
    
    # Select specific road types
    if selected_types is None:
        selected_types = [
            'Trunk Roads',
            'Primary Roads',
            'Secondary Roads',
            'Tertiary Roads',
            'Residential Streets'
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
            elif metric['id'] == 'error_distribution':
                val = filtered_metrics[road_type].get('error_distribution', 68)  # Default to 68 if not found
                values.append(metric['transform'](val))
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
    
    # Legend
    ax.legend(loc='center left', 
            bbox_to_anchor=(1.1, 0.5),
            fontsize=15,
            markerscale=2,
            frameon=True,
            framealpha=0.9,
            edgecolor='gray',
            borderpad=1,
            labelspacing=1.2,
            handlelength=3)
    
    if save_it:
        plt.savefig(result_path + "radar_plot.png", bbox_inches='tight', dpi=300)
        
    plt.show()

def create_error_vs_variability_scatterplots(metrics_by_type, result_path=None, save_it=False, selected_types=None, colors=None):
    """
    Create scatter plot for MSE vs Variance with blue best-fit line and larger labels
    """
    plt.rcParams["font.family"] = "Times New Roman"
    
    if selected_types is None:
        selected_types = [
            'All Roads',
            'Trunk Roads',
            'Primary Roads',
            'Secondary Roads',
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
    
    # ax.plot(x_line, p(x_line), '--', color='#0277bd', alpha=0.8, linewidth=2, 
    #         label=r'Best fit line for $\sigma^2_{t,b}$ vs. MSE')  # Added label with math formatting
    ax.plot(x_line, p(x_line), '--', color='#0277bd', alpha=0.8, linewidth=2, 
            label=r'Best fit line for variance in the base case vs. MSE')
    
    # ax.plot(x_line, x_line, '--', color='gray', alpha=0.9, linewidth=2, 
    #         label=r'$\sigma^2_{t,b}$ (lower bound for MSE)')  # Updated label for clarity
    ax.plot(x_line, x_line, '--', color='gray', alpha=0.9, linewidth=2, 
        label='Estimation of impact of stochasticity on MSE')  # Updated label for clarity
    
    # Plot points
    for i, rt in enumerate(selected_types):
        ax.scatter(variance_values[i], mse_values[i], color=colors[rt], s=150, label='_nolegend_')
        ax.annotate(rt, (variance_values[i], mse_values[i]), 
                   xytext=(10, 10), textcoords='offset points', fontsize=16)
    
    # Labels with larger font size
    # ax.set_xlabel(r'$\sigma^2_{t,b}$', fontsize=16)
    ax.set_xlabel('Variance in the Base Case', fontsize=16)
    ax.set_ylabel('MSE', fontsize=16)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Tick labels
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add correlation coefficient - moved to lower left corner
    # pearson_mse_var = stats.pearsonr(variance_values, mse_values)[0]
    # ax.text(0.02, 0.02, f'Pearson r = {pearson_mse_var:.2f}', 
    #         transform=ax.transAxes, 
    #         verticalalignment='bottom',  # Changed from 'top' to 'bottom'
    #         fontsize=16)
    
    # Add legend
    ax.legend(fontsize=14, frameon=True)
    
    # # Add legend - explicitly positioned in upper right
    # ax.legend(fontsize=14, frameon=True, 
    #          loc='upper right',
    #          bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    if save_it:
        plt.savefig(result_path + "error_vs_variability_scatterplot_with_line.png", bbox_inches='tight', dpi=300)
    plt.show()

def plot_prediction_errors_discrete(
    gdf_inputs: list,
    discrete_thresholds: list,
    cmap_name: str = 'RdYlGn_r',
    use_percentage: bool = False,
    save_path: str = None,
    font: str = 'Times New Roman',
    highlight_disagreement: bool = False,
    disagreement_threshold: float = None
):
    """
    Creates a map visualization of prediction errors using discrete color categories.
    
    Parameters
    ----------
    gdf_inputs : list
        List of GeoDataFrames, each containing predictions from one model
    discrete_thresholds : list
        Thresholds for error categories (e.g., [5, 10, 20] creates categories 0-5, 5-10, 10-20, >20)
    cmap_name : str
        Name of the matplotlib colormap to use (e.g., 'RdYlGn_r', 'viridis_r')
    use_percentage : bool
        If True, display errors as percentages
    save_path : str, optional
        If provided, save the plot to this path
    font : str
        Font to use for text elements
    highlight_disagreement : bool
        If True, highlight areas where models disagree significantly
    disagreement_threshold : float
        CV threshold for highlighting disagreement (only used if highlight_disagreement is True)
    """
    # Setup
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    base_gdf = gdf_inputs[0].copy()
    
    # Calculate errors and statistics
    all_errors = []
    for gdf in gdf_inputs:
        error = abs(gdf['vol_car_change_predicted'] - gdf['vol_car_change_actual'])
        if use_percentage:
            error = error / (abs(gdf['vol_car_change_actual'] + gdf['vol_base_case']) + 1e-10) * 100
        all_errors.append(error)
    
    errors_df = pd.concat(all_errors, axis=1)
    base_gdf['mean_error'] = errors_df.mean(axis=1)
    base_gdf['std_error'] = errors_df.std(axis=1)
    
    # Create discrete categories
    def assign_category(x, thresholds):
        for i, threshold in enumerate(thresholds):
            if abs(x) <= threshold:
                return i
        return len(thresholds)
    
    base_gdf['error_category'] = base_gdf['mean_error'].apply(
        lambda x: assign_category(x, discrete_thresholds))
    
    # Set up colors
    n_categories = len(discrete_thresholds) + 1
    base_cmap = plt.cm.get_cmap(cmap_name)
    colors = base_cmap(np.linspace(0, 1, n_categories))
    category_cmap = plt.cm.colors.ListedColormap(colors)
    
    # Prepare data for plotting
    base_gdf, x_min, y_min, x_max, y_max = filter_for_geographic_section(base_gdf)
    base_gdf['linewidth'] = base_gdf["highway"].apply(get_linewidth)
    
    # Plot roads
    norm = plt.Normalize(vmin=-0.5, vmax=n_categories - 0.5)
    for width in sorted(base_gdf['linewidth'].unique()):
        mask = base_gdf['linewidth'] == width
        base_gdf[mask].plot(
            column='error_category',
            cmap=category_cmap,
            linewidth=width,
            ax=ax,
            legend=False,
            norm=norm,
            zorder=int(width)
        )
    
    # Add disagreement highlights if requested
    if highlight_disagreement and disagreement_threshold is not None:
        cv = base_gdf['std_error'] / (abs(base_gdf['mean_error']) + 1e-10)
        disagreement_mask = cv > disagreement_threshold
        if disagreement_mask.any():
            disagreement_lines = base_gdf[disagreement_mask]
            disagreement_lines.plot(
                color='none',
                edgecolor='black',
                linewidth=disagreement_lines['linewidth']*1.5,
                linestyle='--',
                ax=ax,
                zorder=10
            )
            legend_line = Line2D([], [], color='black', linestyle='--',
                               label=f'High model disagreement (CV > {disagreement_threshold})')
            ax.legend(handles=[legend_line], fontsize=12)
    
    # Style the plot
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Longitude", fontname=font, fontsize=15)
    ax.set_ylabel("Latitude", fontname=font, fontsize=15)
    
    # Style the axes
    ax.tick_params(axis='both', which='major', labelsize=15)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(font)
    
    # Add colorbar
    ax.set_position([0.1, 0.1, 0.75, 0.75])
    cax = fig.add_axes([0.87, 0.22, 0.03, 0.5])
    sm = plt.cm.ScalarMappable(cmap=category_cmap, norm=norm)
    cbar = plt.colorbar(sm, cax=cax, ticks=range(n_categories))
    
    # Create category labels
    labels = []
    for i in range(n_categories):
        if i == 0:
            labels.append(f'0-{discrete_thresholds[0]}')
        elif i == n_categories - 1:
            labels.append(f'>{discrete_thresholds[-1]}')
        else:
            labels.append(f'{discrete_thresholds[i-1]}-{discrete_thresholds[i]}')
    
    # Style the colorbar
    cbar.ax.set_yticklabels(labels, fontsize=15, font=font)
    units = "%" if use_percentage else "vehicles"
    cbar.set_label(f'Prediction Error ({units})', fontname=font, fontsize=15)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    return base_gdf