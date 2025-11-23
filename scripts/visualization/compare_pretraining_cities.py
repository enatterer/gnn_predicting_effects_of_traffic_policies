#!/usr/bin/env python3
"""
Script to compare model performance across different pretraining scenarios (2, 6, 12 cities)
for each city and metric.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Define directories
BASE_DIR = Path("/home/enatterer/Development/elena_gnn_predicting_effects_of_traffic_policies/data")
DIRS = {
    "2_cities": BASE_DIR / "analysis_results_2_cities_in_training",
    "6_cities": BASE_DIR / "analysis_results_6_cities_in_training",
    "12_cities": BASE_DIR / "analysis_results_12_cities_in_training"
}

# File patterns for each directory
FILE_PATTERNS = {
    "2_cities": "evaluation_general_surrogate_2_cities_{city}.json",
    "6_cities": "evaluation_general_surrogate_6_cities_{city}.json",
    "12_cities": "evaluation_general_surrogate_v0_{city}.json"
}

# Metrics to plot (excluding metadata fields)
METRICS = [
    "val_loss",
    "r_squared",
    "spearman",
    "pearson",
    "top_1_hit_rate",
    "bottom_1_hit_rate",
    "top_5_hit_rate",
    "bottom_5_hit_rate",
    "top_10_hit_rate",
    "bottom_10_hit_rate"
]

def load_city_data(city_name):
    """Load data for a city from all three pretraining scenarios."""
    data = {}
    
    for scenario, dir_path in DIRS.items():
        pattern = FILE_PATTERNS[scenario].format(city=city_name)
        file_path = dir_path / pattern
        
        if file_path.exists():
            with open(file_path, 'r') as f:
                content = json.load(f)
                # Handle both single dict and list of dicts
                if isinstance(content, list):
                    if len(content) > 0:
                        data[scenario] = content[0]
                    else:
                        data[scenario] = None
                else:
                    data[scenario] = content
        else:
            data[scenario] = None
    
    return data

def find_all_cities():
    """Find all cities that have at least one result file."""
    cities = set()
    
    for scenario, dir_path in DIRS.items():
        if dir_path.exists():
            for file in dir_path.glob("*.json"):
                # Extract city name from filename
                if scenario == "2_cities":
                    if "all_cities" not in file.name:
                        city = file.name.replace("evaluation_general_surrogate_2_cities_", "").replace(".json", "")
                        cities.add(city)
                elif scenario == "6_cities":
                    if "all_cities" not in file.name:
                        city = file.name.replace("evaluation_general_surrogate_6_cities_", "").replace(".json", "")
                        cities.add(city)
                elif scenario == "12_cities":
                    if "all_cities" not in file.name:
                        city = file.name.replace("evaluation_general_surrogate_v0_", "").replace(".json", "")
                        cities.add(city)
    
    return sorted(cities)

def create_bar_charts_for_city(city_name, output_dir):
    """Create bar charts for all metrics for a given city."""
    data = load_city_data(city_name)
    
    # Check if we have any data
    if all(v is None for v in data.values()):
        print(f"No data found for city: {city_name}")
        return
    
    # Create a figure with subplots for each metric
    n_metrics = len(METRICS)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        
        # Prepare data for bar chart
        scenarios = []
        values = []
        colors = []
        
        color_map = {'2_cities': '#1f77b4', '6_cities': '#ff7f0e', '12_cities': '#2ca02c'}
        
        for scenario in ["2_cities", "6_cities", "12_cities"]:
            if data[scenario] is not None and metric in data[scenario]:
                scenarios.append(scenario.replace("_", " ").title())
                values.append(data[scenario][metric])
                colors.append(color_map[scenario])
            else:
                scenarios.append(scenario.replace("_", " ").title())
                values.append(0)  # Use 0 for missing data
                colors.append('#cccccc')  # Gray for missing data
        
        # Create bar chart
        bars = ax.bar(scenarios, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, val, scenario) in enumerate(zip(bars, values, ["2_cities", "6_cities", "12_cities"])):
            if data[scenario] is not None and metric in data[scenario]:
                height = bar.get_height()
                # Format based on metric type
                if abs(val) < 0.01:
                    label = f'{val:.4f}'
                elif abs(val) < 1:
                    label = f'{val:.3f}'
                elif abs(val) < 100:
                    label = f'{val:.2f}'
                else:
                    label = f'{val:.1f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label,
                       ha='center', va='bottom', fontsize=9, rotation=0)
            else:
                # Show "N/A" for missing data
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       'N/A',
                       ha='center', va='bottom', fontsize=9, style='italic', color='gray')
        
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_title(f'{city_name.title()} - {metric.replace("_", " ").title()}', 
                    fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Set y-axis limits - only consider non-zero values (non-missing data)
        valid_values = [v for i, v in enumerate(values) 
                       if data[["2_cities", "6_cities", "12_cities"][i]] is not None 
                       and metric in data[["2_cities", "6_cities", "12_cities"][i]]]
        if valid_values:
            if min(valid_values) >= 0:
                ax.set_ylim(bottom=0)
            # Let matplotlib auto-scale otherwise
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"{city_name}_metrics_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def create_summary_table(all_cities_data, output_dir):
    """Create a summary table/CSV with all the data."""
    import pandas as pd
    
    rows = []
    for city in all_cities_data:
        for scenario in ["2_cities", "6_cities", "12_cities"]:
            if all_cities_data[city][scenario] is not None:
                row = {
                    "city": city,
                    "pretraining_scenario": scenario.replace("_", " ").title(),
                    **{metric: all_cities_data[city][scenario].get(metric, None) 
                       for metric in METRICS}
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    output_file = output_dir / "summary_comparison.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved summary table: {output_file}")

def main():
    """Main function to generate all visualizations."""
    # Create output directory
    output_dir = BASE_DIR.parent / "scripts" / "visualization" / "results" / "metric_comparisons_depending_on_available_training_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all cities
    cities = find_all_cities()
    print(f"Found cities: {cities}")
    
    # Load all data
    all_cities_data = {}
    for city in cities:
        all_cities_data[city] = load_city_data(city)
        print(f"\n{city}:")
        for scenario in ["2_cities", "6_cities", "12_cities"]:
            if all_cities_data[city][scenario] is not None:
                print(f"  {scenario}: ✓")
            else:
                print(f"  {scenario}: ✗")
    
    # Create bar charts for each city
    print("\n" + "="*50)
    print("Creating bar charts...")
    print("="*50)
    for city in cities:
        create_bar_charts_for_city(city, output_dir)
    
    # Create summary table
    print("\n" + "="*50)
    print("Creating summary table...")
    print("="*50)
    create_summary_table(all_cities_data, output_dir)
    
    print("\n" + "="*50)
    print("Done! All visualizations saved to:", output_dir)
    print("="*50)

if __name__ == "__main__":
    main()

