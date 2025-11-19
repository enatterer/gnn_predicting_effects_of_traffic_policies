#!/usr/bin/env python3
"""
Visualize and tabulate finetuning vs training from scratch comparison results.

This script creates publication-ready visualizations and tables comparing:
- Finetuning from pretrained model vs training from scratch
- Different data availability scenarios (10/20/50 graphs per city)
- Multiple cities
- Metrics: R², Validation Loss (MSE), Spearman, Pearson correlation

Usage:
    python visualize_finetuning_results.py --results_dir <path> --output_dir <path>
    python visualize_finetuning_results.py --wandb_project <project_name> --output_dir <path>
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add scripts directory to path
scripts_path = Path(__file__).resolve().parents[1]
if str(scripts_path) not in sys.path:
    sys.path.append(str(scripts_path))

# Set style for publication-quality plots
try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    try:
        plt.style.use('seaborn-paper')
    except OSError:
        plt.style.use('seaborn-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'pdf.fonttype': 42,  # TrueType fonts for PDF
    'ps.fonttype': 42,
})


class ResultsCollector:
    """Collect results from various sources (WandB, local files, etc.)."""
    
    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        # Structure: results[data_availability][city][method][metric] = value
    
    def add_result(self, 
                   data_availability: int,
                   city: str,
                   method: str,  # 'finetune' or 'scratch'
                   metrics: Dict[str, float]):
        """Add a result entry."""
        self.results[data_availability][city][method] = metrics
    
    def from_wandb(self, project_name: str, api_key: Optional[str] = None):
        """Extract results from WandB. Requires wandb package."""
        try:
            import wandb
            if api_key:
                wandb.login(key=api_key)
            
            api = wandb.Api()
            runs = api.runs(project_name)
            
            for run in runs:
                run_name = run.name
                summary = run.summary
                config = run.config
                
                # Extract metrics (WandB uses 'r^2' not 'r2')
                metrics = {
                    'r2': summary.get('r^2', summary.get('r2', None)),
                    'val_loss': summary.get('best_val_loss', summary.get('val_loss', None)),
                    'spearman': summary.get('spearman', None),
                    'pearson': summary.get('pearson', None),
                }
                
                # Skip if essential metrics are missing
                if metrics['r2'] is None or metrics['val_loss'] is None:
                    continue
                
                # Determine method from run name
                # Format: finetune_{city}__parent-... or run_from_scratch_{city}__parent-...
                method = None
                if 'finetune' in run_name.lower() and 'scratch' not in run_name.lower():
                    method = 'finetune'
                elif 'run_from_scratch' in run_name.lower() or ('scratch' in run_name.lower() and 'finetune' not in run_name.lower()):
                    method = 'scratch'
                else:
                    continue  # Skip runs that don't match our pattern
                
                # Extract city from run name
                # Pattern: finetune_{city}__parent-... or run_from_scratch_{city}__parent-...
                city = None
                if method == 'finetune':
                    # Extract from "finetune_{city}__parent-..."
                    parts = run_name.split('__')
                    if len(parts) > 0:
                        city_part = parts[0].replace('finetune_', '')
                        city = city_part.strip()
                elif method == 'scratch':
                    # Extract from "run_from_scratch_{city}__parent-..."
                    parts = run_name.split('__')
                    if len(parts) > 0:
                        city_part = parts[0].replace('run_from_scratch_', '')
                        city = city_part.strip()
                
                # Fallback: try to get city from config
                if not city:
                    cities_str = config.get('cities', '')
                    if cities_str:
                        city = cities_str.split(',')[0].strip()
                
                # Get data availability from config
                data_avail = config.get('limit_train_graphs', None)
                if data_avail is None or data_avail == 0:
                    # Try alternative config keys
                    data_avail = config.get('finetune_limit_train_graphs', None)
                
                # Only add if we have all required information
                if city and data_avail and data_avail > 0 and method:
                    # Filter out None values from metrics but keep the entry if we have at least r2 and val_loss
                    filtered_metrics = {k: v for k, v in metrics.items() if v is not None}
                    if len(filtered_metrics) >= 2:  # At least r2 and val_loss
                        self.add_result(data_avail, city, method, filtered_metrics)
        
        except ImportError:
            print("Warning: wandb not installed. Install with: pip install wandb")
        except Exception as e:
            print(f"Error extracting from WandB: {e}")
            import traceback
            traceback.print_exc()
    
    def from_json(self, json_path: str):
        """Load results from a JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for entry in data:
            self.add_result(
                entry['data_availability'],
                entry['city'],
                entry['method'],
                entry['metrics']
            )
    
    def from_directory(self, results_dir: str, pattern: str = "*.json"):
        """Load results from a directory of JSON files."""
        results_path = Path(results_dir)
        for json_file in results_path.glob(pattern):
            self.from_json(str(json_file))
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        rows = []
        for data_avail, cities in self.results.items():
            for city, methods in cities.items():
                for method, metrics in methods.items():
                    row = {
                        'data_availability': data_avail,
                        'city': city,
                        'method': method,
                        **metrics
                    }
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def to_json(self, output_path: str):
        """Export results to JSON format."""
        json_data = []
        for data_avail, cities in self.results.items():
            for city, methods in cities.items():
                for method, metrics in methods.items():
                    json_data.append({
                        'data_availability': data_avail,
                        'city': city,
                        'method': method,
                        'metrics': metrics
                    })
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Exported results to {output_path}")


def create_val_loss_improvement_heatmap(results_df: pd.DataFrame, output_dir: str):
    """Create heatmap showing validation loss improvement percentage."""
    data_availabilities = sorted(results_df['data_availability'].unique())
    improvement_data = []
    
    for data_avail in data_availabilities:
        subset = results_df[results_df['data_availability'] == data_avail]
        cities = sorted(subset['city'].unique())
        
        for city in cities:
            city_data = subset[subset['city'] == city]
            finetune = city_data[city_data['method'] == 'finetune']['val_loss'].values
            scratch = city_data[city_data['method'] == 'scratch']['val_loss'].values
            
            if len(finetune) > 0 and len(scratch) > 0:
                # For loss, improvement is reduction (lower is better)
                improvement = ((scratch[0] - finetune[0]) / scratch[0]) * 100
                improvement_data.append({
                    'data_availability': f'{data_avail} graphs',
                    'city': city,
                    'improvement_pct': improvement
                })
    
    if improvement_data:
        df_improve = pd.DataFrame(improvement_data)
        pivot = df_improve.pivot(index='city', columns='data_availability', values='improvement_pct')
        
        plt.figure(figsize=(max(8, len(pivot.columns) * 2.5), max(6, len(pivot) * 0.5)))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGn', 
                   cbar_kws={'label': 'Validation Loss Reduction (%)'}, 
                   vmin=0, vmax=100, linewidths=0.5, linecolor='gray')
        plt.title('Validation Loss Reduction: Finetune vs Training from Scratch', 
                 fontweight='bold', fontsize=14, pad=20)
        plt.ylabel('City', fontweight='bold', fontsize=12)
        plt.xlabel('Data Availability (Number of Training Graphs)', fontweight='bold', fontsize=12)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'val_loss_improvement_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"Saved validation loss improvement heatmap to {output_path}")
    else:
        print("Warning: No improvement data found for validation loss heatmap")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and tabulate finetuning comparison results"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        help='Directory containing JSON result files'
    )
    parser.add_argument(
        '--results_json',
        type=str,
        help='Path to a single JSON file with results'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        help='WandB project name to extract results from'
    )
    parser.add_argument(
        '--wandb_api_key',
        type=str,
        default=None,
        help='WandB API key (optional, can use environment variable)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for visualizations and tables'
    )
    parser.add_argument(
        '--export_json',
        type=str,
        default=None,
        help='Export collected results to JSON file (useful for saving WandB results)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect results
    collector = ResultsCollector()
    
    if args.results_json:
        collector.from_json(args.results_json)
    elif args.results_dir:
        collector.from_directory(args.results_dir)
    elif args.wandb_project:
        collector.from_wandb(args.wandb_project, args.wandb_api_key)
    else:
        print("Error: Must provide --results_dir, --results_json, or --wandb_project")
        return
    
    # Convert to DataFrame
    results_df = collector.to_dataframe()
    
    if len(results_df) == 0:
        print("Error: No results found. Please check your input source.")
        return
    
    print(f"Loaded {len(results_df)} result entries")
    print(f"Cities: {sorted(results_df['city'].unique())}")
    print(f"Data availability scenarios: {sorted(results_df['data_availability'].unique())}")
    
    # Export to JSON if requested
    if args.export_json:
        collector.to_json(args.export_json)
    
    # Create validation loss improvement heatmap
    create_val_loss_improvement_heatmap(results_df, str(output_dir))
    
    print(f"\nOutput saved to {output_dir}")


if __name__ == '__main__':
    main()

