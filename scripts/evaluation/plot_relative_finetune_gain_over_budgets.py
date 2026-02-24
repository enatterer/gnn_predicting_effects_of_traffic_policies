#!/usr/bin/env python3
"""
Plot relative finetuning gain (% of scratch) over training budgets.

This script creates a plot showing how relative finetuning gain varies with
the number of target-city simulations (training budget) for multiple metrics.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import sys

# Add scripts directory to path
CURRENT_FILE = Path(__file__).resolve()
SCRIPTS_DIR = CURRENT_FILE.parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def compute_statistics_no_outliers(values: List[float]) -> Tuple[float, float]:
    """
    Compute mean and standard deviation WITHOUT removing outliers.
    
    Args:
        values: List of values
        
    Returns:
        Tuple of (mean, std)
    """
    if not values:
        return np.nan, np.nan
    
    values_arr = np.array(values)
    mean = np.mean(values_arr)
    std = np.std(values_arr, ddof=1) if len(values_arr) > 1 else 0.0
    return mean, std


def remove_top_5_percent(values: List[float]) -> List[float]:
    """
    Remove the top 5% highest values from a list.
    
    Args:
        values: List of values
        
    Returns:
        List with top 5% highest values removed
    """
    if not values or len(values) < 2:
        return values
    
    values_arr = np.array(values)
    # Calculate number of values to remove (top 5%)
    n_to_remove = max(1, int(np.ceil(len(values_arr) * 0.05)))
    
    # Sort and remove top n_to_remove values
    sorted_indices = np.argsort(values_arr)
    # Remove the highest n_to_remove values
    keep_indices = sorted_indices[:-n_to_remove]
    
    return values_arr[keep_indices].tolist()


def compute_statistics_remove_top_5_percent(values: List[float]) -> Tuple[float, float]:
    """
    Compute mean and standard deviation after removing top 5% highest values.
    
    Args:
        values: List of values
        
    Returns:
        Tuple of (mean, std)
    """
    if not values:
        return np.nan, np.nan
    
    filtered_values = remove_top_5_percent(values)
    
    if not filtered_values:
        return np.nan, np.nan
    
    values_arr = np.array(filtered_values)
    mean = np.mean(values_arr)
    std = np.std(values_arr, ddof=1) if len(values_arr) > 1 else 0.0
    return mean, std


def calculate_relative_gain(finetune_values: List[float], scratch_values: List[float], 
                           higher_is_better: bool = True) -> List[float]:
    """
    Calculate relative finetuning gain: (finetune - scratch) / scratch * 100
    
    Args:
        finetune_values: List of finetune metric values
        scratch_values: List of scratch metric values
        higher_is_better: If True, positive gain means finetune is better.
                         If False (e.g., for MSE), positive gain means finetune is better (lower is better).
    
    Returns:
        List of relative gains in percentage
    """
    if len(finetune_values) != len(scratch_values):
        raise ValueError(f"Mismatch: {len(finetune_values)} finetune vs {len(scratch_values)} scratch values")
    
    gains = []
    for finetune_val, scratch_val in zip(finetune_values, scratch_values):
        if scratch_val == 0 or np.isnan(finetune_val) or np.isnan(scratch_val):
            continue
        
        if higher_is_better:
            # For metrics where higher is better (R², Spearman, Pearson, hit rates)
            gain = ((finetune_val - scratch_val) / scratch_val) * 100
        else:
            # For metrics where lower is better (MSE/loss)
            # Positive gain means finetune is better (lower MSE)
            gain = ((scratch_val - finetune_val) / scratch_val) * 100
        
        gains.append(gain)
    
    return gains


def load_results_for_budget(results_dir: Path, budget: int, cities: List[str], 
                            seed_idxs: List[int]) -> Dict[str, Dict[str, List[float]]]:
    """
    Load finetune and scratch results for a given training budget.
    
    Args:
        results_dir: Base directory containing results
        budget: Training budget (total train+val simulations, e.g., 13, 25, 50, 100, 200)
        cities: List of city names
        seed_idxs: List of seed indices (e.g., [1, 2, 3, 4, 5])
    
    Returns:
        Dict mapping metric -> {"finetune": [values], "scratch": [values]}
    """
    # Map budget to train_val configs
    budget_to_configs = {
        13: (10, 3),
        25: (20, 5),
        50: (40, 10),
        100: (80, 20),
        200: (160, 40),
    }
    
    if budget not in budget_to_configs:
        raise ValueError(f"Unknown budget: {budget}. Supported: {list(budget_to_configs.keys())}")
    
    train_count, val_count = budget_to_configs[budget]
    
    results = defaultdict(lambda: {"finetune": [], "scratch": []})
    
    for city in cities:
        for seed_idx in seed_idxs:
            # Construct JSON filename
            seed = 42 + (seed_idx - 1)
            json_filename = (
                f"{city}_rs{seed_idx}_t{train_count}_v{val_count}_seed{seed}_"
                f"train{train_count}_val{val_count}_test100_random_metrics.json"
            )
            
            # Load finetune results
            finetune_run_name = f"{city}_finetune_rs_{seed_idx}_t{train_count}_v{val_count}"
            finetune_json_path = results_dir / finetune_run_name / "evaluation" / json_filename
            
            # Load scratch results
            scratch_run_name = f"{city}_scratch_rs_{seed_idx}_t{train_count}_v{val_count}"
            scratch_json_path = results_dir / scratch_run_name / "evaluation" / json_filename
            
            if finetune_json_path.exists() and scratch_json_path.exists():
                try:
                    with open(finetune_json_path, 'r') as f:
                        finetune_data = json.load(f)
                    
                    with open(scratch_json_path, 'r') as f:
                        scratch_data = json.load(f)
                except Exception as e:
                    print(f"  Warning: Error loading {finetune_json_path} or {scratch_json_path}: {e}")
                    continue
                
                # Extract metrics
                # MSE (loss)
                if 'loss' in finetune_data and 'loss' in scratch_data:
                    results['loss']['finetune'].append(finetune_data['loss'])
                    results['loss']['scratch'].append(scratch_data['loss'])
                
                # R²
                if 'r2' in finetune_data and 'r2' in scratch_data:
                    results['r2']['finetune'].append(finetune_data['r2'])
                    results['r2']['scratch'].append(scratch_data['r2'])
                
                # Spearman
                if 'spearman' in finetune_data and 'spearman' in scratch_data:
                    results['spearman']['finetune'].append(finetune_data['spearman'])
                    results['spearman']['scratch'].append(scratch_data['spearman'])
                
                # Pearson (handle both 'pearson' and 'pearman' typo)
                pearson_key = None
                if 'pearson' in finetune_data and 'pearson' in scratch_data:
                    pearson_key = 'pearson'
                elif 'pearman' in finetune_data and 'pearman' in scratch_data:
                    pearson_key = 'pearman'
                
                if pearson_key:
                    results['pearson']['finetune'].append(finetune_data[pearson_key])
                    results['pearson']['scratch'].append(scratch_data[pearson_key])
                
                # Hit rates
                if 'hit_rates' in finetune_data and 'hit_rates' in scratch_data:
                    finetune_hr = finetune_data['hit_rates']
                    scratch_hr = scratch_data['hit_rates']
                    
                    if 'top_1_hit_rate' in finetune_hr and 'top_1_hit_rate' in scratch_hr:
                        results['top_1_hit_rate']['finetune'].append(finetune_hr['top_1_hit_rate'])
                        results['top_1_hit_rate']['scratch'].append(scratch_hr['top_1_hit_rate'])
                    
                    if 'bottom_1_hit_rate' in finetune_hr and 'bottom_1_hit_rate' in scratch_hr:
                        results['bottom_1_hit_rate']['finetune'].append(finetune_hr['bottom_1_hit_rate'])
                        results['bottom_1_hit_rate']['scratch'].append(scratch_hr['bottom_1_hit_rate'])
    
    return dict(results)


def create_plot(all_budget_results: Dict[int, Dict[str, Dict[str, List[float]]]], 
                output_path: Path):
    """
    Create plot showing relative finetuning gain over budgets.
    
    Args:
        all_budget_results: Dict mapping budget -> metric -> {"finetune": [values], "scratch": [values]}
        output_path: Path to save the plot
    """
    budgets = sorted(all_budget_results.keys())
    
    # Define metrics and their properties
    metrics_config = [
        ('loss', 'MSE', False, '#1f77b4', 'o'),  # Blue, circle - lower is better
        ('spearman', 'Spearman', True, '#ff7f0e', 's'),  # Orange, square - higher is better
        ('r2', 'R²', True, '#2ca02c', '^'),  # Green, triangle - higher is better
        ('top_1_hit_rate', 'Top 1% hit rate', True, '#d62728', 'D'),  # Red, diamond - higher is better
        ('bottom_1_hit_rate', 'Bottom 1% hit rate', True, '#9467bd', 'v'),  # Purple, inverted triangle - higher is better
        ('pearman', 'Pearson', True, '#8c564b', 'X'),  # Brown, X - higher is better
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for metric_key, metric_label, higher_is_better, color, marker in metrics_config:
        if metric_key not in all_budget_results[budgets[0]]:
            continue
        
        gains_by_budget = []
        stds_by_budget = []
        
        for budget in budgets:
            if metric_key not in all_budget_results[budget]:
                gains_by_budget.append(np.nan)
                stds_by_budget.append(np.nan)
                continue
            
            finetune_values = all_budget_results[budget][metric_key]['finetune']
            scratch_values = all_budget_results[budget][metric_key]['scratch']
            
            if len(finetune_values) == 0 or len(scratch_values) == 0:
                gains_by_budget.append(np.nan)
                stds_by_budget.append(np.nan)
                continue
            
            # Calculate relative gains for all pairs
            relative_gains = calculate_relative_gain(finetune_values, scratch_values, higher_is_better)
            
            if len(relative_gains) == 0:
                gains_by_budget.append(np.nan)
                stds_by_budget.append(np.nan)
                continue
            
            # Compute mean and std after removing top 5% highest values (per metric and budget)
            mean_gain, std_gain = compute_statistics_remove_top_5_percent(relative_gains)
            gains_by_budget.append(mean_gain)
            stds_by_budget.append(std_gain)
        
        # Convert to numpy arrays
        gains_arr = np.array(gains_by_budget)
        stds_arr = np.array(stds_by_budget)
        
        # Create mask for valid data
        valid_mask = ~np.isnan(gains_arr)
        
        if np.any(valid_mask):
            # Plot shaded error band
            ax.fill_between(
                np.array(budgets)[valid_mask],
                gains_arr[valid_mask] - stds_arr[valid_mask],
                gains_arr[valid_mask] + stds_arr[valid_mask],
                color=color,
                alpha=0.2,
                linewidth=0,
                label='_nolegend_'
            )
            
            # Plot line with markers
            ax.plot(
                np.array(budgets)[valid_mask],
                gains_arr[valid_mask],
                marker=marker,
                color=color,
                label=metric_label,
                markersize=8,
                linewidth=1.5,
                markerfacecolor=color,
                markeredgecolor='white',
                markeredgewidth=1
            )
    
    # Formatting
    ax.set_xlabel("# of target-city simulations", fontsize=14)
    ax.set_ylabel("Mean relative finetuning gain (% of scratch)", fontsize=14)
    ax.set_xscale('log')
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ Saved plot to: {pdf_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot relative finetuning gain over training budgets"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory containing evaluation results (default: auto-detect)"
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="13,25,50,100,200",
        help="Comma-separated list of training budgets"
    )
    parser.add_argument(
        "--cities",
        type=str,
        default="regensburg,landshut,bayreuth,schweinfurt,wuerzburg,bamberg",
        help="Comma-separated list of cities"
    )
    parser.add_argument(
        "--seed_idxs",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated list of seed indices"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="scripts/evaluation/plots/relative_finetune_gain_over_budgets_remove_top5pct.png",
        help="Output path for the plot"
    )
    
    args = parser.parse_args()
    
    # Parse arguments
    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            results_dir = project_root / results_dir
    else:
        # Auto-detect: try multiple possible locations
        project_root = Path(__file__).resolve().parents[2]
        possible_dirs = [
            project_root / "data" / "inductive_gnn_data_results" / "transductive" / "Scratch_vs_Finetune",
            project_root / "evaluation_metrics" / "Scratch_vs_Finetune",
        ]
        results_dir = None
        for possible_dir in possible_dirs:
            if possible_dir.exists():
                results_dir = possible_dir
                break
        
        if results_dir is None:
            raise ValueError(f"Could not find results directory. Tried: {possible_dirs}")
    
    budgets = [int(b.strip()) for b in args.budgets.split(',')]
    cities = [c.strip() for c in args.cities.split(',')]
    seed_idxs = [int(s.strip()) for s in args.seed_idxs.split(',')]
    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        output_path = project_root / output_path
    
    print("=" * 80)
    print("PLOTTING RELATIVE FINETUNING GAIN OVER BUDGETS (REMOVE TOP 5% PER METRIC/BUDGET)")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Budgets: {budgets}")
    print(f"Cities: {cities}")
    print(f"Seed indices: {seed_idxs}")
    print("=" * 80)
    
    # Load results for each budget
    all_budget_results = {}
    for budget in budgets:
        print(f"\nLoading results for budget {budget}...")
        budget_results = load_results_for_budget(results_dir, budget, cities, seed_idxs)
        all_budget_results[budget] = budget_results
        
        # Print summary
        for metric_key in budget_results:
            finetune_count = len(budget_results[metric_key]['finetune'])
            scratch_count = len(budget_results[metric_key]['scratch'])
            print(f"  {metric_key}: {finetune_count} finetune, {scratch_count} scratch values")
    
    # Create plot
    print("\nCreating plot...")
    create_plot(all_budget_results, output_path)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
