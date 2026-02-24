#!/usr/bin/env python3
"""
Evaluate varying pretrain sources experiment results.

This script loads models, evaluates them on test splits, and creates visualizations
comparing finetuned vs scratch models across different numbers of pretraining cities.

For i in [1, 4]: Uses runs from run_varying_pretrain_sources_regensburg.py
For i = 5: Uses runs from Scratch_vs_Finetune project (regensburg_scratch_rs_* and regensburg_finetune_rs_*)

All evaluations use the random test set (as specified in spatial_maps.ipynb).

Example usage:
    nohup python scripts/evaluation/evaluate_varying_pretrain_sources.py \
        --project_name VaryingPretrainSources \
        --all_cities_project_name GNN_Transductive \
        --train_val_size '40:10' --seed_idx 1 > output_evaluation.log 2>&1 &
"""

import argparse
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch

# Add scripts directory to path
CURRENT_FILE = Path(__file__).resolve()
SCRIPTS_DIR = CURRENT_FILE.parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from training.help_functions import (
    prepare_data_with_graph_features,
    set_cuda_visible_device,
    str_to_bool,
)
from gnn.help_functions import GNN_Loss, validate_model_during_training
from training import run_models as run_models_module
from evaluation.evaluate_pretrained_on_cities import load_model_from_checkpoint

# Base directory for results
BASE_DIR = Path(run_models_module.base_dir).resolve()

# Target city
TARGET_CITY = "regensburg"


def extract_num_pretrain_cities(run_name: str) -> Optional[int]:
    """
    Extract number of pretraining cities from run name.
    
    For run_varying_pretrain_sources_regensburg.py:
    - Pattern: regensburg_finetune_n{i}_c{j}_* or regensburg_scratch_n{i}_c{j}_*
    - Returns i (1-4)
    
    For i=5 runs (from Scratch_vs_Finetune project):
    - Pattern: regensburg_scratch_rs_{seed_idx}_t{train}_v{val} or regensburg_finetune_rs_{seed_idx}_t{train}_v{val}
    - These use all other cities for pretraining, so count = 5
    """
    # Check for n{i} pattern (i=1-4)
    match = re.search(r'_n(\d+)_', run_name)
    if match:
        return int(match.group(1))
    
    # Check for Scratch_vs_Finetune patterns (i=5)
    # Pattern: regensburg_scratch_rs_* or regensburg_finetune_rs_*
    # Only accept the specific pattern with rs_*, t*, v* format
    if re.match(r'regensburg_(scratch|finetune)_rs_\d+_t\d+_v\d+', run_name):
        # These use all other cities for pretraining, so count = 5
        return 5
    
    return None


def is_finetune_run(run_name: str) -> bool:
    """Check if run is a finetune run (vs scratch)."""
    return "finetune" in run_name and "scratch" not in run_name


def is_scratch_run(run_name: str) -> bool:
    """Check if run is a scratch run."""
    # Pattern: regensburg_scratch_rs_* or run_from_scratch_*
    return "scratch" in run_name or run_name.startswith("run_from_scratch_")


def normalize_path_in_split(path_str: str, project_root: Path) -> str:
    """
    Convert absolute paths from other users/machines to relative paths.
    Minimal change: just extract the relative part after the project name.
    """
    path_str = str(path_str)
    
    # If already a relative path, return as-is
    if not os.path.isabs(path_str):
        return path_str
    
    # Extract relative part from common absolute path patterns
    # Pattern 1: /home/rrao/development/gnn_predicting_effects_of_traffic_policies/...
    if 'gnn_predicting_effects_of_traffic_policies/' in path_str:
        rel_part = path_str.split('gnn_predicting_effects_of_traffic_policies/')[1]
        return rel_part
    
    # Pattern 2: /mnt/repo/... (LRZ paths)
    if path_str.startswith('/mnt/repo/'):
        return path_str.replace('/mnt/repo/', '')
    
    # Pattern 3: Extract data/... part if present
    if '/data/' in path_str:
        rel_part = 'data/' + path_str.split('/data/')[1]
        return rel_part
    
    # Last resort: try to make relative to project_root
    try:
        rel_path = os.path.relpath(path_str, project_root)
        if not rel_path.startswith('..'):
            return rel_path
    except ValueError:
        pass
    
    # Return original if can't convert (will error later)
    return path_str


def normalize_split_paths(split_data: dict, project_root: Path) -> dict:
    """
    Normalize all paths in a split file to relative paths.
    """
    def normalize_path_list(path_list):
        return [normalize_path_in_split(p, project_root) for p in path_list]
    
    # Normalize train_data paths
    if 'train_data' in split_data and 'path' in split_data['train_data']:
        split_data['train_data']['path'] = normalize_path_list(split_data['train_data']['path'])
    
    # Normalize val_data paths
    if 'val_data' in split_data and 'path' in split_data['val_data']:
        split_data['val_data']['path'] = normalize_path_list(split_data['val_data']['path'])
    
    # Normalize test_data paths
    if 'test_data' in split_data and 'path' in split_data['test_data']:
        split_data['test_data']['path'] = normalize_path_list(split_data['test_data']['path'])
    
    # Normalize train_paths and val_paths if they exist
    if 'train_paths' in split_data:
        split_data['train_paths'] = normalize_path_list(split_data['train_paths'])
    
    if 'val_paths' in split_data:
        split_data['val_paths'] = normalize_path_list(split_data['val_paths'])
    
    return split_data


def find_model_path(run_dir: Path) -> Optional[Path]:
    """Find the model checkpoint file."""
    model_path = run_dir / "finetuned_model" / "model.pth"
    if model_path.exists():
        return model_path
    return None


def evaluate_model_on_test_split(
    run_name: str,
    project_name: str,
    model_path: Path,
    split_file_path: Path,
    eval_params: Dict[str, object],
) -> Optional[Dict[str, object]]:
    """Load a trained model and evaluate it on the test split."""
    if not split_file_path.exists():
        print(f"  ✗ Test split missing: {split_file_path}")
        return None
    if not model_path.exists():
        print(f"  ✗ Model file missing: {model_path}")
        return None

    with open(split_file_path, "r") as f:
        split_data = json.load(f)

    # Normalize paths in split file (convert absolute paths from other users/machines to relative paths)
    project_root = Path(__file__).resolve().parents[2]
    split_data = normalize_split_paths(split_data, project_root)

    test_data = split_data.get("test_data") or {}
    if not test_data.get("path"):
        print(f"  ✗ Split file has no test_data paths: {split_file_path}")
        return None

    train_data = split_data.get("train_data") or {}
    val_data = split_data.get("val_data") or {}

    set_cuda_visible_device(eval_params["device_nr"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve run_dir - check both possible locations
    project_root = Path(__file__).resolve().parents[2]
    run_dir_main = BASE_DIR / project_name / run_name
    run_dir_alt = project_root / "data" / "inductive_gnn_data_results" / "transductive" / project_name / run_name
    
    if run_dir_main.exists():
        run_dir = run_dir_main
    elif run_dir_alt.exists():
        run_dir = run_dir_alt
    else:
        print(f"  ✗ Run directory not found: {run_name}")
        print(f"    Checked: {run_dir_main}")
        print(f"    Checked: {run_dir_alt}")
        return None
    
    # Load model
    model, inferred_config = load_model_from_checkpoint(
        checkpoint_path=model_path,
        run_dir=run_dir,
        gnn_arch=eval_params["gnn_arch"],
        device=device,
    )
    model = model.to(device)

    config_dict = {
        "target_type": eval_params["target_type"],
        "target_normalization": eval_params.get("target_normalization"),
        "use_all_features": eval_params["use_all_features"],
        "use_destination_activity": inferred_config.get("use_destination_activity", False),
    }
    for key, value in inferred_config.items():
        config_dict.setdefault(key, value)
    config_obj = SimpleNamespace(**config_dict)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        _, _, test_loader = prepare_data_with_graph_features(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            use_inductive_variant=False,
            batch_size=eval_params["batch_size"],
            path_to_save_dataloader=str(tmp_dir) + "/",
            use_all_features=config_obj.use_all_features,
            use_weighted_batches=False,
            use_nested_neighbor_loader=False,
            neighbor_sizes=eval_params["neighbor_sizes"],
            subgraphs_per_graph=eval_params["subgraphs_per_graph"],
            seed_size=eval_params["seed_size"],
            sampling_strategy=eval_params["sampling_strategy"],
            min_subgraph_nodes=eval_params["min_subgraph_nodes"],
            max_subgraph_nodes=eval_params["max_subgraph_nodes"],
            aug_pos_rotation=False,
            aug_feature_noise=False,
            aug_node_masking_probability=0.0,
            use_destination_activity_param=config_obj.use_destination_activity,
            return_test_loader=True,
            x_scaler_path=run_dir / "data_created_during_finetuning" / "train_x_scaler.pkl"
        )

        if test_loader is None or len(test_loader) == 0:
            print(f"  ✗ No test graphs available")
            return None
        else:
            print(f"  ✓ Test loader ready — graphs: {len(test_data.get('path', []))}, batches: {len(test_loader)}")

        loss_func = GNN_Loss(loss_fct="mse", device=device, weighted=False)
        loss, r2, spearman, pearson, hit_rates = validate_model_during_training(
            config=config_obj,
            model=model,
            dataset=test_loader,
            loss_func=loss_func,
            device=device,
        )

    # Convert paths to relative paths (project_root already defined above)
    try:
        model_path_rel = model_path.relative_to(project_root)
    except ValueError:
        model_path_rel = model_path
    
    try:
        split_file_path_rel = split_file_path.relative_to(project_root)
    except ValueError:
        split_file_path_rel = split_file_path
    
    metrics = {
        "run_name": run_name,
        "project_name": project_name,
        "model_path": str(model_path_rel),
        "split_file": str(split_file_path_rel),
        "city": split_data.get("city"),
        "test_graphs": len(test_data.get("path", [])),
        "loss": float(loss),
        "r2": float(r2),
        "spearman": float(spearman),
        "pearman": float(pearson),
        "hit_rates": {k: float(v) for k, v in (hit_rates or {}).items()},
    }

    output_dir = run_dir / "evaluation"
    output_path = output_dir / f"{split_file_path.stem}_metrics.json"
    
    # Try to save to original location first
    saved = False
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Check if we can write to the directory
        if output_dir.exists() and not os.access(output_dir, os.W_OK):
            raise PermissionError(f"Cannot write to evaluation directory '{output_dir}': permission denied")
        
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  ✓ Saved test metrics to: {output_path}")
        saved = True
    except (PermissionError, OSError) as e:
        # Try fallback location in project root where user likely has permissions
        fallback_dir = project_root / "evaluation_metrics" / project_name / run_name
        fallback_path = fallback_dir / f"{split_file_path.stem}_metrics.json"
        
        try:
            fallback_dir.mkdir(parents=True, exist_ok=True)
            with open(fallback_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  ⚠️  Warning: Could not save to '{output_path}' (permission denied)")
            print(f"  ✓ Saved test metrics to fallback location: {fallback_path}")
            saved = True
        except (PermissionError, OSError) as e:
            # Even fallback failed - just log warning but continue
            print(f"  ⚠️  Warning: Cannot save metrics file (permission denied at both locations)")
            print(f"     Original: {output_path}")
            print(f"     Fallback: {fallback_path}")
            print(f"     Metrics computed but not saved. Results will still be included in statistics.")
    
    return metrics


def collect_and_evaluate_runs(
    project_name: str,
    target_city: str,
    split_file_path: Path,
    eval_params: Dict[str, object],
    skip_if_exists: bool = False
) -> Dict[int, Dict[str, List[float]]]:
    """
    Collect metrics by evaluating runs, grouped by number of pretraining cities.
    
    Returns:
        Dict mapping num_cities -> {"finetuned": [loss1, loss2, ...], "scratch": [loss1, loss2, ...]}
    """
    # Resolve project directory - check both possible locations
    project_root = Path(__file__).resolve().parents[2]
    
    # Try main location first
    project_dir = BASE_DIR / project_name
    if not project_dir.exists():
        # Try alternative location in data/ subdirectory (for Scratch_vs_Finetune)
        alt_project_dir = project_root / "data" / "inductive_gnn_data_results" / "transductive" / project_name
        if alt_project_dir.exists():
            print(f"  Note: Using alternative path: {alt_project_dir}")
            project_dir = alt_project_dir
        else:
            print(f"Warning: Project directory not found: {project_dir}")
            print(f"  Also checked: {alt_project_dir}")
            return {}
    
    # Convert to relative path for consistency
    try:
        project_dir_rel = project_dir.relative_to(project_root)
        print(f"  Using project directory (relative): {project_dir_rel}")
    except ValueError:
        # If can't make relative, use absolute but log it
        print(f"  Using project directory (absolute): {project_dir}")
    
    results = defaultdict(lambda: {"finetuned": [], "scratch": []})
    
    # Scan all run directories
    for run_dir in project_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        run_name = run_dir.name
        
        # Skip if not related to target city
        if target_city not in run_name.lower():
            continue
        
        # For Scratch_vs_Finetune project (i=5), only use t40_v10 runs (rs_1 through rs_5)
        if project_name == "Scratch_vs_Finetune":
            if not re.search(r'_rs_[1-5]_t40_v10$', run_name):
                continue
        
        # Extract number of pretraining cities
        num_cities = extract_num_pretrain_cities(run_name)
        if num_cities is None:
            continue
        
        # Determine run type
        if is_finetune_run(run_name):
            run_type = "finetuned"
        elif is_scratch_run(run_name):
            run_type = "scratch"
        else:
            continue
        
        # Find model path
        model_path = find_model_path(run_dir)
        if model_path is None:
            print(f"  ⚠️  No model found for {run_name} (expected at: {run_dir / 'finetuned_model' / 'model.pth'}), skipping...")
            continue
        
        # Always evaluate fresh (don't load existing metrics)
        print(f"  🔄 Evaluating {run_name} (n={num_cities}, {run_type})...")
        try:
            metrics = evaluate_model_on_test_split(
                run_name=run_name,
                project_name=project_name,
                model_path=model_path,
                split_file_path=split_file_path,
                eval_params=eval_params,
            )
            if metrics and "loss" in metrics:
                results[num_cities][run_type].append(metrics["loss"])
                print(f"    ✓ loss={metrics['loss']:.4f}, r2={metrics.get('r2', 'N/A'):.4f}")
            else:
                print(f"    ⚠️  Evaluation returned no metrics")
        except Exception as e:
            print(f"  ✗ Evaluation failed for {run_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return dict(results)


def remove_outliers_iqr(values: List[float]) -> List[float]:
    """
    Remove outliers using IQR method (1.5 × IQR rule).
    
    Args:
        values: List of values
        
    Returns:
        List of values with outliers removed
    """
    if not values or len(values) < 3:
        return values
    
    values_arr = np.array(values)
    q1 = np.percentile(values_arr, 25)
    q3 = np.percentile(values_arr, 75)
    iqr = q3 - q1
    
    if iqr == 0:
        return values
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Keep values within bounds
    filtered_values = [v for v in values if lower_bound <= v <= upper_bound]
    return filtered_values


def compute_statistics(values: List[float], remove_outliers: bool = True) -> Tuple[Optional[float], Optional[float], int]:
    """
    Compute mean and standard deviation, optionally removing outliers using IQR method.
    
    Args:
        values: List of values
        remove_outliers: If True, remove outliers using IQR method before computing statistics
        
    Returns:
        Tuple of (mean, std, n) where n is the number of values used after outlier removal
    """
    if not values:
        return None, None, 0
    
    # Remove outliers if requested
    if remove_outliers:
        filtered_values = remove_outliers_iqr(values)
    else:
        filtered_values = values
    
    if not filtered_values:
        return None, None, 0
    
    mean = np.mean(filtered_values)
    std = np.std(filtered_values, ddof=1) if len(filtered_values) > 1 else 0.0
    return mean, std, len(filtered_values)


def create_mse_plot(
    results: Dict[int, Dict[str, List[float]]],
    output_path: Path,
    metric_name: str = "MSE"
):
    """
    Create plot comparing finetuned vs scratch across number of pretraining cities.
    
    Args:
        results: Dict mapping num_cities -> {"finetuned": [values], "scratch": [values]}
        output_path: Path to save the plot
        metric_name: Name of metric to plot (default: "MSE")
    """
    # Set font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    # Compute overall scratch statistics across ALL runs (constant line)
    # Remove outliers using IQR method
    all_scratch_values = []
    for num_cities, data in results.items():
        all_scratch_values.extend(data.get("scratch", []))
    scratch_overall_mean, scratch_overall_std, scratch_n = compute_statistics(all_scratch_values, remove_outliers=True)
    
    # Prepare data for per-city finetuned values
    # Remove outliers using IQR method for each n value
    num_cities_list = sorted([k for k in results.keys() if k <= 5])
    finetune_means = []
    finetune_stds = []
    
    for num_cities in num_cities_list:
        finetune_values = results[num_cities].get("finetuned", [])
        finetune_mean, finetune_std, finetune_n = compute_statistics(finetune_values, remove_outliers=True)
        
        finetune_means.append(finetune_mean if finetune_mean is not None else np.nan)
        finetune_stds.append(finetune_std if finetune_std is not None else np.nan)
    
    # Add x=0 case where Scratch = Finetune (both use overall scratch mean)
    num_cities_list_with_zero = [0] + num_cities_list
    finetune_means_with_zero = [scratch_overall_mean if scratch_overall_mean is not None else np.nan] + finetune_means
    finetune_stds_with_zero = [scratch_overall_std if scratch_overall_std is not None else np.nan] + finetune_stds
    
    # Scratch line is constant (overall mean) for all x values including 0
    scratch_means_constant = [scratch_overall_mean if scratch_overall_mean is not None else np.nan] * len(num_cities_list_with_zero)
    scratch_stds_constant = [scratch_overall_std if scratch_overall_std is not None else np.nan] * len(num_cities_list_with_zero)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Colorblind-friendly colors (blue and orange)
    finetune_color = '#1f77b4'  # Blue
    scratch_color = '#ff7f0e'   # Orange
    
    # Convert to numpy arrays for easier handling
    x_positions = np.array(num_cities_list_with_zero)
    finetune_means_arr = np.array(finetune_means_with_zero)
    finetune_stds_arr = np.array(finetune_stds_with_zero)
    scratch_means_arr = np.array(scratch_means_constant)
    scratch_stds_arr = np.array(scratch_stds_constant)
    
    # Create masks for valid data
    finetune_mask = ~np.isnan(finetune_means_arr)
    scratch_mask = ~np.isnan(scratch_means_arr)
    
    # Plot shaded bands first (so they appear behind the lines)
    if np.any(finetune_mask):
        # Finetuned shaded band
        ax.fill_between(
            x_positions[finetune_mask],
            finetune_means_arr[finetune_mask] - finetune_stds_arr[finetune_mask],
            finetune_means_arr[finetune_mask] + finetune_stds_arr[finetune_mask],
            color=finetune_color,
            alpha=0.2,
            linewidth=0,
            label='_nolegend_'  # Don't show in legend
        )
    
    if np.any(scratch_mask):
        # Scratch shaded band (constant)
        ax.fill_between(
            x_positions[scratch_mask],
            scratch_means_arr[scratch_mask] - scratch_stds_arr[scratch_mask],
            scratch_means_arr[scratch_mask] + scratch_stds_arr[scratch_mask],
            color=scratch_color,
            alpha=0.2,
            linewidth=0,
            label='_nolegend_'  # Don't show in legend
        )
    
    # Plot lines on top of bands
    if np.any(finetune_mask):
        # Finetuned line
        ax.plot(
            x_positions[finetune_mask],
            finetune_means_arr[finetune_mask],
            'o-',
            color=finetune_color,
            label='Finetuned',
            markersize=8,
            linewidth=1.5,
            markerfacecolor=finetune_color,
            markeredgecolor='white',
            markeredgewidth=1
        )
    
    if np.any(scratch_mask):
        # Scratch line (constant horizontal line, no markers)
        ax.plot(
            x_positions[scratch_mask],
            scratch_means_arr[scratch_mask],
            '-',
            color=scratch_color,
            label='Scratch',
            linewidth=1.5,
        )
    
    # Formatting
    ax.set_xlabel("# of cities in pretraining", fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    ax.set_xticks(num_cities_list_with_zero)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=12)
    
    # No title
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
        description="Evaluate varying pretrain sources experiment results."
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="VaryingPretrainSources",
        help="Project name for run_varying_pretrain_sources_regensburg.py results"
    )
    parser.add_argument(
        "--all_cities_project_name",
        type=str,
        default="Scratch_vs_Finetune",
        help="Project name for i=5 runs (regensburg_scratch_rs_* and regensburg_finetune_rs_* from Scratch_vs_Finetune project)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots. Defaults to evaluation/plots/"
    )
    parser.add_argument(
        "--target_city",
        type=str,
        default=TARGET_CITY,
        help="Target city name (default: regensburg)"
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="data/splits",
        help="Directory containing split files"
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Path to split file. If not provided, auto-detects from splits_dir."
    )
    parser.add_argument(
        "--train_val_size",
        type=str,
        default="40:10",
        help="Train:val size (e.g., '40:10'). Used to locate split file."
    )
    parser.add_argument(
        "--seed_idx",
        type=int,
        default=1,
        help="Seed index (1-5). Used to locate split file."
    )
    parser.add_argument(
        "--test_count",
        type=int,
        default=100,
        help="Test count used in split file name."
    )
    parser.add_argument(
        "--test_set_type",
        type=str,
        default="random",
        choices=["distant_iou", "random"],
        help="Test set type (default: random, as per spatial_maps.ipynb)"
    )
    parser.add_argument(
        "--gnn_arch",
        type=str,
        default="trans_encoder",
        help="GNN architecture"
    )
    parser.add_argument(
        "--use_all_features",
        type=str_to_bool,
        default=False,
        help="Use all features"
    )
    parser.add_argument(
        "--target_type",
        type=str,
        default="abs_vol_car",
        help="Target type"
    )
    parser.add_argument(
        "--target_normalization",
        type=str,
        default=None,
        help="Target normalization"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--neighbor_sizes",
        type=str,
        default="5,5,5",
        help="Neighbor sizes (comma-separated)"
    )
    parser.add_argument(
        "--subgraphs_per_graph",
        type=int,
        default=2,
        help="Subgraphs per graph"
    )
    parser.add_argument(
        "--seed_size",
        type=int,
        default=10,
        help="Seed size"
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="neighbor_sampling",
        choices=["neighbor_sampling", "random_walk"],
        help="Sampling strategy"
    )
    parser.add_argument(
        "--min_subgraph_nodes",
        type=int,
        default=500,
        help="Min subgraph nodes"
    )
    parser.add_argument(
        "--max_subgraph_nodes",
        type=int,
        default=50000,
        help="Max subgraph nodes"
    )
    parser.add_argument(
        "--device_nr",
        type=int,
        default=0,
        help="Device number"
    )
    parser.add_argument(
        "--skip_if_exists",
        type=str_to_bool,
        default=False,
        help="Skip evaluation if metrics file already exists (default: False - always recompute metrics)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).resolve().parents[2]
    
    # Set output directory
    if args.output_dir is None:
        output_dir = Path(__file__).parent / "plots"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine split file
    if args.split_file:
        split_file_path = Path(args.split_file).resolve()
        if not split_file_path.exists():
            raise ValueError(f"Split file does not exist: {split_file_path}")
    else:
        # Auto-detect split file
        if ":" not in args.train_val_size:
            raise ValueError(f"Invalid train_val_size '{args.train_val_size}'. Use format 'train:val'")
        train_size, val_size = args.train_val_size.split(":")
        train_size, val_size = int(train_size), int(val_size)
        
        shuffle_seed = 42
        seed = shuffle_seed + (args.seed_idx - 1)
        
        if not Path(args.splits_dir).is_absolute():
            splits_dir = project_root / args.splits_dir
        else:
            splits_dir = Path(args.splits_dir)
        splits_dir = splits_dir.resolve()
        
        split_subdir = splits_dir / args.target_city / f"rs_{args.seed_idx}" / f"t{train_size}_v{val_size}"
        split_filename = (
            f"{args.target_city}_rs{args.seed_idx}_t{train_size}_v{val_size}_seed{seed}_"
            f"train{train_size}_val{val_size}_test{args.test_count}_{args.test_set_type}.json"
        )
        split_file_path = split_subdir / split_filename
        
        if not split_file_path.exists():
            raise ValueError(f"Split file not found: {split_file_path}")
    
    print("=" * 80)
    print("EVALUATING VARYING PRETRAIN SOURCES EXPERIMENT")
    print("=" * 80)
    print(f"Target city: {args.target_city}")
    print(f"Project (i=1-4): {args.project_name}")
    print(f"Project (i=5): {args.all_cities_project_name}")
    print(f"Split file: {split_file_path}")
    print(f"Test set type: {args.test_set_type}")
    print("=" * 80)
    
    # Parse neighbor sizes
    neighbor_sizes = [int(x.strip()) for x in args.neighbor_sizes.split(",") if x.strip()]
    
    # Evaluation parameters
    eval_params = {
        "gnn_arch": args.gnn_arch,
        "use_all_features": args.use_all_features,
        "target_type": args.target_type,
        "target_normalization": args.target_normalization,
        "batch_size": args.batch_size,
        "neighbor_sizes": neighbor_sizes,
        "subgraphs_per_graph": args.subgraphs_per_graph,
        "seed_size": args.seed_size,
        "sampling_strategy": args.sampling_strategy,
        "min_subgraph_nodes": args.min_subgraph_nodes,
        "max_subgraph_nodes": args.max_subgraph_nodes,
        "device_nr": args.device_nr,
    }
    
    # Collect and evaluate runs from both projects (always recompute metrics)
    print(f"\nEvaluating runs from {args.project_name} (i=1-4)...")
    results_1_4 = collect_and_evaluate_runs(
        args.project_name, args.target_city, split_file_path, eval_params, skip_if_exists=False
    )
    
    # For i=5, always use Scratch_vs_Finetune project (enforce correct project)
    i5_project_name = "Scratch_vs_Finetune"
    if args.all_cities_project_name != i5_project_name:
        print(f"⚠️  Warning: Overriding --all_cities_project_name '{args.all_cities_project_name}' with '{i5_project_name}' for i=5 runs")
    print(f"\nEvaluating runs from {i5_project_name} (i=5)...")
    results_5 = collect_and_evaluate_runs(
        i5_project_name, args.target_city, split_file_path, eval_params, skip_if_exists=False
    )
    
    # Merge results
    all_results = defaultdict(lambda: {"finetuned": [], "scratch": []})
    
    for num_cities, data in results_1_4.items():
        all_results[num_cities]["finetuned"].extend(data["finetuned"])
        all_results[num_cities]["scratch"].extend(data["scratch"])
    
    for num_cities, data in results_5.items():
        if num_cities == 5:  # Only use i=5 from all_cities project
            all_results[num_cities]["finetuned"].extend(data["finetuned"])
            all_results[num_cities]["scratch"].extend(data["scratch"])
    
    # Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS SUMMARY")
    print("=" * 80)
    
    # Compute overall scratch statistics across ALL runs (constant baseline)
    all_scratch_values = []
    for num_cities, data in all_results.items():
        all_scratch_values.extend(data.get("scratch", []))
    scratch_overall_mean, scratch_overall_std, scratch_n = compute_statistics(all_scratch_values, remove_outliers=True)
    scratch_original_n = len(all_scratch_values)
    scratch_removed = scratch_original_n - scratch_n
    
    for num_cities in sorted(all_results.keys()):
        finetune_values = all_results[num_cities]["finetuned"]
        
        finetune_mean, finetune_std, finetune_n = compute_statistics(finetune_values, remove_outliers=True)
        
        # Show original count and outlier removal info
        finetune_original_n = len(finetune_values)
        finetune_removed = finetune_original_n - finetune_n
        
        print(f"\n{num_cities} cities in pretraining:")
        if finetune_mean is not None:
            outlier_info = f" (removed {finetune_removed} outlier(s))" if finetune_removed > 0 else ""
            print(f"  Finetuned: MSE = {finetune_mean:.4f} ± {finetune_std:.4f} (n={finetune_n}{outlier_info}, original n={finetune_original_n})")
        else:
            print(f"  Finetuned: No data")
    
    # Print scratch statistics once (constant across all n values)
    print(f"\nScratch (baseline, constant across all n values):")
    if scratch_overall_mean is not None:
        outlier_info = f" (removed {scratch_removed} outlier(s))" if scratch_removed > 0 else ""
        print(f"  Scratch: MSE = {scratch_overall_mean:.4f} ± {scratch_overall_std:.4f} (n={scratch_n}{outlier_info}, original n={scratch_original_n})")
    else:
        print(f"  Scratch: No data")
    
    print("=" * 80)
    
    # Create plot
    print("\nCreating MSE plot...")
    plot_path = output_dir / f"{args.target_city}_varying_pretrain_sources_mse.png"
    create_mse_plot(all_results, plot_path, metric_name="MSE")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
