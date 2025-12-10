#!/usr/bin/env python3
"""
Visualize metrics for finetuning vs training from scratch per city.

Reads evaluation JSON files under the PretrainFinetune_Comparison results
directory and produces:
- A CSV summary with hit rate metrics (top/bottom) and core metrics for each city and method.
- One bar plot per metric showing finetune vs scratch across cities.

Usage:
    python scripts/analysis/visualize_pretrain_finetune_comparison.py \
        --results_dir /path/to/PretrainFinetune_Comparison \
        --output_dir /path/to/output

Defaults match the repository layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Default locations for this repository
DEFAULT_RESULTS_DIR = Path(
    "/home/enatterer/Development/elena_gnn_predicting_effects_of_traffic_policies/"
    "inductive_gnn_data_results/transductive/PretrainFinetune_Comparison"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/enatterer/Development/elena_gnn_predicting_effects_of_traffic_policies/"
    "scripts/visualization/results/pretrain_finetune_comparison"
)

# Metrics to keep at the top level before we append the hit rate keys
BASE_METRICS = ["loss", "r2", "spearman", "pearson"]


def _find_evaluation_file(run_dir: Path) -> Optional[Path]:
    """Return the first evaluation JSON found in the run directory."""
    eval_dir = run_dir / "evaluation"
    if not eval_dir.exists():
        return None
    json_files = sorted(eval_dir.glob("*.json"))
    return json_files[0] if json_files else None


def _load_metrics(run_dir: Path) -> Optional[Dict[str, float]]:
    """
    Load and flatten metrics from a run directory.
    """
    eval_file = _find_evaluation_file(run_dir)
    if eval_file is None:
        return None

    with eval_file.open("r") as f:
        data = json.load(f)

    metrics: Dict[str, float] = {}
    for key in BASE_METRICS:
        if key in data:
            metrics[key] = data[key]

    hit_rates = data.get("hit_rates", {})
    for key, value in hit_rates.items():
        metrics[key] = value

    return metrics


def collect_city_results(results_dir: Path) -> pd.DataFrame:
    """
    Collect metrics for each city where both scratch and finetune runs exist.

    Returns a DataFrame with columns: city, method, <metrics>.
    """
    rows: List[Dict[str, float]] = []

    for scratch_dir in results_dir.glob("run_from_scratch_*"):
        if not scratch_dir.is_dir():
            continue
        city = scratch_dir.name.replace("run_from_scratch_", "")
        finetune_dir = results_dir / f"finetune_{city}"
        if not finetune_dir.exists():
            print(f"Skipping {city}: missing finetune directory")
            continue

        scratch_metrics = _load_metrics(scratch_dir)
        finetune_metrics = _load_metrics(finetune_dir)
        if scratch_metrics is None or finetune_metrics is None:
            print(f"Skipping {city}: missing evaluation file")
            continue

        rows.append({"city": city, "method": "scratch", **scratch_metrics})
        rows.append({"city": city, "method": "finetune", **finetune_metrics})

    if not rows:
        raise ValueError(
            f"No comparable runs found in {results_dir}. "
            "Ensure both run_from_scratch_* and finetune_* exist with evaluation JSON files."
        )

    return pd.DataFrame(rows)


def _plot_metric(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    """Create a grouped bar plot for a single metric across cities."""
    subset = df[["city", "method", metric]].dropna()
    if subset.empty:
        print(f"Skipping plot for {metric}: no data.")
        return

    subset["city"] = subset["city"].str.title()
    subset = subset.sort_values("city")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=subset, x="city", y=metric, hue="method", palette=["#6baed6", "#fd8d3c"])
    plt.title(metric.replace("_", " ").title())
    plt.ylabel(metric.replace("_", " ").title())
    plt.xlabel("City")
    plt.legend(title=None)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    output_path = output_dir / f"{metric}_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize finetune vs scratch results for PretrainFinetune_Comparison."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing run_from_scratch_* and finetune_* subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save plots and summary CSV.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_city_results(args.results_dir)

    # Save summary CSV
    summary_path = args.output_dir / "metrics_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Saved summary CSV to {summary_path}")

    # Determine metric columns (all except city/method)
    metric_columns = [col for col in df.columns if col not in {"city", "method"}]

    # Plot each metric
    for metric in metric_columns:
        _plot_metric(df, metric, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()

