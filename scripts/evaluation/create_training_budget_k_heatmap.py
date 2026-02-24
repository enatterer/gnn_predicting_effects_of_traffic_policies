#!/usr/bin/env python3
"""
Create a heatmap with training budget (x) vs k (y) for a fixed
capacity reduction threshold (default: 100%, i.e., no constraint).
"""

import argparse
import glob
import json
import os
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "Times New Roman"

SPLIT_MAPPING = {
    (10, 3): 13,
    (20, 5): 25,
    (40, 10): 50,
    (80, 20): 100,
    (160, 40): 200,
}


def parse_int_list(arg: str) -> List[int]:
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def find_latest_overlap_files(overlaps_dir: str, city: str, k: int, test_count: int) -> Dict[Tuple[int, int], str]:
    pattern = os.path.join(
        overlaps_dir,
        f"overlap_results_varying_capacity_reduction_{city}_t*_v*_test{test_count}_norm_to_k_k{k}_*.json",
    )
    files = glob.glob(pattern)
    split_to_file: Dict[Tuple[int, int], str] = {}

    for path in files:
        name = os.path.basename(path)
        parts = name.split("_")
        train_count = None
        val_count = None
        for part in parts:
            if part.startswith("t") and part[1:].isdigit():
                train_count = int(part[1:])
            elif part.startswith("v") and part[1:].isdigit():
                val_count = int(part[1:])
        if train_count is None or val_count is None:
            continue
        key = (train_count, val_count)
        if key not in split_to_file:
            split_to_file[key] = path
        else:
            if os.path.getmtime(path) > os.path.getmtime(split_to_file[key]):
                split_to_file[key] = path

    return split_to_file


def build_matrix(
    overlaps_dir: str,
    city: str,
    ks: List[int],
    test_count: int,
    threshold_key: str,
    model_type: str,
    training_budgets: List[int],
) -> np.ndarray:
    matrix = np.full((len(ks), len(training_budgets)), np.nan)

    for k_idx, k in enumerate(ks):
        split_to_file = find_latest_overlap_files(overlaps_dir, city, k, test_count)
        for (train_count, val_count), path in split_to_file.items():
            total = SPLIT_MAPPING.get((train_count, val_count))
            if total is None:
                continue
            if total not in training_budgets:
                continue
            col_idx = training_budgets.index(total)
            with open(path, "r") as f:
                data = json.load(f)
            results = data.get("results", {})
            seed_results = results.get(threshold_key, {})
            values = []
            for seed_idx in sorted(seed_results.keys(), key=int):
                seed_data = seed_results[seed_idx]
                if seed_data is None:
                    continue
                key = f"{model_type}_overlap"
                if key in seed_data:
                    values.append(seed_data[key])
            if values:
                matrix[k_idx, col_idx] = float(np.mean(values))

    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Create training-budget vs k heatmap")
    parser.add_argument("--city", type=str, default="regensburg")
    parser.add_argument("--overlaps_dir", type=str, default="scripts/evaluation/overlaps")
    parser.add_argument("--output_dir", type=str, default="scripts/evaluation/plots")
    parser.add_argument("--test_count", type=int, default=1000)
    parser.add_argument("--ks", type=str, default="10,50,100")
    parser.add_argument("--threshold", type=float, default=100.0,
                        help="Capacity reduction percentage to use (default: 100.0)")
    parser.add_argument("--model_type", type=str, default="finetune",
                        choices=["scratch", "finetune"])
    parser.add_argument("--training_budgets", type=str, default="13,25,50,100,200")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    overlaps_dir = os.path.join(project_root, args.overlaps_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ks = parse_int_list(args.ks)
    training_budgets = parse_int_list(args.training_budgets)
    threshold_key = f"{float(args.threshold)}"

    matrix = build_matrix(
        overlaps_dir=overlaps_dir,
        city=args.city,
        ks=ks,
        test_count=args.test_count,
        threshold_key=threshold_key,
        model_type=args.model_type,
        training_budgets=training_budgets,
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0, interpolation="nearest")

    ax.set_xticks(np.arange(len(training_budgets)))
    ax.set_xticklabels(training_budgets)
    ax.set_xlabel("# of target-city data available")

    y_labels_map = {
        10: "Top-1%",
        50: "Top-5%",
        100: "Top-10%",
        200: "Top-20%",
    }
    # Baseline overlap for each top-k row (random guess: k% of scenarios)
    baseline_by_k = {10: 0.01, 50: 0.05, 100: 0.1, 200: 0.2}
    ax.set_yticks(np.arange(len(ks)))
    ax.set_yticklabels([y_labels_map.get(k, str(k)) for k in ks])
    ax.set_ylabel("Top scenario coverage")

    for i in range(len(ks)):
        baseline = baseline_by_k.get(ks[i], 0.0)
        for j in range(len(training_budgets)):
            val = matrix[i, j]
            if not np.isnan(val):
                improvement = val - baseline
                ax.text(j, i, f"{val:.2f} [+{improvement:.2f}]", ha="center", va="center",
                        color="black", fontsize=10)

    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(0.0, 1.1, 0.1))
    cbar.set_label("Overlap", rotation=270, labelpad=15)

    plt.tight_layout()
    output_name = (
        f"overlap_heatmap_{args.model_type}_{args.city}_norm_to_k_no_cap_"
        f"k{'_'.join(str(k) for k in ks)}.png"
    )
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved heatmap to: {output_path}")


if __name__ == "__main__":
    main()
