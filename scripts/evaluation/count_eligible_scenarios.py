#!/usr/bin/env python3
"""
Count eligible scenarios under capacity reduction thresholds.

Eligibility is defined as capacity_reduction_per_scenario <= threshold,
where capacity_reduction_per_scenario = sum(capacity_reduction) / num_links.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.evaluation.compute_overlaps_varying_capacity_reduction import (
    replace_path_for_retina,
)


def parse_percentages(percentages_str: str) -> List[float]:
    return [float(p.strip()) for p in percentages_str.split(",") if p.strip()]


def load_test_paths(split_path: Path, project_root: str) -> List[str]:
    with split_path.open("r") as f:
        split_data = json.load(f)
    test_data = replace_path_for_retina(split_data.get("test_data"), project_root)
    return test_data.get("path", [])


def compute_reduction_fraction(graph_path: str) -> float:
    graph = torch.load(graph_path, map_location="cpu")
    node_features = graph.x.cpu().numpy()
    reductions = node_features[:, 2]
    num_links = reductions.shape[0]
    if num_links == 0:
        return 0.0
    return float(reductions.sum() / num_links)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count eligible scenarios per threshold.")
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--train_count", type=int, required=True)
    parser.add_argument("--val_count", type=int, required=True)
    parser.add_argument("--test_count", type=int, required=True)
    parser.add_argument("--test_set_type", type=str, default="random")
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--percentages", type=str, required=True)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    splits_dir = Path(args.splits_dir)
    if not splits_dir.is_absolute():
        splits_dir = project_root / splits_dir

    thresholds = parse_percentages(args.percentages)
    thresholds_sorted = sorted(thresholds)

    results: Dict[str, Dict[str, int]] = {}

    for seed_idx in range(1, args.num_seeds + 1):
        seed = args.seed_base + seed_idx - 1
        split_path = (
            splits_dir
            / args.city
            / f"rs_{seed_idx}"
            / f"t{args.train_count}_v{args.val_count}"
            / f"{args.city}_rs{seed_idx}_t{args.train_count}_v{args.val_count}_"
            f"seed{seed}_train{args.train_count}_val{args.val_count}_"
            f"test{args.test_count}_{args.test_set_type}.json"
        )
        test_paths = load_test_paths(split_path, str(project_root))
        reduction_fractions = [compute_reduction_fraction(p) for p in test_paths]
        reduction_fractions = np.array(reduction_fractions, dtype=float)

        seed_key = f"seed_{seed_idx}"
        results[seed_key] = {}
        for pct in thresholds_sorted:
            thr = pct / 100.0
            eligible = int((reduction_fractions <= thr).sum())
            results[seed_key][str(pct)] = eligible

    summary = {}
    for pct in thresholds_sorted:
        values = [results[f"seed_{i}"][str(pct)] for i in range(1, args.num_seeds + 1)]
        summary[str(pct)] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }

    payload = {
        "city": args.city,
        "train_count": args.train_count,
        "val_count": args.val_count,
        "test_count": args.test_count,
        "test_set_type": args.test_set_type,
        "thresholds": thresholds_sorted,
        "eligible_counts": results,
        "summary": summary,
    }

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = (
            project_root
            / "scripts"
            / "evaluation"
            / "overlaps"
            / f"eligible_counts_{args.city}_t{args.train_count}_v{args.val_count}_"
            f"test{args.test_count}_{args.test_set_type}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved eligible counts to: {output_path}")


if __name__ == "__main__":
    main()
