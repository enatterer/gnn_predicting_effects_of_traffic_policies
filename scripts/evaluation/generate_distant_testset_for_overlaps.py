#!/usr/bin/env python3
"""
Generate distant (IoU-based) test sets for overlap evaluation.

This script builds test splits compatible with
compute_overlaps_varying_capacity_reduction.py, for multiple seeds.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to Python path so `scripts.*` imports work.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.analysis.generate_distant_splits import find_distant_iou_test_split
from scripts.evaluation.compute_overlaps_varying_capacity_reduction import (
    replace_path_for_retina,
)
from scripts.training.help_functions import load_metadata_from_disk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate distant test splits for overlap evaluation."
    )
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--train_count", type=int, required=True)
    parser.add_argument("--val_count", type=int, required=True)
    parser.add_argument("--test_count", type=int, required=True)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument(
        "--test_set_type",
        type=str,
        default="distant_iou",
        choices=["distant_iou", "random"],
        help="Test set selection strategy.",
    )
    parser.add_argument(
        "--input_splits_dir",
        type=str,
        default="data/splits",
        help="Directory containing existing train/val splits.",
    )
    parser.add_argument(
        "--output_splits_dir",
        type=str,
        default="tmp_distant_splits",
        help="Directory to write distant test splits.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    project_root = PROJECT_ROOT
    city = args.city
    train_count = args.train_count
    val_count = args.val_count
    test_count = args.test_count
    seed_base = args.seed_base
    num_seeds = args.num_seeds

    input_splits_dir = Path(args.input_splits_dir)
    if not input_splits_dir.is_absolute():
        input_splits_dir = project_root / input_splits_dir
    output_splits_dir = Path(args.output_splits_dir)
    if not output_splits_dir.is_absolute():
        output_splits_dir = project_root / output_splits_dir

    input_splits_dir = input_splits_dir / city
    output_splits_dir = output_splits_dir / city
    metadata_path = (
        project_root
        / "data"
        / "bavaria"
        / "inductive_data"
        / "training_data"
        / "kreisfreistadt"
        / city
        / "metadata.json"
    )

    all_data = {"path": [], "policy_region": [], "scenario": [], "city": []}
    load_metadata_from_disk(all_data, str(metadata_path))
    all_data = replace_path_for_retina(all_data, str(project_root))

    for seed_idx in range(1, num_seeds + 1):
        seed = seed_base + seed_idx - 1
        split_dir = input_splits_dir / f"rs_{seed_idx}" / f"t{train_count}_v{val_count}"
        train_val_file = (
            split_dir
            / f"{city}_rs{seed_idx}_t{train_count}_v{val_count}_seed{seed}_"
            f"train{train_count}_val{val_count}_random.json"
        )
        output_file = (
            output_splits_dir
            / f"rs_{seed_idx}"
            / f"t{train_count}_v{val_count}"
            / f"{city}_rs{seed_idx}_t{train_count}_v{val_count}_seed{seed}_"
            f"train{train_count}_val{val_count}_test{test_count}_{args.test_set_type}.json"
        )

        if output_file.exists():
            print(f"[seed {seed_idx}] Test split exists: {output_file}")
            continue

        if not train_val_file.exists():
            raise FileNotFoundError(f"Missing train/val split: {train_val_file}")

        with open(train_val_file, "r") as f:
            split_data = json.load(f)

        split_data["train_data"] = replace_path_for_retina(
            split_data.get("train_data"), str(project_root)
        )
        split_data["val_data"] = replace_path_for_retina(
            split_data.get("val_data"), str(project_root)
        )

        train_paths = split_data.get("train_data", {}).get("path", [])
        val_paths = split_data.get("val_data", {}).get("path", [])

        if args.test_set_type == "distant_iou":
            (
                test_paths,
                test_distances_when_picked,
                test_distances_from_train,
                test_distances_from_val,
            ) = find_distant_iou_test_split(
                train_paths,
                val_paths,
                all_data["path"],
                test_count,
            )
        else:
            from scripts.analysis.generate_distant_splits import find_random_test_split

            (
                test_paths,
                test_distances_from_train,
                test_distances_from_val,
            ) = find_random_test_split(
                train_paths,
                val_paths,
                all_data["path"],
                test_count,
                seed=seed,
            )
            test_distances_when_picked = None

        path_to_index = {path: idx for idx, path in enumerate(all_data["path"])}
        missing_paths = [p for p in test_paths if p not in path_to_index]
        if missing_paths:
            raise ValueError(f"Missing paths in metadata: {len(missing_paths)}")

        test_data = {
            "path": test_paths,
            "policy_region": [
                all_data["policy_region"][path_to_index[p]] for p in test_paths
            ],
            "scenario": [
                all_data["scenario"][path_to_index[p]] for p in test_paths
            ],
            "city": [city] * len(test_paths),
        }

        split_data["test_count"] = len(test_paths)
        split_data["test_paths"] = test_paths
        if test_distances_when_picked is not None:
            split_data["test_distances_when_picked"] = test_distances_when_picked
        split_data["test_distances_from_train"] = test_distances_from_train
        split_data["test_distances_from_val"] = test_distances_from_val
        split_data["test_data"] = test_data

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(split_data, f, indent=2)

        print(f"[seed {seed_idx}] Wrote test split: {output_file}")

    print("Done.")


if __name__ == "__main__":
    main()
