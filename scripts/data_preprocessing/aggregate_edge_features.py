#!/usr/bin/env python3
"""
Script to aggregate all files from edge_features_with_net_flow_1000, edge_features_with_net_flow_2000,
..., edge_features_with_net_flow_5000 into a single folder, renaming them with incrementing batch numbers.

This script will:
1. Find all datalist_batch_*.pt files in the source directories
2. Copy them to the destination directory with new sequential batch numbers
3. Create a mapping file showing the original and new filenames
"""

import os
import shutil
import re
from pathlib import Path
import argparse
from typing import List, Tuple


def find_batch_files(source_dirs: List[str]) -> List[Tuple[str, int]]:
    """
    Find all datalist_batch_*.pt files in the source directories.

    Args:
        source_dirs: List of source directory paths

    Returns:
        List of (file_path, batch_number) tuples sorted by batch number
    """
    batch_files = []
    batch_pattern = re.compile(r"datalist_batch_(\d+)\.pt$")

    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory {source_dir} does not exist, skipping...")
            continue

        print(f"Scanning directory: {source_dir}")
        for filename in os.listdir(source_dir):
            match = batch_pattern.match(filename)
            if match:
                batch_num = int(match.group(1))
                file_path = os.path.join(source_dir, filename)
                batch_files.append((file_path, batch_num))

    # Sort by directory order and then by batch number
    # This ensures consistent ordering across runs
    batch_files.sort(key=lambda x: (os.path.dirname(x[0]), x[1]))

    print(f"Found {len(batch_files)} batch files total")
    return batch_files


def aggregate_files(
    source_dirs: List[str], destination_dir: str, create_mapping: bool = True
):
    """
    Aggregate all batch files from source directories into destination directory.

    Args:
        source_dirs: List of source directory paths
        destination_dir: Destination directory path
        create_mapping: Whether to create a mapping file
    """
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Find all batch files
    batch_files = find_batch_files(source_dirs)

    if not batch_files:
        print("No batch files found in any source directory!")
        return

    # Copy files with new sequential batch numbers
    mapping = []
    new_batch_num = 1

    print(f"Copying files to {destination_dir}...")
    for file_path, original_batch_num in batch_files:
        # Create new filename
        new_filename = f"datalist_batch_{new_batch_num}.pt"
        new_file_path = os.path.join(destination_dir, new_filename)

        # Copy file
        shutil.copy2(file_path, new_file_path)

        # Record mapping
        original_filename = os.path.basename(file_path)
        source_dir = os.path.basename(os.path.dirname(file_path))
        mapping.append(
            {
                "source_dir": source_dir,
                "original_filename": original_filename,
                "original_batch_num": original_batch_num,
                "new_filename": new_filename,
                "new_batch_num": new_batch_num,
                "original_path": file_path,
            }
        )

        print(f"  {source_dir}/{original_filename} -> {new_filename}")
        new_batch_num += 1

    # Create mapping file if requested
    if create_mapping:
        mapping_file = os.path.join(destination_dir, "file_mapping.txt")
        with open(mapping_file, "w") as f:
            f.write("# File mapping for aggregated edge features\n")
            f.write(
                "# Format: source_dir/original_filename -> new_filename (original_batch -> new_batch)\n"
            )
            f.write("\n")
            for item in mapping:
                f.write(
                    f"{item['source_dir']}/{item['original_filename']} -> {item['new_filename']} "
                    f"({item['original_batch_num']} -> {item['new_batch_num']})\n"
                )

        print(f"\nMapping file created: {mapping_file}")

    print(
        f"\nAggregation complete! Copied {len(batch_files)} files with sequential batch numbers 1-{len(batch_files)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Aggregate edge feature batch files")
    parser.add_argument(
        "--project-root",
        default="/home/dnguyen/gnn_predicting_effects_of_traffic_policies",
        help="Path to project root directory",
    )
    parser.add_argument(
        "--dest-name",
        default="edge_features_with_net_flow_aggregated",
        help="Name of the destination directory",
    )
    parser.add_argument(
        "--start-range",
        type=int,
        default=1000,
        help="Starting range for source directories (default: 1000)",
    )
    parser.add_argument(
        "--end-range",
        type=int,
        default=5000,
        help="Ending range for source directories (default: 5000)",
    )
    parser.add_argument(
        "--step", type=int, default=1000, help="Step size for range (default: 1000)"
    )
    parser.add_argument(
        "--no-mapping", action="store_true", help="Don't create mapping file"
    )

    args = parser.parse_args()

    # Build source directory paths
    train_data_dir = os.path.join(args.project_root, "data", "train_data")
    source_dirs = []

    for i in range(args.start_range, args.end_range + 1, args.step):
        source_dir = os.path.join(train_data_dir, f"edge_features_with_net_flow_{i}")
        source_dirs.append(source_dir)

    # Set destination directory
    destination_dir = os.path.join(train_data_dir, args.dest_name)

    print("Edge Features Aggregation Script")
    print("=" * 40)
    print(f"Project root: {args.project_root}")
    print(f"Source directories:")
    for source_dir in source_dirs:
        print(f"  - {source_dir}")
    print(f"Destination: {destination_dir}")
    print("")

    # Check if destination exists and ask for confirmation
    if os.path.exists(destination_dir) and os.listdir(destination_dir):
        response = input(
            f"Destination directory {destination_dir} exists and is not empty. "
            "Do you want to continue? (y/N): "
        )
        if response.lower() != "y":
            print("Aborted.")
            return

    # Perform aggregation
    aggregate_files(source_dirs, destination_dir, create_mapping=not args.no_mapping)


if __name__ == "__main__":
    main()
