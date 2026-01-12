#!/usr/bin/env python3
"""
Compute average variance of traffic volume over all seeds, grouped by road type.

For each city, this script:
1. Loads output_links.csv.gz from all seed directories
2. Computes variance of vol_car for each link across seeds
3. Aggregates variance by road type (osm:way:highway)
4. Reports average variance per road type and overall
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def load_all_seeds(city_path: Path) -> pd.DataFrame:
    """
    Load output_links.csv.gz from all seed directories for a city.
    
    Returns a DataFrame with columns: [link, vol_car, osm:way:highway, seed]
    """
    seed_dirs = sorted([d for d in city_path.iterdir() if d.is_dir() and 'seed' in d.name])
    
    if not seed_dirs:
        print(f"Warning: No seed directories found in {city_path}")
        return pd.DataFrame()
    
    all_data = []
    
    for seed_dir in seed_dirs:
        seed_num = seed_dir.name.split('_')[-1]  # Extract seed number
        links_file = seed_dir / 'output_links.csv.gz'
        
        if not links_file.exists():
            print(f"Warning: {links_file} not found, skipping")
            continue
        
        try:
            df = pd.read_csv(links_file, delimiter=';', low_memory=False)
            
            # Extract relevant columns
            if 'vol_car' not in df.columns or 'osm:way:highway' not in df.columns:
                print(f"Warning: Missing required columns in {links_file}")
                continue
            
            df_seed = df[['link', 'vol_car', 'osm:way:highway']].copy()
            df_seed['seed'] = seed_num
            all_data.append(df_seed)
            
        except Exception as e:
            print(f"Error loading {links_file}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def create_merged_road_type_mapping(road_types: pd.Series) -> Dict[str, str]:
    """
    Create a mapping from original road types to merged road types.
    Merges base types with their _link variants (e.g., 'primary' + 'primary_link' -> 'primary_merged').
    
    Returns a dictionary mapping original road type to merged road type.
    """
    mapping = {}
    # Filter out NaN/None values
    road_type_set = set(rt for rt in road_types.unique() if pd.notna(rt))
    
    # Find all base types that have corresponding _link variants
    base_types_with_links = set()
    for rt in road_type_set:
        if isinstance(rt, str) and rt.endswith('_link'):
            base_type = rt[:-5]  # Remove '_link'
            if base_type in road_type_set:
                base_types_with_links.add(base_type)
    
    # Create mapping: both base and _link types map to base_merged
    for rt in road_type_set:
        if not isinstance(rt, str):
            mapping[rt] = rt  # Keep non-string types as-is
            continue
            
        if rt.endswith('_link'):
            base_type = rt[:-5]
            if base_type in base_types_with_links:
                mapping[rt] = base_type + '_merged'
            else:
                mapping[rt] = rt  # No base type found, keep original
        elif rt in base_types_with_links:
            mapping[rt] = rt + '_merged'
        else:
            mapping[rt] = rt  # No _link variant, keep original
    
    return mapping


def compute_variance_by_road_type(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute variance of vol_car for each link, then aggregate by road type.
    Also compute variance of aggregated volumes per road type across seeds.
    Includes both original road types and merged categories (base + _link variants).
    
    Returns:
        - DataFrame with average variance per road type (per link) - original
        - Overall average variance (per link)
        - DataFrame with variance of aggregated volumes per road type - original
        - DataFrame with average variance per road type (per link) - merged
        - DataFrame with variance of aggregated volumes per road type - merged
    """
    if df.empty:
        return pd.DataFrame(), 0.0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Compute variance for each link across seeds
    link_variances = df.groupby('link').agg({
        'vol_car': 'var',  # Variance across seeds
        'osm:way:highway': 'first'  # Road type (should be same for all seeds)
    }).reset_index()
    
    link_variances.columns = ['link', 'variance', 'road_type']
    
    # Remove links with NaN variance (only one seed or all same values)
    link_variances = link_variances.dropna(subset=['variance'])
    
    # Group by road type and compute average variance (original)
    road_type_stats = link_variances.groupby('road_type').agg({
        'variance': ['mean', 'count']
    }).reset_index()
    
    # Flatten column names
    road_type_stats.columns = ['road_type', 'avg_variance', 'link_count']
    road_type_stats = road_type_stats.sort_values('avg_variance', ascending=False)
    
    # Create merged road types for _link variants
    merged_mapping = create_merged_road_type_mapping(link_variances['road_type'])
    link_variances_merged = link_variances.copy()
    link_variances_merged['road_type_merged'] = link_variances_merged['road_type'].map(merged_mapping)
    
    # Group by merged road type and compute average variance
    road_type_stats_merged = link_variances_merged.groupby('road_type_merged').agg({
        'variance': ['mean', 'count']
    }).reset_index()
    road_type_stats_merged.columns = ['road_type', 'avg_variance', 'link_count']
    road_type_stats_merged = road_type_stats_merged.sort_values('avg_variance', ascending=False)
    
    # Overall average variance
    overall_avg_variance = link_variances['variance'].mean()
    
    # Compute variance of aggregated volumes per road type across seeds
    # For each road type and seed: sum volumes across all links of that type
    # Filter out NaN road types
    df_clean = df.dropna(subset=['osm:way:highway'])
    road_type_volumes = df_clean.groupby(['osm:way:highway', 'seed'])['vol_car'].sum().reset_index()
    road_type_volumes.columns = ['road_type', 'seed', 'total_volume']
    
    # Compute variance of total volume across seeds for each road type (original)
    road_type_variance = road_type_volumes.groupby('road_type').agg({
        'total_volume': ['var', 'mean', 'count']
    }).reset_index()
    
    # Flatten column names
    road_type_variance.columns = ['road_type', 'variance_of_total', 'mean_total_volume', 'num_seeds']
    # Remove NaN variances (shouldn't happen, but just in case)
    road_type_variance = road_type_variance.dropna(subset=['variance_of_total'])
    road_type_variance = road_type_variance.sort_values('variance_of_total', ascending=False)
    
    # Create merged road types for variance of total volume
    merged_mapping_volumes = create_merged_road_type_mapping(road_type_volumes['road_type'])
    road_type_volumes_merged = road_type_volumes.copy()
    road_type_volumes_merged['road_type_merged'] = road_type_volumes_merged['road_type'].map(merged_mapping_volumes)
    
    # First, group by merged road type AND seed to sum volumes (one row per seed per merged road type)
    road_type_volumes_by_seed = road_type_volumes_merged.groupby(['road_type_merged', 'seed'])['total_volume'].sum().reset_index()
    
    # Then group by merged road type and compute variance of total volume across seeds
    road_type_variance_merged = road_type_volumes_by_seed.groupby('road_type_merged').agg({
        'total_volume': ['var', 'mean', 'count']
    }).reset_index()
    road_type_variance_merged.columns = ['road_type', 'variance_of_total', 'mean_total_volume', 'num_seeds']
    road_type_variance_merged = road_type_variance_merged.dropna(subset=['variance_of_total'])
    road_type_variance_merged = road_type_variance_merged.sort_values('variance_of_total', ascending=False)
    
    return road_type_stats, overall_avg_variance, road_type_variance, road_type_stats_merged, road_type_variance_merged


def analyze_city(city_path: Path, city_name: str) -> Dict:
    """
    Analyze volume variance for a single city.
    
    Returns dictionary with results.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing city: {city_name}")
    print(f"{'='*60}")
    
    # Load all seed data
    print("Loading seed data...")
    df = load_all_seeds(city_path)
    
    if df.empty:
        print(f"No data loaded for {city_name}")
        return {}
    
    print(f"Loaded {len(df)} link-seed combinations from {df['seed'].nunique()} seeds")
    print(f"Total unique links: {df['link'].nunique()}")
    
    # Compute variance statistics
    print("\nComputing variance statistics...")
    road_type_stats, overall_avg_variance, road_type_variance, road_type_stats_merged, road_type_variance_merged = compute_variance_by_road_type(df)
    
    if road_type_stats.empty:
        print("No variance data computed")
        return {}
    
    # Print results - average variance per link by road type (original)
    print(f"\n{'='*60}")
    print("Average Variance per Link by Road Type (Original)")
    print(f"{'='*60}")
    print(f"{'Road Type':<20} {'Avg Variance':<15} {'Link Count':<12}")
    print("-" * 50)
    for _, row in road_type_stats.iterrows():
        print(f"{row['road_type']:<20} {row['avg_variance']:<15.2f} {int(row['link_count']):<12}")
    
    print(f"\n{'Overall Average Variance (per link):':<35} {overall_avg_variance:.2f}")
    
    # Print results - average variance per link by road type (merged)
    print(f"\n{'='*60}")
    print("Average Variance per Link by Road Type (Merged)")
    print(f"{'='*60}")
    print(f"{'Road Type':<20} {'Avg Variance':<15} {'Link Count':<12}")
    print("-" * 50)
    for _, row in road_type_stats_merged.iterrows():
        print(f"{row['road_type']:<20} {row['avg_variance']:<15.2f} {int(row['link_count']):<12}")
    
    # Print results - variance of total volume per road type across seeds (original)
    print(f"\n{'='*60}")
    print("Variance of Total Volume per Road Type (across seeds) - Original")
    print(f"{'='*60}")
    print(f"{'Road Type':<20} {'Variance':<15} {'Mean Total Vol':<15} {'Seeds':<8}")
    print("-" * 60)
    for _, row in road_type_variance.iterrows():
        print(f"{row['road_type']:<20} {row['variance_of_total']:<15.2f} {row['mean_total_volume']:<15.0f} {int(row['num_seeds']):<8}")
    
    # Print results - variance of total volume per road type across seeds (merged)
    print(f"\n{'='*60}")
    print("Variance of Total Volume per Road Type (across seeds) - Merged")
    print(f"{'='*60}")
    print(f"{'Road Type':<20} {'Variance':<15} {'Mean Total Vol':<15} {'Seeds':<8}")
    print("-" * 60)
    for _, row in road_type_variance_merged.iterrows():
        print(f"{row['road_type']:<20} {row['variance_of_total']:<15.2f} {row['mean_total_volume']:<15.0f} {int(row['num_seeds']):<8}")
    
    return {
        'city': city_name,
        'road_type_stats': road_type_stats,
        'road_type_stats_merged': road_type_stats_merged,
        'road_type_variance': road_type_variance,
        'road_type_variance_merged': road_type_variance_merged,
        'overall_avg_variance': overall_avg_variance,
        'num_seeds': df['seed'].nunique(),
        'num_links': df['link'].nunique()
    }


def save_city_results(results: Dict, city_name: str, output_dir: Path):
    """Save results for a single city."""
    if not results:
        return
    
    # Save average variance per link by road type (original)
    output_file = output_dir / f'{city_name}_avg_variance_per_link_by_road_type.csv'
    results['road_type_stats'].to_csv(output_file, index=False)
    print(f"Average variance per link by road type (original) saved to: {output_file}")
    
    # Save average variance per link by road type (merged)
    output_file_merged = output_dir / f'{city_name}_avg_variance_per_link_by_road_type_merged.csv'
    results['road_type_stats_merged'].to_csv(output_file_merged, index=False)
    print(f"Average variance per link by road type (merged) saved to: {output_file_merged}")
    
    # Save variance of total volume per road type (original)
    output_file2 = output_dir / f'{city_name}_variance_of_total_volume_by_road_type.csv'
    results['road_type_variance'].to_csv(output_file2, index=False)
    print(f"Variance of total volume per road type (original) saved to: {output_file2}")
    
    # Save variance of total volume per road type (merged)
    output_file2_merged = output_dir / f'{city_name}_variance_of_total_volume_by_road_type_merged.csv'
    results['road_type_variance_merged'].to_csv(output_file2_merged, index=False)
    print(f"Variance of total volume per road type (merged) saved to: {output_file2_merged}")
    
    # Save summary
    summary_file = output_dir / f'{city_name}_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"City: {city_name}\n")
        f.write(f"Number of seeds: {results['num_seeds']}\n")
        f.write(f"Number of links: {results['num_links']}\n")
        f.write(f"Overall average variance (per link): {results['overall_avg_variance']:.2f}\n")
    print(f"Summary saved to: {summary_file}")


def main():
    """Main function to analyze all cities."""
    project_root = Path(__file__).parent.parent
    basecases_path = project_root / 'data' / 'bavaria' / 'basecases'
    
    if not basecases_path.exists():
        print(f"Error: {basecases_path} does not exist")
        return
    
    # Get all city directories
    city_dirs = sorted([d for d in basecases_path.iterdir() if d.is_dir()])
    
    if not city_dirs:
        print(f"No city directories found in {basecases_path}")
        return
    
    print(f"Found {len(city_dirs)} cities to process")
    print(f"Cities: {[d.name for d in city_dirs]}")
    
    output_dir = project_root / 'results' / 'volume_variance'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each city
    all_results = []
    for city_dir in city_dirs:
        city_name = city_dir.name
        
        try:
            # Analyze the city
            results = analyze_city(city_dir, city_name)
            
            if results:
                # Save results
                save_city_results(results, city_name, output_dir)
                all_results.append(results)
            else:
                print(f"Warning: No results for {city_name}, skipping...")
                
        except Exception as e:
            print(f"Error processing {city_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary across all cities
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY ACROSS ALL CITIES")
        print(f"{'='*60}")
        print(f"{'City':<20} {'Seeds':<8} {'Links':<10} {'Overall Avg Variance':<20}")
        print("-" * 60)
        for r in all_results:
            print(f"{r['city']:<20} {r['num_seeds']:<8} {r['num_links']:<10} {r['overall_avg_variance']:<20.2f}")
        
        # Save combined summary
        summary_all_file = output_dir / 'all_cities_summary.csv'
        summary_df = pd.DataFrame([
            {
                'city': r['city'],
                'num_seeds': r['num_seeds'],
                'num_links': r['num_links'],
                'overall_avg_variance': r['overall_avg_variance']
            }
            for r in all_results
        ])
        summary_df.to_csv(summary_all_file, index=False)
        print(f"\nCombined summary saved to: {summary_all_file}")
    
    print(f"\n{'='*60}")
    print(f"Processing complete! Processed {len(all_results)} cities successfully.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

