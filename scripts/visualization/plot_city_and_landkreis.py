#!/usr/bin/env python3
"""
Generate one figure per city highlighting the simulation city boundary and the
corresponding Landkreis (commuter catchment) boundary.

The script is intentionally opinionated about a few things:

1. It expects a boundary file per city that includes the polygon for the
   kreisfreie Stadt as well as the polygon for the Landkreis (or commuter catchment).
   - By default we look for files in ``data/inductive_data/links_and_stats/city_boundaries/{city}/{city}.json``.
   - Override this layout with ``--boundary-template`` such as
     ``data/links_and_stats/city_boundaries/{city}.geojson``.
2. If the boundary file contains multiple polygons, the smaller area is treated
   as the city core and the largest as the Landkreis. This matches the data
   organisation of the Bavarian simulations where agents live in the Landkreis
   and policy interventions target the kreisfreie Stadt.
3. The list of cities defaults to the ones present in ``data/city_embedding.csv``
   (first column named ``city``). Provide ``--cities`` to supply an explicit list.

Example:

    python scripts/visualization/plot_city_and_landkreis.py \\
        --boundary-template "data/inductive_data/links_and_stats/city_boundaries/{city}/{city}.json" \\
        --output-dir plots/city_landkreis

Dependencies: geopandas, shapely, matplotlib.
Install them via the project environment (``conda env update --file traffic-gnn.yml``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

try:
    import geopandas as gpd
except ImportError as exc:  # pragma: no cover - dependency missing at runtime
    raise SystemExit(
        "geopandas is required. Install the project environment or run "
        "`pip install geopandas` in your active environment."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOUNDARY_TEMPLATE = "data/inductive_data/links_and_stats/city_boundaries/{city}/{city}.json"
DEFAULT_CITY_EMBEDDING = PROJECT_ROOT / "data/city_embedding.csv"


def _resolve_path(path: Path) -> Path:
    """Return absolute path, anchoring relative paths at the project root."""
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_city_list(csv_path: Path) -> List[str]:
    csv_path = _resolve_path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"City embedding csv not found at {csv_path}. "
            "Provide --cities explicitly or place the csv."
        )
    df = pd.read_csv(csv_path)
    if "city" not in df.columns:
        raise ValueError(f"'city' column missing in {csv_path}")
    cities = df["city"].astype(str).str.strip().str.lower().tolist()
    if not cities:
        raise ValueError(f"No cities found in {csv_path}")
    return cities


def resolve_boundary_path(template: str, city: str) -> Path:
    path = Path(template.format(city=city))
    path = _resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Boundary file for '{city}' not found: {path}. "
            "Adjust --boundary-template to match your data layout."
        )
    return path


def split_city_and_landkreis(gdf: gpd.GeoDataFrame, city_name: str) -> Tuple[gpd.GeoSeries, gpd.GeoSeries]:
    """Split GeoDataFrame into city and Landkreis polygons.

    Heuristics:
      1. Drop non-polygon geometries and empty rows.
      2. If an attribute column already marks the type, use it.
      3. Otherwise, use area (smaller polygon -> city, larger -> Landkreis).
    """
    if gdf.empty:
        raise ValueError(f"Boundary GeoDataFrame for '{city_name}' is empty.")

    # Keep only polygonal geometries
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        raise ValueError(f"No polygon geometries found for '{city_name}'.")

    # Normalize geometry CRS to visualize consistently (prefer EPSG:3857 for plotting)
    if gdf.crs is None:
        # default to WGS84 assumptions
        gdf = gdf.set_crs(epsg=4326)
    try:
        gdf = gdf.to_crs(epsg=3857)
    except Exception:
        # Fall back silently; plotting still works in native CRS.
        pass

    # Try attribute-based identification first
    candidate_columns = [c for c in gdf.columns if c not in gdf.geometry.name]
    type_series = None
    for col in candidate_columns:
        series = gdf[col].astype(str).str.lower()
        has_city = series.str.contains("stadt").any() or series.str.contains(city_name).any()
        has_landkreis = series.str.contains("kreis").any()
        if has_city or has_landkreis:
            type_series = series
            break

    if type_series is not None:
        city_mask = type_series.str.contains("stadt") | type_series.str.contains(city_name)
        landkreis_mask = type_series.str.contains("kreis") & ~city_mask
        city_gdf = gdf[city_mask]
        landkreis_gdf = gdf[landkreis_mask]
        if not city_gdf.empty and not landkreis_gdf.empty:
            return city_gdf.geometry, landkreis_gdf.geometry

    # Fallback: infer by area (smallest polygon -> city, largest -> Landkreis)
    areas = gdf.geometry.area
    if len(areas) < 2:
        # If only one polygon is provided, use it for both layers.
        # This happens for cities whose Landkreis equals the Stadt extent.
        warning = (
            f"Only one boundary polygon found for '{city_name}'. "
            "Using the same geometry for city and Landkreis."
        )
        print(f"[INFO] {warning}")
        geom = gdf.geometry
        return geom, geom
    city_idx = areas.idxmin()
    landkreis_idx = areas.idxmax()
    return gdf.loc[[city_idx], "geometry"], gdf.loc[[landkreis_idx], "geometry"]


def plot_city(city: str, boundary_path: Path, output_dir: Path) -> Path:
    gdf = gpd.read_file(boundary_path)
    city_geom, landkreis_geom = split_city_and_landkreis(gdf, city)

    fig, ax = plt.subplots(figsize=(6, 6))
    landkreis_geom.plot(ax=ax, color="#f1f1f1", edgecolor="#666666", linewidth=1.0, label="Landkreis")
    city_geom.plot(ax=ax, color="#f08080", edgecolor="#8b0000", linewidth=1.2, alpha=0.7, label="City")

    ax.set_title(f"{city.title()}: Stadt vs. Landkreis", fontsize=12)
    ax.set_axis_off()
    ax.legend(loc="upper right")

    output_path = output_dir / f"{city}_stadt_landkreis.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot each city with its Stadt and Landkreis boundaries."
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        help="Explicit list of city folder names (override city_embedding.csv).",
    )
    parser.add_argument(
        "--city-embedding",
        type=Path,
        default=DEFAULT_CITY_EMBEDDING,
        help="Path to city_embedding.csv (used when --cities is not given).",
    )
    parser.add_argument(
        "--boundary-template",
        default=DEFAULT_BOUNDARY_TEMPLATE,
        help="Template for boundary files. Use {city} placeholder. "
             "Default: data/inductive_data/links_and_stats/city_boundaries/{city}/{city}.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/city_landkreis"),
        help="Directory to store generated figures.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if args.cities:
        cities = [c.strip().lower() for c in args.cities]
    else:
        cities = read_city_list(args.city_embedding)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []
    for city in cities:
        try:
            boundary_path = resolve_boundary_path(args.boundary_template, city)
            output_path = plot_city(city, boundary_path, output_dir)
            print(f"[OK] {city}: saved {output_path}")
        except Exception as exc:  # pylint: disable=broad-except
            failures.append(f"{city}: {exc}")
            print(f"[WARN] {city}: {exc}", file=sys.stderr)

    if failures:
        print(
            "\nSome cities could not be plotted. "
            "Check boundary files or adjust --boundary-template.",
            file=sys.stderr,
        )
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

