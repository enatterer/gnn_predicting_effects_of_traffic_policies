# Data Preprocessing

[`process_simulations_for_gnn.py`](../scripts/data_preprocessing/process_simulations_for_gnn.py) can be used to preprocess the raw MATSim simulation data for GNNs. The following paths need to be specified in the script before running it:

- `sim_input_path`: Path to the MATSim simulation data, multiple directories are allowed.
- `result_path`: Path to save the preprocessed data.
- `basecase_links_path`: Basecase graph (mean over 50 runs) in geojson format, without any policies applied.
- `basecase_stats_path`: Travel mode statistics for the basecase, in csv format. Includes:
    - avg_trav_time_seconds
    - avg_traveled_distance
    - average_trip_count
- `discricts_gdf_path`: Path to the districts shapefile, in geojson format.

Additionaly, the flag `use_allowed_modes` can be set to `True` if needed. This then adds the allowed transport modes (per road segment, bool values) to the features of the output graphs.

## Input Format

The script expects folders named `output_networks_*` in each directory specified in `sim_input_path`. For example, we kept 1000 simulations per folder, naming them `output_networks_1000`, `output_networks_2000`, etc.

Each such folder then contains subdirectories named `network_d_*`, where `*` are the districts where the policy was applied, separated by uderscores. For example, `network_d_1_2_3_4_16_17`. Each one of these scenarios (with policy being applied to different combinations of districts) have the following files:
- `eqasim_pt.csv`
- `eqasim_trips.csv`
- `output_links.csv.gz`

These scenarios act as individual samples for the GNN, i.e. runs simulating the effects of applying the policies. The script will create an output graph for each of these scenarios. Please refer to [eqasim](https://github.com/eqasim-org/eqasim-java) for more information on preparing the simulation data.

## Output Format

PyTorch Geometric (PyG) data batches are saved in the `result_path` as `.pt` tensor files. Each sample is a homogenous graph (representing a scenario as decribed above) with the following attributes per node (road segment):
- `x`: Node features:
    - Volume Base Case
    - Capacity Base Case
    - Capacity Reduction
    - Maximum Speed
    - Road Type
    - Length
    - And additionally, if `use_allowed_modes` is `True`, then booleans indicating whether `Car`, `Bus`, `Public Transport`, `Train`, `Rail`, and `Subway` are allowed on the road segment.
- `y`: Target for the GNN, difference in traffic volume between the base case and the simulation run (with policy applied).
- `pos`: x and y coordinates of the start, middle, and end of the road segment.

And the following attributes for the entire graph:
- `edge_index`: Edges of the graph, defined by the start and end nodes of each edge.
- `mode_stats_diff`: Difference in travel mode statistics between the base case and the simulation run (with policy applied).
- `mode_stats_diff_per`: `mode_stats_diff` in percentage (compared to the base case).

## Dataset

The data is too large to be shared publicly. Please contact us if you would like to access it, or see some samples :-)