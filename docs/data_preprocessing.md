# Data Preprocessing

[`process_simulations_for_gnn.py`](../scripts/data_preprocessing/process_simulations_for_gnn.py) can be used to preprocess the raw MATSim simulation data for GNNs. The following paths need to be specified in the script before running it:

- `sim_input_path`: Path to the MATSim simulation data, multiple directories are allowed.
- `result_path`: Path to save the preprocessed data.
- `basecase_links_path`: Basecase graph (mean over 50 runs) in geojson format, without any policies applied.
- `basecase_stats_path`: Travel mode statistics for the basecase, in csv format. Includes:
    - avg_trav_time_seconds
    - avg_traveled_distance
    - average_trip_count

Additionaly, the following flags can be set:

- `use_allowed_modes`: If `True`, adds the allowed transport modes (per road segment, bool values) to the features of the output graphs.
- `use_linegraph`: Defaults to `True`, where dual graph is created: Roads segments are the nodes, and the edges are the connections between them.

## Input Format

Each directory specified in `sim_input_path` should contain subdirectories named `network_d_*`, where `*` are the districts where the policy was applied, separated by uderscores. For example, `network_d_1_2_3_4_16_17`. Each one of these scenarios (with policy being applied to different combinations of districts) have the following files:
- `eqasim_pt.csv`
- `eqasim_trips.csv`
- `output_links.csv.gz`

These scenarios act as individual samples for the GNN, i.e. runs simulating the effects of applying the policies. The script will create an output graph for each of these scenarios. Please refer to [eqasim](https://github.com/eqasim-org/eqasim-java) for more information on preparing the simulation data.

## Output Format

PyTorch Geometric (PyG) data batches are saved in the `result_path` as `.pt` tensor files. Each sample is a homogenous graph (representing a scenario as decribed above) with the following attributes per road segment:
- `x`: Node/Edge features (depending on `use_linegraph`) including:
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