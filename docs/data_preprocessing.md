# Data Preprocessing

`scripts/data_preprocessing/process_simulations_for_gnn.py` can be used to preprocess the raw MATSim simulation data for GNNs. The following paths need to be specified in the script before running it:

- `sim_input_path`: Path to the MATSim simulation data.
- `result_path`: Path to save the preprocessed data.
- `basecase_links_path`: Basecase graph (mean over 50 runs) in geojson format, without any policies applied.
- `basecase_stats_path`: Some travel mode statistics for the basecase, in csv format.
- `discricts_gdf_path`: Path to the districts shapefile, in geojson format.

Additionaly, the flag `use_allowed_modes` can be set to `True` if needed. This then adds the allowed transport modes (per road segment, bool values) to the features of the output graphs.

## Input Format

[TODO] Update this section!

## Output Format

PyTorch Geometric (PyG) data batches are saved in the `result_path` as `.pt` tensor files. Each sample is a homogenous graph with the following attributes per node (road segment):
- `x`: Node features:
    - Volume Base Case
    - Capacity Base Case
    - Capacity Reduction
    - Maximum Speed
    - Road Type
    - Length
    And additionally, if `use_allowed_modes` is `True`, then booleans indicating whether `Car`, `Bus`, `Public Transport`, `Train`, `Rail`, and `Subway` are allowed on the road segment.
- `y`: Target for the GNN, difference in traffic volume between the base case and the simulation run (with policy applied).
- `pos`: x and y coordinates of the start, middle, and end of the road segment.