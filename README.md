# Machine Learning Surrogates for Agent-Based Models in Transportation Policy Analysis

Effective traffic policies are crucial for managing congestion and reducing emissions. Agent-based transportation models (ABMs) offer a detailed analysis of how these policies affect travel behaviour at a granular level. However, computational constraints limit the number of scenarios that can be tested with ABMs and therefore their ability to find optimal policy settings.In this proof-of-concept study, we propose a machine learning (ML)-based surrogate model to efficiently explore this vast solution space. Leveraging a Graph Neural Network (GNN), the model predicts the effects of traffic policies on the road network at the link level.  We implement our approach in a large-scale MATSim simulation of Paris, France, covering over 30,000 road segments and 10,000 simulations, applying a policy involving capacity reduction on main roads. The ML surrogate model achieves an overall R2 of 0.76, with significantly lower Mean Squared Error and Mean Absolute Error than naive baselines. On primary roads where the policy applies, it reaches an R2 of 0.95. This study therefore shows that GNNs can act as surrogates for complex agent-based transportation models with the potential to enable large-scale policy optimization, helping urban planners explore a broader range of interventions more efficiently.

Read more about the project in our [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5182100#).

## Getting Started

To create a virtual environment and install the required packages (using conda), run the following:

```conda env create -f traffic-gnn.yml```

Then, activate the environment:

```conda activate traffic-gnn```

We use Python 3.10.13 along with CUDA 12.7 and CuDNN 8.7 for this project.

## Usage

The project is structured as follows:
- `gnn_predicting_effects_of_traffic_policies`:

    - `data`: Contains data used for training and testing the model, including raw MATSim simulations. Also used to save results from the experiments. (not included in the repo)

    - `docs`: Documentation for the project, including how to run the experiments and use the code.
    
    - `scripts`: 
        - `data_preprocessing`: Data prepration scripts, read more [here](docs/data_preprocessing.md).
        - `gnn`: Define GNN models, and associated helper functions. Read more [here](docs/gnn.md).
        - `training`: Run training experiments, read more [here](docs/training.md).
        - `evaluation`: Benchmark trained models.
        - `misc`: Miscellaneous scripts. (Under Development) 

    - [`run_models.sbatch`](run_models.sbatch): Example batch job script for LRZ AI cluster.

    - [`traffic-gnn.yml`](traffic-gnn.yml): Conda environment file.

</br>The general workflow would look something like:

1. Run MATSim simulations to generate raw data.
2. Preprocess the data for GNNs using [`process_simulations_for_gnn.py`](scripts/data_preprocessing/process_simulations_for_gnn.py).
3. Review GNN models available in [`scripts/gnn/models`](scripts/gnn/models/), or define your own.
4. Train the GNN model using [`run_models.py`](scripts/training/run_models.py).
5. Check model performance with [`test_model.ipynb`](scripts/evaluation/test_model.ipynb). Use [`in_depth_analysis.ipynb`](scripts/evaluation/in_depth_analysis.ipynb) for a more detailed evaluation.
6. (Optional) Use [`feature_importance.py`](scripts/misc/feature_importance.py) to interpret the model's predictions and understand features that influence them.

## License
This project is licensed under the MIT License, see the [LICENSE](LICENSE) for details. Kindly note that the code is provided for research purposes only and may not be suitable for commercial use. Please contact us for more information :-)