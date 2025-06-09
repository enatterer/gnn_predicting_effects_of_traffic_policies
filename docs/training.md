# Running Training Experiments

[`run_models.py`](../scripts/training/run_models.py) can be used to for GNN model training with configurable architecture and hyperparameters. [`run_models.ipynb`](../scripts/training/run_models.ipynb) can be used to run the same code in an interactive environment.

Before staring, the following paths need to be adjusted:
- `dataset_path`: Path to preprocessed data, the output from [`process_simulations_for_gnn.py`](../scripts/data_preprocessing/process_simulations_for_gnn.py).
- `base_dir`: Base directory to save the results.

All the other parameters can be passed as command line arguments. Run `python run_models.py --help` to see the list of available arguments.

Example usage with default architecture, dropout, and most significant features found using ablation tests:
</br> `python run_models.py --in_channels 5 --use_all_features False --num_epochs 500 --lr 0.003 --early_stopping_patience 25 --use_dropout True --dropout 0.3`

## Results

The results are primarily logged using [WandB](https://wandb.ai/), structured as decribed by the arguments `project_name` and `unique_model_description` (an initial api setup is required). Locally, the results are saved under `base_dir/project_name/unique_model_description/`, it contains:
- `data_created_during_training`:
    - `x` and `pos` scalers (normalization params) for train, validation, and test sets.
    - Test set in `.pt` format.
    - Test data loader and parameters.

- `trained_models`:
    - `model.pth`: Best model state.
    - Checkpoints every 20 epochs.