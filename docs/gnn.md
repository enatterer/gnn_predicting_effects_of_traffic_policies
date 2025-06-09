# Graph Neural Networks (GNNs)

The `gnn/models/` directory can be used to define GNN model classes. These classes must inherit from [`BaseGNN`](../scripts/gnn/models/base_gnn.py) and implement the following methods:

- `__init__`: This method should initialize the model parameters and layers. It should also call the `super().__init__()` method to ensure that the base class is properly initialized. In addition, model specific parameters can be logged to WandB using `wandb.config.model_kwargs = model_kwargs` where `model_kwargs` is a dictionary.
- `define_layers`: Define the model architecture. It should create instances of the layers and store them as attributes of the model class.
- `forward`: Define the forward pass of the model. It should take the input `data` and pass it through the model layers to produce the output.

[`BaseGNN`](../scripts/gnn/models/base_gnn.py) already defines some basic parameters in the `__init__` method. Additionaly, `initialize_wights` and `train_model` (training pipeline) methods are also provided, which can be overridden if needed. Please see [`PointNetTransfGAT`](../scripts/gnn/models/point_net_transf_gat.py) for an example. Note that if `predict_mode_stats` is intended to be used, the model should also return predicted travel mode statistics (as present in the dataset).

Some I/O functions and helper functions are provided in [`gnn_io.py`](../scripts/gnn/gnn_io.py) and [`help_functions.py`](../scripts/gnn/help_functions.py) respectively.