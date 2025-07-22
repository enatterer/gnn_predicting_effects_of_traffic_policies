import os
import sys

import tqdm as tqdm
import wandb
import torch
from torch import nn
from torch_geometric.nn import SAGEConv

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.base_gnn import BaseGNN

class GraphSAGE(BaseGNN):
    """
    GraphSAGE model for inductive learning with flexible aggregator and update functions.
    Supports mean aggregation by default with an option for a learnable aggregator.
    Uses a simple ReLU-based update function with an option for a learnable MLP-based update.
    """
    def __init__(self, 
                 in_channels: int = 5, 
                 out_channels: int = 1,
                 hidden_channels: list[int] = [128, 128],
                 aggregator: str = 'mean',  # Options: 'mean', 'learnable'
                 update_function: str = 'relu',  # Options: 'relu', 'mlp'
                 mlp_hidden_dim: int = 64,  # Hidden dim for learnable MLP update
                 dropout: float = 0.3, 
                 use_dropout: bool = False,
                 predict_mode_stats: bool = False,
                 dtype: torch.dtype = torch.float32,
                 log_to_wandb: bool = False):
        
        # Call parent class constructor
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            use_dropout=use_dropout,
            predict_mode_stats=predict_mode_stats,
            dtype=dtype,
            log_to_wandb=log_to_wandb)
        
        # Model-specific parameters
        self.hidden_channels = hidden_channels
        self.aggregator = aggregator
        self.update_function = update_function
        self.mlp_hidden_dim = mlp_hidden_dim
        
        if self.log_to_wandb:
            wandb.config.update({
                'hidden_channels': hidden_channels,
                'aggregator': aggregator,
                'update_function': update_function,
                'mlp_hidden_dim': mlp_hidden_dim,
                'in_channels': self.in_channels
            }, allow_val_change=True)

        # Validate aggregator and update function options
        if aggregator not in ['mean', 'learnable']:
            raise ValueError(f"Unsupported aggregator: {aggregator}. Choose 'mean' or 'learnable'.")
        if update_function not in ['relu', 'mlp']:
            raise ValueError(f"Unsupported update function: {update_function}. Choose 'relu' or 'mlp'.")

        # Define the layers of the model
        self.define_layers()

        # Initialize weights
        self.initialize_weights()

    def define_layers(self):
        """
        Define the GraphSAGE layers, including flexible aggregators and update functions.
        """
        self.layers = nn.ModuleList()
        
        for i in range(len(self.hidden_channels)):
            if i == 0:
                in_channels = self.in_channels
            else:
                in_channels = self.hidden_channels[i - 1]
            out_channels = self.hidden_channels[i]

            if self.aggregator == 'mean':
                # Use standard SAGEConv with mean aggregation
                conv = SAGEConv(in_channels, out_channels, aggr='mean')
            else:  # learnable aggregator
                # Custom learnable aggregator: Linear transformation + mean aggregation
                conv = SAGEConv(in_channels, out_channels, aggr='mean')
                # Optionally, add a learnable layer for neighbor aggregation
                aggr_layer = nn.Linear(in_channels, out_channels)
                setattr(self, f'aggr_layer{i + 1}', aggr_layer)

            self.layers.append(conv)

            if self.update_function == 'mlp':
                # Define a small MLP for learnable update
                update_mlp = nn.Sequential(
                    nn.Linear(out_channels, self.mlp_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.mlp_hidden_dim, out_channels)
                )
                setattr(self, f'update_mlp{i + 1}', update_mlp)

        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)

        self.fc = nn.Linear(self.hidden_channels[-1], self.out_channels)

    def forward(self, data):
        """
        Forward pass with flexible aggregator and update function.
        """
        # Unpack data
        x = data.x.to(self.dtype)
        edge_index = data.edge_index

        for i, conv in enumerate(self.layers):
            if self.aggregator == 'learnable':
                # Apply learnable aggregator (linear transformation before SAGEConv)
                aggr_layer = getattr(self, f'aggr_layer{i + 1}')
                x = aggr_layer(x)  # Transform node features before aggregation

            # Apply GraphSAGE convolution
            x = conv(x, edge_index)

            # Apply update function
            if self.update_function == 'relu':
                x = nn.functional.relu(x)
            else:  # mlp
                update_mlp = getattr(self, f'update_mlp{i + 1}')
                x = update_mlp(x)

            # Apply dropout if enabled
            if self.use_dropout:
                x = self.dropout_layer(x)

        # Final linear layer for predictions
        x = self.fc(x)
        
        return x