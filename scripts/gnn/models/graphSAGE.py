import os
import sys

import wandb

import torch
from torch import nn
from torch_geometric.nn import SAGEConv, GraphNorm

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
                 dropout: float = 0.3, 
                 use_dropout: bool = False,
                 dtype: torch.dtype = torch.float32,
                 log_to_wandb: bool = True,
                 use_target_standardization: bool = False,
                 target_normalization: str = None,
                 
                 # POSITIONAL ENCODING PARAMETERS
                 use_pos: bool = True,
                 pos_dim: int = 6,
                 use_lap_pe: bool = True,
                 lap_pe_dim: int = 8,
                 
                 # Graph Structure Parameters
                 hidden_channels: list[int] = [256, 512, 1024, 512, 256],
                 use_graph_norm: bool = False,
                 use_residuals: bool = False,
                 
                 # GraphSAGE specific parameters
                 aggregator: str = 'mean', # Options: 'max', 'mean'
                 update_function: str = 'relu',
                 mlp_hidden_dim: int = 1024):

        # Calculate effective input channels
        effective_in_channels = in_channels
        if use_pos:
            effective_in_channels += pos_dim
        if use_lap_pe:
            effective_in_channels += lap_pe_dim

        # Call parent class constructor
        super().__init__(
            in_channels=effective_in_channels,
            out_channels=out_channels,
            dropout=dropout,
            use_dropout=use_dropout,
            dtype=dtype,
            log_to_wandb=log_to_wandb,
            use_target_standardization=use_target_standardization,
            target_normalization=target_normalization)
        
        # Model-specific parameters
        self.hidden_channels = hidden_channels
        self.use_graph_norm = use_graph_norm
        self.use_residuals = use_residuals
        self.mlp_hidden_dim = mlp_hidden_dim
        self.use_pos = use_pos
        self.pos_dim = pos_dim
        self.use_lap_pe = use_lap_pe
        self.lap_pe_dim = lap_pe_dim

        if aggregator not in ['max', 'mean']:
            raise ValueError(f"Unsupported aggregator: {aggregator}. Choose 'max' or 'mean'.")
        else:
            self.aggregator = aggregator
            
        if update_function not in ['relu', 'mlp']:
            raise ValueError(f"Unsupported update function: {update_function}. Choose 'relu' or 'mlp'.")
        else:
            self.update_function = update_function

        if self.log_to_wandb:
            wandb.config.update({
                'in_channels': self.in_channels,
                'feature_in_channels': in_channels,
                'hidden_channels': hidden_channels,
                'use_graph_norm': use_graph_norm,
                'use_residuals': use_residuals,
                'aggregator': aggregator,
                'update_function': update_function,
                'mlp_hidden_dim': mlp_hidden_dim,
                'use_pos': use_pos,
                'pos_dim': pos_dim,
                'use_lap_pe': use_lap_pe,
                'lap_pe_dim': lap_pe_dim,
            }, allow_val_change=True)

        # Define the layers
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
                
            # Define the convolutional layer
            conv = SAGEConv(
                in_channels, self.hidden_channels[i],
                aggr=self.aggregator,   # 'mean', 'max', or 'lstm'
                root_weight=True,
                normalize=True,
                bias=True,
                project=(in_channels != self.hidden_channels[i])) #TODO: To change to True if we want to project the features to the hidden_channels[i]
            
            setattr(self, f'conv{i + 1}', conv)

            if self.use_graph_norm:
                graph_norm = GraphNorm(self.hidden_channels[i-1] if i > 0 else self.in_channels)
                setattr(self, f'graph_norm{i + 1}', graph_norm)

            if self.update_function == 'mlp': # Use this if we want a second layer of non-linearity
                
                # Define a small MLP for learnable update
                update_mlp = nn.Sequential(
                    nn.Linear(self.hidden_channels[i], self.mlp_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.mlp_hidden_dim, self.hidden_channels[i])
                ) # TODO: Missing end activation?
                
                setattr(self, f'update_mlp{i + 1}', update_mlp)
        
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)
        
        self.fc = nn.Linear(self.hidden_channels[-1], self.out_channels)

    def forward(self, data):

        # Unpack data
        x = data.x
        edge_index = data.edge_index

        # Add pos coordinates
        if self.use_pos:
            
            start_pos = data.pos[:, 0, :]
            end_pos = data.pos[:, 1, :]
            mid_pos = data.pos[:, 2, :]
            
            # Concatenate along the feature dimension
            if self.pos_dim == 2:
                x = torch.cat((x, mid_pos), dim=-1)
            elif self.pos_dim == 4:
                x = torch.cat((x, start_pos, end_pos), dim=-1)
            elif self.pos_dim == 6:
                x = torch.cat((x, start_pos, mid_pos, end_pos), dim=-1)  
            else:
                raise ValueError(f"Unsupported pos_dim: {self.pos_dim}. Supported: 2, 4, 6.")

        # Add Laplacian PE
        if self.use_lap_pe:
            if hasattr(data, 'lap_pe') and data.lap_pe is not None:
                if self.lap_pe_dim > data.lap_pe.size(1):
                    raise ValueError(f"{self.lap_pe_dim} is greater than LAP PE dimensions available in data = {data.lap_pe.size(1)}") 
                else:
                    lap_pe = data.lap_pe[:, :self.lap_pe_dim]
                    x = torch.cat((x, lap_pe), dim=-1)
            else:
                raise ValueError("Laplacian positional encodings not found in data object!")
        
        x = x.to(self.dtype)

        # TODO: Check the flow, seems dubious!
        # Apply GraphSAGE layers
        for i in range(len(self.hidden_channels)):
            
            x_0 = x
                
            conv = getattr(self, f'conv{i + 1}')
            x = conv(x, edge_index)
            
            # Graph normalization
            if self.use_graph_norm:
                graph_norm = getattr(self, f'graph_norm{i + 1}')
                x = graph_norm(x)

            # Apply your chosen "update function"
            if self.update_function == 'relu':
                x = nn.functional.relu(x)
            else:
                x = getattr(self, f'update_mlp{i + 1}')(x)
            
            # Dropout
            if self.use_dropout:
                x = self.dropout_layer(x)
                
            # Residual connection
            if self.use_residuals and i > 0 and self.hidden_channels[i] == self.hidden_channels[i - 1]:
                x = x + x_0

        # Read out predictions
        return self.fc(x)