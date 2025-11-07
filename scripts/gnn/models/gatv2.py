import os
import sys

import wandb

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GraphNorm

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.base_gnn import BaseGNN

class GATv2(BaseGNN):
    def __init__(self, 
                in_channels: int = 5, 
                out_channels: int = 1,
                dropout: float = 0.3, 
                use_dropout: bool = False,
                dtype: torch.dtype = torch.float32,
                log_to_wandb: bool = True,
                use_target_standardization: bool = False,
                
                # Graph Structure Parameters
                hidden_channels: list[int] = [256,512,1024,512,256],
                num_heads: int = 4,
                use_graph_norm: bool = False,
                use_residuals: bool = False,
                
                # GATv2 Specific Parameters
                share_weights: bool = False,
                negative_slope: float = 0.2,
                add_self_loops: bool = True,
                
                # POSITIONAL ENCODING PARAMETERS
                use_pos: bool = False,
                use_lap_pe: bool = False,
                lap_pe_dim: int = 8):

        # CALCULATE EFFECTIVE INPUT CHANNELS
        effective_in_channels = in_channels
        if use_pos:
            effective_in_channels += 2  # x, y coordinates
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
            use_target_standardization=use_target_standardization)
        
        # Model specific parameters
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.use_graph_norm = use_graph_norm
        self.use_residuals = use_residuals
        self.use_pos = use_pos
        self.use_lap_pe = use_lap_pe
        self.lap_pe_dim = lap_pe_dim
        self.share_weights = share_weights
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops

        if self.log_to_wandb:
            wandb.config.update({
                'in_channels': self.in_channels,
                'feature_in_channels': in_channels,
                'hidden_channels': hidden_channels,
                'num_heads': num_heads,
                'use_pos': use_pos,
                'use_lap_pe': use_lap_pe,
                'lap_pe_dim': lap_pe_dim,
                'use_graph_norm': use_graph_norm,
                'use_residuals': use_residuals,
                'share_weights': share_weights,
                'negative_slope': negative_slope,
                'add_self_loops': add_self_loops,
            }, allow_val_change=True)
        
        # Define the layers of the model
        self.define_layers()
        
        # Initialize weights
        self.initialize_weights()

    def define_layers(self):
        
        for i in range(len(self.hidden_channels)):
            if i == 0:
                in_channels = self.in_channels
            else:
                in_channels = self.hidden_channels[i - 1]

            # Define the convolutional layer
            conv = GATv2Conv(in_channels, int(self.hidden_channels[i]/self.num_heads), heads=self.num_heads,
                             share_weights=self.share_weights, negative_slope=self.negative_slope, add_self_loops=self.add_self_loops)    
            
            setattr(self, f'conv{i + 1}', conv)

            if self.use_graph_norm:
                graph_norm = GraphNorm(self.hidden_channels[i-1] if i > 0 else self.in_channels)
                setattr(self, f'graph_norm{i + 1}', graph_norm)

        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)

        self.fc = nn.Linear(self.hidden_channels[-1], self.out_channels)

    def forward(self, data):

        # Unpack data
        x = data.x
        edge_index = data.edge_index

        # Add pos coordinates
        if self.use_pos:
            pos = data.pos[:, 2, :]  # Middle position
            x = torch.cat((x, pos), dim=-1)  # Concatenate along the feature dimension

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

        for i in range(len(self.hidden_channels)):
            
            # Residual connection
            if self.use_residuals and i > 0 and self.hidden_channels[i] == self.hidden_channels[i - 1]:
                x_0 = x

            # Graph normalization
            if self.use_graph_norm:
                graph_norm = getattr(self, f'graph_norm{i + 1}')
                x = graph_norm(x)
                
            conv = getattr(self, f'conv{i + 1}')
            x = conv(x, edge_index)

            # Residual connection
            if self.use_residuals and i > 0 and self.hidden_channels[i] == self.hidden_channels[i - 1]:
                x = x + x_0

            x = nn.functional.relu(x)

            # Dropout
            if self.use_dropout:
                x = self.dropout_layer(x)

        # Read out predictions
        x = self.fc(x)
        
        return x