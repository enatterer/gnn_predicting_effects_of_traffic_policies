import os
import sys

import tqdm as tqdm
import wandb

import torch
from torch import nn

from torch_geometric.nn import TransformerConv, GraphNorm

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.base_gnn import BaseGNN

class TransConv(BaseGNN):
    def __init__(self, 
                in_channels: int = 5, 
                use_pos: bool = False,
                out_channels: int = 1,
                hidden_channels: list[int] = [128,256,512,256,128],
                use_graph_norm: bool = False,
                use_residuals: bool = False,
                num_heads: int = 4,
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
        
        # Model specific parameters
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.use_pos = use_pos
        self.use_graph_norm = use_graph_norm
        self.use_residuals = use_residuals

        if self.use_pos:
            self.in_channels += 4 # x and y for start and end points
        
        if self.log_to_wandb:
            wandb.config.update({'hidden_channels': hidden_channels,
                                'num_heads': num_heads,
                                'use_pos': use_pos,
                                'in_channels': self.in_channels,
                                'use_graph_norm': use_graph_norm,
                                'use_residuals': use_residuals},
                                allow_val_change=True)
        
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
            conv = TransformerConv(in_channels, int(self.hidden_channels[i]/self.num_heads), heads=self.num_heads)
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

        if self.use_pos:
            pos1 = data.pos[:, 0, :]  # Start position
            pos2 = data.pos[:, 1, :]  # End position
            x = torch.cat((x, pos1, pos2), dim=1)  # Concatenate along the feature dimension

        x = x.to(self.dtype)

        for i in range(len(self.hidden_channels)):

            if self.use_residuals and i > 0 and self.hidden_channels[i] == self.hidden_channels[i - 1]:
                x_0 = x

            if self.use_graph_norm:
                graph_norm = getattr(self, f'graph_norm{i + 1}')
                x = graph_norm(x)

            conv = getattr(self, f'conv{i + 1}')
            x = conv(x, edge_index)

            # Makes sense?
            if self.use_residuals and i > 0 and self.hidden_channels[i] == self.hidden_channels[i - 1]:
                x = x + x_0

            x = nn.functional.relu(x)
            
            if self.use_dropout:
                x = self.dropout_layer(x)

        # Read out predictions
        x = self.fc(x)
        
        return x