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
    Incomplete Masterpiece for Inductive Learning.
    Missing Neighbor Sampling (NeighborSampler is confusing, check internal workings).
    
    Also, make compatible with other baselines.
    """
    def __init__(self, 
                in_channels: int = 5, 
                out_channels: int = 1,
                hidden_channels: list[int] = [128,128],
                aggregator: str = 'mean',
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
        self.aggregator = aggregator
        
        if self.log_to_wandb:
            wandb.config.update({'hidden_channels': hidden_channels,
                                'aggregator': aggregator,
                                'in_channels': self.in_channels},
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
            conv = SAGEConv(in_channels, self.hidden_channels[i], aggr=self.aggregator)
            setattr(self, f'conv{i + 1}', conv)
        
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)

        self.fc = nn.Linear(self.hidden_channels[-1], self.out_channels)

    def forward(self, data):

        # Unpack data
        x = data.x.to(self.dtype)
        edge_index = data.edge_index

        for i in range(len(self.hidden_channels)):
            conv = getattr(self, f'conv{i + 1}')
            x = conv(x, edge_index)
            x = nn.functional.relu(x)
            if self.use_dropout:
                x = self.dropout_layer(x)

        # Read out predictions
        x = self.fc(x)
        
        return x