import os
import sys

import tqdm as tqdm
import wandb

import torch
from torch import nn
from torch_geometric.data import Batch, Data

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.base_gnn import BaseGNN

class FC_NN(BaseGNN):
    def __init__(self, 
                in_channels: int = 5,
                out_channels: int = 1,
                hidden_channels: list[int] = [32, 8],
                num_nodes: int = 31635,
                use_pos: bool = False,
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
        self.use_pos = use_pos
        self.num_nodes = num_nodes

        # Might give it some sence of Graph structure
        if self.use_pos:
            self.in_channels += 2 # x and y for mid pos, might blow up!

        if self.log_to_wandb:
            wandb.config.update({'hidden_channels': hidden_channels,
                                'use_pos': use_pos,
                                'in_channels': self.in_channels,
                                'num_nodes': num_nodes},
                                allow_val_change=True)
        
        # Define the layers of the model
        self.define_layers()

        # Initialize weights
        self.initialize_weights()

    def define_layers(self):
        
        for i in range(len(self.hidden_channels)):
            if i == 0:
                in_channels = self.in_channels*self.num_nodes
            else:
                in_channels = self.hidden_channels[i - 1]

            out_channels = self.hidden_channels[i]

            # Define the fully connected layer
            setattr(self, f'fc{i + 1}', nn.Linear(in_channels, out_channels))

        self.read_out = nn.Linear(self.hidden_channels[-1], self.out_channels*self.num_nodes)
        
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)


    def forward(self, data):

        # Unpack data
        if isinstance(data, Batch):
            datalist = data.to_data_list()
        elif isinstance(data, Data):
            datalist = [data]
        else:
            raise ValueError("Input data must be a Batch or Data object")

        # Reshape x to (batch_size, num_nodes * in_channels)
        x = [data.x.flatten() for data in datalist]
        x = torch.stack(x)

        if self.use_pos:
            flattened_pos = [data.pos[:,2,:].flatten() for data in datalist]
            flattened_pos = torch.stack(flattened_pos)
            x = torch.cat((x, flattened_pos), dim=1)  # Concatenate along the feature dimension

        x = x.to(self.dtype)

        for i in range(len(self.hidden_channels)):
            fc_layer = getattr(self, f'fc{i + 1}')
            x = fc_layer(x)
            x = nn.functional.relu(x)
            if self.use_dropout:
                x = self.dropout_layer(x)

        # Read out predictions
        x = self.read_out(x)
        
        return x.reshape(-1, 1)