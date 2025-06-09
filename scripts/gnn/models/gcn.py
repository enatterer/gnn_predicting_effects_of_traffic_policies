import os
import sys

import tqdm as tqdm
import wandb

import torch
from torch import nn

from torch_geometric.nn import GCNConv, GCN2Conv, GraphNorm

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.base_gnn import BaseGNN

class GCN(BaseGNN):
    def __init__(self, 
                in_channels: int = 5,
                use_pos: bool = False,
                out_channels: int = 1,
                hidden_channels: list[int] = [256,512,1024,512,256],
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

        if self.use_pos:
            self.in_channels += 4 # x and y for start and end points

        if self.log_to_wandb:
            wandb.config.update({'hidden_channels': hidden_channels,
                                'use_pos': use_pos,
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
            conv = GCNConv(in_channels, self.hidden_channels[i])
            setattr(self, f'conv{i + 1}', conv)
        
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
            conv = getattr(self, f'conv{i + 1}')
            x = conv(x, edge_index)
            x = nn.functional.relu(x)
            if self.use_dropout:
                x = self.dropout_layer(x)

        # Read out predictions
        x = self.fc(x)
        
        return x
    

class GCN2(BaseGNN):
    """
    See https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/itr2.12551 for reference.
    Taken from https://github.com/nikita6187/4_step_model_surrogate
    <experiments/regression/put_only/munich_10k_linear_15_80/direct_graph/gcn2/2021_11_22_regr_put_munich_dg_gcn2.py>
    """

    def __init__(self,
                 in_channels: int = 5,
                 use_pos: bool = False,
                 out_channels: int = 1,
                 dropout: float = 0.25,
                 use_dropout: bool = True,
                 predict_mode_stats: bool = False,
                 dtype: torch.dtype = torch.float32,
                 hidden_channels: int = 512,
                 num_layers: int = 70,
                 alpha: float = 0.1,
                 theta: float = 1.5,
                 num_feed_forward: int = 3,
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
        self.num_layers = num_layers
        self.alpha = alpha
        self.theta = theta
        self.num_feed_forward = num_feed_forward
        self.use_pos = use_pos

        if self.use_pos:
            self.in_channels += 4 # x and y for start and end points

        if self.log_to_wandb:
            wandb.config.update({'hidden_channels': hidden_channels,
                                'num_layers': num_layers,
                                'alpha': alpha,
                                'theta': theta,
                                'num_feed_forward': num_feed_forward,
                                'use_pos': use_pos,
                                'in_channels': self.in_channels},
                                allow_val_change=True)

        # Define the layers of the model
        self.define_layers()

    def define_layers(self):

        self.lin1 = torch.nn.Linear(self.in_channels, self.hidden_channels)

        self.convs = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            self.convs.extend([
                GraphNorm(self.hidden_channels),
                GCN2Conv(self.hidden_channels, alpha=self.alpha, theta=self.theta,
                         layer=layer+1, shared_weights=True, normalize=True)])

        self.linears = torch.nn.ModuleList()
        for layer in range(self.num_feed_forward):
            self.linears.extend([
                torch.nn.Linear(self.hidden_channels, self.hidden_channels),
                torch.nn.Linear(self.hidden_channels, self.hidden_channels)])
            
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)

        self.lin_prt_final = torch.nn.Linear(self.hidden_channels, self.out_channels)

    def forward(self, data):

        # Unpack data
        x_0 = data.x
        edge_index = data.edge_index

        if self.use_pos:
            pos1 = data.pos[:, 0, :]  # Start position
            pos2 = data.pos[:, 1, :]  # End position
            x_0 = torch.cat((x_0, pos1, pos2), dim=1)  # Concatenate along the feature dimension

        x_0 = x_0.to(self.dtype)

        x = x_1 = self.lin1(x_0)

        for idx in range(0, len(self.convs), 2):

            norm = self.convs[idx]
            conv = self.convs[idx + 1]

            if self.use_dropout:
                x = self.dropout_layer(x)

            x = conv(x, x_1, edge_index)
            x = norm(x, batch=data.batch)
            x = x.relu()

        # Do final residual feed forward
        for idx in range(0, len(self.linears), 2):

            prev_x = x
            x = self.linears[idx](x)
            x = x.relu()

            x = self.linears[idx + 1](x)
            x = (1 - self.alpha) * x + (self.alpha * prev_x)

            x = x.relu()

        y = self.lin_prt_final(x)

        return y