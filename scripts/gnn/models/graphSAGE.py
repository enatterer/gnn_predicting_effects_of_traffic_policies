import os
import sys

import tqdm as tqdm
import wandb
import torch
from torch import nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops

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
                 hidden_channels: list[int] = [256, 256],
                 aggregator: str = 'mean',  # Options: 'mean' | 'gcn' | 'pool'
                 update_function: str = 'relu',  # Options: 'relu', 'mlp'
                 mlp_hidden_dim: int = 64,  # Hidden dim for learnable MLP update
                 dropout: float = 0.3, 
                 use_dropout: bool = True,
                 predict_mode_stats: bool = False,
                 dtype: torch.dtype = torch.float32,
                 log_to_wandb: bool = True):
        
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
        if aggregator not in ['mean', 'gcn', 'pool']:
            raise ValueError(f"Unsupported aggregator: {aggregator}. Choose 'mean' or 'gcn' or 'pool'.")
        else:
            self.aggregator = aggregator
        if update_function not in ['relu', 'mlp']:
            raise ValueError(f"Unsupported update function: {update_function}. Choose 'relu' or 'mlp'.")
        else:
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

        # Define the layers of the model
        self.define_layers()

        # Initialize weights
        self.initialize_weights()

    def define_layers(self):
        """
        Define the GraphSAGE layers, including flexible aggregators and update functions.
        """
        self.layers = nn.ModuleList()
        
        if self.aggregator == 'pool':
            self.post_concat = nn.ModuleList() #list of MLPs to concatenate the features of the source and target nodes
        
        for i in range(len(self.hidden_channels)):
            if i == 0:
                in_channels = self.in_channels
            else:
                in_channels = self.hidden_channels[i - 1]
            out_channels = self.hidden_channels[i]

            if self.aggregator == 'mean':
                # Use standard SAGEConv with mean aggregation
                conv = SAGEConv(in_channels, out_channels, aggr='mean')
            elif self.aggregator == 'gcn': #TODO: check if this is correct
                # Use standard SAGEConv with gcn aggregation
                conv = SAGEConv(in_channels, out_channels, aggr='gcn',root_weight=False)
            else: # pool
                #a small MLP to transform the neghbour features before pooling
                aggr_mlp = nn.Sequential(
                    nn.Linear(in_channels, self.mlp_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.mlp_hidden_dim, out_channels)
                )
                setattr(self, f'aggr_mlp{i + 1}', aggr_mlp)
                # Use standard SAGEConv with pool aggregation
                conv = SAGEConv(in_channels, out_channels, aggr='max') #TODO: check if this is correct
                post_concat = nn.Linear(out_channels+in_channels, out_channels) #concatenate the features of the source and target nodes
                self.post_concat.append(post_concat)
            self.layers.append(conv)

            if self.update_function == 'mlp': #TODO: check if this is correct
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
        if self.aggregator == 'gcn':
            edge_index = add_self_loops(edge_index, num_nodes=x.size(0),fill_value=1)
        # Apply GraphSAGE layers
        for i, conv in enumerate(self.layers):
            # (a)GCN: add self-loops only on the first layer if using gcn
            if self.aggregator == 'gcn' and i == 0:
                edge_index, _ = add_self_loops(edge_index,
                                               num_nodes=x.size(0),fill_value=1)
            # (b) Pool: run the aggr_mlp first, then concat & post‐MLP
            if self.aggregator == 'pool':
                aggr_mlp      = getattr(self, f'aggr_mlp{i+1}')
                x_neigh       = aggr_mlp(x)
                x             = conv((x_neigh, x), edge_index)
                x             = self.post_concat[i](torch.cat([x, x], dim=-1))
            else:  
                # (c) Mean: standard SAGEConv
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
    
    '''def forward(self, data):
        """
        Forward pass with nested aggregator and update function.
        Handles fixed-size x_all by using a manual pointer to extract
        the correct target nodes for each hop.
        """
        # Full feature matrix for all nodes involved in the sampled subgraph
        x_all = data.x.to(self.dtype)
        adjs = data.adjs  # List of (edge_index, e_id, (num_src, num_dst))

        # Step 1: Precompute the slices of target nodes per layer
        ptr = 0  # Start at the beginning of x_all
        target_slices = []
        for i in range(len(adjs)):
            num_src, num_dst = adjs[i][2]
            ptr += num_src  # Skip over source nodes
            target_slices.append((ptr, ptr + num_dst))
        target_slices = target_slices[::-1] #reverse the list to go from farthest to closest hop
        # Go from farthest to closest hop
        for i in reversed(range(len(adjs))):
            edge_index, e_id, (num_src, num_dst) = adjs[i]
            start, end = target_slices[i]
            x_target = x_all[start:end]
            
            # Optional: add self-loops only with seed nodes
            if self.aggregator == 'gcn' and i == 0:
                edge_index, _ = add_self_loops(
                    edge_index,
                    num_nodes=x_all.size(0)
                )

            # Aggregate
            if self.aggregator == 'pool':
                aggr_mlp = getattr(self, f'aggr_mlp{i+1}')
                x_all = self.layers[i]((aggr_mlp(x_all), x_target), edge_index)
            else:
                x_all = self.layers[i]((x_all, x_target), edge_index)

            # Apply update function (either ReLU or a learned MLP)
            if self.update_function == 'relu':
                x_all = nn.functional.relu(x_all)
            else:
                update_mlp = getattr(self, f'update_mlp{i+1}')
                x_all = update_mlp(x_all)

            # Apply dropout if enabled
            if self.use_dropout:
                x_all = self.dropout_layer(x_all)

        # Final output for the seed nodes (which are the last num_dst of the last layer)
        seed_count = adjs[0][2][1]  # num_dst of the *last layer*, i.e., number of seeds
        out = self.fc(x_all[-seed_count:])  # Final predictions for seed nodes
        return out'''