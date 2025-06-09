import wandb

import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv

from gnn.models.base_gnn import BaseGNN

class PNC(BaseGNN):
    def __init__(self, 
                in_channels: int = 5,
                hidden_channels: int = 128,
                out_channels: int = 1,
                num_layers: int = 5,
                pnc_local_mlp: list[int] = [256,512], 
                pnc_global_mlp: list[int] = [256],
                pos_mlp: list[int] = [8,32],
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
        
        # Architecture-specific parameters
        self.pnc_local = pnc_local_mlp
        self.pnc_global = pnc_global_mlp
        self.hidden_channels = hidden_channels
        self.pos_mlp = pos_mlp
        self.num_layers = num_layers

        if self.log_to_wandb:
            wandb.config.update({"pnc_local": self.pnc_local,
                                "pnc_global": self.pnc_global,
                                "hidden_channels": self.hidden_channels,
                                "num_layers": self.num_layers,
                                "pos_mlp": self.pos_mlp},
                                allow_val_change=True)

        # Define the layers of the model
        self.define_layers()

        # Initialize weights
        self.initialize_weights()
        
    def define_layers(self):

        # Initialize dropout if needed
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)

        # MLP to upscale pos, and not make it drown in hidden channels
        self.pos_nn = self.create_pos_mlp()

        # PointNet layers 
        # Use mid pos everytime
        # Is orientation correct? If so, can use end - start pos!
        
        self.pnconv_1 = self.create_point_net_layer(is_first_layer=True)

        for i in range(1, self.num_layers):
            pnconv = self.create_point_net_layer(is_first_layer=False)
            setattr(self, f'pnconv_{i+1}', pnconv)
        
        # Output layer
        self.read_out_node_predictions = nn.Linear(self.hidden_channels, 1)

    def forward(self, data):
        """
        Forward pass for the GNN model.

        Parameters:
        - data (Data): Input data containing node features and edge indices.

        Returns:
        - torch.Tensor: Output features after passing through the model.
        """
        x = data.x.to(self.dtype)
        edge_index = data.edge_index

        # Use middle pos
        pos = data.pos[:, 2, :]
        
        for i in range(self.num_layers):
            
            pnconv = getattr(self, f'pnconv_{i+1}')

            if i == 0:
                x = pnconv(x, pos, edge_index)
            else:
                x = pnconv(x, self.pos_nn(pos), edge_index)
        
        node_predictions = self.read_out_node_predictions(x)
        
        return node_predictions
    
    def create_point_net_layer(self, is_first_layer:bool=False):
        
        offset_due_to_pos = 2
        local_MLP_layers = []
        
        if is_first_layer:
            local_MLP_layers.append(nn.Linear(self.in_channels + offset_due_to_pos, self.pnc_local[0]))
        else:
            local_MLP_layers.append(nn.Linear(self.hidden_channels + self.pos_mlp[-1], self.pnc_local[0]))
        
        local_MLP_layers.append(nn.ReLU())
        if self.use_dropout:
            local_MLP_layers.append(self.dropout_layer)
        
        for idx in range(len(self.pnc_local)-1):
            local_MLP_layers.append(nn.Linear(self.pnc_local[idx], self.pnc_local[idx + 1]))
            local_MLP_layers.append(nn.ReLU())
            if self.use_dropout:
                local_MLP_layers.append(self.dropout_layer)
        
        local_MLP = nn.Sequential(*local_MLP_layers)
        
        global_MLP_layers = []
        global_MLP_layers.append(nn.Linear(self.pnc_local[-1], self.pnc_global[0]))
        global_MLP_layers.append(nn.ReLU())
        if self.use_dropout:
            global_MLP_layers.append(self.dropout_layer)
        
        for idx in range(len(self.pnc_global) - 1):
            global_MLP_layers.append(nn.Linear(self.pnc_global[idx], self.pnc_global[idx + 1]))
            global_MLP_layers.append(nn.ReLU())
            if self.use_dropout:
                global_MLP_layers.append(self.dropout_layer)

        # Final layer        
        global_MLP_layers.append(nn.Linear(self.pnc_global[-1], self.hidden_channels))
        global_MLP_layers.append(nn.ReLU())
        if self.use_dropout:
            global_MLP_layers.append(self.dropout_layer)
        
        global_MLP = nn.Sequential(*global_MLP_layers)
        
        return PointNetConv(local_nn=local_MLP, global_nn=global_MLP)
    
    def create_pos_mlp(self):

        offset_due_to_pos = 2
        pos_mlp_layers = []

        pos_mlp_layers.append(nn.Linear(offset_due_to_pos, self.pos_mlp[0]))
        pos_mlp_layers.append(nn.ReLU())
        if self.use_dropout:
            pos_mlp_layers.append(self.dropout_layer)
        
        for idx in range(len(self.pos_mlp) - 1):
            pos_mlp_layers.append(nn.Linear(self.pos_mlp[idx], self.pos_mlp[idx + 1]))
            pos_mlp_layers.append(nn.ReLU())
            if self.use_dropout:
                pos_mlp_layers.append(self.dropout_layer)
        
        return nn.Sequential(*pos_mlp_layers)