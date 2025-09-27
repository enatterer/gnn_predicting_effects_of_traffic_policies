import os
import sys

import tqdm as tqdm
import wandb
import torch
from torch import nn
from torch_geometric.nn import SAGEConv, GraphNorm
from torch_geometric.utils import add_self_loops

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

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
                 hidden_channels: list[int] = [256, 512, 1024, 512, 256],
                 dropout: float = 0.3, 
                 use_dropout: bool = False,
                 predict_mode_stats: bool = False,
                 dtype: torch.dtype = torch.float32,
                 log_to_wandb: bool = True,
                 use_target_standardization: bool = False,
                 use_city_balanced_loss: bool = False, 
                 message_drop_prob: float = 0.0,  # <-- ADD THIS LINE
                 
                 # ✅ NEW: Position parameters from TransEncoder
                 use_pos: bool = True,
                 pos_dim: int = 6,  # Configurable position dimensions
                 
                 # ✅ NEW: Laplacian PE parameters
                 use_lap_pe: bool = True,
                 lap_pe_dim: int = 8,
                 lap_pe_use_dim: int = 8,  # How many LapPE dimensions to actually use
                 
                 # ✅ NEW: Anchor Distance Encoding parameters  
                 use_anchor_pe: bool = False,
                 anchor_k: int = 12,
                 anchor_m: int = 16,
                 
                 # ✅ NEW: Random Walk Structural Encoding parameters
                 use_rwse: bool = False,
                 rwse_dim: int = 8,
                 
                 # Graph structure parameters
                 use_graph_norm: bool = False,
                 use_residuals: bool = False,
                 
                 # GraphSAGE specific parameters
                 aggregator: str = 'mean', #options: 'max', 'mean'
                 update_function: str = 'relu',
                 mlp_hidden_dim: int = 1024):
        
        # ✅ UPDATED: Store positional encoding parameters
        self.use_pos = use_pos
        self.pos_dim = pos_dim
        self.use_lap_pe = use_lap_pe
        self.lap_pe_dim = lap_pe_dim
        self.lap_pe_use_dim = min(lap_pe_use_dim, lap_pe_dim)  # Can't use more than available
        self.use_anchor_pe = use_anchor_pe
        self.anchor_k = anchor_k
        self.anchor_m = anchor_m
        self.use_rwse = use_rwse
        self.rwse_dim = rwse_dim

        # Calculate effective input channels
        effective_in_channels = in_channels
        if use_pos:
            effective_in_channels += pos_dim
        if use_lap_pe:
            effective_in_channels += self.lap_pe_use_dim
        if use_anchor_pe:
            effective_in_channels += anchor_k * anchor_m
        if use_rwse:
            effective_in_channels += rwse_dim

        # Call parent class constructor
        super().__init__(
            in_channels=effective_in_channels,
            out_channels=out_channels,
            dropout=dropout,
            use_dropout=use_dropout,
            predict_mode_stats=predict_mode_stats,
            dtype=dtype,
            log_to_wandb=log_to_wandb,
            use_target_standardization=use_target_standardization)
        
        # Model-specific parameters
        self.hidden_channels = hidden_channels
        self.mlp_hidden_dim = mlp_hidden_dim
        
        if aggregator not in ['max', 'mean']:
            raise ValueError(f"Unsupported aggregator: {aggregator}. Choose 'max' or 'mean'.")
        else:
            self.aggregator = aggregator
            
        if update_function not in ['relu', 'mlp']:
            raise ValueError(f"Unsupported update function: {update_function}. Choose 'relu' or 'mlp'.")
        else:
            self.update_function = update_function
            
        self.use_graph_norm = use_graph_norm
        self.use_residuals = use_residuals

        # ✅ UPDATED: WANDB LOGGING with all new parameters
        if self.log_to_wandb:
            wandb.config.update({
                'hidden_channels': hidden_channels,
                'in_channels': self.in_channels,
                'effective_in_channels': self.in_channels,  # ✅ NEW
                'mlp_hidden_dim': mlp_hidden_dim,
                
                # ✅ NEW: Position encoding parameters
                'use_pos': use_pos,
                'pos_dim': pos_dim,
                
                # ✅ NEW: Laplacian PE parameters
                'use_lap_pe': use_lap_pe,
                'lap_pe_dim': lap_pe_dim,
                'lap_pe_use_dim': self.lap_pe_use_dim,
                
                # ✅ NEW: Anchor PE parameters
                'use_anchor_pe': use_anchor_pe,
                'anchor_k': anchor_k,
                'anchor_m': anchor_m,
                
                # ✅ NEW: RWSE parameters
                'use_rwse': use_rwse,
                'rwse_dim': rwse_dim,
                
                'aggregator': aggregator,
                'update_function': update_function,
                'use_graph_norm': use_graph_norm,
                'use_residuals': use_residuals,
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
                project=(in_channels != self.hidden_channels[i])) #TODO: to change to True if we want to project the features to the hidden_channels[i]
            setattr(self, f'conv{i + 1}', conv)

            if self.use_graph_norm:
                graph_norm = GraphNorm(self.hidden_channels[i-1] if i > 0 else self.in_channels)
                setattr(self, f'graph_norm{i + 1}', graph_norm)

            if self.update_function == 'mlp': #use this if we want a second layer of non-linearity
                # Define a small MLP for learnable update
                update_mlp = nn.Sequential(
                    nn.Linear(self.hidden_channels[i], self.mlp_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.mlp_hidden_dim, self.hidden_channels[i])
                )
                setattr(self, f'update_mlp{i + 1}', update_mlp)
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)
        
        self.fc = nn.Linear(self.hidden_channels[-1], self.out_channels)

    def forward(self, data):
        """
        Forward pass with positional encodings.
        """
        from torch_geometric.data import Batch, Data
        
        # ✅ HANDLE BATCH/DATA INPUT
        if isinstance(data, Batch):
            datalist = data.to_data_list()
        elif isinstance(data, Data):
            datalist = [data]
        else:
            raise ValueError("Input must be Batch or Data")

        # ✅ ENHANCE FEATURES FOR EACH GRAPH
        enhanced_datalist = []
        device = next(self.parameters()).device
        
        for d in datalist:
            xi = d.x.clone()

            # Add coordinate features
            if self.use_pos:
                if hasattr(d, 'pos') and d.pos is not None:
                    pos = d.pos.to(device=device, dtype=self.dtype)
                    if pos.dim() == 3:
                        if self.pos_dim == 2:
                            pos_features = pos[:, 1, :]  # Middle position
                        elif self.pos_dim == 6:
                            pos_features = pos.reshape(pos.size(0), -1)  # All positions
                        elif self.pos_dim == 4:
                            pos_features = torch.cat([pos[:, 0, :], pos[:, 2, :]], dim=-1)  # Start + end
                        else:
                            raise ValueError("pos_dim must be 2, 4, or 6 when pos is 3D")
                    else:
                        raise ValueError("Position features are enabled but 'pos' attribute is missing in data.")
                    xi = torch.cat([xi, pos_features], dim=-1)
                else:
                    raise ValueError("Position features are enabled but 'pos' attribute is missing in data.")

            # Add Laplacian PE
            if self.use_lap_pe:
                if hasattr(d, 'lap_pe') and d.lap_pe is not None:
                    lap_pe = d.lap_pe.to(device)
                    if lap_pe.size(1) < self.lap_pe_use_dim:
                        raise ValueError(f"Laplacian PE has fewer dimensions ({lap_pe.size(1)}) than lap_pe_use_dim ({self.lap_pe_use_dim})")
                    else:
                        lap_pe_reqd_dim = lap_pe[:, :self.lap_pe_use_dim]
                    xi = torch.cat([xi, lap_pe_reqd_dim], dim=-1)
                else:
                    raise ValueError("Laplacian PE is enabled but 'lap_pe' attribute is missing in data.")

            # Add Anchor Distance Encoding
            if self.use_anchor_pe:
                if hasattr(d, 'pos') and d.pos is not None:
                    pos = d.pos.to(device)
                    if pos.dim() == 3:
                        pos = pos[:, 1, :]  # Use middle position
                    
                    # Simple anchor distance encoding (basic version)
                    num_nodes = pos.size(0)
                    num_anchors = min(self.anchor_k, num_nodes)
                    anchor_indices = torch.randperm(num_nodes, device=device)[:num_anchors]
                    anchors = pos[anchor_indices]
                    
                    # Compute distances to anchors
                    distances = torch.cdist(pos, anchors)  # [num_nodes, num_anchors]
                    
                    # Pad or truncate to desired size
                    if distances.size(1) < self.anchor_k:
                        pad_size = self.anchor_k - distances.size(1)
                        distances = torch.cat([distances, torch.zeros(num_nodes, pad_size, device=device)], dim=1)
                    else:
                        distances = distances[:, :self.anchor_k]
                    
                    # Simple encoding (repeat to match anchor_k * anchor_m dimensions)
                    ade = distances.repeat(1, self.anchor_m // self.anchor_k + 1)[:, :self.anchor_k * self.anchor_m]
                    xi = torch.cat([xi, ade], dim=-1)

            # Add Random Walk Structural Encoding
            if self.use_rwse:
                if hasattr(d, 'rw_pe') and d.rw_pe is not None:
                    rwse = d.rw_pe.to(device)
                    # Pad or truncate to desired dimension
                    if rwse.size(1) < self.rwse_dim:
                        pad = torch.zeros(rwse.size(0), self.rwse_dim - rwse.size(1), device=device, dtype=self.dtype)
                        rwse = torch.cat([rwse, pad], dim=1)
                    else:
                        rwse = rwse[:, :self.rwse_dim]
                    xi = torch.cat([xi, rwse], dim=-1)

            # Create enhanced data object
            d_enhanced = d.clone()
            d_enhanced.x = xi.to(self.dtype)
            enhanced_datalist.append(d_enhanced)

        # ✅ RECONSTRUCT BATCH
        if len(enhanced_datalist) > 1:
            data = Batch.from_data_list(enhanced_datalist)
        else:
            data = enhanced_datalist[0]

        # ✅ STANDARD GRAPHSAGE FORWARD PASS
        x, edge_index = data.x, data.edge_index
        x = x.to(self.dtype)

        # Apply GraphSAGE layers
        for i in range(len(self.hidden_channels)):
            x_0 = x
                
            conv = getattr(self, f'conv{i + 1}')
            x = conv(x, edge_index)
            
            # Graph normalization
            if self.use_graph_norm:
                graph_norm = getattr(self, f'graph_norm{i + 1}')
                x = graph_norm(x)

            # apply your chosen "update function"
            if self.update_function == 'relu':
                x = F.relu(x)
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