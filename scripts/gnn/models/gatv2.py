import os
import sys

import tqdm as tqdm
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
                hidden_channels: list[int] = [256,512,1024,512,256],
                num_heads: int = 4,
                dropout: float = 0.3, 
                use_dropout: bool = False,
                predict_mode_stats: bool = False,
                dtype: torch.dtype = torch.float32,
                log_to_wandb: bool = True,
                use_target_standardization: bool = False,  # ✅ ADD THIS
                
                # ✅ ADD POSITIONAL ENCODING PARAMETERS
                use_pos: bool = False,
                use_pos_encoding: bool = False,
                use_lap_pe: bool = False,
                lap_pe_dim: int = 8,
                use_anchor_pe: bool = False,
                anchor_k: int = 12,
                anchor_m: int = 16,
                use_rwse: bool = False,
                rwse_dim: int = 8,
                
                # Graph structure parameters
                use_graph_norm: bool = False,
                use_residuals: bool = False,
                
                # GATv2 specific parameters
                share_weights: bool = False,
                negative_slope: float = 0.2,
                add_self_loops: bool = True):
    
        # ✅ STORE POSITIONAL ENCODING PARAMETERS
        self.use_pos = use_pos
        self.use_pos_encoding = use_pos_encoding
        self.use_lap_pe = use_lap_pe
        self.lap_pe_dim = lap_pe_dim
        self.use_anchor_pe = use_anchor_pe
        self.anchor_k = anchor_k
        self.anchor_m = anchor_m
        self.use_rwse = use_rwse
        self.rwse_dim = rwse_dim

        # ✅ CALCULATE EFFECTIVE INPUT CHANNELS
        effective_in_channels = in_channels
        if use_pos:
            effective_in_channels += 2  # x, y coordinates
        if use_lap_pe:
            effective_in_channels += 2  # First 2 Laplacian eigenvectors
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
            use_target_standardization=use_target_standardization)  # ✅ ADD THIS
        
        # Model specific parameters
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.use_graph_norm = use_graph_norm
        self.use_residuals = use_residuals

        # GATv2 specific parameters
        self.share_weights = share_weights
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops

        # ✅ ENHANCED WANDB LOGGING
        if self.log_to_wandb:
            wandb.config.update({
                'in_channels': self.in_channels,
                'original_in_channels': in_channels,
                'hidden_channels': hidden_channels,
                'num_heads': num_heads,
                'use_pos': use_pos,
                'use_pos_encoding': use_pos_encoding,
                'use_lap_pe': use_lap_pe,
                'lap_pe_dim': lap_pe_dim,
                'use_anchor_pe': use_anchor_pe,
                'anchor_k': anchor_k,
                'anchor_m': anchor_m,
                'use_rwse': use_rwse,
                'rwse_dim': rwse_dim,
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
            conv = GATv2Conv(in_channels, int(self.hidden_channels[i]/self.num_heads), heads=self.num_heads, share_weights=self.share_weights, negative_slope=self.negative_slope, add_self_loops=self.add_self_loops)    
            setattr(self, f'conv{i + 1}', conv)

            if self.use_graph_norm:
                graph_norm = GraphNorm(self.hidden_channels[i-1] if i > 0 else self.in_channels)
                setattr(self, f'graph_norm{i + 1}', graph_norm)

        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)

        self.fc = nn.Linear(self.hidden_channels[-1], self.out_channels)

    def forward(self, data):
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
        for d in datalist:
            xi = d.x.clone()

            # Add coordinate features
            if self.use_pos:
                if hasattr(d, 'pos') and d.pos is not None:
                    pos = d.pos.to(self.dtype)
                    if pos.dim() == 3 and pos.shape[1] == 3:
                        pos = pos[:, 1, :]  # Take middle position
                    elif pos.dim() == 2 and pos.shape[1] != 2:
                        pos = pos.view(pos.size(0), -1)  # Flatten if needed
                    xi = torch.cat([xi, pos], dim=-1)
                else:
                    raise ValueError("Position features enabled but 'pos' missing")

            # Add Laplacian PE
            if self.use_lap_pe:
                if hasattr(d, 'lap_pe') and d.lap_pe is not None:
                    lap_pe = d.lap_pe[:, :2].to(self.dtype)  # First 2 eigenvectors
                    xi = torch.cat([xi, lap_pe], dim=-1)

            # Add Anchor Distance Encoding
            if self.use_anchor_pe:
                if hasattr(d, 'pos') and d.pos is not None:
                    pos = d.pos.to(next(self.parameters()).device)
                    if pos.dim() == 3:
                        pos = pos[:, 1, :]  # Middle position
                    
                    # Select random anchor nodes
                    num_nodes = pos.size(0)
                    num_anchors = min(self.anchor_k, num_nodes)
                    anchor_indices = torch.randperm(num_nodes)[:num_anchors]
                    anchors = pos[anchor_indices]
                    
                    # Compute distances to anchors
                    distances = torch.cdist(pos, anchors)  # [num_nodes, num_anchors]
                    
                    # Pad or truncate to desired size
                    if distances.size(1) < self.anchor_k:
                        pad_size = self.anchor_k - distances.size(1)
                        distances = torch.cat([distances, torch.zeros(num_nodes, pad_size, device=distances.device)], dim=1)
                    else:
                        distances = distances[:, :self.anchor_k]
                    
                    # Simple encoding
                    ade = distances.repeat(1, self.anchor_m // self.anchor_k + 1)[:, :self.anchor_k * self.anchor_m]
                    xi = torch.cat([xi, ade], dim=-1)

            # Add Random Walk Structural Encoding
            if self.use_rwse:
                if hasattr(d, 'rw_pe') and d.rw_pe is not None:
                    rwse = d.rw_pe.to(self.dtype)
                    if rwse.size(1) < self.rwse_dim:
                        pad = torch.zeros(rwse.size(0), self.rwse_dim - rwse.size(1), device=rwse.device, dtype=self.dtype)
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

        # ✅ STANDARD GATV2 FORWARD PASS
        x, edge_index = data.x, data.edge_index
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