import os
import sys
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    GCNConv, GATConv, GraphConv, TransformerConv, GraphNorm
)
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import get_laplacian, degree, to_dense_adj
# Remove fps import since it's not available
import torch.linalg as LA
import numpy as np
import wandb

# Add 'scripts' dir to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.base_gnn import BaseGNN


# ---------------- Utility functions ---------------- #

def rbf(d, centers, gamma):
    """Radial Basis Function for distance encoding."""
    return torch.exp(-gamma * (d.unsqueeze(-1) - centers) ** 2)  # [N,K,M]

def degree_based_anchors(pos, edge_index, K):
    """Select high-degree nodes as anchors (for graph structure awareness)"""
    N, device = pos.size(0), pos.device
    
    # Compute node degrees
    degrees = torch.bincount(edge_index[0], minlength=N)  # [N]
    
    # Select top-K/2 degree nodes
    top_k_half = min(K//2, N)
    _, top_degree_indices = degrees.topk(top_k_half)
    
    # Fill remaining with spatial diversity (random from remaining nodes)
    remaining_needed = K - len(top_degree_indices)
    if remaining_needed > 0:
        all_indices = torch.arange(N, device=device)
        remaining_mask = ~torch.isin(all_indices, top_degree_indices)
        remaining_indices = all_indices[remaining_mask]
        
        if len(remaining_indices) >= remaining_needed:
            additional = remaining_indices[torch.randperm(len(remaining_indices))[:remaining_needed]]
        else:
            additional = remaining_indices
        
        anchor_indices = torch.cat([top_degree_indices, additional])
    else:
        anchor_indices = top_degree_indices[:K]
    
    return anchor_indices

def anchor_distance_encoding(pos, edge_index, K=12, M=16):
    """
    Compute Anchor Distance Encoding (ADE) using degree-based anchor selection.
    
    Args:
        pos: [N,2] node coordinates 
        edge_index: [2, E] edge indices
        K: number of anchor points
        M: number of RBF centers per anchor
    
    Returns:
        [N, K*M] distance encodings
    """
    N, device = pos.size(0), pos.device
    
    if N <= K:
        # For small graphs, use all nodes as anchors (with padding if needed)
        anchors = pos[torch.randperm(N)[:min(N, K)]]
        if anchors.size(0) < K:
            pad = torch.zeros(K - anchors.size(0), 2, device=device)
            anchors = torch.cat([anchors, pad], dim=0)
    else:
        # Use degree-based anchor selection
        anchor_indices = degree_based_anchors(pos, edge_index, K)
        anchors = pos[anchor_indices]

    # Compute distances to all anchors
    d = torch.cdist(pos, anchors, p=2)  # [N,K] Euclidean distances
    
    # Normalize distances for stable RBF computation
    scale = d.std() + 1e-6
    d = d / scale
    dmax = d.max().detach() + 1e-6
    
    # Create RBF centers
    centers = torch.linspace(0, dmax, M, device=device)
    gamma = 1.0 / (centers[1] - centers[0]).clamp(min=1e-6) ** 2
    
    # Apply RBF and flatten
    rbf_encodings = rbf(d, centers, gamma)  # [N, K, M]
    return rbf_encodings.reshape(N, K * M)  # [N, K*M]

def compute_laplacian_pe(edge_index, num_nodes, pe_dim=8):
    """
    Compute Laplacian Positional Encoding using eigendecomposition.
    
    Args:
        edge_index: [2, num_edges] 
        num_nodes: number of nodes
        pe_dim: dimension of positional encoding
        
    Returns:
        [num_nodes, pe_dim] positional encodings
    """
    device = edge_index.device
    
    try:
        # Get normalized Laplacian
        edge_index_lap, edge_weight_lap = get_laplacian(
            edge_index, 
            num_nodes=num_nodes, 
            normalization='sym',
            dtype=torch.float32
        )
        
        # Convert to dense for eigendecomposition
        L = to_dense_adj(
            edge_index_lap, 
            edge_attr=edge_weight_lap, 
            max_num_nodes=num_nodes
        ).squeeze(0)  # [num_nodes, num_nodes]
        
        # Eigendecomposition
        eigenvals, eigenvecs = LA.eigh(L)
        
        # Skip first eigenvalue (~0) and take next pe_dim eigenvectors
        start_idx = 1 if eigenvals[0].abs() < 1e-6 else 0
        end_idx = min(start_idx + pe_dim, num_nodes)
        
        pe_eigenvecs = eigenvecs[:, start_idx:end_idx]  # [num_nodes, k]
        
        # Pad or truncate to pe_dim
        if pe_eigenvecs.size(1) < pe_dim:
            padding = torch.zeros(
                num_nodes, 
                pe_dim - pe_eigenvecs.size(1), 
                device=device, 
                dtype=pe_eigenvecs.dtype
            )
            pe_eigenvecs = torch.cat([pe_eigenvecs, padding], dim=1)
        else:
            pe_eigenvecs = pe_eigenvecs[:, :pe_dim]
            
        return pe_eigenvecs.to(device)
        
    except Exception as e:
        print(f"⚠️ LapPE computation failed: {e}, using random features")
        return torch.randn(num_nodes, pe_dim, device=device) * 0.1


# ---------------- Enhanced TransEncoder ---------------- #

class TransEncoder(BaseGNN):
    def __init__(self,
                 in_channels: int = 5,
                 out_channels: int = 1,
                 embed_dim: int = 128,
                 ff_dim: int = 512,
                 num_heads: int = 4,
                 num_layers: int = 5,
                 num_nodes: int = 31635,
                 use_pos: bool = False,
                 use_pos_encoding: bool = False,
                 # LapPE
                 use_lap_pe: bool = False,
                 lap_pe_dim: int = 8,
                 # Anchor Distance Encoding
                 use_anchor_pe: bool = False,
                 anchor_k: int = 12,
                 anchor_m: int = 16,
                 # Random Walk Structural Encoding
                 use_rwse: bool = False,
                 rwse_dim: int = 8,
                 # Training params
                 dropout: float = 0.1,
                 use_dropout: bool = False,
                 use_graph_conv: bool = True,
                 use_graph_norm: bool = False,
                 graph_conv_type: str = 'trans_conv',
                 num_graph_conv_layers: int = 2,
                 predict_mode_stats: bool = False,
                 dtype: torch.dtype = torch.float32,
                 log_to_wandb: bool = False,
                 pad_to: int | None = None,
                 pad_value: float = 0.0):

        # Calculate effective input channels
        effective_in_channels = in_channels
        if use_pos:
            effective_in_channels += 2
        if use_lap_pe:
            effective_in_channels += lap_pe_dim
        if use_anchor_pe:
            effective_in_channels += anchor_k * anchor_m
        if use_rwse:
            effective_in_channels += rwse_dim

        super().__init__(
            in_channels=effective_in_channels,
            out_channels=out_channels,
            dropout=dropout,
            use_dropout=use_dropout,
            predict_mode_stats=predict_mode_stats,
            dtype=dtype,
            log_to_wandb=log_to_wandb)

        # Store all parameters
        self.use_pos = use_pos
        self.use_pos_encoding = use_pos_encoding
        self.use_lap_pe = use_lap_pe
        self.lap_pe_dim = lap_pe_dim
        self.use_anchor_pe = use_anchor_pe
        self.anchor_k = anchor_k
        self.anchor_m = anchor_m
        self.use_rwse = use_rwse
        self.rwse_dim = rwse_dim

        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.use_graph_conv = use_graph_conv
        self.use_graph_norm = use_graph_norm
        self.graph_conv_type = graph_conv_type
        self.num_graph_conv_layers = num_graph_conv_layers

        self.pad_to = pad_to
        self.pad_value = pad_value

        # Log to wandb
        if self.log_to_wandb:
            wandb.config.update({
                'use_pos': use_pos,
                'use_pos_encoding': use_pos_encoding,
                'use_lap_pe': use_lap_pe,
                'lap_pe_dim': lap_pe_dim,
                'use_anchor_pe': use_anchor_pe,
                'anchor_k': anchor_k,
                'anchor_m': anchor_m,
                'use_rwse': use_rwse,
                'rwse_dim': rwse_dim,
                'effective_in_channels': self.in_channels,
                'embed_dim': embed_dim,
                'ff_dim': ff_dim,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'pad_to': pad_to,
            }, allow_val_change=True)

        self.define_layers()

    def define_layers(self):
        self.embed = nn.Linear(self.in_channels, self.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout if self.use_dropout else 0.0,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.output = nn.Linear(self.embed_dim, 1)

        if self.use_pos_encoding:
            self.pos_embedding = nn.Embedding(self.num_nodes, self.embed_dim)

        if self.use_graph_conv:
            for i in range(self.num_graph_conv_layers):
                in_channels = self.in_channels if i == 0 else self.embed_dim
                if self.graph_conv_type == 'gcn':
                    conv = GCNConv(in_channels, self.embed_dim)
                elif self.graph_conv_type == 'gat':
                    conv = GATConv(in_channels, self.embed_dim)
                elif self.graph_conv_type == 'graph':
                    conv = GraphConv(in_channels, self.embed_dim)
                elif self.graph_conv_type == 'trans_conv':
                    conv = TransformerConv(in_channels, self.embed_dim)
                setattr(self, f'conv_{i+1}', conv)

                if self.use_graph_norm:
                    graph_norm = GraphNorm(self.embed_dim if i > 0 else self.in_channels)
                    setattr(self, f'graph_norm_{i+1}', graph_norm)

        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, data):
        device = next(self.parameters()).device

        # Process data into per-graph tensors FIRST (before graph conv)
        if isinstance(data, Batch):
            datalist = data.to_data_list()
        elif isinstance(data, Data):
            datalist = [data]
        else:
            raise ValueError("Input must be Batch or Data")

        # Add all positional encodings BEFORE graph convolution
        enhanced_datalist = []
        for d in datalist:
            xi = d.x.clone()  # [num_nodes, original_features]

            # Add coordinate features
            if self.use_pos:
                pos = d.pos[:, 2, :].to(device)  # [num_nodes, 2]
                xi = torch.cat([xi, pos], dim=-1)

            # Add Laplacian PE
            if self.use_lap_pe:
                lap_pe = compute_laplacian_pe(d.edge_index, d.x.size(0), self.lap_pe_dim)
                xi = torch.cat([xi, lap_pe], dim=-1)

            # Add Anchor Distance Encoding
            if self.use_anchor_pe:
                pos = d.pos[:, 2, :].to(device)
                ade = anchor_distance_encoding(pos, d.edge_index, K=self.anchor_k, M=self.anchor_m)
                xi = torch.cat([xi, ade], dim=-1)

            # Add Random Walk Structural Encoding (if precomputed)
            if self.use_rwse and hasattr(d, 'rw_pe'):
                rwse = d.rw_pe.to(device)
                if rwse.size(1) != self.rwse_dim:
                    if rwse.size(1) < self.rwse_dim:
                        pad = torch.zeros(rwse.size(0), self.rwse_dim - rwse.size(1), device=device)
                        rwse = torch.cat([rwse, pad], dim=1)
                    else:
                        rwse = rwse[:, :self.rwse_dim]
                xi = torch.cat([xi, rwse], dim=-1)

            # Create new data object with enhanced features
            d_enhanced = d.clone()
            d_enhanced.x = xi.to(self.dtype)
            enhanced_datalist.append(d_enhanced)

        # Reconstruct batch with enhanced features
        if len(enhanced_datalist) > 1:
            data = Batch.from_data_list(enhanced_datalist)
        else:
            data = enhanced_datalist[0]

        # NOW apply graph convolution with correct input dimensions
        if self.use_graph_conv:
            x, edge_index = data.x, data.edge_index

            for i in range(self.num_graph_conv_layers):
                if self.use_graph_norm:
                    graph_norm = getattr(self, f'graph_norm_{i+1}')
                    x = graph_norm(x)
                conv = getattr(self, f'conv_{i+1}')
                x = conv(x, edge_index)
                x = nn.functional.relu(x)
                if self.use_dropout:
                    x = self.dropout_layer(x)
            data.x = x

        # Process for transformer (now x has embed_dim from graph conv)
        if isinstance(data, Batch):
            datalist = data.to_data_list()
        elif isinstance(data, Data):
            datalist = [data]

        x_list, lengths = [], []
        for d in datalist:
            xi = d.x  # [num_nodes, embed_dim] after graph conv
            lengths.append(xi.size(0))
            x_list.append(xi.to(device))

        # Pad sequences
        lengths = torch.tensor(lengths, device=device)
        max_len = int(lengths.max().item())
        target_len = self.pad_to if self.pad_to is not None else max_len

        if self.use_pos_encoding and target_len > self.num_nodes:
            raise ValueError(
                f"pad_to/max_len ({target_len}) exceeds num_nodes ({self.num_nodes})"
            )

        padded = pad_sequence(x_list, batch_first=True, padding_value=self.pad_value)
        B, cur_S, C = padded.shape
        
        if target_len > cur_S:
            extra = torch.full((B, target_len-cur_S, C), self.pad_value,
                               dtype=padded.dtype, device=device)
            padded = torch.cat([padded, extra], dim=1)
        elif target_len < cur_S:
            padded = padded[:, :target_len, :]
            lengths = torch.clamp(lengths, max=target_len)

        # Create attention mask
        arange_S = torch.arange(padded.size(1), device=device).unsqueeze(0)
        key_padding_mask = arange_S >= lengths.unsqueeze(1)

        # Features are already embedded by graph conv to embed_dim
        x = padded.to(self.dtype)
        if x.size(-1) != self.embed_dim:
            x = self.embed(x)

        # Add learnable positional encoding
        if self.use_pos_encoding:
            node_indices = torch.arange(x.size(1), device=device).long()
            pos_emb = self.pos_embedding(node_indices)
            pos_emb = pos_emb.unsqueeze(0).expand(x.size(0), -1, -1)
            x = x + pos_emb

        # Transformer processing
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.output(x)

        # Unpad predictions
        lengths = (~key_padding_mask).sum(dim=1)
        preds_list = [x[i, :lengths[i]] for i in range(x.size(0))]
        preds_compact = torch.cat(preds_list, dim=0)

        return preds_compact