import os
import sys
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    GCNConv, GATConv, GraphConv, TransformerConv, GraphNorm, GATv2Conv
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


# ---------------- DANN Components ---------------- #

class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for Domain Adversarial Training"""
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.alpha, None

def gradient_reversal(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


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


# ---------------- Enhanced TransEncoder with DANN ---------------- #

class TransEncoder(BaseGNN):
    def __init__(self,
                 in_channels: int = 5,
                 out_channels: int = 1,
                 hidden_channels: list[int] = [128, 256, 128],
                 ff_dim: int = 512, # 512,
                 num_heads: int = 4,
                 num_layers: int = 5, # 5
                 num_nodes: int = 31635,
                 use_pos: bool = True,
                 pos_dim: int = 6,  # ✅ NEW: Configurable position dimensions
                 use_pos_encoding: bool = False,
                 # LapPE
                 use_lap_pe: bool = True,
                 lap_pe_dim: int = 8,
                 lap_pe_use_dim: int = 8,  # ✅ NEW: How many LapPE dimensions to actually use
                 # Anchor Distance Encoding
                 use_anchor_pe: bool = False,
                 anchor_k: int = 12,
                 anchor_m: int = 16,
                 # Random Walk Structural Encoding
                 use_rwse: bool = False,
                 rwse_dim: int = 8,
                 # DANN settings
                 use_dann: bool = False,
                 city_list: list = None,
                 domain_classifier_layers: list = None,
                 # Training params
                 dropout: float = 0.1,
                 use_dropout: bool = False,
                 use_graph_conv: bool = True,
                 use_graph_norm: bool = False,
                 graph_conv_type: str = 'trans_conv', # 'gcn', 'gatv2', 'graph'
                 use_residuals: bool = False,
                 predict_mode_stats: bool = False,
                 dtype: torch.dtype = torch.float32,
                 log_to_wandb: bool = False,
                 use_target_standardization: bool = False, 
                 use_city_balanced_loss: bool = False,
                 pad_to: int | None = None,
                 pad_value: float = 0.0,
                 # ✅ NEW: Edge dropout parameter
                 message_drop_prob: float = 0.0):

        # DANN setup
        self.use_dann = use_dann
        if city_list is None:
            # Default city list based on your data
            self.city_list = ['wuerzburg', 'aschaffenburg', 'regensburg', 'bayreuth', 
                             'rosenheim', 'landshut', 'schweinfurt']
        else:
            self.city_list = city_list
        
        self.city_to_idx = {city: idx for idx, city in enumerate(self.city_list)}
        self.num_cities = len(self.city_list)

        # Store new parameters
        self.pos_dim = pos_dim
        self.lap_pe_use_dim = min(lap_pe_use_dim, lap_pe_dim)  # Can't use more than available

        # Calculate effective input channels
        effective_in_channels = in_channels
        if use_pos:
            effective_in_channels += pos_dim  # ✅ UPDATED: Use configurable pos_dim
        if use_lap_pe:
            effective_in_channels += self.lap_pe_use_dim  # ✅ UPDATED: Use configurable lap_pe_use_dim
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
            log_to_wandb=log_to_wandb,
            use_target_standardization=use_target_standardization,
            use_city_balanced_loss=use_city_balanced_loss
        )

        # Store parameters for forward pass
        self.use_pos = use_pos
        self.use_pos_encoding = use_pos_encoding
        self.use_lap_pe = use_lap_pe
        self.lap_pe_dim = lap_pe_dim
        self.use_anchor_pe = use_anchor_pe
        self.anchor_k = anchor_k
        self.anchor_m = anchor_m
        self.use_rwse = use_rwse
        self.rwse_dim = rwse_dim

        self.hidden_channels = hidden_channels
        self.embed_dim = hidden_channels[-1]
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.use_graph_conv = use_graph_conv
        self.use_graph_norm = use_graph_norm
        self.graph_conv_type = graph_conv_type
        self.num_graph_conv_layers = len(hidden_channels) if use_graph_conv else 0
        self.use_residuals = use_residuals

        self.pad_to = pad_to
        self.pad_value = pad_value

        # Domain classifier architecture
        if domain_classifier_layers is None:
            self.domain_classifier_layers = [self.embed_dim // 2]
        else:
            self.domain_classifier_layers = domain_classifier_layers
            
        # ✅ Store edge dropout probability
        self.message_drop_prob = message_drop_prob

        # Log to wandb
        if self.log_to_wandb:
            wandb.config.update({
                'in_channels': self.in_channels,
                'use_pos': use_pos,
                'use_pos_encoding': use_pos_encoding,
                'use_lap_pe': use_lap_pe,
                'lap_pe_dim': lap_pe_dim,
                'use_anchor_pe': use_anchor_pe,
                'anchor_k': anchor_k,
                'anchor_m': anchor_m,
                'use_rwse': use_rwse,
                'rwse_dim': rwse_dim,
                'use_dann': use_dann,
                'num_cities': self.num_cities,
                'domain_classifier_layers': self.domain_classifier_layers,
                'effective_in_channels': self.in_channels,
                'embed_dim': self.embed_dim,
                'ff_dim': ff_dim,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'num_nodes': num_nodes,
                'use_graph_conv': use_graph_conv,
                'use_graph_norm': use_graph_norm,
                'graph_conv_type': graph_conv_type,
                'num_graph_conv_layers': self.num_graph_conv_layers,
                'pad_to': pad_to,
                'pad_value': pad_value,
                'message_drop_prob': message_drop_prob,  # ✅ NEW
                'pos_dim': pos_dim,  # ✅ NEW
                'lap_pe_dim': lap_pe_dim,
                'lap_pe_use_dim': self.lap_pe_use_dim,  # ✅ NEW
            }, allow_val_change=True)

        self.define_layers()

    def define_layers(self):
        # Graph convolution backbone with variable hidden_channels
        self.graph_convs = nn.ModuleList()
        self.graph_norms = nn.ModuleList() if self.use_graph_norm else None

        for i, hidden_dim in enumerate(self.hidden_channels):
            in_dim = self.in_channels if i == 0 else self.hidden_channels[i - 1]

            if self.graph_conv_type == 'gcn':
                conv = GCNConv(in_dim, hidden_dim)

            elif self.graph_conv_type == 'gatv2':
                conv = GATv2Conv(in_dim,
                                hidden_dim // self.num_heads,
                                heads=self.num_heads,
                                dropout=self.message_drop_prob)

            elif self.graph_conv_type == 'graph':
                conv = GraphConv(in_dim, hidden_dim)

            elif self.graph_conv_type == 'trans_conv':
                conv = TransformerConv(in_dim,
                                    hidden_dim // self.num_heads,
                                    heads=self.num_heads,
                                    dropout=self.message_drop_prob)
            else:
                raise ValueError(f"Unsupported conv type: {self.graph_conv_type}")

            self.graph_convs.append(conv)

            if self.use_graph_norm:
                self.graph_norms.append(GraphNorm(in_dim))

        # Transformer encoder (operates after graph conv backbone)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout if self.use_dropout else 0.0,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Output regression layer
        self.output = nn.Linear(self.embed_dim, 1)

        # Positional embedding (optional)
        if self.use_pos_encoding:
            self.pos_embedding = nn.Embedding(self.num_nodes, self.embed_dim)

        # DANN Domain Classifier (optional)
        if self.use_dann:
            domain_layers = []
            prev_dim = self.embed_dim
            for hidden_dim in self.domain_classifier_layers:
                domain_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout if self.use_dropout else 0.0)
                ])
                prev_dim = hidden_dim
            
            # Final classification layer
            domain_layers.append(nn.Linear(prev_dim, self.num_cities))
            
            self.domain_classifier = nn.Sequential(*domain_layers)

        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)


    def get_city_labels(self, datalist):
        """Convert city names to indices for domain classification."""
        city_labels = []
        for d in datalist:
            if hasattr(d, 'city'):
                city_idx = self.city_to_idx.get(d.city, 0)  # Default to 0 if city not found
            else:
                city_idx = 0  # Default city index
            city_labels.append(city_idx)
        return torch.tensor(city_labels, dtype=torch.long, device=next(self.parameters()).device)

    def forward(self, data, alpha=1.0):
        """
        Forward pass with optional DANN.
        
        Args:
            data: Input batch
            alpha: Gradient reversal strength for DANN (0.0 = no reversal, 1.0 = full reversal)
        
        Returns:
            If use_dann=False: predictions [total_nodes_across_batch, 1]
            If use_dann=True: (predictions, domain_logits) where domain_logits [batch_size, num_cities]
        """
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
                if hasattr(d, 'pos') and d.pos is not None:
                    pos = d.pos.to(self.dtype)
                    if pos.dim() == 3:
                        if self.pos_dim == 2:
                            pos_features = pos[:, 1, :]
                        elif self.pos_dim == 6:
                            pos_features = pos.reshape(pos.size(0), -1)
                        elif self.pos_dim == 4:
                            pos_features = torch.cat([pos[:, 0, :], pos[:, 2, :]], dim=-1)
                        else:
                            raise ValueError("pos_dim must be 2, 4, or 6 when pos is 3D")
                    else:
                        raise ValueError("Position features are enabled but 'pos' attribute is missing in data.")
                    xi = torch.cat([xi, pos_features], dim=-1)
                else:
                    raise ValueError("Position features are enabled but 'pos' attribute is missing in data.")

            # Laplacian PE
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
                    # pos is already [N, 2] from collate function
                    pos = d.pos.to(device)
                    ade = anchor_distance_encoding(pos, d.edge_index, K=self.anchor_k, M=self.anchor_m)
                    xi = torch.cat([xi, ade], dim=-1)
                else:
                    raise ValueError("Anchor PE is enabled but 'pos' attribute is missing in data.")

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
            for i, conv in enumerate(self.graph_convs):
                if self.use_graph_norm:
                    x = self.graph_norms[i](x, batch=data.batch if hasattr(data, 'batch') else None)

                x_in = x
                x = conv(x, edge_index)

                # residual connection (if enabled and dims match)
                if self.use_residuals and x.shape == x_in.shape:
                    x = x + x_in

                x = nn.functional.relu(x)
                if self.use_dropout:
                    x = self.dropout_layer(x)

            data.x = x

        # ----- Transformer + prediction -----
        datalist = data.to_data_list() if isinstance(data, Batch) else [data]
        all_predictions, all_graph_embeddings = [], []

        for d in datalist:
            x = d.x.unsqueeze(0)  # [1, N, embed_dim]

            # optional learned positional encoding
            if self.use_pos_encoding:
                seq_len = x.size(1)
                if seq_len <= self.num_nodes:
                    node_indices = torch.arange(seq_len, device=device).long()
                    pos_emb = self.pos_embedding(node_indices).unsqueeze(0)
                    x = x + pos_emb

            # Transformer encoder
            x_transformed = self.transformer(x)

            # Node-level regression
            preds = self.output(x_transformed.squeeze(0))  # [N, 1]
            all_predictions.append(preds.squeeze(-1))

            # domain classifier
            if self.use_dann:
                graph_embedding = x_transformed.mean(dim=1).squeeze(0)
                all_graph_embeddings.append(graph_embedding)

        result = torch.cat(all_predictions, dim=0)  # [total_nodes]

        if self.use_dann:
            graph_embeddings = torch.stack(all_graph_embeddings, dim=0)
            reversed_embeddings = gradient_reversal(graph_embeddings, alpha)
            domain_logits = self.domain_classifier(reversed_embeddings)
            return result.unsqueeze(-1), domain_logits

        return result.unsqueeze(-1)
