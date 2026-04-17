import os
import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GraphNorm

# Add 'scripts' dir to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.base_gnn import BaseGNN


class TPB(BaseGNN):
    """
    TPB-inspired model adapted for this repository's graph regression setup.

    Core mechanisms mirrored from paper:
    - Source pretraining learns transferable patch/node encoder.
    - Learnable Traffic Pattern Bank queried by target/source nodes.
    - Pattern aggregation yields metaknowledge.
    - Metaknowledge-based graph reconstruction guides forecasting.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_dim: int = 128,
        bank_size: int = 64,
        key_dim: int = 64,
        dropout: float = 0.1,
        use_dropout: bool = False,
        dtype: torch.dtype = torch.float32,
        log_to_wandb: bool = False,
        use_target_standardization: bool = False,
        target_normalization: str = None,
        adjacency_temperature: float = 1.0,
        use_transformer_aggregator: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            use_dropout=use_dropout,
            dtype=dtype,
            log_to_wandb=log_to_wandb,
            use_target_standardization=use_target_standardization,
            target_normalization=target_normalization,
        )
        self.hidden_dim = hidden_dim
        self.bank_size = bank_size
        self.key_dim = key_dim
        self.adjacency_temperature = float(adjacency_temperature)
        self.use_transformer_aggregator = bool(use_transformer_aggregator)
        self.define_layers()
        self.initialize_weights()

    def define_layers(self):
        # Traffic patch/node encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout if self.use_dropout else 0.0),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Pattern bank + query keys
        self.key_bank = nn.Parameter(torch.randn(self.bank_size, self.key_dim) * 0.02)
        self.pattern_bank = nn.Parameter(torch.randn(self.bank_size, self.hidden_dim) * 0.02)
        self.query_proj = nn.Linear(self.hidden_dim, self.key_dim)

        # Pattern aggregation (default fast feed-forward path).
        self.aggregator_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout if self.use_dropout else 0.0),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        # Optional slower transformer path for smaller graphs / ablations.
        self.aggregator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=4,
                dim_feedforward=4 * self.hidden_dim,
                dropout=self.dropout if self.use_dropout else 0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1,
        )

        # Metaknowledge-based graph reconstruction
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Downstream spatial-temporal proxy (graph model)
        self.conv1 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.norm1 = GraphNorm(self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.norm2 = GraphNorm(self.hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.out_channels),
        )

    def _query_pattern_bank(self, z: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(z)  # [N, key_dim]
        attn_logits = torch.matmul(q, self.key_bank.t())  # [N, K]
        attn = torch.softmax(attn_logits, dim=-1)
        return torch.matmul(attn, self.pattern_bank)  # [N, hidden]

    def _reconstruct_edge_weight(self, m: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(m)
        k = self.k_proj(m)
        src, dst = edge_index[0], edge_index[1]
        logits = (q[src] * k[dst]).sum(dim=-1) / max(self.adjacency_temperature, 1e-6)
        return torch.sigmoid(logits)

    def forward(self, data):
        x, edge_index = data.x.to(self.dtype), data.edge_index
        z = self.encoder(x)

        retrieved = self._query_pattern_bank(z)
        if self.use_transformer_aggregator:
            # Treat node set as a single sequence. This is expressive but can be slow on large graphs.
            m = self.aggregator(retrieved.unsqueeze(0)).squeeze(0)
        else:
            # Fast default path: per-node feed-forward aggregation with residual connection.
            m = retrieved + self.aggregator_ffn(retrieved)

        edge_weight = self._reconstruct_edge_weight(m, edge_index)
        h = self.conv1(z, edge_index, edge_weight=edge_weight)
        h = self.norm1(F.gelu(h))
        if self.use_dropout:
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        h = self.norm2(F.gelu(h))
        if self.use_dropout:
            h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.head(torch.cat([m, h], dim=-1))
        return out.reshape(-1, 1)
