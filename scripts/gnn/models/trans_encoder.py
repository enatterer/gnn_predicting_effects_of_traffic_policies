import os
import sys

import tqdm as tqdm
import wandb

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, GATConv, GraphConv, TransformerConv, GraphNorm
from torch.nn.utils.rnn import pad_sequence

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.base_gnn import BaseGNN


"""
Transformer Encoder for Graph Neural Networks

This model implements a transformer encoder that can optionally incorporate graph structure
through various mechanisms. The base transformer operates on node features without explicit
graph awareness, but can be enhanced with:

1. Positional Information: Add node positions as additional features
2. Positional Encoding: Learn embeddings for node positions (like in transformers)
3. Graph Convolution Layers: Pre-process features using graph structure before transformer
"""

class TransEncoder(BaseGNN):
    def __init__(self, 
                in_channels: int = 5,
                out_channels: int = 1,
                embed_dim: int = 128,
                ff_dim: int = 512,
                num_heads: int = 4,
                num_layers: int = 5,
                num_nodes: int = 31635,
                use_pos: bool = True,
                use_pos_encoding: bool = True,
                dropout: float = 0.1,
                use_dropout: bool = False,
                use_graph_conv: bool = True,
                use_graph_norm: bool = False,
                graph_conv_type: str = 'trans_conv', #Options: gatv2,gcn,trans_conv
                num_graph_conv_layers: int = 2,
                predict_mode_stats: bool = False,
                dtype: torch.dtype = torch.float32,
                log_to_wandb: bool = False,
                # NEW ↓↓↓
                pad_to: int | None = None,      # None -> pad to max in batch; int -> pad to this many nodes
                pad_value: float = 0.0):        # value used for padding
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            use_dropout=use_dropout,
            predict_mode_stats=predict_mode_stats,
            dtype=dtype,
            log_to_wandb=log_to_wandb)

        self.use_pos = use_pos
        self.use_pos_encoding = use_pos_encoding
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.use_graph_conv = use_graph_conv
        self.use_graph_norm = use_graph_norm
        self.graph_conv_type = graph_conv_type
        self.num_graph_conv_layers = num_graph_conv_layers

        # NEW
        self.pad_to = pad_to
        self.pad_value = pad_value

        if self.use_pos:
            self.in_channels += 2

        if self.log_to_wandb:
            wandb.config.update({'use_pos': use_pos,
                                 'use_pos_encoding': use_pos_encoding,
                                 'in_channels': self.in_channels,
                                 'embed_dim': embed_dim,
                                 'ff_dim': ff_dim,
                                 'num_heads': num_heads,
                                 'num_layers': num_layers,
                                 'num_nodes': num_nodes,
                                 'use_graph_conv': use_graph_conv,
                                 'use_graph_norm': use_graph_norm,
                                 'graph_conv_type': graph_conv_type,
                                 'num_graph_conv_layers': num_graph_conv_layers,
                                 'pad_to': pad_to,
                                 'pad_value': pad_value},
                                 allow_val_change=True)
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
            # Keep capacity up to num_nodes. We'll index per-batch length in forward().
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
                setattr(self, f'conv_{i + 1}', conv)

                if self.use_graph_norm:
                    graph_norm = GraphNorm(self.embed_dim if i > 0 else self.in_channels)
                    setattr(self, f'graph_norm_{i + 1}', graph_norm)

        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, data):
        device = next(self.parameters()).device

        # ----- Optional pre-transform conv stack (node-wise features remain variable length) -----
        if self.use_graph_conv:
            x = data.x
            edge_index = data.edge_index
            if self.use_pos:
                pos = data.pos[:, 2, :]
                x = torch.cat((x, pos), dim=1)
            x = x.to(self.dtype)

            for i in range(self.num_graph_conv_layers):
                if self.use_graph_norm:
                    graph_norm = getattr(self, f'graph_norm_{i + 1}')
                    x = graph_norm(x)
                conv = getattr(self, f'conv_{i + 1}')
                x = conv(x, edge_index)
                x = nn.functional.relu(x)
                if self.use_dropout:
                    x = self.dropout_layer(x)
            data.x = x

        # ----- Gather per-graph tensors (variable Ni x C) -----
        if isinstance(data, Batch):
            datalist = data.to_data_list()
        elif isinstance(data, Data):
            datalist = [data]
        else:
            raise ValueError("Input data must be a Batch or Data object")

        x_list = []
        lengths = []
        for d in datalist:
            xi = d.x
            if self.use_pos and not self.use_graph_conv:
                pos = d.pos[:, 2, :]
                xi = torch.cat((xi, pos), dim=1)
            lengths.append(xi.size(0))
            x_list.append(xi.to(device))

        lengths = torch.tensor(lengths, device=device)  # [B]
        max_len_in_batch = int(lengths.max().item())
        target_len = self.pad_to if self.pad_to is not None else max_len_in_batch

        if self.use_pos_encoding and target_len > self.num_nodes:
            raise ValueError(
                f"pad_to/max_len ({target_len}) exceeds num_nodes ({self.num_nodes}) "
                "used to size the positional embedding. Increase num_nodes or set pad_to accordingly."
            )

        # ----- Pad to [B, S, C] -----
        # pad_sequence pads to the max length of the list; if we need larger fixed pad_to,
        # we append extra padding manually.
        padded = pad_sequence(x_list, batch_first=True, padding_value=self.pad_value)  # [B, max_len_in_batch, C]
        B, cur_S, C = padded.shape
        if target_len > cur_S:
            # add extra pad on the right to reach target_len
            extra = torch.full((B, target_len - cur_S, C), self.pad_value, dtype=padded.dtype, device=device)
            padded = torch.cat([padded, extra], dim=1)
        elif target_len < cur_S:
            # truncate if someone set pad_to smaller than max
            padded = padded[:, :target_len, :]
            lengths = torch.clamp(lengths, max=target_len)

        # ----- Build key padding mask: True = ignore -----
        # shape [B, S]
        arange_S = torch.arange(padded.size(1), device=device).unsqueeze(0)  # [1, S]
        key_padding_mask = arange_S >= lengths.unsqueeze(1)                  # [B, S], bool

        # ----- Project & (optional) positional encodings -----
        x = padded.to(self.dtype)
        if not self.use_graph_conv:
            x = self.embed(x)  # [B, S, D]
        else:
            # If conv emitted features of size embed_dim already, ensure shapes match embed layer input.
            if x.size(-1) != self.embed_dim:
                x = self.embed(x)

        if self.use_pos_encoding:
            # indices: [S], broadcast to [B, S, D]
            node_indices = torch.arange(x.size(1), device=device).long()
            pos_emb = self.pos_embedding(node_indices)           # [S, D]
            pos_emb = pos_emb.unsqueeze(0).expand(x.size(0), -1, -1)
            x = x + pos_emb

        # ----- Transformer with mask -----
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # [B, S, D]

        # ----- Per-node scalar -----
        x = self.output(x)   # [B, S, 1]
        
        # lengths per graph (number of real nodes)
        lengths = (~key_padding_mask).sum(dim=1)            # [B], on same device

        # compact predictions back to PyG's concatenated layout: [sum(Ni), 1]
        preds_list = [x[i, :lengths[i]] for i in range(x.size(0))]   # each [Ni, 1]
        preds_compact = torch.cat(preds_list, dim=0)                  # [sum(Ni), 1]

        # (optional) stash masks if you want to inspect later
        data.pad_mask = key_padding_mask.detach()
        data.pad_mask_flat = key_padding_mask.view(-1).detach()

        return preds_compact