import os
import sys

import wandb

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, GraphConv, TransformerConv, GraphNorm, GATv2Conv

# Add 'scripts' dir to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.base_gnn import BaseGNN

class TransEncoder(BaseGNN):
    def __init__(self,
                 in_channels: int = 5,
                 out_channels: int = 1,
                 dropout: float = 0.1,
                 use_dropout: bool = False,
                 dtype: torch.dtype = torch.float32,
                 log_to_wandb: bool = False,
                 use_target_standardization: bool = False,
                 target_normalization: str = None,

                 # Transformer Parameters
                 ff_dim: int = 256,
                 num_layers: int = 3,
                 num_heads: int = 4, # Also for GNN
                 
                 # GNN Parameters
                 use_graph_conv: bool = True,
                 graph_conv_type: str = 'trans_conv', # 'gcn', 'gatv2', 'graph'
                 hidden_channels: list[int] = [128, 256, 128],
                 use_graph_norm: bool = False,
                 use_residuals: bool = False,
                 message_drop_prob: float = 0.0,

                 # POSITIONAL ENCODING PARAMETERS
                 use_pos: bool = True,
                 pos_dim: int = 6,
                 use_lap_pe: bool = False,
                 lap_pe_dim: int = 8):

        # Calculate effective input channels
        effective_in_channels = in_channels
        if use_pos:
            effective_in_channels += pos_dim
        if use_lap_pe:
            effective_in_channels += lap_pe_dim

        super().__init__(
            in_channels=effective_in_channels,
            out_channels=out_channels,
            dropout=dropout,
            use_dropout=use_dropout,
            dtype=dtype,
            log_to_wandb=log_to_wandb,
            use_target_standardization=use_target_standardization,
            target_normalization=target_normalization
        )

        # Model specific parameters
        self.hidden_channels = hidden_channels
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_graph_conv = use_graph_conv
        self.use_graph_norm = use_graph_norm
        self.graph_conv_type = graph_conv_type
        self.use_residuals = use_residuals
        self.use_pos = use_pos
        self.use_lap_pe = use_lap_pe
        self.lap_pe_dim = lap_pe_dim
        self.message_drop_prob = message_drop_prob
        self.pos_dim = pos_dim

        # Infered parameters
        self.embed_dim = hidden_channels[-1] # d_model for transformer
        self.num_graph_conv_layers = len(hidden_channels) if use_graph_conv else 0

        # Log to wandb
        if self.log_to_wandb:
            wandb.config.update({
                'in_channels': self.in_channels,
                'feature_in_channels': in_channels,
                'use_pos': use_pos,
                'pos_dim': pos_dim,
                'use_lap_pe': use_lap_pe,
                'lap_pe_dim': lap_pe_dim,
                'embed_dim': self.embed_dim,
                'ff_dim': ff_dim,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'use_graph_conv': use_graph_conv,
                'graph_conv_type': graph_conv_type,
                'hidden_channels': hidden_channels,
                'use_graph_norm': use_graph_norm,
                'use_residuals': use_residuals,
                'num_graph_conv_layers': self.num_graph_conv_layers,
                'message_drop_prob': message_drop_prob,
            }, allow_val_change=True)

        self.define_layers()

    def define_layers(self):
        
        # Graph convolution backbone with variable hidden_channels
        if self.use_graph_conv:
            
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
            batch_first=True, norm_first=True) # Set norm_first to False for post-normalization
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Output regression layer
        self.output = nn.Linear(self.embed_dim, 1)

        # Optional: Dropout between Convolutional layers
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)

        # Optional: Embed input into higher dimension (only if no graph conv backbone)
        self.embed = nn.Linear(self.in_channels, self.embed_dim)

    
    def forward(self, data):

        # Add extra POS features to nodes
        if self.use_pos or self.use_lap_pe:
            
            x_new = data.x

            # Add pos coordinates
            if self.use_pos:
                
                start_pos = data.pos[:, 0, :]
                end_pos = data.pos[:, 1, :]
                mid_pos = data.pos[:, 2, :]
                
                # Concatenate along the feature dimension
                if self.pos_dim == 2:
                    x_new = torch.cat((x_new, mid_pos), dim=-1)
                elif self.pos_dim == 4:
                    x_new = torch.cat((x_new, start_pos, end_pos), dim=-1)
                elif self.pos_dim == 6:
                    x_new = torch.cat((x_new, start_pos, mid_pos, end_pos), dim=-1)  
                else:
                    raise ValueError(f"Unsupported pos_dim: {self.pos_dim}. Supported: 2, 4, 6.")

            # Add Laplacian PE
            if self.use_lap_pe:
                if hasattr(data, 'lap_pe') and data.lap_pe is not None:
                    if self.lap_pe_dim > data.lap_pe.size(1):
                        raise ValueError(f"{self.lap_pe_dim} is greater than LAP PE dimensions available in data = {data.lap_pe.size(1)}") 
                    else:
                        lap_pe = data.lap_pe[:, :self.lap_pe_dim]
                        x_new = torch.cat((x_new, lap_pe), dim=-1)
                else:
                    raise ValueError("Laplacian positional encodings not found in data object!")

            # Updated node features
            data.x = x_new

        # NOW apply graph convolutional backbone
        if self.use_graph_conv:
            
            # Unpack data
            x, edge_index = data.x, data.edge_index
            x = x.to(self.dtype)
            
            for i, conv in enumerate(self.graph_convs):

                if self.use_residuals and i > 0 and self.hidden_channels[i] == self.hidden_channels[i - 1]:
                    x_0 = x
                
                if self.use_graph_norm:
                    x = self.graph_norms[i](x)
    
                x = conv(x, edge_index)

                # Residual connection
                if self.use_residuals and i > 0 and self.hidden_channels[i] == self.hidden_channels[i - 1]:
                    x = x + x_0

                x = nn.functional.relu(x)
                
                if self.use_dropout:
                    x = self.dropout_layer(x)

            data.x = x

        
        ### Transformer Encoder ###

        # Unpack data
        if isinstance(data, Batch):
            datalist = data.to_data_list()
        elif isinstance(data, Data):
            datalist = [data]
        else:
            raise ValueError("Input data must be a Batch or Data object")

        outputs = []
        for graph_data in datalist:
            x = graph_data.x.unsqueeze(0)  # [1, num_nodes, embed_dim]

            if not self.use_graph_conv:
                x = self.embed(x.to(self.dtype))

            out_graph = self.transformer(x).squeeze(0)
            outputs.append(out_graph)

        out = torch.cat(outputs, dim=0)
        
        out = self.output(out)
        return out.reshape(-1, 1)