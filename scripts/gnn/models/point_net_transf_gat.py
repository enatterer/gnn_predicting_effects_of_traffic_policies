import wandb

import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import Sequential as GeoSequential, TransformerConv, GATConv, PointNetConv, global_mean_pool

from gnn.models.base_gnn import BaseGNN

"""
This architecture is a Graph Neural Network (GNN) model that combines PointNet Convolutions, Graph Attention Networks, and Transformer layers.
It is designed to predict the effects of traffic policies using graph-based data.
The model includes configurations for dropout, Monte Carlo dropout, and mode statistics prediction. Mode statistics prediction is not finetuned. 
This architecture was used for the paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5182100
The experiments in the paper were conducted using 10,000 simulations of a 1% downsampled population of Paris.
"""

class PointNetTransfGAT(BaseGNN):
    def __init__(self, 
                in_channels: int = 5, 
                out_channels: int = 1, 
                point_net_conv_layer_structure_local_mlp: list = [256], 
                point_net_conv_layer_structure_global_mlp: list = [512], 
                gat_conv_layer_structure: list = [128, 256, 512], 
                dropout: float = 0.3, 
                use_dropout: bool = False,
                predict_mode_stats: bool = False,
                dtype: torch.dtype = torch.float32,
                log_to_wandb: bool = False):
        
        """
        Initialize the GNN model with specified configurations.

        Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - point_net_conv_layer_structure_local_mlp (list): Layer structure for local MLP in PointNetConv.
        - point_net_conv_layer_structure_global_mlp (list): Layer structure for global MLP in PointNetConv.
        - gat_conv_layer_structure (list): Layer structure for GATConv layers.
        - dropout (float, optional): Dropout rate. Default is 0.0.
        - use_dropout (bool, optional): Whether to use dropout. Default is False.
        - predict_mode_stats (bool, optional): Whether to predict mode stats. Default is False.
        """
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
        self.pnc_local = point_net_conv_layer_structure_local_mlp
        self.pnc_global = point_net_conv_layer_structure_global_mlp
        self.gat_conv = gat_conv_layer_structure

        if self.log_to_wandb:
            wandb.config.update({"pnc_local": self.pnc_local,
                                "pnc_global": self.pnc_global,
                                "gat_conv": self.gat_conv},
                                allow_val_change=True)

        # Define the layers of the model
        self.define_layers()

        # Initialize weights
        self.initialize_weights()
        
    def define_layers(self):

        # Initialize dropout if needed
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)

        # PointNet layers 
        # Use start + end pos
        self.point_net_conv_1 = self.create_point_net_layer(
            gat_conv_starts_with_layer=self.gat_conv[0], 
            is_first_layer=True, 
            is_last_layer=False
        )
        self.point_net_conv_2 = self.create_point_net_layer(
            gat_conv_starts_with_layer=self.gat_conv[0], 
            is_first_layer=False, 
            is_last_layer=True
        )
        
        # GAT layers
        layers_global = self.define_gat_layers()
        self.gat_graph_layers = GeoSequential('x, edge_index', layers_global)
        
        # Output layers
        self.read_out_node_predictions = nn.Linear(64, 1)
        
        # Mode stats predictor (if enabled)
        if self.predict_mode_stats:
            self.mode_stat_predictor = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=4), num_layers=2),
                nn.Linear(64, 2)
            )

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

        # Use start + end pos
        pos1 = data.pos[:, 0, :]  # Start position
        pos2 = data.pos[:, 1, :]  # End position
        x = self.point_net_conv_1(x, pos1, edge_index)
        x = self.point_net_conv_2(x, pos2, edge_index)
        
        x = self.gat_graph_layers(x, edge_index)
        node_predictions = self.read_out_node_predictions(x)
        
        if self.predict_mode_stats:
            
            mode_stats = data.mode_stats
            batch = data.batch
            pooled_node_predictions = global_mean_pool(x, batch)
            shape_node_preds = pooled_node_predictions.shape[0]
            shape_mode_stats = int(mode_stats.shape[0]/shape_node_preds)
            
            tensor_for_pooling = torch.repeat_interleave(torch.arange(shape_node_preds), shape_mode_stats).to(x.device)
            mode_stats_pooled = global_mean_pool(mode_stats, tensor_for_pooling)
            
            mode_stats_pred = self.mode_stat_predictor(mode_stats_pooled)
            mode_stats_pred = mode_stats_pred.repeat_interleave(shape_mode_stats, dim=0)
            
            return node_predictions, mode_stats_pred
        
        return node_predictions
    
    def define_gat_layers(self):
        """
        Define layers for GATConv based on configuration.

        Returns:
        - List: Layers for GATConv.
        """
        layers = []
        for idx in range(len(self.gat_conv) - 1):      
            # Transformer layer
            layers.append((TransformerConv(self.gat_conv[idx], int(self.gat_conv[idx + 1]/4), heads=4), 'x, edge_index -> x'))
            layers.append(nn.ReLU(inplace=True))
            if self.use_dropout:
                layers.append(self.dropout_layer)
        layers.append((GATConv(self.gat_conv[-1], 64), 'x, edge_index -> x'))
        return layers
    
    def create_point_net_layer(self, gat_conv_starts_with_layer:int, is_first_layer:bool=False, is_last_layer:bool=False):
        """
        Create PointNetConv layers with specified configurations.

        Parameters:
        - gat_conv_starts_with_layer (int): Starting layer size for GATConv.

        Returns:
        - Tuple[nn.Sequential, nn.Sequential]: Local and global MLP layers.
        """
        offset_due_to_pos = 2
        local_MLP_layers = []
        if is_first_layer:
            local_MLP_layers.append(nn.Linear(self.in_channels + offset_due_to_pos, self.pnc_local[0]))
        else:
            local_MLP_layers.append(nn.Linear(self.pnc_global[-1] + offset_due_to_pos, self.pnc_local[0]))
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
        for idx in range(len(self.pnc_global) - 1):
            global_MLP_layers.append(nn.Linear(self.pnc_global[idx], self.pnc_global[idx + 1]))
            global_MLP_layers.append(nn.ReLU())
                
        if is_last_layer:
            global_MLP_layers.append(nn.Linear(self.pnc_global[- 1], gat_conv_starts_with_layer))
        else:
            global_MLP_layers.append(nn.Linear(self.pnc_global[-1], self.pnc_global[-1]))
        
        global_MLP_layers.append(nn.ReLU())
        if self.use_dropout:
            global_MLP_layers.append(self.dropout_layer)
        global_MLP = nn.Sequential(*global_MLP_layers)
        return PointNetConv(local_nn=local_MLP, global_nn=global_MLP)
    
    # WEIGHT INITIALIZION (Override)
    def initialize_weights(self):
        """
        Initialize model weights using Xavier and Kaiming initialization.
        """
        super().initialize_weights() # Call parent class method for Linear Layers
        for m in self.modules():
            if isinstance(m, PointNetConv):
                self._initialize_pointnetconv(m)
            elif isinstance(m, GATConv):
                self._initialize_gatconv(m)

    def _initialize_pointnetconv(self, m: PointNetConv):
        """
        Initialize weights for PointNetConv layers.

        Parameters:
        - m (PointNetConv): PointNetConv layer to initialize.
        """
        for name, param in m.local_nn.named_parameters():
            if param.dim() > 1:  # weight parameters
                init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            else:  # bias parameters
                init.zeros_(param)
        for name, param in m.global_nn.named_parameters():
            if param.dim() > 1:  # weight parameters
                init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            else:  # bias parameters
                init.zeros_(param)

    def _initialize_gatconv(self, m: GATConv):
        """
        Initialize weights for GATConv layers.

        Parameters:
        - m (GATConv): GATConv layer to initialize.
        """
        if hasattr(m, 'lin') and m.lin is not None:
            init.xavier_normal_(m.lin.weight)
            if m.lin.bias is not None:
                init.zeros_(m.lin.bias)
        if hasattr(m, 'att_src') and m.att_src is not None:
            init.xavier_normal_(m.att_src)
        if hasattr(m, 'att_dst') and m.att_dst is not None:
            init.xavier_normal_(m.att_dst)