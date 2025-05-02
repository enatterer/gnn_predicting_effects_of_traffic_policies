import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

from gnn.models.base_gnn import BaseGNN

class GraphSAGE(BaseGNN):
    def __init__(self, 
                in_channels: int = 0, 
                out_channels: int = 0,
                hidden_channels: int = 32,
                num_layers: int = 3,
                dropout: float = 0.3, 
                use_dropout: bool = False,
                predict_mode_stats: bool = False,
                dtype: torch.dtype = torch.float32,
                verbose: bool = True):
    
        # Call parent class constructor
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            use_dropout=use_dropout,
            predict_mode_stats=predict_mode_stats,
            dtype=dtype,
            verbose=verbose)
        
        # Model specific parameters
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # Define the layers of the model
        self.define_layers()

        # Initialize weights
        self.initialize_weights()

        if verbose:
            print("Model initialized")
            print(self) # Print the model architecture

    def define_layers(self):
        
        for i in range(self.num_layers):
            if i == 0:
                in_channels = self.in_channels
            else:
                in_channels = self.hidden_channels

            if i == self.num_layers - 1:
                out_channels = self.out_channels
            else:
                out_channels = self.hidden_channels

            # Define the convolutional layer
            conv = SAGEConv(in_channels, out_channels)
            setattr(self, f'conv{i + 1}', conv)
        
        if self.use_dropout:
            self.dropout = torch.nn.Dropout(self.dropout)

    def forward(self, data):

        # Unpack data
        x = data.x.to(self.dtype)
        edge_index = data.edge_index

        for i in range(self.num_layers):
            conv = getattr(self, f'conv{i + 1}')
            x = conv(x, edge_index)

            if i != self.num_layers - 1:
                x = F.relu(x)
                if self.use_dropout:
                    x = self.dropout(x)
        
        return x