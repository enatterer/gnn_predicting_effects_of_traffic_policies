'''
This file implements the architecture from the paper:
"EIGN: Efficient and Interpretable Graph Neural Networks" (https://arxiv.org/abs/2410.16935)
The implementation can be found here: https://github.com/dfuchsgruber/eign/tree/main
As all models in this repository, this model is a Graph Neural Network (GNN) that predicts the effects of traffic policies.

The parameters UseMonteCarloDropout and PredictModeStats may be implemented in the future.
'''

import torch
from base_gnn import BaseGNN

class Eign(BaseGNN):
    def __init__(self, 
                in_channels: int = 0, 
                out_channels: int = 0, 
                dropout: float = 0.3, 
                use_dropout: bool = False,
                predict_mode_stats: bool = False,
                dtype: torch.dtype = torch.float32,
                ):
        """
        Initialize the GNN model with specified configurations.

        Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - dropout (float, optional): Dropout rate. Default is 0.3.
        - use_dropout (bool, optional): Whether to use dropout. Default is False.
        """
        # Call parent class constructor
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            use_dropout=use_dropout,
            predict_mode_stats=predict_mode_stats,
            dtype=dtype
        )
        
    def forward(self, data):
        pass
    
    def define_layers(self):
        pass
    
    def initialize_weights(self):
        pass