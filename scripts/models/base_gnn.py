from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class BaseGNN(nn.Module, ABC):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.0,
                 use_dropout: bool = False,
                 use_monte_carlo_dropout: bool = False,
                 predict_mode_stats: bool = False,
                 dtype: torch.dtype = torch.float32):
        """
        Base class for all GNN implementations.
        
        Parameters match your current architecture's core requirements.
        Additional parameters can be added in child classes.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_dropout = use_dropout
        self.use_monte_carlo_dropout = use_monte_carlo_dropout
        self.predict_mode_stats = predict_mode_stats
        self.dtype = dtype
        
        # Validate monte carlo dropout usage
        if self.use_monte_carlo_dropout and not self.use_dropout:
            raise ValueError("use_monte_carlo_dropout requires use_dropout to be True")
            
    @abstractmethod
    def forward(self, data):
        """
        Forward pass of the model.
        Must be implemented by all child classes.
        """
        pass

    def initialize_weights(self):
        """
        Initialize model weights. Can be overridden by child classes.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)