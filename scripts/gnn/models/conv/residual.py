"""Wrapper for residual connections around the convolution operators."""

import torch
import torch.nn as nn


class ResidualWrapper(nn.Module):
    """Wrapper for residual connections around the convolution operators."""

    def __init__(
        self,
        conv: nn.Module,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.conv = conv
        self.reset_parameters()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.lin(x) + self.conv(x, *args, **kwargs)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.conv.reset_parameters()  # pyright: ignore
