"""Fusion layers to model interactions between signed (orientation equivariant) and unsinged (orientation invariant) edge signals."""

import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    def __init__(
        self,
        in1_channels: int,
        in2_channels: int,
        out_channels: int,
        **kwargs,
    ):
        """
        Parameter-efficient bi-linear fusion layer.

        Args:
            in1_channels (int): Input dimension for the first sample.
            in2_channels (int): Input dimension for the second sample.
            out_channels (int): Output dimension.
        """
        super().__init__()

        self.in1_channels = in1_channels
        self.in2_channels = in2_channels
        self.out_channels = out_channels

        self.lin_layer1 = nn.Linear(
            in_features=self.in1_channels,
            out_features=self.out_channels,
            bias=(kwargs.get('bias', True)),
        )

        self.lin_layer2 = nn.Linear(
            in_features=self.in2_channels,
            out_features=self.out_channels,
            bias=(kwargs.get('bias', True)),
        )

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input1 = self.lin_layer1(input1)
        input2 = self.lin_layer2(input2)
        fused_representation = input1 * input2
        return fused_representation

    def reset_parameters(self):
        self.lin_layer1.reset_parameters()
        self.lin_layer2.reset_parameters()
