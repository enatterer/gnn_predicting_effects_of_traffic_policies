import os
import sys
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.conv import (
    FusionLayer,
    ResidualWrapper,
)


class EIGNBlock(nn.Module):
    r"""Abstract block within the EIGN architecture that models signed (orientation signedvariant) and unsigned (orientation unsignedariant) modalities."""

    def __init__(
        self,
        in_channels_signed: int,
        out_channels_signed: int,
        in_channels_unsigned: int,
        out_channels_unsigned: int,
        use_fusion: bool = True,
        use_unsigned_to_signed_conv: bool = True,
        use_signed_to_unsigned_conv: bool = True,
        use_residual: bool = True,
        signed_activation_fn=F.tanh,
        unsigned_activation_fn=F.relu,
        **kwargs,
    ):
        super().__init__()
        self._in_channels_signed = in_channels_signed
        self._out_channels_signed = out_channels_signed
        self._in_channels_unsigned = in_channels_unsigned
        self._out_channels_unsigned = out_channels_unsigned
        self.use_residual = use_residual
        self.use_fusion = use_fusion
        self.use_unsigned_to_signed_conv = use_unsigned_to_signed_conv
        self.use_signed_to_unsigned_conv = use_signed_to_unsigned_conv
        self.signed_activation_fn = signed_activation_fn
        self.unsigned_activation_fn = unsigned_activation_fn

        # Fusion can only be used once we somehow obtain signed and unsignedariant inputs of the same size
        self.use_fusion = (
            use_fusion
            and (
                in_channels_signed > 0
                or (use_unsigned_to_signed_conv and in_channels_unsigned > 0)
            )
            and (
                in_channels_unsigned > 0
                or (use_signed_to_unsigned_conv and in_channels_signed > 0)
            )
            and out_channels_signed == out_channels_unsigned
        )

        if self.use_fusion:
            self.signed_fusion_layer = FusionLayer(
                in1_channels=out_channels_signed,
                in2_channels=out_channels_unsigned,
                out_channels=out_channels_signed,
                bias=False,
            )
            self.unsigned_fusion_layer = FusionLayer(
                in1_channels=out_channels_signed,
                in2_channels=out_channels_unsigned,
                out_channels=out_channels_unsigned,
            )
        if in_channels_unsigned > 0:
            self.unsigned_conv = self._wrap_residual(
                conv=self.initialize_convolution(
                    in_channels=in_channels_unsigned,
                    out_channels=out_channels_unsigned,
                    signed_in=False,
                    signed_out=False,
                    **kwargs,
                ),
                in_channels=in_channels_unsigned,
                out_channels=out_channels_unsigned,
            )
            if self.use_unsigned_to_signed_conv:
                self.unsigned_signed_conv = self.initialize_convolution(
                    in_channels=in_channels_unsigned,
                    out_channels=out_channels_signed,
                    signed_in=False,
                    signed_out=True,
                    **kwargs,
                )
        if in_channels_signed > 0:
            self.signed_conv = self._wrap_residual(
                conv=self.initialize_convolution(
                    in_channels=in_channels_signed,
                    out_channels=out_channels_signed,
                    signed_in=True,
                    signed_out=True,
                    **kwargs,
                ),
                in_channels=in_channels_signed,
                out_channels=out_channels_signed,
            )
            if self.use_signed_to_unsigned_conv:
                self.signed_unsigned_conv = self.initialize_convolution(
                    in_channels=in_channels_signed,
                    out_channels=out_channels_unsigned,
                    signed_in=True,
                    signed_out=False,
                    **kwargs,
                )

    @abstractmethod
    def initialize_convolution(
        self,
        in_channels: int,
        out_channels: int,
        signed_in: bool,
        signed_out: bool,
        **kwargs,
    ) -> nn.Module:
        """Creates a convolution operator for edge signals."""
        raise NotImplementedError

    @property
    def out_channels_signed(self) -> int:
        """The effective output dimension of the block for signedvariant features."""
        if self._in_channels_signed > 0 or (
            self.use_unsigned_to_signed_conv and self._in_channels_unsigned > 0
        ):
            return self._out_channels_signed
        else:
            return 0

    @property
    def out_channels_unsigned(self) -> int:
        """The effective output dimension of the block for unsignedariant features."""
        if self._in_channels_unsigned > 0 or (
            self.use_signed_to_unsigned_conv and self._in_channels_signed > 0
        ):
            return self._out_channels_unsigned
        else:
            return 0

    def _wrap_residual(
        self, conv: nn.Module, in_channels: int, out_channels: int
    ) -> nn.Module:
        """Wraps a convolution in a residual connection if desired."""
        if self.use_residual:
            return ResidualWrapper(
                in_channels=in_channels, out_channels=out_channels, conv=conv
            )
        else:
            return conv

    def fusion(self, x_signed, x_unsigned):
        new_x_signed = (
            self.signed_fusion_layer(
                x_signed,
                x_unsigned,
            )
            + x_signed
        )
        # Absolute value to keep representations orientation-unsignedariant.
        new_x_unsigned = (
            self.unsigned_fusion_layer(
                torch.abs(x_signed),
                x_unsigned,
            )
            + x_unsigned
        )
        return new_x_signed, new_x_unsigned

    def _mix(
        self, tensor1: torch.Tensor | None, tensor2: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Combines two tensors by element-wise addition if they are both not `None`."""
        match tensor1, tensor2:
            case None, None:
                return None
            case None, tensor2:
                return tensor2
            case tensor1, None:
                return tensor1
            case tensor1, tensor2:
                return tensor1 + tensor2

    def forward(
        self,
        x_signed: torch.Tensor | None,
        x_unsigned: torch.Tensor | None,
        edge_index: torch.Tensor,
        is_directed: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        h_signed_signed, h_unsigned_signed, h_signed_unsigned, h_unsigned_unsigned = (
            None,
            None,
            None,
            None,
        )

        # Convolutions
        if x_signed is not None and x_signed.size(-1) > 0:
            h_signed_signed = self.signed_conv(
                edge_index=edge_index,
                x=x_signed,
                is_directed=is_directed,
            )
            if self.use_signed_to_unsigned_conv:
                h_signed_unsigned = self.signed_unsigned_conv(
                    edge_index=edge_index,
                    x=x_signed,
                    is_directed=is_directed,
                )

        if x_unsigned is not None and x_unsigned.size(-1) > 0:
            h_unsigned_unsigned = self.unsigned_conv(
                edge_index=edge_index,
                x=x_unsigned,
                is_directed=is_directed,
            )
            if self.use_unsigned_to_signed_conv:
                h_unsigned_signed = self.unsigned_signed_conv(
                    edge_index=edge_index,
                    x=x_unsigned,
                    is_directed=is_directed,
                )

        # Mixing and activation
        h_signed_signed = self._mix(h_signed_signed, h_unsigned_signed)
        h_unsigned_unsigned = self._mix(h_unsigned_unsigned, h_signed_unsigned)

        if h_signed_signed is not None:
            h_signed_signed = self.signed_activation_fn(h_signed_signed)
        if h_unsigned_unsigned is not None:
            h_unsigned_unsigned = self.unsigned_activation_fn(h_unsigned_unsigned)

        # Fusion operators for local interactions
        if self.use_fusion:
            h_signed_signed, h_unsigned_unsigned = self.fusion(
                h_signed_signed, h_unsigned_unsigned
            )

        if h_signed_signed is None:
            assert x_signed is None or x_signed.size(-1) == 0
            h_signed_signed = x_signed

        if h_unsigned_unsigned is None:
            assert x_unsigned is None or x_unsigned.size(-1) == 0
            h_unsigned_unsigned = x_unsigned

        return h_signed_signed, h_unsigned_unsigned
