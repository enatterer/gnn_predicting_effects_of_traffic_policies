"""Convolution operator that uses the Laplacian as graph shift operator."""

import os
import sys
from typing import Protocol

import torch
import torch.nn as nn

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.conv.laplacian import degree_normalization, magnetic_edge_laplacian


class NodeFeatureTransformation(Protocol):
    """Protocol for node feature transformations."""

    def __call__(
        self, in_channels: int, out_channels: int, *args, **kwargs
    ) -> nn.Module: ...


class MagneticEdgeLaplacianWithNodeTransformationConv(nn.Module):
    r"""The magnetic edge Laplacian convolutional operator that does not transform node features.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        cached (bool, optional): Whether to cache the computed Laplacian. Defaults to False.
        bias (bool, optional): Whether to include a bias term. Defaults to None.
        normalize (bool, optional): Whether to normalize the Laplacian. Defaults to True.
        signed_in (bool, optional): Whether the edge inputs are signed (orientation equivariant). Defaults to True.
        signed_out (bool, optional): Whether the edge outputs are signed (orientation equivariant). Defaults to True.
        q (float, optional): Phase shift parameter for the magnetic Laplacian. Defaults to 1.0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        initialize_node_feature_transformation: NodeFeatureTransformation,
        cached: bool = False,
        bias: bool | None = None,
        normalize: bool = True,
        signed_in: bool = True,
        signed_out: bool = True,
        q: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        assert (
            out_channels % 2 == 0
        ), "out_channels must be even to model a real and complex part"
        if bias and signed_out:
            raise ValueError("Bias is not supported for signed output")
        if bias is None:
            bias = not signed_out

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.normalize = normalize
        self.signed_in = signed_in
        self.signed_out = signed_out
        self.q = q

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.node_feature_transformation = initialize_node_feature_transformation(
            out_channels, out_channels, *args, **kwargs
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if hasattr(self.node_feature_transformation, "reset_parameters"):
            self.node_feature_transformation.reset_parameters()  # pyright: ignore
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self._cached_laplacian = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        is_directed: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self._cached_laplacian is not None:
            laplacian, incidence_in, incidence_out = self._cached_laplacian
        else:
            laplacian, incidence_in, incidence_out = magnetic_edge_laplacian(
                edge_index,
                is_directed,
                q=self.q,
                signed_in=self.signed_in,
                signed_out=self.signed_out,
                return_incidence=True,
            )
            if self.normalize:
                laplacian, deg_inv_sqrt = degree_normalization(
                    laplacian, return_deg_inv_sqrt=True
                )
                incidence_in *= deg_inv_sqrt.reshape(1, -1)
                incidence_out *= deg_inv_sqrt.reshape(1, -1)
            if self.cached:
                self._cached_laplacian = laplacian, incidence_in, incidence_out

        x = self.lin(x)
        if self.q == 0.0:
            x = incidence_in @ x
            x = self.node_feature_transformation(x)
            x = incidence_out.t() @ x
        else:
            x_nodes = incidence_in @ torch.view_as_complex(
                x.reshape(-1, self.out_channels // 2, 2)
            )
            x_nodes = self.node_feature_transformation(
                torch.view_as_real(x_nodes).reshape(-1, self.out_channels),
                *args,
                **kwargs,
            )
            x = torch.view_as_real(
                incidence_out.t().conj()
                @ torch.view_as_complex(x_nodes.reshape(-1, self.out_channels // 2, 2))
            ).reshape(-1, self.out_channels)

        if self.bias is not None:
            x = x + self.bias

        return x
