"""Convolution operator that uses the Laplacian as graph shift operator."""

import torch
import torch.nn as nn

from ..laplacian import degree_normalization, magnetic_edge_laplacian


class MagneticEdgeLaplacianConv(nn.Module):
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
        cached: bool = False,
        bias: bool | None = None,
        normalize: bool = True,
        signed_in: bool = True,
        signed_out: bool = True,
        q: float = 1.0,
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
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self._cached_laplacian = None

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, is_directed: torch.Tensor
    ) -> torch.Tensor:
        if self._cached_laplacian is not None:
            laplacian = self._cached_laplacian
        else:
            laplacian = magnetic_edge_laplacian(
                edge_index,
                is_directed,
                return_incidence=False,
                q=self.q,
                signed_in=self.signed_in,
                signed_out=self.signed_out,
            )
            if self.normalize:
                laplacian = degree_normalization(laplacian)
            if self.cached:
                self._cached_laplacian = laplacian

        x = self.lin(x).to(torch.float32)
        if self.q == 0.0:
            x = laplacian @ x
        else:
            x = torch.view_as_real(
                laplacian
                @ torch.view_as_complex(x.reshape(-1, self.out_channels // 2, 2))
            ).reshape(-1, self.out_channels)

        if self.bias is not None:
            x = x + self.bias

        return x
