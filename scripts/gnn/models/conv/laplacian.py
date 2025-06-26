from typing import Literal, overload

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

"""Laplacian operators for the Magnetic Edge GNN."""

def magnetic_incidence_matrix(
    edge_index: Tensor,
    is_directed: Tensor,
    num_nodes: int | None = None,
    q: float = 0.0,
    signed: bool = True,
) -> torch.Tensor:
    """Compute the incidence matrix for the magnetic edge graph Laplacian.

    Args:
        edge_index (Tensor): Edge index tensor of shape [2, num_edges].
        is_undirected (Tensor): Boolean tensor indicating whether the edges are undirected.
        num_nodes (int, optional): Number of nodes in the graph. Defaults to None.
        q (float, optional): Phase shift parameter for the magnetic Laplacian. Defaults to 0.0.
        signed (bool, optional): Whether to use signed edge weights. Defaults to True.

    Returns:
        Tensor: Incidence matrix of shape [num_nodes, num_edges], a sparse tensor.
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    num_edges = edge_index.size(1)

    row = edge_index.flatten()
    col = torch.arange(
        num_edges, device=edge_index.device, dtype=edge_index.dtype
    ).repeat(2)
    values = torch.ones_like(col, dtype=torch.float if q == 0.0 else torch.cfloat)
    if q != 0.0:
        values[:num_edges][is_directed] = np.exp(1j * np.pi * q)
        values[num_edges:][is_directed] = np.exp(-1j * np.pi * q)
    if signed:
        values[:num_edges] *= -1

    return torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),
        values=values,
        dtype=values.dtype,
        size=(num_nodes, num_edges),
    )


@overload
def magnetic_edge_laplacian(
    edge_index: Tensor,
    is_directed: Tensor,
    return_incidence: Literal[False],
    num_nodes: int | None = None,
    q: float = 0.0,
    signed_in: bool = True,
    signed_out: bool = True,
) -> Tensor: ...
@overload
def magnetic_edge_laplacian(
    edge_index: Tensor,
    is_directed: Tensor,
    return_incidence: Literal[True],
    num_nodes: int | None = None,
    q: float = 0.0,
    signed_in: bool = True,
    signed_out: bool = True,
) -> tuple[Tensor, Tensor, Tensor]: ...


def magnetic_edge_laplacian(
    edge_index: Tensor,
    is_directed: Tensor,
    return_incidence: bool = False,
    num_nodes: int | None = None,
    q: float = 0.0,
    signed_in: bool = True,
    signed_out: bool = True,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """Compute the magnetic edge Laplacian for the graph.

    Args:
        edge_index (Tensor): Edge index tensor of shape [2, num_edges].
        is_undirected (Tensor): Boolean tensor indicating whether the edges are undirected.
        num_nodes (int, optional): Number of nodes in the graph. Defaults to None.
        q (float, optional): Phase shift parameter for the magnetic Laplacian. Defaults to 0.0.
        signed (bool, optional): Whether to use signed edge weights. Defaults to True.

    Returns:
        Tensor: Magnetic edge Laplacian of shape [num_edges, num_edges], a sparse tensor.
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    incidence_in = magnetic_incidence_matrix(
        edge_index=edge_index,
        is_directed=is_directed,
        num_nodes=num_nodes,
        q=q,
        signed=signed_in,
    )
    incidence_out = magnetic_incidence_matrix(
        edge_index=edge_index,
        is_directed=is_directed,
        num_nodes=num_nodes,
        q=q,
        signed=signed_out,
    )
    laplacian = (
        incidence_out.t() if q == 0.0 else incidence_out.t().conj()
    ) @ incidence_in
    if return_incidence:
        return laplacian, incidence_in, incidence_out
    return laplacian


def degree_normalization(
    matrix: torch.Tensor, return_deg_inv_sqrt: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Normalizes the matrix based on the square roots of the out-degrees like GCN.

    Args:
        matrix (torch.Tensor): Matrix to normalize, shape [N, N].

    Returns:
        torch.Tensor: Degree normalized matrix.
    """
    deg = torch.abs(matrix).sum(dim=-1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    normalized_matrix = (
        deg_inv_sqrt.reshape(-1, 1) * matrix * deg_inv_sqrt.reshape(1, -1)
    )
    if return_deg_inv_sqrt:
        return normalized_matrix, deg_inv_sqrt
    else:
        return normalized_matrix


"""Convolution operator that uses the Laplacian as graph shift operator."""

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
