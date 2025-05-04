"""Laplacian operators for the Magnetic Edge GNN."""

from typing import Literal, overload

import numpy as np
import torch
from torch import Tensor


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
