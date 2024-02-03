import numpy as np
from math import prod
from itertools import product
from scipy.sparse import csr_matrix
from ed_lgt.tools import get_time

__all__ = ["abelian_sector_indices", "abelian_sector_indices1", "nbody_sector"]


def apply_basis_projection(op, basis_label, site_basis):
    op = site_basis[basis_label].transpose() * op * site_basis[basis_label]
    return op.toarray()


@get_time
def abelian_sector_indices(loc_dims, op_list, op_sectors_list, sym_type):
    # Precompute the diagonals of the operators
    op_diagonals = [np.diag(op) for op in op_list]
    # Precompute the ranges for each dimension
    ranges = [range(dim) for dim in loc_dims]
    # Create configurations using meshgrid with 'ij' indexing
    configs = np.transpose(np.meshgrid(*ranges, indexing="ij")).reshape(
        -1, len(loc_dims)
    )
    # Vectorize the symmetry sector check
    if sym_type == "U":
        # Sum the diagonal elements for each operator and check against sector values
        checks = np.all(
            [
                np.sum(op_diag[configs], axis=1) == sector
                for op_diag, sector in zip(op_diagonals, op_sectors_list)
            ],
            axis=0,
        )
    elif sym_type == "P":
        # Multiply the diagonal elements for each operator and check against sector values
        checks = np.all(
            [
                np.prod(op_diag[configs], axis=1) == sector
                for op_diag, sector in zip(op_diagonals, op_sectors_list)
            ],
            axis=0,
        )
    else:
        raise ValueError(
            f"For now sym_type can only be P (parity) or U(1), not {sym_type}"
        )
    # Filter configs based on checks
    sector_configs = configs[checks]
    sector_indices = np.ravel_multi_index(sector_configs.T, loc_dims)
    # Use the sorting indices to reorder both sector_indices and sector_configs
    return (
        sector_indices[np.argsort(sector_indices)],
        sector_configs[np.argsort(sector_indices)],
    )


def abelian_sector_indices1(
    loc_dims,
    op_list,
    op_sectors_list,
    sym_type="U",
    site_basis=None,
    lattice_labels=None,
):
    # For the moment it works only with same loc_dim
    sector_indices = []
    sector_basis = []
    # Precompute the ranges for each dimension
    ranges = [range(dim) for dim in loc_dims]
    # Precompute the diagonals of the operators
    if site_basis is not None:
        op_diagonals = []
        for op in op_list:
            single_op_list = []
            for label in lattice_labels:
                single_op_list.append(
                    np.diag(apply_basis_projection(op, label, site_basis))
                )
            op_diagonals.append(single_op_list)
    else:
        op_diagonals = [np.diag(op) for op in op_list]
    for n, config in enumerate(list(product(*ranges))):
        if sym_type == "U":
            check = all(
                sum(op_diag[i][c] for i, c in enumerate(config)) == sector
                for op_diag, sector in zip(op_diagonals, op_sectors_list)
            )
        elif sym_type == "P":
            check = all(
                prod(op_diag[i][c] for i, c in enumerate(config)) == sector
                for op_diag, sector in zip(op_diagonals, op_sectors_list)
            )
        else:
            raise ValueError(
                f"For now sym_type can only be P (parity) or U(1), not {sym_type}"
            )
        if check:
            sector_indices.append(n)
            sector_basis.append(list(config))
    return np.array(sector_indices, dtype=int), np.array(sector_basis, dtype=int)


def nbody_sector(op_list, op_sites_list, sector_basis):
    # Dimension of the symmetry sector
    sector_dim = sector_basis.shape[0]
    # Preallocate arrays with estimated sizes
    row_list = np.zeros(sector_dim, dtype=int)
    col_list = np.zeros(sector_dim, dtype=int)
    value_list = np.zeros(sector_dim, dtype=op_list[0].dtype)
    # Run over pairs of states config
    nnz = 0
    # Remove entries
    m_states = np.delete(sector_basis, op_sites_list, axis=1)
    for row, mstate1 in enumerate(m_states):
        mstate_good = np.equal(m_states, mstate1).all(axis=1)
        idxs = np.nonzero(mstate_good)[0]
        for col in idxs:
            element = prod(
                [
                    op[sector_basis[row, ii], sector_basis[col, ii]]
                    for op, ii in zip(op_list, op_sites_list)
                ]
            )
            if abs(element) != 0:
                row_list[nnz] = row
                col_list[nnz] = col
                value_list[nnz] = element
                nnz += 1
    # Trim the arrays to actual size
    value_list = value_list[:nnz]
    row_list = row_list[:nnz]
    col_list = col_list[:nnz]
    return csr_matrix(
        (value_list, (row_list, col_list)),
        shape=(sector_dim, sector_dim),
    )
