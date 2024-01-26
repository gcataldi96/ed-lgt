import numpy as np
from math import prod
from itertools import product

__all__ = ["abelian_sector_indices"]


def abelian_sector_indices(loc_dims, op_list, op_sectors_list, sym_type="U"):
    # For the moment it works only with same loc_dim
    sector_indices = []
    sector_basis = []
    # Precompute the ranges for each dimension
    ranges = [range(dim) for dim in loc_dims]
    # Precompute the diagonals of the operators
    op_diagonals = [np.diag(op) for op in op_list]
    for n, config in enumerate(list(product(*ranges))):
        if sym_type == "U":
            check = all(
                sum(op_diag[c] for c in config) == sector
                for op_diag, sector in zip(op_diagonals, op_sectors_list)
            )
        elif sym_type == "P":
            check = all(
                prod(op_diag[c] for c in config) == sector
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
