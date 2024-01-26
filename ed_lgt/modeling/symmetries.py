import numpy as np
from math import prod
from itertools import product

__all__ = ["abelian_sector_indices"]


def abelian_sector_indices(loc_dims, op_list, op_sectors_list, sym_type="U"):
    # For the moment it works only with same loc_dim
    basis = []
    config_basis = []
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
        if check:
            basis.append(n)
            config_basis.append(list(config))
    return np.array(basis, dtype=int), np.array(config_basis, dtype=int)
