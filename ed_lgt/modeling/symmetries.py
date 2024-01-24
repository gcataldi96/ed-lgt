import numpy as np
from math import prod
from itertools import product

__all__ = ["U_sector_indices", "P_sector_indices"]


def P_sector_indices(loc_dims, op_list, sectors_list):
    # For the moment it works only with same loc_dim
    basis = []
    config_basis = []
    # Precompute the ranges for each dimension
    ranges = [range(dim) for dim in loc_dims]
    # Precompute the diagonals of the operators
    op_diagonals = [np.diag(op) for op in op_list]
    for n, config in enumerate(list(product(*ranges))):
        if all(
            prod(op_diag[c] for c in config) == sector
            for op_diag, sector in zip(op_diagonals, sectors_list)
        ):
            basis.append(n)
            config_basis.append(list(config))
    return basis, config_basis


def U_sector_indices(loc_dims, op_list, sectors_list):
    # For the moment it works only with same loc_dim
    basis = []
    config_basis = []
    # Precompute the ranges for each dimension
    ranges = [range(dim) for dim in loc_dims]
    # Precompute the diagonals of the operators
    op_diagonals = [np.diag(op) for op in op_list]
    for n, config in enumerate(list(product(*ranges))):
        if all(
            sum(op_diag[c] for c in config) == sector
            for op_diag, sector in zip(op_diagonals, sectors_list)
        ):
            basis.append(n)
            config_basis.append(list(config))
    return basis, config_basis
