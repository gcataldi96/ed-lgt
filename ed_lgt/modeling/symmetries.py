# %%
import numpy as np
from itertools import product
from scipy.sparse import isspmatrix, lil_matrix


def sector_indices(loc_dims, op_list, sectors_list):
    basis = []
    for n, config in enumerate(list(product(*[range(dim) for dim in loc_dims]))):
        include_state = True
        for op, sector_value in zip(op_list, sectors_list):
            # Calculate the value of the operator in the configuration config
            tmp = sum(op[c, c] for c in config)
            # Check if the state belongs to the sector
            if tmp != sector_value:
                include_state = False
                break
        if include_state:
            # Save the configuration state in the basis projector
            basis.append(n)
            print(n, config)
    print(len(basis))
    return basis


def build_H_block(H, loc_dims, site_basis, op_list, sectors_list):
    basis = sector_indices(loc_dims, site_basis, op_list, sectors_list)
    H_sector = H[basis, :][:, basis]
    return H_sector


def get_submatrix_from_sparse(matrix, rows_list, cols_list):
    if not isspmatrix(matrix):
        raise TypeError(f"matrix must be a SPARSE MATRIX, not a {type(matrix)}")
    if not isinstance(rows_list, list):
        raise TypeError(f"rows_list must be a LIST, not a {type(rows_list)}")
    if not isinstance(cols_list, list):
        raise TypeError(f"cols_list must be a LIST, not a {type(cols_list)}")
    matrix = lil_matrix(matrix)
    # Get the Submatrix out of the list of rows and columns
    sub_matrix = matrix[rows_list, :][:, cols_list]
    return sub_matrix


"""
loc_dims = [4, 4, 4, 4]
op1 = np.diag([0, 1, 1, 2])
op2 = np.diag([0, 1, 0, 1])
sectors = [6, 3]
op_list = [op1, op2]
basis = sector_indices(loc_dims, op_list, sectors)
"""
# %%
