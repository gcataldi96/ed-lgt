import numpy as np
from numba import njit, prange
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "rowcol_to_index",
    "get_nonzero_indices",
    "precompute_nonzero_indices",
    "arrays_equal",
    "exclude_columns",
    "filter_compatible_rows",
]


@njit(cache=True)
def rowcol_to_index(row, col, loc_dims):
    """
    Compute the global index from row and column indices, considering
    the number of nonzero columns for each row (loc_dims).

    Args:
        row (int): Index of the current row in the valid rows list.
        col (int): Index of the current column within the valid columns for the current row.
        loc_dims (np.ndarray of ints): The number of nonzero columns for each valid row.

    Returns:
        int: The flattened global index corresponding to the (row, col) pair.
    """
    index = 0
    # Compute the cumulative sum of nonzero columns up to the current row
    for ii in range(row):
        index += loc_dims[ii]
    # Add the column index within the current row
    return index + col


@njit
def get_nonzero_indices(arr):
    return np.nonzero(arr)[0]


@njit
def precompute_nonzero_indices(momentum_basis):
    basis_dim = momentum_basis.shape[1]
    nonzero_indices = [
        get_nonzero_indices(momentum_basis[:, i]) for i in range(basis_dim)
    ]
    return nonzero_indices


@njit(cache=True)
def arrays_equal(arr1, arr2):
    if arr1.shape != arr2.shape:
        return False
    for ii in range(arr1.shape[0]):
        if not np.isclose(arr1[ii], arr2[ii], atol=1e-14):
            return False
    return True


@njit(parallel=True, cache=True)
def exclude_columns(data_matrix, exclude_indices):
    """
    Exclude columns from a integer matrix based on a list of indices, parallelized with prange.

    Args:
        data_matrix (np.ndarray): Input 2D array (matrix) from which to exclude columns.
        exclude_indices (list of ints): List of column indices to exclude from the matrix.

    Returns:
        reduced_matrix (np.ndarray): Resulting matrix with specified columns excluded.
    """
    num_rows = data_matrix.shape[0]
    num_cols_remaining = data_matrix.shape[1] - len(exclude_indices)
    reduced_matrix = np.zeros((num_rows, num_cols_remaining), dtype=np.uint8)

    # Parallel loop over rows (configurations)
    for row in prange(num_rows):
        new_col_idx = 0
        for col in range(data_matrix.shape[1]):
            if col not in exclude_indices:
                reduced_matrix[row, new_col_idx] = data_matrix[row, col]
                new_col_idx += 1

    return reduced_matrix


@njit
def filter_compatible_rows(matrix, row):
    # Find indices corresponding to matrix of ints rows equal to row
    check = np.zeros(matrix.shape[0], dtype=np.bool_)
    matching_indices = np.arange(matrix.shape[0], dtype=np.uint16)
    for idx in range(matrix.shape[0]):
        if arrays_equal(matrix[idx], row):
            check[idx] = True
    return matching_indices[check]
