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


@njit(cache=True)
def compare_int_vectors(vec_a: np.ndarray, vec_b: np.ndarray) -> int:
    """
    Lexicographically compare two 1D arrays 'vec_a' and 'vec_b'.

    For each position, the function compares the corresponding elements:
      - Returns -1 if vec_a[i] < vec_b[i] at the first differing index.
      - Returns 1 if vec_a[i] > vec_b[i].
      - Returns 0 if all elements are equal.

    Args:
        vec_a (np.ndarray): 1D array (e.g. np.uint8).
        vec_b (np.ndarray): 1D array (same length and type as vec_a).

    Returns:
        int: -1, 0, or 1, indicating the lexicographical order of vec_a and vec_b.
    """
    for ii in range(vec_a.shape[0]):
        # If you know that the arrays are integers, you can use simple comparison:
        if vec_a[ii] < vec_b[ii]:
            return -1
        elif vec_a[ii] > vec_b[ii]:
            return 1
    return 0


@njit(cache=True)
def find_equal_rows(m_states, target):
    """
    Given a sorted 2D array m_states and a target row, return the indices of all rows
    in m_states that are identical to target.

    The function assumes that m_states is sorted in lexicographical order.
    It first uses binary search to locate one occurrence of the target row,
    then scans backwards and forwards from that location to collect all indices
    where the row equals the target.

    Detailed Steps:
      1. **Binary Search:**
         Set `lo` = 0 and `hi` = number of rows - 1. Then, while `lo <= hi`,
         compute `mid` as the midpoint between `lo` and `hi`. Compare `m_states[mid]` with
         `target` using `compare_rows`:
           - If they are equal (i.e. compare_rows returns 0), save `mid` in `found` and break.
           - If `m_states[mid]` is less than `target`, move the lower bound up (`lo = mid + 1`).
           - Otherwise, move the upper bound down (`hi = mid - 1`).
         If no match is found, return an empty array.

      2. **Backward Scan:**
         Starting from the found index, decrement until you find a row that is no longer equal to target.
         This index is the first occurrence.

      3. **Forward Scan:**
         Starting from the found index, increment until you find a row that is no longer equal to target.
         This index is the last occurrence.

      4. **Return All Indices:**
         Allocate an array of appropriate size and fill it with the indices from the first to the last occurrence.

    Args:
        m_states (np.ndarray): 2D array of shape (num_rows, num_cols) with type np.uint8.
        target (np.ndarray): 1D array of length num_cols (also np.uint8) representing the target row.

    Returns:
        np.ndarray: 1D array of int32 indices where each row of m_states equals target.
                   If no match is found, returns an empty array.
    """
    num_rows = m_states.shape[0]
    lo = 0
    hi = num_rows - 1
    found = -1
    # Binary search for one occurrence of target in m_states
    while lo <= hi:
        mid = (lo + hi) // 2
        cmp_val = compare_int_vectors(m_states[mid], target)
        if cmp_val == 0:
            found = mid
            break
        elif cmp_val < 0:
            lo = mid + 1
        else:
            hi = mid - 1

    if found == -1:
        # No match found; return an empty array.
        return np.empty(0, dtype=np.int32)

    # Scan backwards from the found index to get the first occurrence.
    first_index = found
    while (
        first_index > 0 and compare_int_vectors(m_states[first_index - 1], target) == 0
    ):
        first_index -= 1

    # Scan forwards from the found index to get the last occurrence.
    last_index = found
    while (
        last_index < num_rows - 1
        and compare_int_vectors(m_states[last_index + 1], target) == 0
    ):
        last_index += 1

    size = last_index - first_index + 1
    result = np.empty(size, dtype=np.int32)
    for i in range(size):
        result[i] = first_index + i
    return result


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
def filter_compatible_rows(matrix: np.ndarray, row):
    # Find indices corresponding to matrix of ints rows equal to row
    check = np.zeros(matrix.shape[0], dtype=np.bool_)
    matching_indices = np.arange(matrix.shape[0], dtype=np.uint16)
    for idx in range(matrix.shape[0]):
        if arrays_equal(matrix[idx], row):
            check[idx] = True
    return matching_indices[check]
