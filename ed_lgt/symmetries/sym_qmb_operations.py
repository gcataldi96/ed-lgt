import numpy as np
from scipy.sparse import csr_matrix
from numba import njit, prange
from ed_lgt.tools import (
    exclude_columns,
    arrays_equal,
    rowcol_to_index,
    precompute_nonzero_indices,
)
import logging

logger = logging.getLogger(__name__)


__all__ = ["nbody_term", "nbody_data", "nbody_data_par", "get_operators_nbody_term"]


def nbody_term(op_list, op_sites_list, sector_configs, momentum_basis=None, k=0):
    if momentum_basis is not None:
        row_list, col_list, value_list = nbody_data_momentum_basis(
            op_list, op_sites_list, sector_configs, momentum_basis
        )
        sector_dim = momentum_basis.shape[1]
    else:
        sector_dim = sector_configs.shape[0]
        row_list, col_list, value_list = nbody_data_par(
            op_list, op_sites_list, sector_configs
        )
    return csr_matrix(
        (value_list, (row_list, col_list)),
        shape=(sector_dim, sector_dim),
    )


@njit
def nbody_data(op_list, op_sites_list, sector_configs):
    """
    Compute the nonzero elements of an nbody-operator.

    Args:
        sector_configs (np.ndarray): Array of sector configurations for lattice sites.
        op_list (np.ndarray): List of operator matrices acting on the lattice sites.
        op_sites_list (list of ints): List of site indices where the operator acts.

    Returns:
        (row_list, col_list, value_list):
            - row_list (np.ndarray of ints): The row indices of nonzero elements.
            - col_list (np.ndarray of ints): The column indices of nonzero elements.
            - value_list (np.ndarray of floats): The nonzero values of the operator elements.
    """
    # Step 1: Initialize the problem dimensions and arrays
    sector_dim = sector_configs.shape[0]
    # Exclude specified sites where the operators act from the list of configs
    m_states = exclude_columns(sector_configs, op_sites_list)
    # Assuming 90% of sparsity we define allow for 10% of nonzero entries
    max_elements = int(0.05 * sector_dim**2)
    row_list = np.zeros(max_elements, dtype=np.int32)
    col_list = np.zeros(max_elements, dtype=np.int32)
    value_list = np.zeros(max_elements, dtype=np.float64)
    count = 0

    for row in range(sector_dim):
        for col in range(sector_dim):
            if arrays_equal(m_states[row], m_states[col]):
                element = 1.0
                for ii, site in enumerate(op_sites_list):
                    op = op_list[ii, site]
                    element *= op[sector_configs[row, site], sector_configs[col, site]]

                if not np.isclose(element, 0, atol=1e-10):
                    row_list[count] = row
                    col_list[count] = col
                    value_list[count] = element
                    count += 1

    # Trim arrays to actual size
    row_list = row_list[:count]
    col_list = col_list[:count]
    value_list = value_list[:count]

    return row_list, col_list, value_list


@njit(parallel=True)
def nbody_data_par(op_list, op_sites_list, sector_configs):
    """
    Compute the nonzero elements of an nbody-operator in a parallelized manner.
    First, it checks the nonzero rows and cols of the operator, and then construct
    the corresponding nonzero operator entries.

    Args:
        sector_configs (np.ndarray): Array of sector configurations for lattice sites.
        op_list (np.ndarray): List of operator matrices acting on the lattice sites.
        op_sites_list (list of ints): List of site indices where the operator acts.

    Returns:
        (row_list, col_list, value_list):
            - row_list (np.ndarray of ints): The row indices of nonzero elements.
            - col_list (np.ndarray of ints): The column indices of nonzero elements.
            - value_list (np.ndarray of floats): The nonzero values of the operator elements.
    """
    # Step 1: Initialize the problem dimensions and arrays
    sector_dim = sector_configs.shape[0]
    # Exclude specified sites where the operators act from the list of configs
    m_states = exclude_columns(sector_configs, op_sites_list)

    # Initialize boolean arrays: check_array tracks valid (row, col) pairs, check_rows tracks valid rows
    check_array = np.zeros((sector_dim, sector_dim), dtype=np.bool_)
    check_rows = np.zeros(sector_dim, dtype=np.bool_)
    # Array with all row/col indices (it will be used to shrink it to the nonzero rows/cols
    sector_dim_array = np.arange(sector_dim, dtype=np.int32)

    # Step 2: Parallelize the row-column equality check to fill check_array and check_rows
    for row in prange(sector_dim):
        m_states_row = m_states[row]
        for col in range(sector_dim):
            # Check if the input (row) and arrival (col) configurations are equal and mark them
            # (it means that the action of the nbody operator which only modify the excluded site configs)
            if arrays_equal(m_states_row, m_states[col]):
                check_array[row, col] = True
                # Mark the row as having at least one nonzero element
                check_rows[row] = True

    # Step 3: Extract valid rows and compute loc_dims (the number of nonzero columns per row)
    # Filter rows with at least one nonzero column
    valid_rows = sector_dim_array[check_rows]
    # Initialize array for storing nonzero columns per row
    loc_dims = np.zeros(len(valid_rows), dtype=np.int32)
    # Loop over valid rows and count the number of nonzero columns for each row (loc_dims)
    for row_idx in prange(len(valid_rows)):
        row = valid_rows[row_idx]
        # Count the nonzero columns in the current row
        loc_dims[row_idx] = np.sum(check_array[row, :])

    # Step 4: Initialize arrays to store the results (row, column indices, and operator values)
    max_elements = np.sum(loc_dims)  # Total number of nonzero (row, col) pairs
    row_list = np.zeros(max_elements, dtype=np.int32)  # To store row indices
    col_list = np.zeros(max_elements, dtype=np.int32)  # To store column indices
    value_list = np.ones(max_elements, dtype=np.float64)  # To store matrix values

    # Step 5: Loop over valid rows and columns, and compute the global index for each (row, col) pair
    for row_idx in prange(len(valid_rows)):
        row = valid_rows[row_idx]
        # Extract valid columns for the current row
        valid_cols = sector_dim_array[check_array[row, :]]

        # Loop over valid columns for this row
        for col_idx in range(loc_dims[row_idx]):
            col = valid_cols[col_idx]
            # Compute the global index using rowcol_to_index
            index = rowcol_to_index(row_idx, col_idx, loc_dims)

            # Step 6: Assign values to row_list, col_list, and value_list at the computed index
            row_list[index] = row
            col_list[index] = col
            for ii, site in enumerate(op_sites_list):
                # Multiply the corresponding operator elements for the (row, col) pair
                op = op_list[ii, site]
                value_list[index] *= op[
                    sector_configs[row, site], sector_configs[col, site]
                ]

    # Return the final lists of nonzero elements: row indices, column indices, and values
    return row_list, col_list, value_list


@njit
def nbody_data_momentum_basis(op_list, op_sites_list, sector_configs, momentum_basis):
    # Step 1: Initialize the problem dimensions and arrays
    basis_dim = momentum_basis.shape[1]
    # Estimated maximum possible non-zero elements based on 90% sparsity
    max_elements = int(0.1 * basis_dim**2)
    row_list = np.zeros(max_elements, dtype=np.int32)
    col_list = np.zeros(max_elements, dtype=np.int32)
    value_list = np.zeros(max_elements, dtype=np.float64)
    count = 0
    # Precompute non-zero indices for each column of the momentum_basis
    nonzero_indices = precompute_nonzero_indices(momentum_basis)
    # Exclude specified sites where the operators act from the list of configs
    m_states = exclude_columns(sector_configs, op_sites_list)

    for row in range(basis_dim):
        nonzero_indices_row = nonzero_indices[row]
        for col in range(basis_dim):
            nonzero_indices_col = nonzero_indices[col]
            element = 0
            for config_ind1 in nonzero_indices_row:
                for config_ind2 in nonzero_indices_col:
                    if arrays_equal(m_states[config_ind1], m_states[config_ind2]):
                        transition_amplitude = 1.0
                        for ii, site in enumerate(op_sites_list):
                            op = op_list[ii, site]
                            transition_amplitude *= op[
                                sector_configs[config_ind1, site],
                                sector_configs[config_ind2, site],
                            ]
                        element += (
                            momentum_basis[config_ind1, row]
                            * transition_amplitude
                            * momentum_basis[config_ind2, col]
                        )
            if not np.isclose(element, 0, atol=1e-10):
                row_list[count] = row
                col_list[count] = col
                value_list[count] = element
                count += 1

    # Trim arrays to actual size
    row_list = row_list[:count]
    col_list = col_list[:count]
    value_list = value_list[:count]

    return row_list, col_list, value_list


def get_operators_nbody_term(op_list, loc_dims, gauge_basis=None, lattice_labels=None):
    def apply_basis_projection(op, basis_label, gauge_basis):
        return (gauge_basis[basis_label].T @ op @ gauge_basis[basis_label]).toarray()

    n_sites = len(loc_dims)
    new_op_list = np.zeros(
        (len(op_list), n_sites, max(loc_dims), max(loc_dims)), dtype=op_list[0].dtype
    )
    for ii, op in enumerate(op_list):
        for jj, loc_dim in enumerate(loc_dims):
            # For Lattice Gauge Theories where sites have different Hilbert Bases
            if gauge_basis is not None:
                # Get the projected operator
                proj_op = apply_basis_projection(op, lattice_labels[jj], gauge_basis)
            # For Theories where all the sites have the same Hilber basis
            else:
                proj_op = op.toarray()
            # Save it inside the new list of operators
            new_op_list[ii, jj, :loc_dim, :loc_dim] = proj_op
    return new_op_list
