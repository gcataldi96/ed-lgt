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


__all__ = [
    "nbody_term",
    "nbody_data",
    "nbody_data_par",
    "get_operators_nbody_term",
    "nbody_data_momentum_basis",
    "nbody_data_momentum_basis_par",
    "process_batches_with_nbody",
]


def nbody_term(
    op_list,
    op_sites_list,
    sector_configs: np.ndarray,
    momentum_basis: np.ndarray = None,
    k=0,
):
    if momentum_basis is not None:
        if k > 0:
            return nbody_data_momentum_basis_finitek(
                op_list, op_sites_list, sector_configs, momentum_basis, k
            )
        else:
            return nbody_data_momentum_basis_par(
                op_list, op_sites_list, sector_configs, momentum_basis
            )
    else:
        r, c, v = process_batches_with_nbody(op_list, op_sites_list, sector_configs)
        return r, c, v


@njit(cache=True)
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


@njit(cache=True)
def process_batches_with_nbody(
    op_list: list[np.ndarray],
    op_sites_list: list[int],
    sector_configs: np.ndarray,
    batch_size: int = int(2**20),
):
    """
    Process nbody_data_par in batches to handle large sector dimensions.

    Args:
        op_list (list[np.ndarray]): List of operator matrices acting on the lattice sites.
        op_sites_list (list[int]): List of site indices where the operator acts.
        sector_configs (np.ndarray): Array of sector configurations for lattice sites.
        batch_size (int): Number of rows to process in each batch.

    Returns:
        (row_list, col_list, value_list):
            - row_list (np.ndarray of ints): The row indices of nonzero elements.
            - col_list (np.ndarray of ints): The column indices of nonzero elements.
            - value_list (np.ndarray of floats): The nonzero values of the operator elements.
    """
    sector_dim = sector_configs.shape[0]
    if len(op_list)==1:
        # Use a dedicated function for local operators
        return localbody_data_par(op_list[0], op_sites_list[0], sector_configs)
    if sector_dim <= batch_size:
        # Directly use `nbody_data_par` if the sector fits in one batch
        return nbody_data_par(op_list, op_sites_list, sector_configs)

    # Step 1: Allocate conservatively large arrays for final results
    n_sites = sector_configs.shape[1]
    sparsity = 5.9 * np.exp(-0.92 * n_sites) / n_sites
    estimated_elements = int(0.0001 * sector_dim**2)  # Estimate sparsity
    final_row_list = np.zeros(estimated_elements, dtype=np.int32)
    final_col_list = np.zeros(estimated_elements, dtype=np.int32)
    final_value_list = np.zeros(estimated_elements, dtype=np.float64)
    # Determine the number of batches
    num_batches = (sector_dim + batch_size - 1) // batch_size
    offset = 0
    for row_batch_idx in range(num_batches):
        # Define batch start and end indices
        start_row = row_batch_idx * batch_size
        end_row = min((row_batch_idx + 1) * batch_size, sector_dim)
        # Extract the batch from sector_configs
        batch_row_configs = sector_configs[start_row:end_row]

        for col_batch_idx in range(num_batches):
            # Define column batch start and end indices
            start_col = col_batch_idx * batch_size
            end_col = min((col_batch_idx + 1) * batch_size, sector_dim)
            batch_col_configs = sector_configs[start_col:end_col]
            # Process the batch with nbody_data_par
            row_list, col_list, value_list = nbody_data_batch(
                op_list,
                op_sites_list,
                batch_row_configs,
                batch_col_configs,
            )
            # Adjust row/col indices to the global sector_configs
            row_list += start_row
            col_list += start_col
            # Evaluate the number of nonzero elements in the batch
            nnzero_elems = len(row_list)
            # Append the batch results into the final arrays
            final_row_list[offset : offset + nnzero_elems] = row_list
            final_col_list[offset : offset + nnzero_elems] = col_list
            final_value_list[offset : offset + nnzero_elems] = value_list
            # Update the offset for the next batch
            offset += nnzero_elems
    # Step 3: Truncate final arrays to the actual used size
    return final_row_list[:offset], final_col_list[:offset], final_value_list[:offset]


@njit(parallel=True, cache=True)
def nbody_data_batch(
    op_list: list[np.ndarray],
    op_sites_list: list[int],
    batch_row_configs: np.ndarray,
    batch_col_configs: np.ndarray,
):
    # Step 1: Initialize the problem dimensions and arrays
    row_dim = batch_row_configs.shape[0]
    col_dim = batch_col_configs.shape[0]
    # Exclude specified sites where the operators act from the list of configs
    ex_row_configs = exclude_columns(batch_row_configs, op_sites_list)
    ex_col_configs = exclude_columns(batch_col_configs, op_sites_list)
    # Initialize boolean arrays: check_array tracks valid (row, col) pairs
    check_array = np.zeros((row_dim, col_dim), dtype=np.bool_)
    # check_rows tracks valid rows (rows with at least one nonzero element)
    check_rows = np.zeros(row_dim, dtype=np.bool_)

    # Step 2: Parallelize the row-column equality check to fill check_array and check_rows
    for row in prange(row_dim):
        # Access the excluded site configurations for the current row
        row_state = ex_row_configs[row]
        for col in range(col_dim):
            # Check if the input (row) and arrival (col) configurations are equal
            # (the only way to have a nonzero element)
            if arrays_equal(row_state, ex_col_configs[col]):
                # Measure the nbody operator element for the (row, col) pair
                element = 1.0
                for ii, site in enumerate(op_sites_list):
                    op = op_list[ii, site]
                    element *= op[
                        batch_row_configs[row, site], batch_col_configs[col, site]
                    ]
                # Check that the element is nonzero
                if not np.isclose(element, 0, atol=1e-10):
                    check_array[row, col] = True
                    # Mark the row as having at least one nonzero element
                    check_rows[row] = True

    # Step 3: Extract valid rows and compute ncols_per_row (the number of nonzero columns per row)
    # Extract rows with at least one nonzero column
    valid_rows = np.arange(row_dim, dtype=np.int32)[check_rows]
    # Initialize array for storing the number of nonzero columns on each valid row
    ncols_per_row = np.zeros(len(valid_rows), dtype=np.int32)
    # Loop over valid rows and count the number of nonzero columns for each row (ncols_per_row)
    for row_idx in prange(len(valid_rows)):
        row = valid_rows[row_idx]
        # Count the nonzero columns in the current row
        ncols_per_row[row_idx] = np.sum(check_array[row, :])

    # Step 4: Initialize arrays to store the results (row, column indices, and operator values)
    max_elements = np.sum(ncols_per_row)  # Total number of nonzero (row, col) pairs
    row_list = np.zeros(max_elements, dtype=np.int32)  # To store row indices
    col_list = np.zeros(max_elements, dtype=np.int32)  # To store column indices
    value_list = np.ones(max_elements, dtype=np.float64)  # To store matrix values

    # Step 5: Loop over valid rows and columns and compute the global value for each (row, col) pair
    # Create a vector from which to extract the valid columns
    col_array = np.arange(col_dim, dtype=np.int32)
    for row_idx in prange(len(valid_rows)):
        row = valid_rows[row_idx]
        # Extract valid columns for the current row
        valid_cols = col_array[check_array[row, :]]
        # Loop over valid columns for this row
        for col_idx in range(ncols_per_row[row_idx]):
            col = valid_cols[col_idx]
            # Compute the global index using rowcol_to_index
            index = rowcol_to_index(row_idx, col_idx, ncols_per_row)

            # Step 6: Assign values to row_list, col_list, and value_list at the computed index
            row_list[index] = row
            col_list[index] = col
            for ii, site in enumerate(op_sites_list):
                # Multiply the corresponding operator elements for the (row, col) pair
                op = op_list[ii, site]
                value_list[index] *= op[
                    batch_row_configs[row, site], batch_col_configs[col, site]
                ]
    # Return the final lists of nonzero elements: row indices, column indices, and values
    return row_list, col_list, value_list


@njit(parallel=True, cache=True)
def nbody_data_par(
    op_list: list[np.ndarray], op_sites_list: list[int], sector_configs: np.ndarray
):
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
    # Array with all row/col indices (it will be used to shrink it to the nonzero rows/cols)
    sector_dim_array = np.arange(sector_dim, dtype=np.int32)

    # Step 2: Parallelize the row-column equality check to fill check_array and check_rows
    for row in prange(sector_dim):
        m_states_row = m_states[row]
        for col in range(sector_dim):
            # Check if the input (row) and arrival (col) configurations are equal and mark them
            # (it means that the action of the nbody operator which only modify the excluded site configs)
            if arrays_equal(m_states_row, m_states[col]):
                element = 1.0
                for ii, site in enumerate(op_sites_list):
                    op = op_list[ii, site]
                    element *= op[sector_configs[row, site], sector_configs[col, site]]
                # Check that the element is nonzero
                if not np.isclose(element, 0, atol=1e-10):
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

@njit(parallel=True,cache=True)
def localbody_data_par(op: np.ndarray, op_site: int, sector_configs: np.ndarray):
    """
    Efficiently process a diagonal operator on a given sector of configurations.

    Args:
        op (np.ndarray): A single-site diagonal operator matrix.
        op_sites_list (int): site index where the operator acts.
        sector_configs (np.ndarray): Array of sector configurations for lattice sites.

    Returns:
        (row_list, col_list, value_list):
            - row_list (np.ndarray of ints): The row indices of diagonal elements.
            - col_list (np.ndarray of ints): Same as row_list (since diagonal).
            - value_list (np.ndarray of floats): The diagonal elements of the operator.
    """
    sector_dim = sector_configs.shape[0]
    # Initialize row_list and col_list as the diagonal indices
    row_list = np.arange(sector_dim, dtype=np.int32)
    check_rows = np.zeros(sector_dim, dtype=np.bool_)
    value_list = np.zeros(sector_dim, dtype=np.float64)
    # Isolate the action of the operator on the site
    op_diag = op[op_site]
    for row in prange(len(row_list)):
        value_list[row] = op_diag[sector_configs[row, op_site], sector_configs[row, op_site]]
        # Check that the element is nonzero
        if not np.isclose(value_list[row], 0, atol=1e-10):
            # Mark the row as having at least one nonzero element
            check_rows[row] = True
    # Filter out zero elements
    row_list = row_list[check_rows]
    value_list = value_list[check_rows]
    return row_list, row_list, value_list

@njit(cache=True)
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
                m_states_row = m_states[config_ind1]
                for config_ind2 in nonzero_indices_col:
                    if arrays_equal(m_states_row, m_states[config_ind2]):
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


@njit(parallel=True, cache=True)
def nbody_data_momentum_basis_par(
    op_list, op_sites_list, sector_configs, momentum_basis
):
    # Step 1: Initialize the problem dimensions and arrays
    basis_dim = momentum_basis.shape[1]
    # Precompute non-zero indices for each column of the momentum_basis
    nonzero_indices = precompute_nonzero_indices(momentum_basis)
    # Exclude specified sites where the operators act from the list of configs
    m_states = exclude_columns(sector_configs, op_sites_list)
    # Initialize boolean arrays: check_array tracks valid (row, col) pairs, check_rows tracks valid rows
    check_array = np.zeros((basis_dim, basis_dim), dtype=np.bool_)
    check_rows = np.zeros(basis_dim, dtype=np.bool_)
    # Array with all row/col indices (it will be used to shrink it to the nonzero rows/cols)
    sector_dim_array = np.arange(basis_dim, dtype=np.int32)

    # Step 2: Parallelize the row-column equality check to fill check_array and check_rows
    for row in prange(basis_dim):
        nonzero_indices_row = nonzero_indices[row]
        for col in range(basis_dim):
            nonzero_indices_col = nonzero_indices[col]
            found_match = False
            for config_ind1 in nonzero_indices_row:
                if found_match:  # Exit early if match is already found
                    break
                m_states_row = m_states[config_ind1]
                for config_ind2 in nonzero_indices_col:
                    # Check if the input (row) and arrival (col) configurations are equal and mark them
                    # (it means that the action of the nbody operator which only modify the excluded site configs)
                    if arrays_equal(m_states_row, m_states[config_ind2]):
                        check_array[row, col] = True
                        # Mark the row as having at least one nonzero element
                        check_rows[row] = True
                        # Set flag to exit outer loops
                        found_match = True
                        break  # Breaks out of `config_ind2` loop

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
    value_list = np.zeros(max_elements, dtype=np.float64)  # To store matrix values

    # Step 5: Loop over valid rows and columns, and compute the global index for each (row, col) pair
    for row_idx in prange(len(valid_rows)):
        row = valid_rows[row_idx]
        nonzero_indices_row = nonzero_indices[row]
        # Extract valid columns for the current row
        valid_cols = sector_dim_array[check_array[row, :]]
        # Loop over valid columns for this row
        for col_idx in range(loc_dims[row_idx]):
            col = valid_cols[col_idx]
            nonzero_indices_col = nonzero_indices[col]
            # Compute the global index using rowcol_to_index
            index = rowcol_to_index(row_idx, col_idx, loc_dims)
            # Step 6A: Assign values to row_list, col_list at the computed index
            row_list[index] = row
            col_list[index] = col
            # Step 6B: Assign values to value_list at the computed index
            for config_ind1 in nonzero_indices_row:
                m_states_row = m_states[config_ind1]
                for config_ind2 in nonzero_indices_col:
                    if arrays_equal(m_states_row, m_states[config_ind2]):
                        transition_amplitude = 1.0
                        for ii, site in enumerate(op_sites_list):
                            op = op_list[ii, site]
                            transition_amplitude *= op[
                                sector_configs[config_ind1, site],
                                sector_configs[config_ind2, site],
                            ]
                        value_list[index] += (
                            momentum_basis[config_ind1, row]
                            * transition_amplitude
                            * momentum_basis[config_ind2, col]
                        )
    return (
        row_list[np.nonzero(value_list)],
        col_list[np.nonzero(value_list)],
        value_list[np.nonzero(value_list)],
    )


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


@njit(cache=True)
def nbody_data_momentum_basis_finitek(
    op_list, op_sites_list, sector_configs, momentum_basis, k
):
    # Step 1: Initialize the problem dimensions and arrays
    basis_dim = momentum_basis.shape[1]
    system_dim = sector_configs.shape[0]
    # Estimated maximum possible non-zero elements based on 90% sparsity
    max_elements = int(0.1 * basis_dim**2)
    row_list = np.zeros(max_elements, dtype=np.int32)
    col_list = np.zeros(max_elements, dtype=np.int32)
    value_list = np.zeros(max_elements, dtype=np.complex128)
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
                            # Calculate phase difference due to position 'site' in the momentum basis
                            phase_diff = np.exp(-1j * 2 * np.pi * k * site / system_dim)
                            # Measure the transition Amplitude
                            op = op_list[ii, site]
                            transition_amplitude *= op[
                                sector_configs[config_ind1, site],
                                sector_configs[config_ind2, site],
                            ]
                            transition_amplitude *= phase_diff
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
