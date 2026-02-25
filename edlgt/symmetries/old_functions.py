import numpy as np
from numba import njit, prange
from edlgt.tools import (
    exclude_columns,
    arrays_equal,
    rowcol_to_index,
    precompute_nonzero_indices,
)
from .generate_configs import config_to_index_binarysearch
from .sym_qmb_operations import localbody_data_par
import logging


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

                if np.abs(element) > 1e-10:
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
    batch_size: int = int(2**25),
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
    if len(op_list) == 1:
        # Use a dedicated function for local operators
        return localbody_data_par(op_list[0], op_sites_list[0], sector_configs)
    if sector_dim <= batch_size:
        # Directly use `nbody_data_par` if the sector fits in one batch
        return nbody_data_par_v2(op_list, op_sites_list, sector_configs)

    # Step 1: Allocate conservatively large arrays for final results
    n_sites = sector_configs.shape[1]
    estimated_elements = int(0.005 * sector_dim**2)  # Estimate sparsity
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
    return (
        final_row_list[:offset],
        final_col_list[:offset],
        final_value_list[:offset],
    )


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
                if np.abs(element) > 1e-10:
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
    col_list = np.arange(col_dim, dtype=np.int32)
    for row_idx in prange(len(valid_rows)):
        row = valid_rows[row_idx]
        # Extract valid columns for the current row
        valid_cols = col_list[check_array[row, :]]
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
                if np.abs(element) > 1e-10:
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


@njit(parallel=True, cache=True)
def nbody_data_par_v2(
    op_list: list[np.ndarray], op_sites_list: list[int], sector_configs: np.ndarray
):
    """
    Compute the nonzero elements of an N-body operator in a parallel, two-pass fashion.

    This routine avoids any O(N²) scratch storage by first counting, per row, how many
    nonzero entries will appear; then it allocates exactly the right amount of space
    and in a second parallel pass writes out (row, col, value) triples.

    In a lattice gauge context, the operator only acts on a small subset of “sites”
    in each full-system configuration. We exploit that all other degrees of freedom
    (the complement of ``op_sites_list``) must match exactly between bra and ket.

    :Parameters:
        op_list (List[np.ndarray])
            A list of shape-(*n_sites*,) operator matrices. Each ``op_list[i]``
            is a 2D array of shape *(loc_dim, loc_dim)* giving the diagonal
            action of the *i*th operator on the *i*th site.
        op_sites_list (List[int])
            The subset of site-indices (length *M*) on which the N-body operator
            acts. All other sites must match exactly between bra and ket.
        sector_configs (np.ndarray)
            Shape *(N, n_sites)*, dtype *int*. Each row is a full-system
            configuration in the symmetry sector, with local state-indices for
            all *n_sites*.

    :Returns:
        (row_list, col_list, value_list):
            - row_list (np.ndarray of ints): The row indices of nonzero elements.
            - col_list (np.ndarray of ints): The column indices of nonzero elements.
            - value_list (np.ndarray of floats): The nonzero values of the operator elements.

    :Notes:
        **Two-pass**:

        1. **Pass 1** (parallel over rows): for each row ``ii``, scan all
           ``jj = 0..N-1``, test “match on untouched sites” then form the product
           of the *M* local diagonal entries; if nonzero, increment
           ``nnz_cols_per_row[ii]``.
        2. **Pass 2** (parallel over rows): given an exclusive-prefix sum of
           ``nnz_cols_per_row``, each row knows its unique output-array offset;
           we repeat the same test+multiply loop, but now “fill” the
           pre-allocated slots.

        **Complexity**:
        :math:`O(N^2 \cdot (S + M))` work, where *S* is the check-cost on the
        matched sites and *M* the number of operator multiplications. Both passes
        are fully data-parallel over rows.

        **Memory**:
        Only :math:`O(N)` for the per-row ``nnz_cols_per_row`` plus
        :math:`O(nnz)` for the final three output arrays—no *NxN* boolean temp arrays.

    :Examples:

    ::

        >>> # Suppose 1000 sector-configs, each of 5 sites, and our op acts on sites [1,3]
        >>> rows, cols, vals = nbody_data_par(op_list, [1,3], sector_configs)
    """
    sector_dim = sector_configs.shape[0]
    # 1) Precompute the “reduced” configs where operator does NOT act:
    m_states = exclude_columns(sector_configs, op_sites_list)
    # 2) Count how many nonzero cols each row will contribute
    nnz_cols_per_row = np.zeros(sector_dim, dtype=np.int32)
    for row_idx in prange(sector_dim):
        row_config = sector_configs[row_idx]
        m_row_state = m_states[row_idx]
        count = 0
        for col_idx in range(sector_dim):
            # Check if the input (row) and arrival (col) configurations are equal and mark them
            # (the action of the nbody operator which only modify the excluded site configs)
            if arrays_equal(m_row_state, m_states[col_idx]):
                element = 1.0
                for ii, site in enumerate(op_sites_list):
                    op = op_list[ii, site]
                    element *= op[row_config[site], sector_configs[col_idx, site]]
                if np.abs(element) > 1e-10:
                    count += 1
        nnz_cols_per_row[row_idx] = count

    # 3) Exclusive prefix-sum in place to get each row’s start offset
    total_nnz = 0
    for ii in range(sector_dim):
        tmp = nnz_cols_per_row[ii]
        nnz_cols_per_row[ii] = total_nnz
        total_nnz += tmp

    # 4) Preallocate output arrays exactly the right size
    row_list = np.empty(total_nnz, dtype=np.int32)
    col_list = np.empty(total_nnz, dtype=np.int32)
    value_list = np.empty(total_nnz, dtype=np.float64)

    # 5) Second pass: fill in the results
    for row_idx in prange(sector_dim):
        row_config = sector_configs[row_idx]
        m_row_state = m_states[row_idx]
        # Offset to be set in the row/col/value list index
        start_idx = nnz_cols_per_row[row_idx]
        insert_idx = start_idx
        for col_idx in range(sector_dim):
            if arrays_equal(m_row_state, m_states[col_idx]):
                element = 1.0
                for ii, site in enumerate(op_sites_list):
                    op = op_list[ii, site]
                    element *= op[row_config[site], sector_configs[col_idx, site]]
                if np.abs(element) > 1e-10:
                    row_list[insert_idx] = row_idx
                    col_list[insert_idx] = col_idx
                    value_list[insert_idx] = element
                    insert_idx += 1

    return row_list, col_list, value_list


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
            if np.abs(element) > 1e-10:
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
            if np.abs(element) > 1e-10:
                row_list[count] = row
                col_list[count] = col
                value_list[count] = element
                count += 1

    # Trim arrays to actual size
    row_list = row_list[:count]
    col_list = col_list[:count]
    value_list = value_list[:count]

    return row_list, col_list, value_list
