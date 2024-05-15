import numpy as np
from numba import njit, prange
from ed_lgt.tools import get_time
from scipy.sparse import csr_matrix
import logging

logger = logging.getLogger(__name__)


__all__ = [
    "nbody_operator_data_sitebased",
    "nbody_operator_data",
    "nbody_term",
    "nbody_term_par",
    "get_operators_nbody_term",
]


@njit
def arrays_equal(arr1, arr2):
    if arr1.shape != arr2.shape:
        return False
    for ii in range(arr1.shape[0]):
        if arr1[ii] != arr2[ii]:
            return False
    return True


@njit
def prod(elements):
    result = 1.0
    for el in elements:
        result *= el
    return result


@njit
def exclude_columns(sector_configs, op_sites_list):
    num_configs = sector_configs.shape[0]
    num_sites = sector_configs.shape[1] - len(op_sites_list)
    excluded_configs = np.zeros((num_configs, num_sites), dtype=np.uint8)
    for ii in range(num_configs):
        new_col_idx = 0
        for jj in range(sector_configs.shape[1]):
            if jj not in op_sites_list:
                excluded_configs[ii, new_col_idx] = sector_configs[ii, jj]
                new_col_idx += 1

    return excluded_configs


@njit(parallel=True)
def nbody_operator_data_sitebased(op_list, op_sites_list, sector_configs):
    sector_dim = np.int32(sector_configs.shape[0])
    # Exclude specified sites from sector_configs
    m_states = exclude_columns(sector_configs, op_sites_list)
    nbody_op = np.zeros((sector_dim, sector_dim), dtype=op_list[0].dtype)
    for row in prange(sector_dim):
        mstate1 = m_states[row]
        for col in range(sector_dim):
            if arrays_equal(mstate1, m_states[col]):
                element = 1.0
                for ii, site in enumerate(op_sites_list):
                    # Fetch the operator for the current site
                    op = op_list[ii, site]
                    # Apply the operator
                    element *= op[sector_configs[row, site], sector_configs[col, site]]
                nbody_op[row, col] = element

    return nbody_op


@get_time
def nbody_term_par(op_list, op_sites_list, sector_configs):
    return csr_matrix(
        nbody_operator_data_sitebased(op_list, op_sites_list, sector_configs)
    )


def get_operators_nbody_term(op_list, loc_dims, gauge_basis=None, lattice_labels=None):
    def apply_basis_projection(op, basis_label, gauge_basis):
        return (
            gauge_basis[basis_label].transpose() @ op @ gauge_basis[basis_label]
        ).toarray()

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


@njit
def nbody_operator_data(op_list, op_sites_list, sector_configs):
    sector_dim = sector_configs.shape[0]
    # Exclude specified sites from sector_configs
    m_states = exclude_columns(sector_configs, op_sites_list)
    row_list = []
    col_list = []
    value_list = []

    for row in range(sector_dim):
        mstate1 = m_states[row]
        for col in range(sector_dim):
            if arrays_equal(mstate1, m_states[col]):
                element = 1.0
                for ii, site in enumerate(op_sites_list):
                    # Fetch the operator for the current site
                    op = op_list[ii, site]
                    # Apply the operator
                    element *= op[sector_configs[row, site], sector_configs[col, site]]

                if not np.isclose(element, 0, atol=1e-10):
                    row_list.append(np.int32(row))
                    col_list.append(np.int32(col))
                    value_list.append(element)

    # Trim arrays to actual size
    row_list = np.array(row_list)
    col_list = np.array(col_list)
    value_list = np.array(value_list)

    return row_list, col_list, value_list


@njit
def nbody_operator_data_v2(op_list, op_sites_list, sector_configs):
    sector_dim = sector_configs.shape[0]
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


@njit
def nbody_operator_data_momentum_basis(
    op_list, op_sites_list, sector_configs, momentum_basis
):
    m_states = exclude_columns(sector_configs, op_sites_list)
    basis_dim = momentum_basis.shape[1]
    max_elements = int(
        0.1 * basis_dim**2
    )  # Estimated maximum possible non-zero elements based on 90% sparsity
    row_list = np.zeros(max_elements, dtype=np.int32)
    col_list = np.zeros(max_elements, dtype=np.int32)
    value_list = np.zeros(max_elements, dtype=np.float64)
    count = 0

    # Precompute non-zero indices for each column of the momentum_basis
    nonzero_indices = precompute_nonzero_indices(momentum_basis)

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


@get_time
def nbody_term(op_list, op_sites_list, sector_configs, momentum_basis=None, k=0):
    if momentum_basis is not None:
        row_list, col_list, value_list = nbody_operator_data_momentum_basis(
            op_list, op_sites_list, sector_configs, momentum_basis
        )
        sector_dim = momentum_basis.shape[1]
    else:
        sector_dim = sector_configs.shape[0]
        row_list, col_list, value_list = nbody_operator_data_v2(
            op_list, op_sites_list, sector_configs
        )
    return csr_matrix(
        (value_list, (row_list, col_list)),
        shape=(sector_dim, sector_dim),
    )
