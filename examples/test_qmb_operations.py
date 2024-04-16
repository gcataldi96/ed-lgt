# %%
from ed_lgt.operators import get_Pauli_operators
from ed_lgt.modeling import abelian_sector_indices
from ed_lgt.tools import get_time
from math import prod
import numpy as np
from scipy.sparse import csr_matrix
from numba import njit, int32


@njit
def numba_all_axis1(array):
    return np.take_along_axis(array, np.expand_dims(np.argmin(array, axis=1), 1), 1)[
        :, 0
    ].astype("bool")


@njit
def delete_cols(arr, indices):
    mask = np.zeros(arr.shape[1]) == 0
    for indx in indices:
        mask[indx] = False
    return arr[:, mask]


@njit
def element_operator(operator, op_site, sector_basis):
    # Dimension of the symmetry sector
    sector_dim = sector_basis.shape[0]
    # Preallocate arrays with estimated sizes
    row_list = np.zeros(sector_dim, dtype=int32)
    col_list = np.zeros(sector_dim, dtype=int32)
    value_list = np.zeros(sector_dim, dtype=operator.dtype)
    # Run over pairs of states config
    nnz = 0
    m_states = delete_cols(sector_basis, np.array([op_site], dtype=int32))
    for row, mstate1 in enumerate(m_states):
        mstate_good = numba_all_axis1(np.equal(m_states, mstate1))
        idxs = np.nonzero(mstate_good)[0]
        for col in idxs:
            element = operator[sector_basis[row, op_site], sector_basis[col, op_site]]
            if element != 0:
                row_list[nnz] = row
                col_list[nnz] = col
                value_list[nnz] = element
                nnz += 1
    return value_list[:nnz], row_list[:nnz], col_list[:nnz]


@get_time
def local_sector_numba(operator, op_site, sector_basis):
    sector_dim = sector_basis.shape[0]
    values, rows, cols = element_operator(operator, op_site, sector_basis)
    return csr_matrix(
        (values, (rows, cols)),
        shape=(sector_dim, sector_dim),
    )


@get_time
def nbody_sector(op_list, op_sites_list, sector_basis):
    # Dimension of the symmetry sector
    sector_dim = sector_basis.shape[0]
    # Preallocate arrays with estimated sizes
    row_list = np.zeros(sector_dim, dtype=int)
    col_list = np.zeros(sector_dim, dtype=int)
    value_list = np.zeros(sector_dim, dtype=op_list[0].dtype)
    # Run over pairs of states config
    nnz = 0
    # Remove entries
    m_states = np.delete(sector_basis, op_sites_list, axis=1)
    for row, mstate1 in enumerate(m_states):
        mstate_good = np.equal(m_states, mstate1).all(axis=1)
        idxs = np.nonzero(mstate_good)[0]
        for col in idxs:
            element = prod(
                [
                    op[sector_basis[row, ii], sector_basis[col, ii]]
                    for op, ii in zip(op_list, op_sites_list)
                ]
            )
            if abs(element) != 0:
                row_list[nnz] = row
                col_list[nnz] = col
                value_list[nnz] = element
                nnz += 1
    # Trim the arrays to actual size
    value_list = value_list[:nnz]
    row_list = row_list[:nnz]
    col_list = col_list[:nnz]
    return csr_matrix(
        (value_list, (row_list, col_list)),
        shape=(sector_dim, sector_dim),
    )


from concurrent.futures import ThreadPoolExecutor as pool


@get_time
def nbody_sector_par(op_list, op_sites_list, sector_basis):
    sector_dim = sector_basis.shape[0]
    row_list = np.zeros(sector_dim, dtype=int)
    col_list = np.zeros(sector_dim, dtype=int)
    value_list = np.zeros(sector_dim, dtype=op_list[0].dtype)
    nnz = 0
    m_states = np.delete(sector_basis, op_sites_list, axis=1)

    def process_row(row):
        nonlocal nnz
        mstate1 = m_states[row]
        mstate_good = np.equal(m_states, mstate1).all(axis=1)
        idxs = np.nonzero(mstate_good)[0]
        for col in idxs:
            element = np.prod(
                [
                    op[sector_basis[row, ii], sector_basis[col, ii]]
                    for op, ii in zip(op_list, op_sites_list)
                ]
            )
            if abs(element) != 0:
                row_list[nnz] = row
                col_list[nnz] = col
                value_list[nnz] = element
                nnz += 1

    import os

    with pool(max_workers=os.cpu_count()) as executor:
        executor.map(process_row, range(sector_dim))

    value_list = value_list[:nnz]
    row_list = row_list[:nnz]
    col_list = col_list[:nnz]

    return csr_matrix(
        (value_list, (row_list, col_list)),
        shape=(sector_dim, sector_dim),
    )


# %%
# LATTICE GEOMETRY
lvals = [12]  # number if sites
dim = len(lvals)
directions = "xyz"[:dim]
n_sites = prod(lvals)
has_obc = [False]  # boundary conditions
loc_dims = np.array(
    [2 for i in range(n_sites)]
)  # local dimensions of the lvals lattice sites
ops = get_Pauli_operators()  # acquire spin pauli operators as sparse csr_matrices
# convert them to np.ndarray
for op in ops.keys():
    ops[op] = ops[op].toarray()

sector_indices, sector_basis = abelian_sector_indices(
    loc_dims, [ops["Sz"]], [1], sym_type="P"
)

op_list = [ops["Sx"]]
op_sites_list = [0]
loc1 = nbody_sector(op_list, op_sites_list, sector_basis)
loc1p = nbody_sector_par(op_list, op_sites_list, sector_basis)
# %%
