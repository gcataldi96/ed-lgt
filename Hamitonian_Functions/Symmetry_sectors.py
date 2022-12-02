import numpy as np
from scipy.sparse import isspmatrix
from scipy.sparse import lil_matrix


class U1_Symmetry:
    def __init__(self, Operator, Operator_name):
        return 10


# ===========================================================================
def single_site_symmetry_sectors(loc_dim, sectors_list, dim_sectors_list):
    # CHECK ON TYPES
    if not np.isscalar(loc_dim) and not isinstance(loc_dim, int):
        raise TypeError(f"loc_dim must be a SCALAR & INTEGER, not a {type(loc_dim)}")
    if not isinstance(sectors_list, list):
        raise TypeError(f"sectors_list must be a LIST, not a {type(sectors_list)}")
    if not isinstance(dim_sectors_list, list):
        raise TypeError(
            f"dim_sectors_list must be a LIST, not a {type(dim_sectors_list)}"
        )
    # Generate a single-site vector
    single_site_vec = np.zeros(loc_dim, dtype=float)
    # TMP variable
    tmp = 0
    # fill the entries of single_site_vec with the values of the charge for each state of the local_basis
    for jj, dim in enumerate(dim_sectors_list):
        for ii in range(dim):
            single_site_vec[tmp] = int(sectors_list[jj])
            tmp += 1
    return single_site_vec


# ===========================================================================
def QMB_local_state(single_site_sym_sector_state, state_site, n_sites):
    # CHECK ON TYPES
    if not np.isscalar(state_site) and not isinstance(state_site, int):
        raise TypeError(
            f"state_site must be a SCALAR & INTEGER, not a {type(state_site)}"
        )
    if not isinstance(single_site_sym_sector_state, np.ndarray):
        raise TypeError(
            f"single_site_sym_sector_state must be a ndarray, not a {type(single_site_sym_sector_state)}"
        )
    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f"n_sites must be a SCALAR & INTEGER, not a {type(n_sites)}")
    # Make a copy of the single site symm sector state
    tmp = single_site_sym_sector_state
    # Make the tensor products with Identities
    ID = np.ones(single_site_sym_sector_state.shape[0])
    for ii in range(state_site):
        tmp = np.kron(ID, tmp)
    for ii in range(n_sites - state_site - 1):
        tmp = np.kron(tmp, ID)
    # print(tmp)
    return tmp


# ===========================================================================
def many_body_symmetry_sectors(single_site_state, n_sites):
    # CHECK ON TYPES
    if not isinstance(single_site_state, np.ndarray):
        raise TypeError(
            f"single_site_state must be a ndarray, not a {type(single_site_state)}"
        )
    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f"n_sites must be a SCALAR & INTEGER, not a {type(n_sites)}")
    QMB_sym_sec_state = np.zeros(single_site_state.shape[0] ** n_sites)
    for ii in range(n_sites):
        QMB_sym_sec_state += QMB_local_state(single_site_state, ii, n_sites)

    return QMB_sym_sec_state


# ===========================================================================
def get_submatrix_from_sparse(matrix, rows_list, cols_list):
    # CHECK ON TYPES
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


# ===========================================================================
def get_indices_from_array(array, value):
    if not isinstance(array, np.ndarray):
        raise TypeError(f"array must be a ndarray, not a {type(array)}")
    index_list = np.nonzero(
        (value - 10 ** (-10) < array) & (array < value + 10 ** (-10))
    )[0]
    return index_list


# GET THE SINGLE SITE SYMMETRY SECTORS WRT THE NUMBER OF FERMIONS
N_tot_sectors = [0, 1, 2]
N_tot_dim_sectors = [9, 12, 9]
single_site_syms = single_site_symmetry_sectors(
    loc_dim, sectors_list=N_tot_sectors, dim_sectors_list=N_tot_dim_sectors
)
# GET THE NBODY SYMMETRY SECTOR STATE
Nbody_syms = many_body_symmetry_sectors(single_site_syms, n) - 4


H_subsector = {}
for ii in range(-4, 5, 2):
    print(ii)
    # GET THE INDICES ASSOCIATED TO EACH SYMMETRY SECTOR
    indices = get_indices_from_array(Nbody_syms, ii)
    indices = indices.tolist()
    print("Computing H subsector of ", ii)
    H_subsector[str(ii)] = get_submatrix_from_sparse(H, indices, indices)
    sub_energy, sub_psi = get_ground_state_from_Hamiltonian(
        csr_matrix(H_subsector[str(ii)]), debug=False
    )
