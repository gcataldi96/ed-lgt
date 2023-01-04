import numpy as np
from scipy.linalg import eigh as array_eigh
from scipy.sparse import csr_matrix, isspmatrix, isspmatrix_csr, save_npz, lil_matrix
from scipy.sparse.linalg import eigsh as sparse_eigh

from simsio import logger
from tools import pause, zig_zag
from .twobody_term import two_body_op

__all__ = [
    "Ground_State",
    "entanglement_entropy",
    "truncation",
    "get_loc_states_from_qmb_state",
    "get_qmb_state_from_loc_state",
    "get_submatrix_from_sparse",
    "get_state_configurations",
    "get_SU2_topological_invariant",
]


class Ground_State:
    def __init__(self, Ham, n_eigs):
        if not isspmatrix_csr(Ham):
            raise TypeError(f"Ham should be a csr_matrix, not a {type(Ham)}")
        if not np.isscalar(n_eigs) and not isinstance(n_eigs, int):
            raise TypeError(f"n_eigs should be a SCALAR INT, not a {type(n_eigs)}")
        # COMPUTE THE LOWEST n_eigs ENERGY VALUES AND THE 1ST EIGENSTATE
        self.Nenergies, self.Npsi = sparse_eigh(Ham, k=n_eigs, which="SA")
        # Save GROUND STATE PROPERTIES
        self.energy = self.Nenergies[0]
        self.psi = self.Npsi[:, 0]

    def normalize(self):
        norm = np.linalg.norm(self.psi)
        logger.info(f"NORM {norm}")
        self.psi /= norm

    def truncate(self, threshold):
        if not np.isscalar(threshold) and not isinstance(threshold, float):
            raise TypeError(f"threshold must be a SCALAR FLOAT, not {type(threshold)}")
        self.psi = np.where(np.abs(self.psi) > threshold, self.psi, 0)


def get_norm(psi):
    if not isinstance(psi, np.ndarray):
        raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
    norm = np.linalg.norm(psi)
    if np.abs(norm - 1) > 1e-14:
        logger.info(f"NORM {norm}")
        psi = psi / norm
    return psi


def truncation(array, threshold):
    if not isinstance(array, np.ndarray):
        raise TypeError(f"array should be an ndarray, not a {type(array)}")
    if not np.isscalar(threshold) and not isinstance(threshold, float):
        raise TypeError(f"threshold should be a SCALAR FLOAT, not a {type(threshold)}")
    return np.where(np.abs(array) > threshold, array, 0)


def diagonalize_density_matrix(rho):
    # Diagonalize a density matrix which is HERMITIAN COMPLEX MATRIX
    if isinstance(rho, np.ndarray):
        rho_eigvals, rho_eigvecs = array_eigh(rho)
    elif isspmatrix(rho):
        rho_eigvals, rho_eigvecs = array_eigh(rho.toarray())
    return rho_eigvals, rho_eigvecs


def get_reduced_density_matrix(psi, loc_dim, nx, ny, site):
    if not isinstance(psi, np.ndarray):
        raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
    if not np.isscalar(loc_dim) and not isinstance(loc_dim, int):
        raise TypeError(f"loc_dim must be an SCALAR & INTEGER, not a {type(loc_dim)}")
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f"nx must be an SCALAR & INTEGER, not a {type(nx)}")
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f"ny must be an SCALAR & INTEGER, not a {type(ny)}")
    if not np.isscalar(site) and not isinstance(site, int):
        raise TypeError(f"site must be an SCALAR & INTEGER, not a {type(site)}")
    # GET THE TOTAL NUMBER OF SITES
    n_sites = nx * ny
    # GET THE COORDINATES OF THE SITE site
    x, y = zig_zag(nx, ny, site)
    logger.info("----------------------------------------------------")
    logger.info(f"DENSITY MATRIX OF SITE ({str(x+1)},{str(y+1)})")
    logger.info("----------------------------------------------------")
    # RESHAPE psi
    psi_copy = psi.reshape(*[loc_dim for ii in range(n_sites)])
    # DEFINE A LIST OF SITES WE WANT TO TRACE OUT
    indices = np.arange(0, n_sites)
    indices = indices.tolist()
    # The index we remove is the one wrt which we get the reduced DM
    indices.remove(site)
    # COMPUTE THE REDUCED DENSITY MATRIX
    rho = np.tensordot(psi_copy, np.conjugate(psi_copy), axes=(indices, indices))
    # TRUNCATE RHO
    rho = truncation(rho, 10 ** (-10))
    # PROMOTE RHO TO A SPARSE MATRIX
    rho = csr_matrix(rho)
    return rho


def get_projector(rho, loc_dim, threshold, debug, name_save):
    # THIS FUNCTION CONSTRUCTS THE PROJECTOR OPERATOR TO REDUCE
    # THE SINGLE SITE DIMENSION ACCORDING TO THE EIGENVALUES
    # THAT MOSTLY CONTRIBUTES TO THE REDUCED DENSITY MATRIX OF THE SINGLE SITE
    if not isinstance(loc_dim, int) and not np.isscalar(loc_dim):
        raise TypeError(f"loc_dim should be INT & SCALAR, not a {type(loc_dim)}")
    if not isinstance(threshold, float) and not np.isscalar(threshold):
        raise TypeError(f"threshold should be FLOAT & SCALAR, not a {type(threshold)}")
    if not isinstance(debug, bool):
        raise TypeError(f"debug should be a BOOL, not a {type(debug)}")
    if not isinstance(name_save, str):
        raise TypeError(f"name_save should be a STR, not a {type(name_save)}")
    # ------------------------------------------------------------------
    # 1) DIAGONALIZE THE SINGLE SITE DENSITY MATRIX rho
    phrase = "DIAGONALIZE THE SINGLE SITE DENSITY MATRIX"
    pause(phrase, debug)
    rho_eigvals, rho_eigvecs = diagonalize_density_matrix(rho, debug)
    # ------------------------------------------------------------------
    # 2) COMPUTE THE PROJECTOR STARTING FROM THE DIAGONALIZATION of rho
    phrase = "COMPUTING THE PROJECTOR"
    pause(phrase, debug)
    # Counts the number of eigenvalues larger than <threshold>
    P_columns = (rho_eigvals > threshold).sum()
    # PREVENT THE CASE OF A SINGLE RELEVANT EIGENVALUE:
    while P_columns < 2:
        threshold = threshold / 10
        # Counts the number of eigenvalues larger than <threshold>
        P_columns = (rho_eigvals > threshold).sum()
    phrase = "TOTAL NUMBER OF SIGNIFICANT EIGENVALUES " + str(P_columns)
    pause(phrase, debug)

    column_indx = -1
    # Define the projector operator Proj: it has dimension (loc_dim,P_columns)
    Proj = np.zeros((loc_dim, P_columns), dtype=complex)
    # Now, recalling that eigenvalues in <reduced_dm> are stored in increasing order,
    # in order to compute the columns of P_proj we proceed as follows
    for ii in range(loc_dim):
        if rho_eigvals[ii] > threshold:
            column_indx += 1
            Proj[:, column_indx] = rho_eigvecs[:, ii]
    # Truncate to 0 all the entries of Proj that are below a certain threshold
    Proj = truncation(Proj, 10 ** (-14))
    # Promote the Projector Proj to a CSR_MATRIX
    Proj = csr_matrix(Proj)
    logger.info(Proj)
    # Save the Projector on a file
    save_npz(f"{name_save}.npz", Proj)
    return Proj


def projection(Proj, Operator):
    # THIS FUNCTION PERFORMS THE PROJECTION OF Operator
    # WITH THE PROJECTOR Proj: Op' = Proj^{dagger} Op Proj
    # NOTE: both Proj and Operator have to be CSR_MATRIX type!
    if not isspmatrix_csr(Proj):
        raise TypeError(f"Proj should be an CSR_MATRIX, not a {type(Proj)}")
    if not isspmatrix_csr(Operator):
        raise TypeError(f"Operator should be an CSR_MATRIX, not a {type(Operator)}")
    # -----------------------------------------------------------------------
    Proj_dagger = csr_matrix(csr_matrix(Proj).conj().transpose())
    Operator = Operator * Proj
    Operator = Proj_dagger * Operator
    return Operator


def entanglement_entropy(psi, loc_dim, n_sites, partition_size):
    if not isinstance(psi, np.ndarray):
        raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
    if not np.isscalar(loc_dim) and not isinstance(loc_dim, int):
        raise TypeError(f"loc_dim must be an SCALAR & INTEGER, not a {type(loc_dim)}")
    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f"n_sites must be an SCALAR & INTEGER, not a {type(n_sites)}")
    if not np.isscalar(partition_size) and not isinstance(partition_size, int):
        raise TypeError(
            f"partition_size must be an SCALAR & INTEGER, not a {type(partition_size)}"
        )
    # COMPUTE THE ENTANGLEMENT ENTROPY OF A SPECIFIC SUBSYSTEM
    tmp = psi.reshape(
        (loc_dim**partition_size, loc_dim ** (n_sites - partition_size))
    )
    S, V, D = np.linalg.svd(tmp)
    tmp = np.array([-(llambda**2) * np.log2(llambda**2) for llambda in V])
    return np.sum(tmp)


def get_loc_states_from_qmb_state(index, loc_dim, n_sites):
    if not np.isscalar(index) and not isinstance(index, int):
        raise TypeError(f"index must be an SCALAR & INTEGER, not a {type(index)}")
    if not np.isscalar(loc_dim) and not isinstance(loc_dim, int):
        raise TypeError(f"loc_dim must be an SCALAR & INTEGER, not a {type(loc_dim)}")
    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f"n_sites must be an SCALAR & INTEGER, not a {type(n_sites)}")
    """
    Compute the state of each single lattice site given the index of the qmb state
    Args:
        index (int): qmb state index of the a specific local sites configurations 
        loc_dim (int): dimension of the local (single site) Hilbert Space
        n_sites (int): number of sites

    Returns:
        ndarray(int): list of the states of the local Hilbert space 
        associated to the given QMB state index
    """
    if index < 0:
        raise ValueError(f"index {index} should be positive")
    if index > (loc_dim**n_sites - 1):
        raise ValueError(f"index {index} is too high")
    loc_states = np.zeros(n_sites, dtype=int)
    for ii in range(n_sites):
        if ii > 0:
            index = index - loc_states[ii - 1] * (loc_dim ** (n_sites - ii))
        loc_states[ii] = index // (loc_dim ** (n_sites - ii - 1))
    return loc_states


# index=22
# loc_dim=3
# n_sites=3
# locs= get_loc_states_from_qmb_state(index,loc_dim,n_sites)
# print(locs)


def get_qmb_state_from_loc_state(loc_states, loc_dim):
    n_sites = len(loc_states)
    qmb_index = 0
    for ii in range(n_sites):
        qmb_index += loc_states[ii] * (loc_dim ** (n_sites - ii - 1))
    print(qmb_index)
    return qmb_index


# loc_states=[1,1,2]
# qmb_state = get_qmb_state_from_loc_state(loc_states, 3)


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


def get_state_configurations(psi, loc_dim, n_sites):
    if not isinstance(psi, np.ndarray):
        raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
    if not np.isscalar(loc_dim) and not isinstance(loc_dim, int):
        raise TypeError(f"loc_dim must be an SCALAR & INTEGER, not a {type(loc_dim)}")
    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f"n_sites must be an SCALAR & INTEGER, not a {type(n_sites)}")
    logger.info("----------------------------------------------------")
    logger.info(" STATE CONFIGURATIONS")
    psi = truncation(psi, 1e-10)
    sing_vals = sorted(csr_matrix(psi).data, key=lambda x: (abs(x), -x), reverse=True)
    indices = [
        x
        for _, x in sorted(
            zip(csr_matrix(psi).data, csr_matrix(psi).indices),
            key=lambda pair: (abs(pair[0]), -pair[0]),
            reverse=True,
        )
    ]
    for ind, alpha in zip(indices, sing_vals):
        loc_states = get_loc_states_from_qmb_state(
            index=ind, loc_dim=loc_dim, n_sites=n_sites
        )
        logger.info(f" {loc_states+1}  {alpha}")
    logger.info("----------------------------------------------------")


def get_SU2_topological_invariant(link_parity_op, lvals, psi, axis):
    n_sites = lvals[0] * lvals[1]
    op_list = [link_parity_op, link_parity_op]
    if axis == "x":
        op_sites_list = [1, 2]
    elif axis == "y":
        op_sites_list = [1, lvals[0] + 1]
    else:
        raise ValueError(f"axis can be only x or y not {axis}")
    sector = np.real(
        np.dot(np.conjugate(psi), two_body_op(op_list, op_sites_list, n_sites).dot(psi))
    )
    logger.info(f" P{axis} TOPOLOGICAL SECTOR: {sector}")
    logger.info("----------------------------------------------------")
    return sector
