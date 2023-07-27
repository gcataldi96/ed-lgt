# %%
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
    "get_reduced_density_matrix",
    "diagonalize_density_matrix",
    "get_state_configurations",
    "get_SU2_topological_invariant",
    "define_measurements",
]


class Ground_State:
    def __init__(self, Ham, n_eigs):
        if not isspmatrix(Ham):
            raise TypeError(f"Ham should be a sparse_matrix, not a {type(Ham)}")
        if not np.isscalar(n_eigs) and not isinstance(n_eigs, int):
            raise TypeError(f"n_eigs should be a SCALAR INT, not a {type(n_eigs)}")
        # COMPUTE THE LOWEST n_eigs ENERGY VALUES AND THE 1ST EIGENSTATE
        logger.info("DIAGONALIZE HAMILTONIAN")
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


def get_reduced_density_matrix(psi, loc_dims, lvals, site):
    """
    This function computes the reduced density matrix (in sparse format)
    of a state psi with respect to sigle site in position "site".
    Args:
        psi (np.ndarray): QMB states
        loc_dims (list, np.ndarray, or int): dimensions of the single site Hilbert space
        lvals (list): list of the lattice spatial dimensions
        site (int): position of the lattice site we want to look at
    Returns:
        sparse: reduced density matrix of the single site
    """
    if not isinstance(psi, np.ndarray):
        raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
    if isinstance(loc_dims, list):
        loc_dims = np.asarray(loc_dims)
        tot_dim = np.prod(loc_dims)
    elif isinstance(loc_dims, np.ndarray):
        tot_dim = np.prod(loc_dims)
    elif np.isscalar(loc_dims):
        if isinstance(loc_dims, int):
            tot_dim = loc_dims**n_sites
            loc_dims = np.asarray([loc_dims for ii in range(n_sites)])
        else:
            raise TypeError(f"loc_dims must be INTEGER, not a {type(loc_dims)}")
    else:
        raise TypeError(f"loc_dims isn't SCALAR/LIST/ARRAY but {type(loc_dims)}")
    if not isinstance(lvals, list):
        raise TypeError(f"lvals should be a list, not a {type(lvals)}")
    else:
        for ii, ll in enumerate(lvals):
            if not isinstance(ll, int):
                raise TypeError(f"lvals[{ii}] should be INTEGER, not {type(ll)}")
    if not np.isscalar(site) and not isinstance(site, int):
        raise TypeError(f"site must be an SCALAR & INTEGER, not a {type(site)}")
    # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
    nx = lvals[0]
    ny = lvals[1]
    n_sites = nx * ny
    # GET THE COORDINATES OF THE SITE site
    x, y = zig_zag(nx, ny, site)
    logger.info("----------------------------------------------------")
    logger.info(f"DENSITY MATRIX OF SITE ({x},{y})")
    # RESHAPE psi
    psi_copy = psi.reshape(*[loc_dim for loc_dim in loc_dims.tolist()])
    # DEFINE A LIST OF SITES WE WANT TO TRACE OUT
    indices = list(np.arange(0, n_sites))
    # The index we remove is the one wrt which we get the reduced DM
    indices.remove(site)
    # COMPUTE THE REDUCED DENSITY MATRIX
    rho = np.tensordot(psi_copy, np.conjugate(psi_copy), axes=(indices, indices))
    # TRUNCATE RHO
    rho = truncation(rho, 10 ** (-10))
    # PROMOTE RHO TO A SPARSE MATRIX
    return csr_matrix(rho)


def entanglement_entropy(psi, loc_dims, n_sites, partition_size):
    if not isinstance(psi, np.ndarray):
        raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
    if isinstance(loc_dims, list):
        loc_dims = np.asarray(loc_dims)
        tot_dim = np.prod(loc_dims)
    elif isinstance(loc_dims, np.ndarray):
        tot_dim = np.prod(loc_dims)
    elif np.isscalar(loc_dims):
        if isinstance(loc_dims, int):
            tot_dim = loc_dims**n_sites
            loc_dims = np.asarray([loc_dims for ii in range(n_sites)])
        else:
            raise TypeError(f"loc_dims must be INTEGER, not a {type(loc_dims)}")
    else:
        raise TypeError(
            f"loc_dims is neither a SCALAR, a LIST or ARRAY but a {type(loc_dims)}"
        )
    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f"n_sites must be an SCALAR & INTEGER, not a {type(n_sites)}")
    if not np.isscalar(partition_size) and not isinstance(partition_size, int):
        raise TypeError(
            f"partition_size must be an SCALAR & INTEGER, not a {type(partition_size)}"
        )
    # COMPUTE THE ENTANGLEMENT ENTROPY OF A SPECIFIC SUBSYSTEM
    partition = 1
    for site in range(partition_size):
        partition *= loc_dims[site]
    S, V, D = np.linalg.svd(psi.reshape((partition, int(tot_dim / partition))))
    tmp = np.array([-(llambda**2) * np.log2(llambda**2) for llambda in V])
    logger.info(f"ENTROPY: {format(np.sum(tmp), '.9f')}")
    return np.sum(tmp)


def get_loc_states_from_qmb_state(qmb_index, loc_dims, n_sites):
    if not np.isscalar(qmb_index) and not isinstance(qmb_index, int):
        raise TypeError(
            f"qmb_index must be an SCALAR & INTEGER, not a {type(qmb_index)}"
        )
    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f"n_sites must be an SCALAR & INTEGER, not a {type(n_sites)}")
    if isinstance(loc_dims, list):
        loc_dims = np.asarray(loc_dims)
        tot_dim = np.prod(loc_dims)
    elif isinstance(loc_dims, np.ndarray):
        tot_dim = np.prod(loc_dims)
    elif np.isscalar(loc_dims):
        if isinstance(loc_dims, int):
            loc_dims = np.asarray([loc_dims for ii in range(n_sites)])
            tot_dim = loc_dims**n_sites
        else:
            raise TypeError(f"loc_dims must be INTEGER, not a {type(loc_dims)}")
    else:
        raise TypeError(
            f"loc_dims is neither a SCALAR, a LIST or ARRAY but a {type(loc_dims)}"
        )
    """
    Compute the state of each single lattice site given the index of the qmb state
    Args:
        index (int): qmb state index corresponding to a specific list 
        of local sites configurations
        loc_dim (int, list, np.array): dimensions of the local (single site) Hilbert Spaces
        n_sites (int): number of sites

    Returns:
        ndarray(int): list of the states of the local Hilbert space 
        associated to the given QMB state index
    """
    if qmb_index < 0:
        raise ValueError(f"index {qmb_index} should be positive")
    if qmb_index > (tot_dim - 1):
        raise ValueError(f"index {qmb_index} is too high")
    loc_states = np.zeros(n_sites, dtype=int)
    # logger.info(f"TOT_DIM {tot_dim}")
    for ii in range(n_sites):
        # logger.info(f"QMB index {qmb_index}")
        tot_dim /= loc_dims[ii]
        # logger.info(f"TOT_DIM {tot_dim}")
        loc_states[ii] = qmb_index // tot_dim
        # logger.info(f"loc_state {loc_states[ii]}")
        qmb_index -= loc_states[ii] * tot_dim
    # logger.info(loc_states)
    return loc_states


"""
qmb_index = 5
loc_dim = [3, 4, 2]
n_sites = 3
locs = get_loc_states_from_qmb_state(qmb_index, loc_dim, n_sites)
"""


def get_qmb_state_from_loc_state(loc_states, loc_dims):
    """
    This function generate the QMB index out the the indices of the
    single lattices sites. The latter ones can display local Hilbert
    space with different dimension.
    The order of the sites must match the order of the dimensionality
    of the local basis
    Args:
        loc_states (list of ints): list of numbered state of the lattice sites
        loc_dims (list of ints): list of lattice site dimensions
        (in the same order as they are stored in the loc_states!)

    Returns:
        int: QMB index
    """
    if np.isscalar(loc_dims):
        if isinstance(loc_dims, int):
            loc_dims = [loc_dims for ii in range(len(loc_states))]
        else:
            raise TypeError(f"loc_dims must be INTEGER, not a {type(loc_dims)}")
    elif isinstance(loc_dims, list):
        if len(loc_dims) != len(loc_states):
            raise ValueError(
                f"DIMENSION MISMATCH: dim loc_states = {len(loc_states)} != dim loc_dims = {len(loc_dims)}"
            )
    else:
        raise TypeError(
            f"loc_dims is neither a SCALAR, a LIST or ARRAY but a {type(loc_dims)}"
        )
    n_sites = len(loc_states)
    qmb_index = 0
    dim_factor = 1
    for ii in reversed(range(n_sites)):
        qmb_index += loc_states[ii] * dim_factor
        dim_factor *= loc_dims[ii]
    return qmb_index


"""
loc_states = [0, 1, 1]
qmb_state = get_qmb_state_from_loc_state(loc_states, [3, 4, 2])
"""


def get_state_configurations(psi, loc_dims, n_sites):
    """
    This function express the main QMB state configurations associated to the
    most relevant coefficients of the QMB state psi. Every state configuration
    is expressed in terms of the single site local Hilber basis
    Args:
        psi (np.ndarray): QMB state
        loc_dims (list/np.ndarray/int: dimension of every lattice site Hilbert space
        n_sites (int): number of sites in the lattice
    """
    if not isinstance(psi, np.ndarray):
        raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f"n_sites must be an SCALAR & INTEGER, not a {type(n_sites)}")
    if not isinstance(loc_dims, np.ndarray):
        if isinstance(loc_dims, list):
            loc_dims = np.asarray(loc_dims)
        elif np.isscalar(loc_dims):
            if isinstance(loc_dims, int):
                loc_dims = np.asarray([loc_dims for ii in range(n_sites)])
            else:
                raise TypeError(f"loc_dims must be INTEGER, not a {type(loc_dims)}")
        else:
            raise TypeError(
                f"loc_dims is neither a SCALAR, a LIST or ARRAY but a {type(loc_dims)}"
            )
    logger.info("----------------------------------------------------")
    logger.info("STATE CONFIGURATIONS")
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
    state_configurations = {"state_config": [], "coeff": []}
    for ind, alpha in zip(indices, sing_vals):
        loc_states = get_loc_states_from_qmb_state(
            qmb_index=ind, loc_dims=loc_dims, n_sites=n_sites
        )
        logger.info(f"{loc_states}  {alpha}")
        state_configurations["state_config"].append(loc_states)
        state_configurations["coeff"].append(alpha)
    logger.info("----------------------------------------------------")
    return state_configurations


def define_measurements(obs_list, stag_obs_list=None, has_obc=False):
    if not isinstance(obs_list, list):
        raise TypeError(f"obs_list must be a LIST, not a {type(obs_list)}")
    else:
        for obs in obs_list:
            if not isinstance(obs, str):
                raise TypeError(f"obs_list elements are STR, not a {type(obs)}")
    if not isinstance(has_obc, bool):
        raise TypeError(f"has_obc should be a BOOL, not a {type(has_obc)}")
    # ===========================================================================
    # Default observables
    measures = {}
    measures["energy"] = []
    measures["energy_density"] = []
    measures["entropy"] = []
    if not has_obc:
        measures["rho_eigvals"] = []
    else:
        measures["state_configurations"] = []
    # ===========================================================================
    # Observables resulting from Operators
    for obs in obs_list:
        measures[obs] = []
        measures[f"delta_{obs}"] = []
    # Observables resulting from STAGGERED Operators
    if stag_obs_list is not None:
        if not isinstance(stag_obs_list, list):
            raise TypeError(
                f"stag_obs_list must be a LIST, not a {type(stag_obs_list)}"
            )
        else:
            for obs in stag_obs_list:
                if not isinstance(obs, str):
                    raise TypeError(
                        f"stag_obs_list elements are STR, not a {type(obs)}"
                    )
        for site in ["even", "odd"]:
            for obs in stag_obs_list:
                measures[f"{obs}_{site}"] = []
                measures[f"delta_{obs}_{site}"] = []
    return measures


# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================


def get_SU2_topological_invariant(link_parity_op, lvals, psi, axis):
    # NOTE: it works only on a 2x2 system
    op_list = [link_parity_op, link_parity_op]
    if axis == "x":
        op_sites_list = [0, 1]
    elif axis == "y":
        op_sites_list = [0, lvals[0]]
    else:
        raise ValueError(f"axis can be only x or y not {axis}")
    sector = np.real(
        np.dot(
            np.conjugate(psi),
            two_body_op(op_list, op_sites_list, lvals, has_obc=True).dot(psi),
        )
    )
    logger.info(f"P{axis} TOPOLOGICAL SECTOR: {sector}")
    logger.info("----------------------------------------------------")
    return sector


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
