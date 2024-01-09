import numpy as np
from math import prod
from scipy.linalg import eigh as array_eigh
from scipy.sparse.linalg import eigsh as sparse_eigh
from scipy.sparse import csr_matrix, isspmatrix
from ed_lgt.tools import zig_zag, validate_parameters

__all__ = [
    "Ground_state",
    "QMB_state",
    "truncation",
    "get_norm",
    "get_loc_states_from_qmb_state",
    "get_qmb_state_from_loc_states",
    "diagonalize_density_matrix",
    "get_projector_for_efficient_density_matrix",
]


class Ground_State:
    def __init__(self, Ham, n_eigs):
        if not isspmatrix(Ham):
            raise TypeError(f"Ham should be a sparse_matrix, not a {type(Ham)}")
        if not np.isscalar(n_eigs) and not isinstance(n_eigs, int):
            raise TypeError(f"n_eigs should be a SCALAR INT, not a {type(n_eigs)}")
        # COMPUTE THE LOWEST n_eigs ENERGY VALUES AND THE 1ST EIGENSTATE
        print("DIAGONALIZE HAMILTONIAN")
        self.Nenergies, self.Npsi = sparse_eigh(Ham, k=n_eigs, which="SA")
        # Save GROUND STATE PROPERTIES
        self.energy = self.Nenergies[0]
        self.psi = self.Npsi[:, 0]


class QMB_state:
    def __init__(self, psi, lvals=None, loc_dims=None):
        """
        Args:
            psi (np.ndarray): QMB states

            lvals (list, optional): list of the lattice spatial dimensions

            loc_dims (list of ints, np.ndarray of ints, or int): list of lattice site dimensions

        Returns:
            sparse: reduced density matrix of the single site
        """
        validate_parameters(psi=psi, lvals=lvals, loc_dims=loc_dims)
        self.psi = psi
        if lvals is not None:
            self.lvals = lvals
        if loc_dims is not None:
            self.loc_dims = loc_dims

    def normalize(self, threshold=1e-14):
        norm = get_norm(self.psi)
        if np.abs(norm - 1) > threshold:
            self.psi /= norm
        return norm

    def truncate(self, threshold=1e-14):
        return truncation(self.psi, threshold)

    def expectation_value(self, operator):
        validate_parameters(op_list=[operator])
        return np.real(np.dot(np.conjugate(self.psi), (operator.dot(self.psi))))

    def reduced_density_matrix(self, qmb_index):
        """
        This function computes the reduced density matrix (in sparse format)
        of a state psi with respect to sigle site in position "qmb_index".

        Args:
            psi (np.ndarray): QMB states

            loc_dims (list of ints, np.ndarray of ints, or int): list of lattice site dimensions

            lvals (list): list of the lattice spatial dimensions

            qmb_index (int): position of the lattice site we want to look at
        Returns:
            sparse: reduced density matrix of the single site
        """
        validate_parameters(index=qmb_index)
        # Get d-dimensional coordinates of
        coords = zig_zag(self.lvals, qmb_index)
        print("----------------------------------------------------")
        print(f"DENSITY MATRIX OF SITE {coords}")
        # RESHAPE psi
        psi_copy = self.psi.reshape(*[loc_dim for loc_dim in self.loc_dims.tolist()])
        # DEFINE A LIST OF SITES WE WANT TO TRACE OUT
        indices = list(np.arange(0, prod(self.lvals)))
        # The index we remove is the one wrt which we get the reduced DM
        indices.remove(qmb_index)
        # Get the reduced density matrix
        rho = np.tensordot(psi_copy, np.conjugate(psi_copy), axes=(indices, indices))
        # Return a truncated sparse version of rho
        return csr_matrix(truncation(rho, 1e-10))

    def entanglement_entropy(self, partition_size):
        """
        This function computes the bipartite entanglement entropy of a portion of a QMB state psi
        related to a lattice model with dimension lvals where single sites have local hilbert spaces of dimensions loc_dims

        Args:
            psi (np.ndarray): QMB states

            loc_dims (list, np.ndarray, or int): dimensions of the single site Hilbert space

            lvals (list): list of the lattice spatial dimensions

            partition_size (int): number of lattice sites to be involved in the partition
        Returns:
            sparse: reduced density matrix of the single site
        """
        if not np.isscalar(partition_size) and not isinstance(partition_size, int):
            raise TypeError(
                f"partition_size must be an SCALAR & INTEGER, not a {type(partition_size)}"
            )
        # COMPUTE THE ENTANGLEMENT ENTROPY OF A SPECIFIC SUBSYSTEM
        partition = 1
        for site in range(partition_size):
            partition *= self.loc_dims[site]
        _, V, _ = np.linalg.svd(
            self.psi.reshape((partition, int(prod(self.loc_dims) / partition)))
        )
        tmp = np.array([-(llambda**2) * np.log2(llambda**2) for llambda in V])
        print(f"ENTROPY: {format(np.sum(tmp), '.9f')}")
        return np.sum(tmp)

    def get_state_configurations(self):
        """
        This function express the main QMB state configurations associated to the
        most relevant coefficients of the QMB state psi. Every state configuration
        is expressed in terms of the single site local Hilber basis
        """
        print("----------------------------------------------------")
        print("STATE CONFIGURATIONS")
        psi = csr_matrix(self.psi.truncate(1e-10))
        sing_vals = sorted(psi.data, key=lambda x: (abs(x), -x), reverse=True)
        indices = [
            x
            for _, x in sorted(
                zip(psi.data, psi.indices),
                key=lambda pair: (abs(pair[0]), -pair[0]),
                reverse=True,
            )
        ]
        state_configurations = {"state_config": [], "coeff": []}
        for ind, alpha in zip(indices, sing_vals):
            loc_states = get_loc_states_from_qmb_state(
                qmb_index=ind, loc_dims=self.loc_dims, lvals=self.lvals
            )
            print(f"{loc_states}  {alpha}")
            state_configurations["state_config"].append(loc_states)
            state_configurations["coeff"].append(alpha)
        print("----------------------------------------------------")
        self.state_configs = state_configurations


def truncation(array, threshold=1e-14):
    if not isinstance(array, np.ndarray):
        raise TypeError(f"array should be an ndarray, not a {type(array)}")
    if not np.isscalar(threshold) and not isinstance(threshold, float):
        raise TypeError(f"threshold should be a SCALAR FLOAT, not a {type(threshold)}")
    return np.where(np.abs(array) > threshold, array, 0)


def get_norm(psi):
    validate_parameters(psi=psi)
    norm = np.linalg.norm(psi)
    return norm


def get_loc_states_from_qmb_state(qmb_index, loc_dims, lvals):
    """
    Compute the state of each single lattice site given the index of the qmb state

    Args:
        qmb_index (int): qmb state index corresponding to a specific list of local sites configurations

        loc_dims (list of ints, np.ndarray of ints, or int): list of lattice site dimensions

        lvals (list): list of the lattice spatial dimensions

    Returns:
        ndarray(int): list of the states of the local Hilbert space associated to the given QMB state index
    """
    validate_parameters(index=qmb_index, lvals=lvals, loc_dims=loc_dims)
    if qmb_index < 0 or qmb_index > (tot_dim - 1):
        raise ValueError(f"index {qmb_index} should be in between 0 and {tot_dim-1}")
    tot_dim = np.prod(loc_dims)
    loc_states = np.zeros(prod(lvals), dtype=int)
    for ii in range(prod(lvals)):
        tot_dim /= loc_dims[ii]
        loc_states[ii] = qmb_index // tot_dim
        qmb_index -= loc_states[ii] * tot_dim
    return loc_states


def get_qmb_state_from_loc_states(loc_states, loc_dims):
    """
    This function generate the QMB index out the the indices of the single lattices sites.
    The latter ones can display local Hilbert space with different dimension.
    The order of the sites must match the order of the dimensionality of the local basis

    Args:
        loc_states (list of ints): list of numbered state of the lattice sites

        loc_dims (list of ints, np.ndarray of ints, or int): list of lattice site dimensions
            (in the same order as they are stored in the loc_states!)

    Returns:
        int: QMB index
    """
    validate_parameters(loc_dims=loc_dims)
    if len(loc_dims) != len(loc_states):
        raise ValueError(
            f"dim loc_states = {len(loc_states)} != dim loc_dims = {len(loc_dims)}"
        )
    n_sites = len(loc_states)
    qmb_index = 0
    dim_factor = 1
    for ii in reversed(range(n_sites)):
        qmb_index += loc_states[ii] * dim_factor
        dim_factor *= loc_dims[ii]
    return qmb_index


def diagonalize_density_matrix(rho):
    # Diagonalize a density matrix which is HERMITIAN COMPLEX MATRIX
    if isinstance(rho, np.ndarray):
        rho_eigvals, rho_eigvecs = array_eigh(rho)
    elif isspmatrix(rho):
        rho_eigvals, rho_eigvecs = array_eigh(rho.toarray())
    return rho_eigvals, rho_eigvecs


def get_projector_for_efficient_density_matrix(rho, loc_dim, threshold):
    """
    This function constructs the projector operator to reduce the single site dimension
    according to the eigenvalues that mostly contributes to the reduced density matrix of the single-site
    """
    if not isinstance(loc_dim, int) and not np.isscalar(loc_dim):
        raise TypeError(f"loc_dim should be INT & SCALAR, not a {type(loc_dim)}")
    if not isinstance(threshold, float) and not np.isscalar(threshold):
        raise TypeError(f"threshold should be FLOAT & SCALAR, not a {type(threshold)}")
    # Diagonalize the single-site density matrix rho
    rho_eigvals, rho_eigvecs = diagonalize_density_matrix(rho)
    # Counts the number of eigenvalues larger than threshold
    P_columns = (rho_eigvals > threshold).sum()
    while P_columns < 2:
        threshold = threshold / 10
        P_columns = (rho_eigvals > threshold).sum()
    print(f"TOTAL NUMBER OF SIGNIFICANT EIGENVALUES {P_columns}")
    column_indx = -1
    # Define the projector operator Proj: it has dimension (loc_dim,P_columns)
    proj = np.zeros((loc_dim, P_columns), dtype=complex)
    # S eigenvalues in <reduced_dm> are stored in increasing order,
    # in order to compute the columns of P_proj we proceed as follows
    for ii in range(loc_dim):
        if rho_eigvals[ii] > threshold:
            column_indx += 1
            proj[:, column_indx] = rho_eigvecs[:, ii]
    # Truncate to 0 the entries below a certain threshold and promote to sparse matrix
    return csr_matrix(truncation(proj, 1e-14))


"""
loc_states = [0, 1, 1]
qmb_state = get_qmb_state_from_loc_state(loc_states, [3, 4, 2])
qmb_index = 5
loc_dim = [3, 4, 2]
n_sites = 3
locs = get_loc_states_from_qmb_state(qmb_index, loc_dim, n_sites)
"""
