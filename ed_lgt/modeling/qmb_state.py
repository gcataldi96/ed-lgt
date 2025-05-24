import numpy as np
from numba import njit, prange
from math import prod
from scipy.linalg import eigh as array_eigh, svd
from scipy.sparse import isspmatrix, csr_array
from ed_lgt.tools import validate_parameters
from ed_lgt.symmetries import index_to_config, config_to_index_linsearch
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "QMB_state",
    "truncation",
    "get_norm",
    "diagonalize_density_matrix",
    "get_projector_for_efficient_density_matrix",
]


class QMB_state:
    def __init__(self, psi: np.ndarray, lvals=None, loc_dims=None):
        """
        Args:
            psi (np.ndarray): QMB states

            lvals (list, optional): list of the lattice spatial dimensions

            loc_dims (list of ints, np.ndarray of ints, or int): list of lattice site dimensions
        """
        validate_parameters(psi=psi, lvals=lvals, loc_dims=loc_dims)
        self.psi = psi
        self.lvals = lvals
        self.loc_dims = loc_dims

    def normalize(self, threshold=1e-14):
        """
        Normalizes the quantum state vector to unit norm, if it is not already.
        If the norm is off by more than the specified threshold, the state vector is scaled down.

        Args:
            threshold (float, optional): The tolerance level for the norm check. Defaults to 1e-14.

        Returns:
            float: The norm of the state before normalization.
        """
        norm = get_norm(self.psi)
        if np.abs(norm - 1) > threshold:
            self.psi /= norm
        return norm

    def truncate(self, threshold=1e-14):
        """
        Truncates small components of the state vector based on a threshold.

        Args:
            threshold (float, optional): Components smaller than this value are set to zero. Defaults to 1e-14.

        Returns:
            np.ndarray: The truncated state vector.
        """
        return truncation(self.psi, threshold)

    def expectation_value(self, operator):
        """
        Calculates the expectation value of the given operator with the current quantum state.
        The operator can be provided in one of the following formats:
            - As a tuple of nonzero elements: (row_list, col_list, value_list).
            - As a dense matrix (np.ndarray).
            - As a sparse matrix (e.g., scipy.sparse.csc_matrix or csr_matrix).

        Args:
            operator (tuple or np.ndarray or sparse_matrix): The operator to apply.
                - If a tuple, it should contain (row_list, col_list, value_list), where:
                    * row_list (np.ndarray): Row indices of nonzero elements.
                    * col_list (np.ndarray): Column indices of nonzero elements.
                    * value_list (np.ndarray): Values corresponding to (row, col) pairs.
                - If a dense matrix, it should be a NumPy array (np.ndarray).
                - If a sparse matrix, it should be a scipy sparse matrix.

        Returns:
            float: The real part of the expectation value.
        """
        if isinstance(operator, tuple) and len(operator) == 3:
            # Case 1: Operator as (row_list, col_list, value_list)
            row_list, col_list, value_list = operator
            return exp_val_data(self.psi, row_list, col_list, value_list)
        elif isinstance(operator, np.ndarray) or isspmatrix(operator):
            # Case 2: Dense or Sparse Matrix
            if self.psi.shape[0] != operator.shape[0]:
                msg = f"The dimensions of the quantum state {self.psi.shape[0]} and the operator {operator.shape[0]} do not match."
                raise ValueError(msg)
            if isinstance(operator, np.ndarray):
                operator = csr_array(operator)
            validate_parameters(op_list=[operator])
            return np.real(np.dot(np.conjugate(self.psi), (operator.dot(self.psi))))
        else:
            raise TypeError(
                "Operator must be provided as a tuple (row_list, col_list, value_list), "
                "a dense matrix (np.ndarray), or a sparse matrix (scipy sparse matrix)."
            )

    def bipartite_psi(self, keep_indices):
        """
        Reshape and reorder psi for partitioning based on keep_indices.

        Args:
            keep_indices (list of ints): Indices of the lattice sites to keep.

        Returns:
            tuple: The reshaped and reordered psi tensor, subsystem dimension, and environment dimension.
        """
        # Ensure psi is reshaped into a tensor with one leg per lattice site
        psi_tensor = self.psi.reshape(self.loc_dims)
        # Determine the environmental indices
        all_indices = list(range(prod(self.lvals)))
        env_indices = [i for i in all_indices if i not in keep_indices]
        new_order = keep_indices + env_indices
        # Rearrange the tensor to group subsystem and environment indices
        psi_tensor = np.transpose(psi_tensor, axes=new_order)
        logger.info(f"Reordered psi_tensor shape: {psi_tensor.shape}")
        # Determine the dimensions of the subsystem and environment for the bipartition
        subsystem_dim = np.prod([self.loc_dims[i] for i in keep_indices])
        env_dim = np.prod([self.loc_dims[i] for i in env_indices])
        # Reshape the reordered tensor to separate subsystem from environment
        psi_partitioned = psi_tensor.reshape((subsystem_dim, env_dim))
        return psi_partitioned, subsystem_dim, env_dim

    def reduced_density_matrix(
        self,
        keep_indices,
        subsystem_configs=None,
        env_configs=None,
        unique_subsys_configs=None,
        unique_env_configs=None,
    ):
        """
        Computes the reduced density matrix of the quantum state for specified lattice sites.
        Optionally handles different symmetry sectors.

        Args:
            keep_indices (list of ints): list of lattice indices to be involved in the partition
            subsystem_configs (np.ndarray): Configurations of the subsystem in the symmetry sector.
            environment_configs (np.ndarray): Configurations of the environment in the symmetry sector.
            unique_subsys_configs (np.ndarray): Unique configurations of the subsystem.
            unique_env_configs (np.ndarray): Unique configurations of the environment.

        Returns:
            np.ndarray: The reduced density matrix in dense format.
        """
        logger.info("----------------------------------------------------")
        logger.info(f"RED. DENSITY MATRIX OF SITES {keep_indices}")
        # CASE OF SYMMETRY SECTOR
        if subsystem_configs is not None:
            # Compute the psi_matrix
            psi_matrix = build_psi_matrix(
                self.psi,
                subsystem_configs,
                env_configs,
                unique_subsys_configs,
                unique_env_configs,
            )
            return psi_matrix.conj().T @ psi_matrix
        else:
            # NO SYMMETRIES
            # Prepare psi tensor sorting subsystem indices close each other to bipartite the system
            psi_tensor, subsystem_dim, _ = self.bipartite_psi(keep_indices)
            # Compute the reduced density matrix by tracing out the env-indices
            RDM = np.tensordot(psi_tensor, np.conjugate(psi_tensor), axes=([1], [1]))
            # Reshape rho to ensure it is a square matrix corresponding to the subsystem
            RDM = RDM.reshape((subsystem_dim, subsystem_dim))
            return RDM

    def entanglement_entropy(
        self,
        keep_indices,
        subsystem_configs=None,
        env_configs=None,
        unique_subsys_configs=None,
        unique_env_configs=None,
    ):
        """
        This function computes the bipartite entanglement entropy of a portion of a QMB state psi
        related to a lattice model with dimension lvals where single sites
        have local hilbert spaces of dimensions loc_dims

        Args:
            keep_indices (list of ints): list of lattice indices to be involved in the partition
            subsystem_configs (np.ndarray): Configurations of the subsystem in the symmetry sector.
            environment_configs (np.ndarray): Configurations of the environment in the symmetry sector.
            unique_subsys_configs (np.ndarray): Unique configurations of the subsystem.
            unique_env_configs (np.ndarray): Unique configurations of the environment.

        Returns:
            float: bipartite entanglement entropy of the lattice subsystem
        """
        if subsystem_configs is not None:
            # Compute the psi_matrix
            psi_matrix = build_psi_matrix(
                self.psi,
                subsystem_configs,
                env_configs,
                unique_subsys_configs,
                unique_env_configs,
            )
        else:
            # Prepare psi tensor sorting subsystem indices close each other to bipartite the system
            psi_matrix, _, _ = self.bipartite_psi(keep_indices)
        # Compute SVD
        _, V, _ = svd(psi_matrix, full_matrices=False)
        llambdas = V**2
        llambdas = llambdas[llambdas > 1e-10]
        entropy = -np.sum(llambdas * np.log2(llambdas))
        logger.info(f"ENTROPY of {keep_indices}: {format(entropy, '.9f')}")
        return entropy

    def get_state_configurations(self, threshold=1e-2, sector_configs=None):
        """
        List out |psi> configurations whose amplitudes exceed `threshold`.

        If `sector_configs` is provided, it must be an (MxL) array of all
        symmetry-sector configurations, and we simply look up rows from it.
        Otherwise we reconstruct each configuration via `index_to_config`.

        Args:
            threshold (float): minimum absolute amplitude to keep
            sector_configs (ndarray or None): if present, shape (M,L)
                mapping each state-index → its L-site configuration.

        Prints to the logger each surviving configuration in order of descending |amplitude|.
        """
        logger.info("----------------------------------------------------")
        logger.info("STATE CONFIGURATIONS")
        # 1) mask off small amplitudes
        psi = self.psi
        mask = np.abs(psi) > threshold
        # 2) collect indices & values
        idx = np.nonzero(mask)[0]  # shape (K,)
        vals = psi[idx]  # shape (K,)
        logger.info(f"{len(idx)} configurations above threshold {threshold}")
        if idx.size == 0:
            logger.info("[no configurations above threshold]")
            return
        # 3) sort by descending absolute value
        order = np.argsort(-np.abs(vals))
        idx = idx[order]
        vals = vals[order]
        # 4) pull out configs
        if sector_configs is not None:
            cfgs = sector_configs[idx, :]
        else:
            # reconstruct each via index_to_config
            cfgs = np.array(
                [index_to_config(i, self.loc_dims) for i in idx], dtype=np.uint8
            )
        # 5) print
        for _, config, amp in zip(idx, cfgs, vals):
            # rescale all the amplitudes to have the first one real and positive
            amp *= np.exp(-1j * np.angle(vals[0]))
            coords = " ".join(f"{c:2d}" for c in config)
            logger.info(f"[{coords}] |psi|={abs(amp):.8f}")  # psi={amp:.8f}")


def truncation(array, threshold=1e-14):
    validate_parameters(array=array, threshold=threshold)
    return np.where(np.abs(array) > threshold, array, 0)


def get_norm(psi):
    validate_parameters(psi=psi)
    norm = np.linalg.norm(psi)
    return norm


def get_sorted_indices(data):
    abs_data = np.abs(data)
    real_data = np.real(data)
    # Lexsort by real part first (secondary key), then by absolute value (primary key)
    sorted_indices = np.lexsort((real_data, abs_data))
    return sorted_indices[::-1]  # Descending order


def diagonalize_density_matrix(rho):
    # Diagonalize a density matrix which is HERMITIAN COMPLEX MATRIX
    if isinstance(rho, np.ndarray):
        rho_eigvals, rho_eigvecs = array_eigh(rho)
    elif isspmatrix(rho):
        rho_eigvals, rho_eigvecs = array_eigh(rho.toarray())
    return rho_eigvals, rho_eigvecs


def get_projector_for_efficient_density_matrix(rho: np.ndarray, threshold: float):
    """
    Build a projector P from the single-site density matrix rho.

    The function diagonalizes the Hermitian matrix rho, sorts the eigenvalues
    in descending order, and selects those eigenvectors with eigenvalues greater than
    the given threshold. If fewer than 2 eigenvectors pass, the threshold is relaxed
    until at least 2 are selected.

    Args:
        rho (np.ndarray): Reduced density matrix (shape (N, N)).
        threshold (float): Initial threshold for eigenvalue significance.

    Returns:
        np.ndarray: Projector matrix P of shape (N, k), where k is the number of selected eigenvectors.
    """
    # Diagonalize the density matrix
    rho_eigvals, rho_eigvecs = diagonalize_density_matrix(rho)
    # Sort eigenvalues and eigenvectors in descending order.
    # Note: np.argsort sorts in ascending order; we reverse to get descending order.
    sorted_indices = np.argsort(rho_eigvals)[::-1]
    rho_eigvals = rho_eigvals[sorted_indices]
    rho_eigvecs = rho_eigvecs[:, sorted_indices]
    logger.info(f"DIAGONALIZED RHO eigvals")
    for ii, eigval in enumerate(rho_eigvals):
        logger.info(f"{ii}  {format(eigval, '.8f')}")
    # Determine how many eigenvectors have eigenvalues greater than the threshold.
    # (If too few are significant, relax the threshold until at least 2 are selected.)
    P_columns = np.sum(rho_eigvals > threshold)
    while P_columns < 2:
        threshold /= 10
        P_columns = np.sum(rho_eigvals > threshold)
    logger.info(f"SIGNIFICANT EIGENVALUES {P_columns} with threshold {threshold}")

    # Build the projector matrix P from the selected eigenvectors.
    # Here we take the first P_columns eigenvectors
    # (which correspond to the largest eigenvalues).
    proj = np.zeros((rho.shape[0], P_columns), dtype=complex)
    for jj in range(P_columns):
        proj[:, jj] = rho_eigvecs[:, jj]
    return proj, rho_eigvals, rho_eigvecs


@njit(parallel=True, cache=True)
def build_psi_matrix(
    psi: np.ndarray,
    subsystem_configs: np.ndarray,
    environment_configs: np.ndarray,
    unique_subsys_configs: np.ndarray,
    unique_env_configs: np.ndarray,
):
    """
    Repack the full wave function vector psi into a matrix whose rows and columns
    label the *environment* and the *subsystem* configs of a given bipartition.

    Args:
        psi (np.ndarray): The wavefunction of the system.
        subsystem_configs (np.ndarray): Configurations of the subsystem in the symmetry sector.
        environment_configs (np.ndarray): Configurations of the environment in the symmetry sector.
        unique_subsys_configs (np.ndarray): Unique configurations of the subsystem.
        unique_env_configs (np.ndarray): Unique configurations of the environment.

    Returns:
        psi_matrix (np.ndarray): The matrix representation of the wavefunction in terms
        of the unique environment and subsystem configurations.
    """
    # allocate the matrix
    unique_env_dim = unique_env_configs.shape[0]
    unique_subsys_dim = unique_subsys_configs.shape[0]
    psi_matrix = np.zeros((unique_env_dim, unique_subsys_dim), dtype=np.complex128)

    for idx in prange(len(psi)):
        # find which row (env) of psi_matrix this idx belongs to
        eidx = config_to_index_linsearch(environment_configs[idx], unique_env_configs)
        # find which col (subsys) of psi_matrix this idx belongs to
        sidx = config_to_index_linsearch(subsystem_configs[idx], unique_subsys_configs)
        psi_matrix[eidx, sidx] = psi[idx]
    return psi_matrix


@njit
def exp_val_data(psi, row_list, col_list, value_list):
    """
    Compute the expectation value directly from the nonzero elements of the operator
    without constructing the full sparse matrix.

    Args:
        psi (np.ndarray): The quantum state.
        row_list (np.ndarray): Row indices of nonzero elements in the operator.
        col_list (np.ndarray): Column indices of nonzero elements in the operator.
        value_list (np.ndarray): Nonzero values of the operator.

    Returns:
        float: The computed expectation value.
    """
    exp_val = 0.0
    psi_dag = np.conjugate(psi)
    for idx in range(len(row_list)):
        row = row_list[idx]
        col = col_list[idx]
        value = value_list[idx]
        exp_val += psi_dag[row] * value * psi[col]
    return np.real(exp_val)
