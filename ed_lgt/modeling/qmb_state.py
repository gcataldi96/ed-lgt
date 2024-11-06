import numpy as np
from numba import njit, prange
from math import prod
from scipy.linalg import eigh as array_eigh, svd
from scipy.sparse import csr_matrix, isspmatrix, csr_array
from ed_lgt.tools import validate_parameters, exclude_columns, filter_compatible_rows
from ed_lgt.symmetries import config_to_index_binarysearch, index_to_config
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
    def __init__(self, psi, lvals=None, loc_dims=None):
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

    def expectation_value(
        self, operator=None, row_list=None, col_list=None, value_list=None
    ):
        """
        Calculates the expectation value of the given operator with the current quantum state.

        Args:
            operator (np.ndarray or sparse_matrix): The operator to apply.

        Returns:
            float: The real part of the expectation value.
        """
        if operator is not None:
            validate_parameters(op_list=[operator])
            return np.real(np.dot(np.conjugate(self.psi), (operator.dot(self.psi))))
        else:
            return exp_val_data(self.psi, row_list, col_list, value_list)

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

    def reduced_density_matrix(self, keep_indices, sector_configs=None):
        """
        Computes the reduced density matrix of the quantum state for specified lattice sites.
        Optionally handles different symmetry sectors.

        Args:
            keep_indices (list of ints): Indices of the lattice sites to keep.
            sector_configs (np.ndarray, optional): Configurations that define symmetry sectors.

        Returns:
            np.ndarray: The reduced density matrix in sparse format.
        """
        logger.info("----------------------------------------------------")
        logger.info(f"RED. DENSITY MATRIX OF SITES {keep_indices}")
        # CASE OF SYMMETRY SECTOR
        if sector_configs is not None:
            # Indices for the environment
            env_indices = [
                i for i in range(sector_configs.shape[1]) if i not in keep_indices
            ]
            # Separate subsystem and environment configurations
            subsystem_configs = exclude_columns(sector_configs, np.array(env_indices))
            env_configs = exclude_columns(sector_configs, np.array(keep_indices))
            # Find unique subsystem and environment configurations
            unique_subsys_configs = np.unique(subsystem_configs, axis=0)
            subsystem_dim = unique_subsys_configs.shape[0]
            # Initialize the RDM with shape = number of unique subsys configs
            unique_env_configs = np.unique(env_configs, axis=0)
            RDM = np.zeros((subsystem_dim, subsystem_dim), dtype=np.complex128)
            # Iterate over unique environment configurations
            return compute_RDM(
                self.psi,
                subsystem_configs,
                env_configs,
                unique_subsys_configs,
                unique_env_configs,
            )
        else:
            # NO SYMMETRIES
            # Prepare psi tensor sorting subsystem indices close each other to bipartite the system
            psi_tensor, subsystem_dim, _ = self.bipartite_psi(keep_indices)
            # Compute the reduced density matrix by tracing out the env-indices
            RDM = np.tensordot(psi_tensor, np.conjugate(psi_tensor), axes=([1], [1]))
            # Reshape rho to ensure it is a square matrix corresponding to the subsystem
            RDM = RDM.reshape((subsystem_dim, subsystem_dim))
            return RDM

    def entanglement_entropy(self, keep_indices, sector_configs=None):
        """
        This function computes the bipartite entanglement entropy of a portion of a QMB state psi
        related to a lattice model with dimension lvals where single sites
        have local hilbert spaces of dimensions loc_dims

        Args:
            keep_indices (list of ints): list of lattice indices to be involved in the partition

        Returns:
            float: bipartite entanglement entropy of the lattice subsystem
        """
        if sector_configs is not None:
            # Compute the RDM for the partition within a symmetry sector
            RDM = self.reduced_density_matrix(keep_indices, sector_configs)
            # Compute its eigenvalues
            llambdas = array_eigh(RDM, eigvals_only=True)
        else:
            # Prepare psi tensor sorting subsystem indices close each other to bipartite the system
            psi_tensor, _, _ = self.bipartite_psi(keep_indices)
            # Compute SVD
            _, V, _ = svd(psi_tensor, full_matrices=False)
            llambdas = V**2
        llambdas = llambdas[llambdas > 1e-10]
        entropy = -np.sum(llambdas * np.log2(llambdas))
        logger.info(f"ENTROPY of {keep_indices}: {format(entropy, '.9f')}")
        return entropy

    def get_state_configurations(self, threshold=1e-2, sector_configs=None):
        """
        This function expresses the main QMB state configurations associated with the
        most relevant coefficients of the QMB state psi. Every state configuration
        is expressed in terms of the single site local Hilber basis
        """
        logger.info("----------------------------------------------------")
        logger.info("STATE CONFIGURATIONS")
        psi = csr_array(self.truncate(threshold))
        if sector_configs is not None:
            psi_configs = sector_configs[psi.indices, :]
        else:
            psi_configs = np.array(
                [index_to_config(i, self.loc_dims) for i in psi.indices], dtype=np.uint8
            )
        # Get the descending order of absolute values of psi_data
        sorted_indices = get_sorted_indices(psi.data)
        # Use sorted indices to reorder data and configurations
        psi_data = psi.data[sorted_indices]
        psi_configs = psi_configs[sorted_indices]
        # Print sorted cofigs with its amplitude
        for config, val in zip(psi_configs, psi_data):
            logger.info(
                f"[{' '.join(f'{s:2d}' for s in config)}]  {format(np.abs(val),'.4f')}  {format(val,'.4f')}"
            )


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
    logger.info(f"TOTAL NUMBER OF SIGNIFICANT EIGENVALUES {P_columns}")
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


@njit(parallel=True)
def compute_RDM(
    psi,
    subsystem_configs,
    environment_configs,
    unique_subsys_configs,
    unique_env_configs,
):
    """
    Computes the reduced density matrix of the quantum state for the specified subsystem
    using symmetry sector configurations, parallelized using Numba.

    Args:
        psi (np.ndarray): The wavefunction of the system.
        subsystem_configs (np.ndarray): Configurations of the subsystem in the symmetry sector.
        environment_configs (np.ndarray): Configurations of the environment in the symmetry sector.
        unique_subsys_configs (np.ndarray): Unique configurations of the subsystem.
        unique_env_configs (np.ndarray): Unique configurations of the environment.

    Returns:
        RDM (np.ndarray): The reduced density matrix.
    """
    # Number of unique subsys configurato
    subsystem_dim = unique_subsys_configs.shape[0]
    # Initialize the reduced density matrix
    RDM = np.zeros((subsystem_dim, subsystem_dim), dtype=np.complex128)
    # Loop over unique environment configurations in parallel
    for env_idx in prange(len(unique_env_configs)):
        env_config = unique_env_configs[env_idx]
        # Find indices where the environment matches the current env_config
        matching_indices = filter_compatible_rows(environment_configs, env_config)
        # Create the subsystem wavefunction slice
        subsystem_psi = np.zeros(subsystem_dim, dtype=np.complex128)
        for ii in range(len(matching_indices)):
            match_idx = matching_indices[ii]
            subsys_config = subsystem_configs[match_idx]
            # Find the index of the subsystem configuration using a binary search
            subsys_index = config_to_index_binarysearch(
                subsys_config, unique_subsys_configs
            )
            # Populate the subsystem wavefunction with the matching psi values
            subsystem_psi[subsys_index] = psi[match_idx]
        # Add the contribution to the RDM for this environment configuration
        RDM += np.outer(subsystem_psi, np.conjugate(subsystem_psi))
    return RDM


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
