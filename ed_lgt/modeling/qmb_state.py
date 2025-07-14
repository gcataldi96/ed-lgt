import numpy as np
from numba import njit, prange
from math import prod
from scipy.linalg import eigh as array_eigh, svd
from scipy.sparse.linalg import svds
from scipy.sparse import isspmatrix, csr_array, csr_matrix
from ed_lgt.tools import validate_parameters, exclude_columns
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

# choose your own sensible thresholds density matrix
_SPARSE_SIZE_THRESH = 100000000  # total elements
_SPARSE_DENSITY_THRESH = 1e-4  # fraction nonzero


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
        self.n_sites = prod(self.lvals)
        # A cache keyed by each bipartition (keep-indices tuple), storing everything you need for that cut.
        self._partition_cache: dict[tuple[int, ...], dict] = {}

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

    def _get_partition(self, keep_indices, sector_configs: np.ndarray = None):
        """
        Lazily build (and cache) all of the bits needed for a given bipartition.

        Args:
            keep_indices:
                List or tuple of site-indices (0..n_sites-1) that you want to keep
                in the “subsystem".
                The complement of these indices form the “environment”.

            sector_configs:
                An (N_states x n_sites) array of basis configurations within your
                symmetry sector. Rows are full-system configurations. Default to be None

        Returns:
            A dict with keys:

            - "subsys": (N_states x len(keep_indices)) array
                The full list of subsystem configurations, one row per symmetry-sector state.

            - "env": (N_states x (n_sites-len(keep_indices))) array
              The full list of environment configurations, complementary to “subsys”.

            - "uniq_sub": (n_unique_sub x len(keep_indices)) array
                Unique configurations of the subsystem.

            - "uniq_env": (n_unique_env x (n_sites-len(keep_indices))) array
                Unique configurations of the environment.

            - "psi_matrix": (n_unique_env x n_unique_sub) array
                Once you’ve actually built the wave-matrix for this cut, you save it here
                so you don’t have to rebuild it on each call.

        Caching behavior:
        -----------------
        We store everything, keyed by the sorted tuple of keep_indices.  That way
        if you ever re-ask for the same cut, we do zero work—just a dict lookup.
        """
        key = tuple(sorted(keep_indices))
        if key not in self._partition_cache:
            # Determine the environmental indices
            env_indices = [ii for ii in range(self.n_sites) if ii not in keep_indices]
            # ---------------------------------------------------------------------------------
            # Distinguish between the case of symmetry sector and the standard case
            if sector_configs is not None:
                # SYMMETRY SECTOR
                # Separate subsystem and environment configurations
                subsys_configs = exclude_columns(sector_configs, np.array(env_indices))
                env_configs = exclude_columns(sector_configs, np.array(keep_indices))
                # Find unique subsystem and environment configurations
                unique_subsys_configs = np.unique(subsys_configs, axis=0)
                # Initialize the RDM with shape = number of unique subsys configs
                unique_env_configs = np.unique(env_configs, axis=0)
                # Dimensions of the partitions
                subsys_dim = unique_subsys_configs.shape[0]
                env_dim = unique_env_configs.shape[0]
                # Compute the psi_matrix
                psi_matrix = build_psi_matrix(
                    self.psi,
                    subsys_configs,
                    env_configs,
                    unique_subsys_configs,
                    unique_env_configs,
                )
            else:
                # NO SYMMETRY SECTOR
                # Reshape and reorder psi for partitioning based on keep_indices.
                # Ensure psi is reshaped into a tensor with one leg per lattice site
                psi_tensor = self.psi.reshape(self.loc_dims)
                # Reorder psi indices
                new_order = keep_indices + env_indices
                # Rearrange the tensor to group subsystem and environment indices
                psi_tensor = np.transpose(psi_tensor, axes=new_order)
                logger.debug(f"Reordered psi_tensor shape: {psi_tensor.shape}")
                # Determine the dimensions of the subsystem and environment for the bipartition
                subsys_dim = np.prod([self.loc_dims[ii] for ii in keep_indices])
                env_dim = np.prod([self.loc_dims[ii] for ii in env_indices])
                # Reshape the reordered tensor to separate subsystem from environment
                psi_matrix = psi_tensor.reshape((subsys_dim, env_dim))
            # ---------------------------------------------------------------------------------
            # According to the size and sparity of psi_matrix, convert it to sparse
            total = psi_matrix.size
            nnz = np.count_nonzero(psi_matrix)
            density = nnz / total
            if total > _SPARSE_SIZE_THRESH and density < _SPARSE_DENSITY_THRESH:
                psi_matrix = csr_matrix(psi_matrix)
            # ---------------------------------------------------------------------------------
            # Save the partition information
            self._partition_cache[key] = {
                "subsys_indices": keep_indices,
                "env_indices": env_indices,
                "subsys_dim": subsys_dim,
                "env_dim": env_dim,
                "psi_matrix": psi_matrix,
            }
        return self._partition_cache[key]

    def reduced_density_matrix(
        self, keep_indices, sector_configs: np.ndarray = None
    ) -> np.ndarray:
        """
        Computes the reduced density matrix of the quantum state for specified lattice sites.
        Optionally handles different symmetry sectors.

        Args:
            keep_indices (list of ints): list of lattice indices to be involved in the partition

            sector_configs:
                An (N_states x n_sites) array of basis configurations within your
                symmetry sector. Rows are full-system configurations. Default to be None

        Returns:
            np.ndarray: The reduced density matrix in dense format.
        """
        logger.info("----------------------------------------------------")
        logger.info(f"RED. DENSITY MATRIX OF SITES {keep_indices}")
        # Call of initialize the partition
        psi_matrix = self._get_partition(keep_indices, sector_configs)["psi_matrix"]
        if sector_configs is not None:
            # CASE OF SYMMETRY SECTOR
            return psi_matrix.conj().T @ psi_matrix
        else:
            # CASE with NO SYMMETRIES
            # Compute the reduced density matrix by tracing out the env-indices
            if hasattr(psi_matrix, "toarray"):
                psi_matrix = psi_matrix.toarray()
            RDM = np.tensordot(psi_matrix, psi_matrix.conj(), axes=([1], [1]))
            # Reshape rho to ensure it is a square matrix corresponding to the subsystem
            subsys_dim = psi_matrix.shape[0]
            RDM = RDM.reshape((subsys_dim, subsys_dim))
            return RDM

    def entanglement_entropy(self, keep_indices, sector_configs: np.ndarray = None):
        """
        This function computes the bipartite entanglement entropy of a portion of a QMB state psi
        related to a lattice model with dimension lvals where single sites
        have local hilbert spaces of dimensions loc_dims

        Args:
            keep_indices (list of ints): list of lattice indices to be involved in the partition

            sector_configs:
                An (N_states x n_sites) array of basis configurations within your
                symmetry sector. Rows are full-system configurations. Default to be None

        Returns:
            float: bipartite entanglement entropy of the lattice subsystem
        """
        logger.debug("computing SVD of psi_matrix")
        # Call of initialize the partition
        psi_matrix = self._get_partition(keep_indices, sector_configs)["psi_matrix"]
        # Get the singular values
        if hasattr(psi_matrix, "tocsr"):
            # pick K until you capture enough weight...
            for n_singvals in [50, 100, 200, 400, 600]:
                SV = svds(psi_matrix, k=n_singvals, return_singular_vectors=False)
                ratio = np.sum(SV**2)
                logger.info(f"ratio of norm {ratio}")
                if ratio > 0.99:
                    break
        else:
            SV = svd(psi_matrix, full_matrices=False, compute_uv=False)
        llambdas = SV**2
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
        mask = np.abs(psi) ** 2 > threshold
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
            rescaled_amp = round(amp * np.exp(-1j * np.angle(vals[0])), 8)
            square_amp = round(np.abs(amp) ** 2, 8)
            coords = " ".join(f"{c:2d}" for c in config)
            logger.info(f"[{coords}] psi={rescaled_amp} |psi|^2={square_amp}")


def truncation(array, threshold=1e-14):
    validate_parameters(array=array, threshold=threshold)
    return np.where(np.abs(array) > threshold, array, 0)


@njit(cache=True)
def get_norm(psi: np.ndarray):
    psi_norm = 0.0
    for ii in range(psi.shape[0]):
        psi_norm += psi[ii].real * psi[ii].real + psi[ii].imag * psi[ii].imag
    return np.sqrt(psi_norm)


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


def get_projector_for_efficient_density_matrix(
    rho_eigvals, rho_eigvecs, threshold: float
):
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
    # Sort eigenvalues and eigenvectors in descending order.
    # Note: np.argsort sorts in ascending order; we reverse to get descending order.
    sorted_indices = np.argsort(rho_eigvals)[::-1]
    rho_eigvals = rho_eigvals[sorted_indices]
    rho_eigvecs = rho_eigvecs[:, sorted_indices]
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
    proj = np.zeros((rho_eigvals.shape[0], P_columns), dtype=complex)
    for jj in range(P_columns):
        proj[:, jj] = rho_eigvecs[:, jj]
    return proj


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
