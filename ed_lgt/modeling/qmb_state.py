import numpy as np
from numba import njit, prange
from scipy.linalg import eigh as array_eigh, svd
from scipy.sparse.linalg import svds
from scipy.sparse import isspmatrix, csr_matrix
from ed_lgt.tools import validate_parameters, encode_all_configs, compute_strides
from ed_lgt.symmetries import index_to_config
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "QMB_state",
    "truncation",
    "get_norm",
    "diagonalize_density_matrix",
    "get_projector_for_efficient_density_matrix",
    "exp_val_data2",
    "extract_support",
]

# choose your own sensible thresholds density matrix
_SPARSE_SIZE_THRESH = 100000000  # total elements
_SPARSE_DENSITY_THRESH = 1e-4  # fraction nonzero


class QMB_state:
    def __init__(
        self,
        psi: np.ndarray,
        lvals=None,
        loc_dims=None,
        symmetry_sector=True,
        debug_mode=False,
    ):
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
        self.symmetry_sector = symmetry_sector
        self.debug_mode = debug_mode
        self._psi_matrix_cache: dict[tuple[int, ...], np.ndarray | csr_matrix] = {}

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
                operator = csr_matrix(operator)
            validate_parameters(op_list=[operator])
            return np.real(np.dot(np.conjugate(self.psi), (operator.dot(self.psi))))
        else:
            raise TypeError(
                "Operator must be provided as a tuple (row_list, col_list, value_list), "
                "a dense matrix (np.ndarray), or a sparse matrix (scipy sparse matrix)."
            )

    def _get_psi_matrix(self, keep_indices, partitions_dict):
        logger.info(f"----------------------------------------------------")
        key = tuple(sorted(keep_indices))
        if key not in partitions_dict:
            raise ValueError(f"{key} partition not yet implemented on the model")
        if key not in self._psi_matrix_cache:
            if self.symmetry_sector:
                # Compute the psi matrix assuming model.sector_configs not to be None
                psi_matrix = build_psi_matrix(
                    psi=self.psi,
                    subsys_config_index=partitions_dict[key]["subsys_map"],
                    env_config_index=partitions_dict[key]["env_map"],
                    subsys_dim=partitions_dict[key]["subsys_dim"],
                    env_dim=partitions_dict[key]["env_dim"],
                )
                logger.info(f"psi matrix {psi_matrix.shape}")
            else:
                # NO SYMMETRY SECTOR
                env_indices = partitions_dict[key]["env_indices"]
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
            # CHECK NORM
            if self.debug_mode:
                norm_vec = np.vdot(self.psi, self.psi).real
                if hasattr(psi_matrix, "toarray"):
                    A = psi_matrix.toarray()
                else:
                    A = psi_matrix
                norm_matrix = np.sum(np.abs(A) ** 2)
                assert np.allclose(norm_vec, norm_matrix, rtol=1e-12, atol=1e-12)
            # ---------------------------------------------------------------------------------
            # According to the size and sparity of psi_matrix, convert it to sparse
            total = psi_matrix.size
            nnz = np.count_nonzero(psi_matrix)
            density = nnz / total
            if total > _SPARSE_SIZE_THRESH and density < _SPARSE_DENSITY_THRESH:
                psi_matrix = csr_matrix(psi_matrix)
            self._psi_matrix_cache[key] = psi_matrix
        # ---------------------------------------------------------------------------------
        return self._psi_matrix_cache[key]

    def reduced_density_matrix(self, keep_indices, partitions_dict: dict) -> np.ndarray:
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
        psi_matrix = self._get_psi_matrix(keep_indices, partitions_dict)
        if self.symmetry_sector:
            # CASE OF SYMMETRY SECTOR
            return psi_matrix @ psi_matrix.conj().T
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

    def entanglement_entropy(self, keep_indices, partitions_dict: dict):
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
        psi_matrix = self._get_psi_matrix(keep_indices, partitions_dict)
        # Get the singular values
        min_dim = min(psi_matrix.shape)
        if min_dim == 1:
            # Rank-1 matrix: no entanglement
            return 0.0
        max_ks = [k for k in [50, 100, 200, 400, 600] if k < min_dim]
        if not max_ks:
            max_ks = [min_dim - 1]
        if hasattr(psi_matrix, "tocsr"):
            # pick K until you capture enough weight...
            for n_singvals in max_ks:
                SV = svds(psi_matrix, k=n_singvals, return_singular_vectors=False)
                ratio = np.sum(SV**2)
                logger.info(f"ratio of norm {ratio}")
                if ratio > 0.99:
                    break
        else:
            SV = svd(psi_matrix, full_matrices=False, compute_uv=False)
        llambdas = SV**2
        llambdas = llambdas[llambdas > 1e-10]
        logger.debug(f"n singvals: {llambdas.size} {llambdas[llambdas > 1e-4].size}")
        logger.debug(f"MAX ENTROPY log2(partitiondim): {np.log2(psi_matrix.shape[0])} ")
        entropy = -np.sum(llambdas * np.log2(llambdas))
        logger.info(f"ENTROPY S of {keep_indices}: {format(entropy, '.12f')}")
        chi_exact = llambdas.size  # exact Schmidt rank at that cut
        chi_min = int(np.ceil(2**entropy))  # lower bound from S alone
        logger.debug(f"BOND DIMENSION \chi=2^{entropy}: {chi_min}<{chi_exact}")
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
            logger.info(
                f"[{coords}] {round(amp,3)} psi={rescaled_amp} |psi|^2={square_amp}"
            )


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


def diagonalize_density_matrix(rho: np.ndarray | csr_matrix):
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


@njit(cache=True, parallel=True)
def build_psi_matrix(
    psi: np.ndarray,  # (N_states,)
    subsys_config_index: np.ndarray,  # (N_states,)
    env_config_index: np.ndarray,  # (N_states,)
    subsys_dim: int,
    env_dim: int,
):
    # rows = subsystem, cols = environment
    psi_matrix = np.zeros((subsys_dim, env_dim), dtype=np.complex128)
    for ii in prange(psi.shape[0]):
        psi_matrix[subsys_config_index[ii], env_config_index[ii]] = psi[ii]
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


@njit(cache=True)
def exp_val_data2(psi1, psi2, row_list, col_list, value_list):
    exp_val = 0.0 + 0.0j
    psi1_dag = np.conjugate(psi1)
    for ii in range(len(row_list)):
        exp_val += psi1_dag[row_list[ii]] * value_list[ii] * psi2[col_list[ii]]
    return exp_val


def extract_support(
    psi: np.ndarray,
    loc_dims: np.ndarray,
    sector_configs: np.ndarray,
    prob_threshold: float = 1e-2,
    sort_for_encoding: bool = True,
):
    """
    Build a truncated-support representation of |psi>.

    Returns
    -------
    support_indices : (K,) int64
        Indices in the *sector basis* (i.e. indices into sector_configs / psi).
    support_coeffs  : (K,) complex128
        Amplitudes psi[support_indices].
    support_configs : (K, N) uint16
        Configurations for each kept basis index, sorted consistently if requested.
    discarded_weight : float
        Sum of probabilities outside the support (delta = 1 - sum(|coeffs|^2)).
    """
    prob = np.abs(psi) ** 2
    mask = prob > prob_threshold
    support_indices = np.nonzero(mask)[0].astype(np.int64)
    probs_kept = prob[support_indices]
    order_desc = np.argsort(-probs_kept)
    support_indices = support_indices[order_desc]
    support_coeffs = psi[support_indices]
    support_configs = sector_configs[support_indices, :]
    # Compute discarded probability mass (delta)
    kept_weight = float(np.sum(np.abs(support_coeffs) ** 2))
    discarded_weight = max(0.0, 1.0 - kept_weight)
    # Sort support by the same encoding order your algorithm expects
    if sort_for_encoding:
        # rightmost-fastest encoding key
        encoding_keys = encode_all_configs(support_configs, compute_strides(loc_dims))
        sort_order = np.argsort(encoding_keys, kind="mergesort")
        support_indices = support_indices[sort_order]
        support_coeffs = support_coeffs[sort_order]
        support_configs = support_configs[sort_order, :]
    return support_indices, support_coeffs, support_configs, discarded_weight
