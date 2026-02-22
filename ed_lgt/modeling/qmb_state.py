import numpy as np
from numba import njit, prange
from scipy.linalg import eigh as array_eigh, svd
from scipy.sparse.linalg import svds
from scipy.sparse import isspmatrix, csr_matrix
from ed_lgt.tools import (
    validate_parameters,
    encode_all_configs,
    compute_strides,
    all_pairwise_pkeys_support,
    unique_sorted_int64,
    stabilizer_renyi_sum,
)
from ed_lgt.symmetries import index_to_config
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "QMB_state",
    "truncation",
    "get_norm",
    "diagonalize_density_matrix",
    "get_projector_for_efficient_density_matrix",
    "mixed_exp_val_data",
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

    def expectation_value(self, operator, component: str = "real"):
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
            exp_val = exp_val_data(self.psi, row_list, col_list, value_list)
            return _select_component(exp_val, component)
        elif isinstance(operator, np.ndarray) or isspmatrix(operator):
            # Case 2: Dense or Sparse Matrix
            if self.psi.shape[0] != operator.shape[0]:
                msg = f"quantum state dim {self.psi.shape[0]} is NOT the operator dim {operator.shape[0]}"
                raise ValueError(msg)
            if isinstance(operator, np.ndarray):
                operator = csr_matrix(operator)
            validate_parameters(op_list=[operator])
            exp_val = np.dot(np.conjugate(self.psi), (operator.dot(self.psi)))
            return _select_component(exp_val, component)
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

    def get_state_configurations(
        self, threshold=1e-2, sector_configs=None, return_configs=False
    ):
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
        prob = np.abs(psi) ** 2
        current_threshold = float(threshold)
        idx = np.empty(0, dtype=np.int64)
        relax_steps = 0
        while True:
            mask = prob > current_threshold
            idx = np.nonzero(mask)[0]
            if idx.size > 0:
                break
            relax_steps += 1
            new_threshold = current_threshold / 10
            msg = f"No configs above threshold {current_threshold:.3e}; relaxe to {new_threshold:.3e}."
            logger.info(msg)
            current_threshold = new_threshold
        # 2) collect indices & values
        vals = psi[idx]
        msg = f"{len(idx)} configs above threshold {current_threshold:.3e}."
        logger.info(msg)
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
            rescaled_amp = amp * np.exp(-1j * np.angle(vals[0]))
            square_amp = np.abs(amp) ** 2
            coords = " ".join(f"{c:>3d}" for c in config)
            msg = f"[{coords}] |psi|^2={square_amp:6f} ({amp:6f})"
            logger.info(msg)
        if return_configs:
            return cfgs, vals

    def participation_renyi_entropy(self, alpha: int = 2) -> float:
        """
        Compute the participation Rényi entropy (order ``alpha``) of the state in the
        current basis.

        The participation Rényi entropy quantifies how delocalized the probability
        distribution ``P_i = |psi_i|^2`` is over the basis states. It is defined as

        - For ``alpha != 1``:
        ``PE_alpha = (1 / (1 - alpha)) * log( sum_i P_i**alpha )``

        - In the limit ``alpha -> 1`` it approaches the Shannon entropy
        ``-sum_i P_i log P_i`` (not implemented here).

        Parameters
        ----------
        alpha:
            Rényi order. Must satisfy ``alpha > 0`` and ``alpha != 1``.
            The default ``alpha=2`` gives the commonly used inverse participation
            ratio (IPR) form:
            ``PE_2 = -log( sum_i P_i**2 )``.

        Returns
        -------
        float
            The participation Rényi entropy of order ``alpha`` (natural logarithm).

        Notes
        -----
        - The result depends on the basis in which ``self.psi`` is represented
        (e.g. full computational basis, symmetry sector basis, momentum basis, ...).
        - This function assumes ``self.psi`` is normalized to 1. If not, probabilities
        do not sum to 1 and the quantity is not an entropy.
        - Numerical stability: a small epsilon is added inside the logarithm to
        avoid ``log(0)`` when underflow or exact zeros occur.

        Raises
        ------
        ValueError
            If ``alpha <= 0`` or ``alpha == 1``.

        Examples
        --------
        Uniform state on a D-dimensional basis (``P_i = 1/D``) has
        ``PE_alpha = log(D)`` for any ``alpha``.
        A basis state (one-hot probability) has ``PE_alpha = 0``.
        """
        if alpha <= 0 or alpha == 1:
            raise ValueError(f"alpha must be > 0 and != 1. Got alpha={alpha}.")
        logger.info("----------------------------------------------------")
        logger.info(f"PARTICIPATION RÉNYI ENTROPY alpha={alpha} (PE_{alpha})")
        prob = np.abs(self.psi) ** 2
        ipr_alpha = np.sum(prob**alpha)
        prefactor = 1.0 / (1.0 - float(alpha))
        PE_value = float(prefactor * np.log(ipr_alpha + 1e-16))
        logger.info(f"PE_{alpha} = {PE_value:.12f}")
        return PE_value

    def stabilizer_renyi_entropy(
        self,
        sector_configs: np.ndarray,
        prob_threshold: float = 1e-2,
    ):
        """
        Compute the Rényi-2 stabilizer entropy using a truncated support in a sector basis.

        Parameters
        ----------
        sector_configs:
            2D uint array (D_sector, n_sites). Basis configurations for the symmetry sector.
            Row i corresponds to basis index i of self.psi.
        prob_threshold:
            Keep basis configurations with |psi[i]|^2 > prob_threshold.
            This controls the support size K (and therefore the cost).

        Returns
        -------
        SRE2:
            float. The (normalized) Rényi-2 stabilizer entropy estimate from the support.

        Notes
        -----
        - This implementation uses the identity that the sum over all Z-strings for a fixed
          X-string can be performed analytically, reducing the problem to overlaps of
          probabilities between shifted configurations.
        - This function does NOT require the X-strings to commute with symmetries. It only
          counts strings that act within the support (and therefore contribute on the support).
        - The dominant cost is O(K^2 * n_sites) to build candidate strings, plus
          O(n_strings * K * n_sites) to accumulate contributions.
        """
        logger.info("----------------------------------------------------")
        logger.info("STABILIZER RENYI-ENTROPY SRE2 on support.")
        # Step 1: extract support arrays (Python-level) ----
        _, support_coeffs, support_configs, support_keys, discarded_weight = (
            extract_support(
                self.psi,
                self.loc_dims,
                sector_configs,
                prob_threshold,
                sort_for_encoding=True,
            )
        )
        n_configs_support = support_configs.shape[0]
        if n_configs_support == 0:
            raise ValueError("Support is empty; cannot compute stabilizer entropy.")
        msg = f"Support size = {n_configs_support}, discarded weight = {discarded_weight:.3e}"
        logger.info(msg)
        # Step 2: compute strides consistent with encoding (rightmost fastest)
        strides = compute_strides(self.loc_dims)
        # Step 3: generate candidate X-strings from support
        pkeys_all = all_pairwise_pkeys_support(support_configs, self.loc_dims, strides)
        pkeys_all.sort()
        pkeys_uniq = unique_sorted_int64(pkeys_all)
        n_strings = int(pkeys_uniq.shape[0])
        msg = f"Generated {n_strings} candidate X-strings from support."
        logger.info(msg)
        # Step 4: compute M2 and per-string contributions in parallel
        # Normalize support probabilities
        kept_weight = float(
            np.sum((np.abs(support_coeffs) ** 2).astype(np.float64, copy=False))
        )
        support_coeffs /= np.sqrt(kept_weight)
        M2 = stabilizer_renyi_sum(
            pkeys_uniq=pkeys_uniq,
            support_configs=support_configs,
            support_coeffs=support_coeffs,
            support_keys=support_keys,
            loc_dims=self.loc_dims,
            strides=strides,
        )
        # Step 5: compute SRE2
        SRE2 = -float(np.log(max(M2, 1e-16)))
        logger.info(f"SRE2 = {SRE2:.12f}")
        return SRE2


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


@njit(cache=True)
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
        complex: The computed expectation value.
    """
    exp_val = 0.0 + 0.0j
    psi_dag = np.conjugate(psi)
    for idx in range(len(row_list)):
        row = row_list[idx]
        col = col_list[idx]
        value = value_list[idx]
        exp_val += psi_dag[row] * value * psi[col]
    return exp_val


@njit(cache=True)
def mixed_exp_val_data(psi1, psi2, row_list, col_list, value_list):
    """
    Compute a mixed expectation value directly from the nonzero elements of the operator
    without constructing the full sparse matrix.

    Args:
        psi1 (np.ndarray): The first quantum state (bra).
        psi2 (np.ndarray): The second quantum state (ket).
        row_list (np.ndarray): Row indices of nonzero elements in the operator.
        col_list (np.ndarray): Column indices of nonzero elements in the operator.
        value_list (np.ndarray): Nonzero values of the operator.

    Returns:
        float: The computed expectation value.
    """
    exp_val = 0.0 + 0.0j
    psi1_dag = np.conjugate(psi1)
    for ii in range(len(row_list)):
        exp_val += psi1_dag[row_list[ii]] * value_list[ii] * psi2[col_list[ii]]
    return exp_val


def _select_component(exp_val: complex, component: str) -> float:
    if component == "real":  # <(A + A†)/2>
        return float(np.real(exp_val))
    if component == "imag":  # <(A - A†)/(2i)>
        return float(np.imag(exp_val))
    raise ValueError(f"Unknown component='{component}'")


def extract_support(
    psi: np.ndarray,
    loc_dims: np.ndarray,
    sector_configs: np.ndarray,
    prob_threshold: float = 1e-2,
    sort_for_encoding: bool = True,
):
    """
    Extract a truncated support of a state represented in a symmetry-sector basis.

    Parameters
    ----------
    psi:
        1D complex array of shape (n_configs,). State coefficients in the sector basis.
    loc_dims:
        1D int array of shape (n_sites,). Local dimensions per site.
    sector_configs:
        2D uint array of shape (n_configs, n_sites). Basis configurations for the sector.
        Row i corresponds to basis index i of psi.
    prob_threshold:
        Tail tolerance delta in [0,1). The support is chosen such that the cumulative
        probability mass kept is at least (1 - delta).
    sort_for_encoding:
        If True, sort the support by the same encoding order used by the X-string code:
        "rightmost site is fastest digit" (i.e. consistent with compute_strides).

    Returns
    -------
    support_indices:
        1D int64 array, indices in the sector basis that are kept.
    support_coeffs:
        1D complex128 array, psi[support_indices].
    support_configs:
        2D uint16 array, sector_configs[support_indices].
    support_keys:
        1D int64 array, encoded keys for support_configs, sorted ascending if sort_for_encoding=True.
    discarded_weight:
        float. Probability mass discarded: 1 - sum(support_probs).

    Notes
    -----
    - Sorting by encoding order is important because we do membership queries with
      binary_search_sorted(support_keys, key).
    """
    # Interpret prob_threshold as a tail tolerance delta
    delta = float(prob_threshold)
    if delta <= 0.0 or delta >= 1:
        raise ValueError("prob_threshold must be 1> delta >0.")
    prob = np.abs(psi) ** 2
    # Sort all sector-basis configurations by descending probability
    order_desc = np.argsort(-prob, kind="mergesort")
    prob_sorted = prob[order_desc]
    # Find smallest prefix length K with cumulative mass >= 1 - delta
    cum = np.cumsum(prob_sorted, dtype=np.float64)
    target = 1.0 - delta
    k_last = int(np.searchsorted(cum, target, side="left"))
    # Keep indices [0, ..., k_last]
    support_indices = order_desc[: k_last + 1].astype(np.int64)
    support_coeffs = psi[support_indices]
    support_configs = sector_configs[support_indices, :].astype(np.uint16, copy=False)
    # Compute discarded probability mass (delta)
    kept_weight = float(np.sum(np.abs(support_coeffs) ** 2))
    discarded_weight = max(0.0, 1.0 - kept_weight)
    # rightmost-fastest encoding key
    strides = compute_strides(loc_dims)
    support_keys = encode_all_configs(support_configs, strides)
    # Sort support by the same encoding order your algorithm expects
    if sort_for_encoding:
        sort_order = np.argsort(support_keys, kind="mergesort")
        support_indices = support_indices[sort_order]
        support_coeffs = support_coeffs[sort_order]
        support_configs = support_configs[sort_order, :]
        support_keys = support_keys[sort_order]
    return (
        support_indices,
        support_coeffs,
        support_configs,
        support_keys,
        discarded_weight,
    )
