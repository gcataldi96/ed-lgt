"""Utilities to analyze and manipulate quantum many-body states.

This module provides the :class:`QMB_state` class and several helper functions
for expectation values, reduced density matrices, entanglement measures, support
extraction, and density-matrix post-processing.
"""

import numpy as np
from numba import njit, prange
from scipy.linalg import eigh as array_eigh, svd
from scipy.sparse.linalg import svds
from scipy.sparse import isspmatrix, csr_matrix
from edlgt.tools import (
    validate_parameters,
    encode_all_configs,
    compute_strides,
    all_pairwise_pkeys_support,
    unique_sorted_int64,
    stabilizer_renyi_sum,
)
from edlgt.symmetries import index_to_config
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
    """Container and analysis methods for a quantum many-body state vector."""

    def __init__(
        self,
        psi: np.ndarray,
        lvals=None,
        loc_dims=None,
        symmetry_sector=True,
        debug_mode=False,
    ):
        """Initialize a many-body state.

        Parameters
        ----------
        psi : numpy.ndarray
            State vector coefficients.
        lvals : list, optional
            Lattice dimensions.
        loc_dims : int or list or numpy.ndarray, optional
            Local Hilbert-space dimensions per site.
        symmetry_sector : bool, optional
            If ``True``, treat ``psi`` as living in a symmetry-reduced basis.
        debug_mode : bool, optional
            If ``True``, perform additional internal consistency checks.

        Returns
        -------
        None
        """
        validate_parameters(psi=psi, lvals=lvals, loc_dims=loc_dims)
        self.psi = psi
        self.lvals = lvals
        self.loc_dims = loc_dims
        self.symmetry_sector = symmetry_sector
        self.debug_mode = debug_mode
        self._psi_matrix_cache: dict[tuple[int, ...], np.ndarray | csr_matrix] = {}

    def normalize(self, threshold=1e-14):
        """Normalize the state vector to unit norm if needed.

        Parameters
        ----------
        threshold : float, optional
            Tolerance used to decide whether renormalization is required.

        Returns
        -------
        float
            Norm of the state before normalization.
        """
        norm = get_norm(self.psi)
        if np.abs(norm - 1) > threshold:
            self.psi /= norm
        return norm

    def truncate(self, threshold=1e-14):
        """Set state-vector entries below a threshold to zero.

        Parameters
        ----------
        threshold : float, optional
            Absolute-value threshold.

        Returns
        -------
        numpy.ndarray
            Truncated state vector.
        """
        return truncation(self.psi, threshold)

    def expectation_value(self, operator, component: str = "real"):
        """Compute an expectation value on the current state.

        Parameters
        ----------
        operator : tuple or numpy.ndarray or scipy.sparse.spmatrix
            Operator to apply. Supported formats are:

            - ``(row_list, col_list, value_list)`` sparse triplets,
            - dense NumPy matrix,
            - SciPy sparse matrix.
        component : str, optional
            Output component selector: ``"real"`` or ``"imag"``.

        Returns
        -------
        float
            Selected component of the expectation value.

        Raises
        ------
        TypeError
            If ``operator`` has an unsupported format.
        ValueError
            If matrix dimensions are incompatible with the state.
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
            # According to the size and sparsity of psi_matrix, convert it to sparse
            total = psi_matrix.size
            nnz = np.count_nonzero(psi_matrix)
            density = nnz / total
            if total > _SPARSE_SIZE_THRESH and density < _SPARSE_DENSITY_THRESH:
                psi_matrix = csr_matrix(psi_matrix)
            self._psi_matrix_cache[key] = psi_matrix
        # ---------------------------------------------------------------------------------
        return self._psi_matrix_cache[key]

    def reduced_density_matrix(self, keep_indices, partitions_dict: dict) -> np.ndarray:
        """Compute the reduced density matrix for a subsystem.

        Parameters
        ----------
        keep_indices : list[int]
            Lattice sites retained in the subsystem.
        partitions_dict : dict
            Partition metadata/cache used to build the subsystem-environment
            factorization.

        Returns
        -------
        numpy.ndarray
            Reduced density matrix in dense format.
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
        """Compute the bipartite entanglement entropy of a subsystem.

        Parameters
        ----------
        keep_indices : list[int]
            Lattice sites retained in the subsystem.
        partitions_dict : dict
            Partition metadata/cache used to build the subsystem-environment
            factorization.

        Returns
        -------
        float
            Von Neumann entanglement entropy (base-2 logarithm convention).
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
        """List or return basis configurations with large amplitudes.

        Parameters
        ----------
        threshold : float, optional
            Minimum probability threshold used to keep configurations.
        sector_configs : numpy.ndarray, optional
            If provided, lookup table mapping basis indices to configurations.
        return_configs : bool, optional
            If ``True``, return the selected configurations and amplitudes.

        Returns
        -------
        tuple or None
            If ``return_configs=True``, returns ``(cfgs, vals)``. Otherwise
            returns ``None`` and logs the selected configurations.
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
            msg = f"No configs above threshold {current_threshold:.3e}; relax to {new_threshold:.3e}."
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
            square_amp = np.abs(amp) ** 2
            coords = " ".join(f"{c:>3d}" for c in config)
            msg = f"[{coords}] |psi|^2={square_amp:6f} ({amp:6f})"
            logger.info(msg)
        if return_configs:
            return cfgs, vals

    def participation_renyi_entropy(self, alpha: int = 2) -> float:
        """Compute the participation Renyi entropy of the state in the current basis.

        Parameters
        ----------
        alpha : int, optional
            Renyi order. Must satisfy ``alpha > 0`` and ``alpha != 1``.
            The default ``alpha=2`` corresponds to the inverse-participation-ratio
            form.

        Returns
        -------
        float
            Participation Renyi entropy computed with the natural logarithm.

        Raises
        ------
        ValueError
            If ``alpha <= 0`` or ``alpha == 1``.

        Notes
        -----
        The result depends on the basis used to represent ``self.psi`` and assumes
        the state is normalized.
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
        """Compute the Renyi-2 stabilizer entropy on a truncated sector-basis support.

        Parameters
        ----------
        sector_configs : numpy.ndarray
            Sector-basis configurations of shape ``(n_configs, n_sites)``. Row
            ``i`` corresponds to basis index ``i`` of ``self.psi``.
        prob_threshold : float, optional
            Tail tolerance used in :func:`extract_support` to choose the
            truncated support.

        Returns
        -------
        float
            Renyi-2 stabilizer entropy estimate computed from the retained
            support.

        Notes
        -----
        The dominant cost scales with the support size, so lowering
        ``prob_threshold`` increases runtime and memory use.
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
    """Set array entries below a threshold to zero.

    Parameters
    ----------
    array : numpy.ndarray
        Input array.
    threshold : float, optional
        Absolute-value threshold.

    Returns
    -------
    numpy.ndarray
        Thresholded array.
    """
    validate_parameters(array=array, threshold=threshold)
    return np.where(np.abs(array) > threshold, array, 0)


@njit(cache=True)
def get_norm(psi: np.ndarray):
    """Compute the Euclidean norm of a complex state vector.

    Parameters
    ----------
    psi : numpy.ndarray
        Complex vector.

    Returns
    -------
    float
        Euclidean norm of ``psi``.
    """
    psi_norm = 0.0
    for ii in range(psi.shape[0]):
        psi_norm += psi[ii].real * psi[ii].real + psi[ii].imag * psi[ii].imag
    return np.sqrt(psi_norm)


def get_sorted_indices(data):
    """Return indices that sort entries by descending magnitude.

    Parameters
    ----------
    data : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Indices that sort ``data`` by descending absolute value. Ties are
        broken using the real part.
    """
    abs_data = np.abs(data)
    real_data = np.real(data)
    # Lexsort by real part first (secondary key), then by absolute value (primary key)
    sorted_indices = np.lexsort((real_data, abs_data))
    return sorted_indices[::-1]  # Descending order


def diagonalize_density_matrix(rho: np.ndarray | csr_matrix):
    """Diagonalize a (dense or sparse) Hermitian density matrix.

    Parameters
    ----------
    rho : numpy.ndarray or scipy.sparse.csr_matrix
        Density matrix to diagonalize.

    Returns
    -------
    tuple
        ``(rho_eigvals, rho_eigvecs)`` from Hermitian diagonalization.
    """
    # Diagonalize a density matrix which is HERMITIAN COMPLEX MATRIX
    if isinstance(rho, np.ndarray):
        rho_eigvals, rho_eigvecs = array_eigh(rho)
    elif isspmatrix(rho):
        rho_eigvals, rho_eigvecs = array_eigh(rho.toarray())
    else:
        raise TypeError("rho must be a NumPy array or SciPy sparse matrix.")
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

    Parameters
    ----------
    rho_eigvals : numpy.ndarray
        Eigenvalues of the density matrix.
    rho_eigvecs : numpy.ndarray
        Corresponding eigenvectors (columns).
    threshold : float
        Initial threshold for eigenvalue significance.

    Returns
    -------
    numpy.ndarray
        Projector matrix ``P`` of shape ``(N, k)`` built from selected eigenvectors.
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
    """Build the subsystem-environment matrix representation of a state vector.

    Parameters
    ----------
    psi : numpy.ndarray
        State coefficients in the basis used by the partition maps.
    subsys_config_index : numpy.ndarray
        Row indices (subsystem configurations) for each basis state.
    env_config_index : numpy.ndarray
        Column indices (environment configurations) for each basis state.
    subsys_dim : int
        Number of subsystem basis states.
    env_dim : int
        Number of environment basis states.

    Returns
    -------
    numpy.ndarray
        Dense matrix ``psi_matrix`` with shape ``(subsys_dim, env_dim)``.
    """
    # rows = subsystem, cols = environment
    psi_matrix = np.zeros((subsys_dim, env_dim), dtype=np.complex128)
    for ii in prange(psi.shape[0]):
        psi_matrix[subsys_config_index[ii], env_config_index[ii]] = psi[ii]
    return psi_matrix


@njit(cache=True)
def exp_val_data(psi, row_list, col_list, value_list):
    """Compute ``<psi|O|psi>`` from a sparse triplet operator representation.

    Parameters
    ----------
    psi : numpy.ndarray
        State coefficients.
    row_list, col_list, value_list : numpy.ndarray
        Sparse triplet representation of the operator.

    Returns
    -------
    complex
        Expectation value ``<psi|O|psi>``.
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
    """Compute a mixed expectation value from sparse triplet data.

    Parameters
    ----------
    psi1 : numpy.ndarray
        Bra-state coefficients.
    psi2 : numpy.ndarray
        Ket-state coefficients.
    row_list, col_list, value_list : numpy.ndarray
        Sparse triplet representation of the operator.

    Returns
    -------
    complex
        Mixed expectation value ``<psi1|O|psi2>``.
    """
    exp_val = 0.0 + 0.0j
    psi1_dag = np.conjugate(psi1)
    for ii in range(len(row_list)):
        exp_val += psi1_dag[row_list[ii]] * value_list[ii] * psi2[col_list[ii]]
    return exp_val


def _select_component(exp_val: complex, component: str) -> float:
    """Select the real or imaginary component of an expectation value.

    Parameters
    ----------
    exp_val : complex
        Complex expectation value.
    component : str
        Component selector, ``"real"`` or ``"imag"``.

    Returns
    -------
    float
        Selected component.

    Raises
    ------
    ValueError
        If ``component`` is not supported.
    """
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
    psi : numpy.ndarray
        1D complex array of shape (n_configs,). State coefficients in the sector basis.
    loc_dims : numpy.ndarray
        1D int array of shape (n_sites,). Local dimensions per site.
    sector_configs : numpy.ndarray
        2D uint array of shape (n_configs, n_sites). Basis configurations for the sector.
        Row i corresponds to basis index i of psi.
    prob_threshold : float, optional
        Tail tolerance delta in [0,1). The support is chosen such that the cumulative
        probability mass kept is at least (1 - delta).
    sort_for_encoding : bool, optional
        If True, sort the support by the same encoding order used by the X-string code:
        "rightmost site is fastest digit" (i.e. consistent with compute_strides).

    Returns
    -------
    tuple
        ``(support_indices, support_coeffs, support_configs, support_keys, discarded_weight)``
        where:

        - ``support_indices`` is a 1D int64 array of kept basis indices,
        - ``support_coeffs`` are the corresponding coefficients,
        - ``support_configs`` are the kept basis configurations,
        - ``support_keys`` are encoded configuration keys,
        - ``discarded_weight`` is the discarded probability mass.

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
