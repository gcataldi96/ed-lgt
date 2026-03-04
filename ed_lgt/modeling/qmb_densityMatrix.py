import numpy as np
from numba import njit, prange
from math import prod
from scipy.sparse import isspmatrix, csr_array, csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import svd
from ed_lgt.tools import validate_parameters
from ed_lgt.symmetries import index_to_config
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "QMB_densityMatrix",
    "truncation",
]


class QMB_densityMatrix:
    def __init__(
        self,
        rho,
        rho_form,
        lvals=None,
        loc_dims=None,
        symmetry_sector=True,
        debug_mode=False,
    ):
        """
        Args:
            rho (np.ndarray): Density matrix, either in (row major) vectorized form or matrix

            rho_form (string):'vec' or 'mat', depending on whether it is rho_vec or rho_mat. The relationsips is rho_vec = rho_mat.flatten(),
            rho_mat = rho_vec.reshape(d,d)

            lvals (list, optional): list of the lattice spatial dimensions

            loc_dims (list of ints, np.ndarray of ints, or int): list of lattice site dimensions
        """
        validate_parameters(psi=rho, lvals=lvals, loc_dims=loc_dims)
        if rho_form not in ("vec", "mat"):
            raise ValueError(
                f"The representation of rho must be 'vec' or 'mat', not {rho_form}"
            )
        self.rho = rho
        self.rho_form = rho_form
        self.lvals = lvals
        self.loc_dims = loc_dims
        self.n_sites = prod(self.lvals)
        self.symmetry_sector = symmetry_sector
        self.debug_mode = debug_mode
        self._rho_tensor_cache: dict[tuple[int, ...], np.ndarray | csr_matrix] = {}

        # By this point we confirm it is either mat or vec
        if self.rho_form == "vec":
            N = int(np.sqrt(self.rho.shape[0]))
            if N * N != self.rho.shape[0]:
                raise ValueError(
                    f"The dimension of a vectorized density matrix must be a perfect square"
                )
            self.hilbertDim = N
        else:
            self.hilbertDim = self.rho.shape[0]

    def normalize(self, threshold=1e-14):
        """
        Normalizes the quantum states to unit trace, if it is not already. Unit trace is not equivalent
        to unit norm. In general the norm of the vector will not be 1.

        Args:
            threshold (float, optional): The tolerance level for the norm check. Defaults to 1e-14.

        Returns:
            float: The norm of the state before normalization.
        """
        trace_func = (
            get_trace_norm_vec if (self.rho_form == "vec") else get_trace_norm_mat
        )

        trace_norm = trace_func(self.rho)
        if np.abs(trace_norm - 1) > threshold:
            self.rho /= trace_norm

    def truncate(self, threshold=1e-14):
        """
        Truncates small components of the states based on a threshold. Set all numerically 0 numbers to 0.

        Args:
            threshold (float, optional): Components smaller than this value are set to zero. Defaults to 1e-14.

        Returns:
            np.ndarray: The truncated state vector.
        """
        return truncation(self.rho, threshold)

    def convert_representation(self, repr):
        """Convert the representation of rho to 'vec' or 'mat'"""
        if repr == self.rho_form:  # Already in format
            return
        elif repr == "vec":
            self.rho = self.rho.flatten()
            self.rho_form = repr
        elif repr == "mat":
            self.rho = self.rho.reshape(self.hilbertDim, self.hilbertDim)
            self.rho_form = repr
        else:
            raise ValueError(f"Desired repr should be 'vec' or 'mat', not {repr}")

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
        sparse_exp_val = (
            sparse_vec_exp_val if (self.rho_form == "vec") else sparse_matrix_exp_val
        )
        csr_exp_val = (
            csr_vec_exp_val if (self.rho_form == "vec") else csr_matrix_exp_val
        )

        if isinstance(operator, tuple) and len(operator) == 3:
            # Case 1: Operator as (row_list, col_list, value_list)
            row_list, col_list, value_list = operator
            return sparse_exp_val(self.rho, row_list, col_list, value_list)
        elif isinstance(operator, np.ndarray) or isspmatrix(operator):
            # Case 2: Dense or Sparse Matrix
            if isinstance(operator, np.ndarray):
                operator = csr_array(operator)
            validate_parameters(op_list=[operator])
            return csr_exp_val(
                self.rho, operator.indptr, operator.indices, operator.data
            )
        else:
            raise TypeError(
                "Operator must be provided as a tuple (row_list, col_list, value_list), "
                "a dense matrix (np.ndarray), or a sparse matrix (scipy sparse matrix)."
            )

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
        self.convert_representation("mat")
        # 1) mask off small amplitudes
        rho_ii = np.diagonal(self.rho).copy()
        mask = rho_ii > threshold
        # 2) collect indices & values
        idx = np.nonzero(mask)[0]  # shape (K,)
        vals = rho_ii[idx, idx]  # shape (K,)
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

    def _get_rho_tensor(self, keep_indices, partitions_dict):
        logger.info(f"----------------------------------------------------")
        logger.info("get_rho_tensor")
        key = tuple(sorted(keep_indices))
        if key not in partitions_dict:
            raise ValueError(f"{key} partition not yet implemented on the model")
        if key not in self._rho_tensor_cache:
            if self.symmetry_sector:
                # Compute the rho tensor assuming model.sector_configs not to be None
                if self.rho_form == "vec":
                    self.convert_representation("mat")
                rho_tensor = build_rho_tensor(
                    rho=self.rho,
                    subsys_config_index=partitions_dict[key]["subsys_map"],
                    env_config_index=partitions_dict[key]["env_map"],
                    subsys_dim=partitions_dict[key]["subsys_dim"],
                    env_dim=partitions_dict[key]["env_dim"],
                )
            else:
                # NO SYMMETRY SECTOR
                raise NotImplementedError
            # CHECK NORM
            if self.debug_mode:
                trace_func = (
                    get_trace_norm_vec
                    if (self.rho_form == "vec")
                    else get_trace_norm_mat
                )
                trace_norm = trace_func(self.rho)
                if hasattr(rho_tensor, "toarray"):
                    A = rho_tensor.toarray()
                else:
                    A = rho_tensor
                norm_tensor = np.trace(np.trace(A, axis1=1, axis2=3), axis1=0, axis2=1)
                assert np.allclose(trace_norm, norm_tensor, rtol=1e-12, atol=1e-12)
            # ---------------------------------------------------------------------------------
            # self._rho_tensor_cache[key] = rho_tensor # Saving the cache is very expensive on memory since it is a 4D matrix
        # ---------------------------------------------------------------------------------
        return rho_tensor

    def reduced_density_matrix(self, keep_indices, partitions_dict: dict) -> np.ndarray:
        """
        Computes the reduced density matrix of the density matrix for specified lattice sites.
        Optionally handles different symmetry sectors.

        Args:
            keep_indices (list of ints): list of lattice indices to be involved in the partition

            partitions_dict:

        Returns:
            np.ndarray: The reduced density matrix in dense format.
        """
        logger.info("----------------------------------------------------")
        logger.info(f"RED. DENSITY MATRIX OF SITES {keep_indices}")
        # Call of initialize the partition
        rho_tensor = self._get_rho_tensor(keep_indices, partitions_dict)
        if self.symmetry_sector:
            # CASE OF SYMMETRY SECTOR
            RDM = np.trace(rho_tensor, axis1=1, axis2=3)
            return RDM
        else:
            raise NotImplementedError

    # NOTE: Is there a way to use the rho_tensor directly instead of having to get the RDM?
    def entanglement_entropy(self, keep_indices, partitions_dict: dict):
        """
        This function computes the bipartite von neumann entropy of a portion of a QMB density matrix rho
        related to a lattice model with dimension lvals where single sites
        have local hilbert spaces of dimensions loc_dims

        Args:
            keep_indices (list of ints): list of lattice indices to be involved in the partition

            partitions_dict:

        Returns:
            float: bipartite entanglement entropy of the lattice subsystem
        """
        logger.debug("getting rho_tensor")
        # Call of initialize the partition
        # rho_tensor = self._get_rho_tensor(keep_indices, partitions_dict)
        RDM = self.reduced_density_matrix(keep_indices, partitions_dict)
        logger.debug("computing SVD of rho_tensor")
        # Get the singular values
        min_dim = min(RDM.shape)
        if min_dim == 1:
            # Rank-1 matrix: no entanglement
            return 0.0
        max_ks = [k for k in [50, 100, 200, 400, 600] if k < min_dim]
        if not max_ks:
            max_ks = [min_dim - 1]
        if hasattr(RDM, "tocsr"):
            # pick K until you capture enough weight...
            for n_singvals in max_ks:
                SV = svds(RDM, k=n_singvals, return_singular_vectors=False)
                ratio = np.sum(SV)
                logger.info(f"ratio of norm {ratio}")
                if ratio > 0.99:
                    break
        else:
            SV = svd(RDM, full_matrices=False, compute_uv=False)
        llambdas = SV
        llambdas = llambdas[llambdas > 1e-10]
        logger.info(
            f"n singular values: {llambdas.size} {llambdas[llambdas > 1e-4].size}"
        )
        logger.info(f"MAX ENTROPY log2(partitiondim): {np.log2(RDM.shape[0])} ")
        entropy = -np.sum(llambdas * np.log2(llambdas))
        logger.info(f"ENTROPY S of {keep_indices}: {format(entropy, '.9f')}")
        chi_exact = llambdas.size  # exact Schmidt rank at that cut
        chi_min = int(np.ceil(2**entropy))  # lower bound from S alone
        logger.info(f"-> BOND DIMENSION \chi=2^{entropy}: {chi_min}<{chi_exact}")
        return entropy

    def mutual_information(self, sites_a, sites_b, partitions_dict: dict):
        S_rhoA = self.entanglement_entropy(sites_a, partitions_dict)
        S_rhoB = self.entanglement_entropy(sites_b, partitions_dict)
        sites_ab = tuple(sites_a) + tuple(sites_b)
        S_rhoAB = self.entanglement_entropy(sites_ab, partitions_dict)

        mutual_information = S_rhoA + S_rhoB - S_rhoAB
        return mutual_information


def truncation(array, threshold=1e-14):
    validate_parameters(array=array, threshold=threshold)
    return np.where(np.abs(array) > threshold, array, 0)


@njit(cache=True, parallel=True)
def build_rho_tensor(
    rho: np.ndarray,  # (N_states,N_states)
    subsys_config_index: np.ndarray,  # (N_states,)
    env_config_index: np.ndarray,  # (N_states,)
    subsys_dim: int,
    env_dim: int,
):
    # C(ket_sub, ket_env, bra_sub, bra_env)
    rho_tensor = np.zeros(
        (subsys_dim, env_dim, subsys_dim, env_dim), dtype=np.complex128
    )
    for i in prange(rho.shape[0]):
        alpha_i = subsys_config_index[i]
        beta_i = env_config_index[i]
        for j in range(rho.shape[1]):
            rho_tensor[alpha_i, beta_i, subsys_config_index[j], env_config_index[j]] = (
                rho[i, j]
            )
    return rho_tensor


@njit(cache=True)
def get_trace_norm_vec(rho_vec: np.ndarray):
    N = int(np.sqrt(rho_vec.size))
    trace = 0.0 + 0.0j
    for ii in range(N):
        trace += rho_vec[ii * (N + 1)]
    return trace


@njit(cache=True)
def get_trace_norm_mat(rho_mat: np.ndarray):
    N = rho_mat.shape[0]
    trace = 0.0 + 0.0j
    for i in range(N):
        trace += rho_mat[i, i]
    return trace


@njit
def sparse_matrix_exp_val(rho_mat, row_list, col_list, value_list):
    """
    Compute the expectation value directly from the nonzero elements of the operator
    without constructing the full sparse matrix of a density matrix.

    Args:
        rho_mat (np.ndarray): The density state in matrix formalism
        row_list (np.ndarray): Row indices of nonzero elements in the operator.
        col_list (np.ndarray): Column indices of nonzero elements in the operator.
        value_list (np.ndarray): Nonzero values of the operator.

    Returns:
        float: The computed expectation value.
    """
    exp_val = 0.0
    for idx in range(len(row_list)):
        i = row_list[idx]
        j = col_list[idx]
        exp_val += value_list[idx] * rho_mat[j, i]  # O_ij * rho_ji
    return np.real(exp_val)


@njit
def sparse_vec_exp_val(rho_vec, row_list, col_list, value_list):
    """
    Compute the expectation value directly from the nonzero elements of the operator
    without constructing the full sparse matrix of a density matrix.

    Args:
        rho_mat (np.ndarray): The density state in matrix formalism
        row_list (np.ndarray): Row indices of nonzero elements in the operator.
        col_list (np.ndarray): Column indices of nonzero elements in the operator.
        value_list (np.ndarray): Nonzero values of the operator.

    Returns:
        float: The computed expectation value.
    """
    dim = int(np.sqrt(rho_vec.size))
    exp_val = 0.0
    for idx in range(len(row_list)):
        i = row_list[idx]
        j = col_list[idx]
        exp_val += value_list[idx] * rho_vec[j * dim + i]  # O_ij * rho_ji
    return np.real(exp_val)


@njit
def csr_matrix_exp_val(rho_mat, indptr, idx, value_list):
    """
    Compute the expectation value directly from the nonzero data already stored
    in a csr_matrix. Assuming O is a csr_matrix or csr_array

    Args:
        rho_mat (np.ndarray): The density state in matrix formalism
        indptr (np.ndarray): Row pointer array (O.indptr)
        idx (np.ndarray): Column indices of nonzero elements (O.indices)
        value_list (np.ndarray): Nonzero values of the operator (O.data)

    Returns:
        float: The computed expectation value.
    """

    exp_val = 0.0
    for i in range(len(indptr) - 1):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        for k in range(row_start, row_end):
            j = idx[k]
            exp_val += value_list[k] * rho_mat[j, i]  # O_ij * rho_ji
    return np.real(exp_val)


@njit
def csr_vec_exp_val(rho_vec, indptr, idx, value_list):
    """
    Compute the expectation value directly from the nonzero data already stored
    in a csr_matrix. Assuming O is a csr_matrix or csr_array

    Args:
        rho_mat (np.ndarray): The density state in matrix formalism
        indptr (np.ndarray): Row pointer array (O.indptr)
        idx (np.ndarray): Column indices of nonzero elements (O.indices)
        value_list (np.ndarray): Nonzero values of the operator (O.data)

    Returns:
        float: The computed expectation value.
    """
    dim = int(np.sqrt(rho_vec.size))
    exp_val = 0.0
    for i in range(len(indptr) - 1):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        for k in range(row_start, row_end):
            j = idx[k]
            exp_val += value_list[k] * rho_vec[j * dim + i]  # O_ij * rho_ji
    return np.real(exp_val)
