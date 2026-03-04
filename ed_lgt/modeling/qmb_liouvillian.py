import numpy as np
from numba import njit, prange
from scipy.linalg import eig as array_eig
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigs as sparse_eig
from scipy.sparse.linalg import expm_multiply
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csc_matrix, isspmatrix, coo_matrix
from ed_lgt.tools import validate_parameters
from .qmb_densityMatrix import QMB_densityMatrix, get_trace_norm_vec
import scipy.sparse.linalg as spla
import scipy.sparse as sp
import logging

logger = logging.getLogger(__name__)

__all__ = ["QMB_liouvillian"]


class QMB_liouvillian:
    def __init__(self, lvals, size):
        """
        Initialize the Liouvillian class.

        Args:
            Liou: The Liouvillian, which can be a dense NumPy array, a sparse matrix (csc_matrix),
                 or a LinearOperator.
            lvals: Additional lattice values or parameters.
            size: Liouvillian space dimension = (Hilbert space dimension)^2
        """
        validate_parameters(lvals=lvals)
        self.lvals = lvals
        self.shape = (size, size)
        self.check_dimension()
        # Initialize row_list, col_list, value_list as empty arrays
        self.row_list = np.array([], dtype=int)
        self.col_list = np.array([], dtype=int)
        self.value_list = np.array([], dtype=np.complex128)
        self.effH = 0

    def add_term(self, term):
        """
        Add a term to the Liouvillian.

        Args:
            term: The term to add, which can be:
                  - A dense NumPy array (np.ndarray).
                  - A sparse matrix (scipy.sparse.csc_matrix or csr_matrix (Prefered)).
                  - Non-zero elements in the form of (row_list, col_list, value_list).
        """
        if isinstance(term, tuple) and len(term) == 3:
            # Case 1: term is (row_list, col_list, value_list)
            row_list, col_list, value_list = term
        elif isinstance(term, np.ndarray):
            # Case 2: term is a dense matrix
            row_list, col_list = np.nonzero(term)
            value_list = term[row_list, col_list]
        elif isspmatrix(term):
            # Case 3: term is a sparse matrix
            coo = coo_matrix(term)
            row_list = coo.row
            col_list = coo.col
            value_list = coo.data
        else:
            raise TypeError(
                "Unsupported term type. Provide a matrix (dense or sparse) "
                "or (row_list, col_list, value_list)."
            )
        self.row_list = np.concatenate((self.row_list, row_list))
        self.col_list = np.concatenate((self.col_list, col_list))
        self.value_list = np.concatenate((self.value_list, value_list))

    def build(self, format):
        """
        Construct the Liouvillian as a sparse matrix or LinearOperator.

        Args:
            format (str): Target type ('dense', 'sparse', or 'linear').
        """
        logger.info(f"Construct {format} Liouvillian")

        # ==========================================================================
        def matvec(psi):
            """Function to compute L @ v."""
            result = csr_spmv(self.indptr, self.indices, self.data, psi)
            if result is None:
                raise ValueError("matvec returned None.")
            if not isinstance(result, np.ndarray):
                raise TypeError(f"matvec must return np.ndarray, got {type(result)}.")
            return result

        # ==========================================================================
        def matmat(psis):
            """Function to compute L @ matrix (batch of vectors)."""
            result = csr_matmat(self.indptr, self.indices, self.data, psis)
            if result is None:
                raise ValueError("matmat returned None.")
            if not isinstance(result, np.ndarray):
                raise TypeError(f"matmat must return np.ndarray, got {type(result)}.")
            return result

        # ==========================================================================
        if format == "sparse":
            self.Liou = csc_matrix(
                (self.value_list, (self.row_list, self.col_list)), shape=self.shape
            )
        # ==========================================================================
        elif format == "linear":
            N = self.shape[0]
            self.indptr, self.indices, self.data = build_csr_numba(
                N, self.row_list, self.col_list, self.value_list
            )
            self.Liou = LinearOperator(shape=self.shape, matvec=matvec, matmat=matmat)
        # ==========================================================================
        elif format == "dense":
            self.Liou = csc_matrix(
                (self.value_list, (self.row_list, self.col_list)), shape=self.shape
            )
            self.Liou = self.Liou.toarray()
        # Measure sparsity
        self.get_sparsity()
        self.Liou_type = format

    def convert_liouvillian(self, format):
        """
        Convert the Liouvillian to the specified representation type.

        Args:
            format (str): Target type ('dense', 'sparse', or 'linear').
        """
        logger.info(f"CONVERT LIOUVILLIAN from {self.Liou_type} to {format}")
        if format == self.Liou_type:
            return  # Already in the desired format
        elif format == "dense":
            if self.Liou_type == "sparse":
                self.Liou = self.Liou.toarray()
            elif self.Liou_type == "linear":
                self.build("dense")
        elif format == "sparse":
            if self.Liou_type == "dense":
                self.Liou = csc_matrix(self.Liou)
            elif self.Liou_type == "linear":
                self.build("sparse")
        elif format == "linear":
            self.build("linear")
        else:
            msg = f"Invalid format: {format}. Use 'dense', 'sparse', or 'linear'."
            raise ValueError(msg)
        self.Liou_type = format

    def diagonalize(self, n_eigs, format, loc_dims):
        """
        Diagonalize the Liouvillian.

        Args:
            n_eigs(int or "full"): Number of Eigenvalues to retrieve
            format: Format in which to diagonalize the Liouvillian
            loc_dims(list of ints, np.ndarray of ints, or int): list of lattice site dimensions
        """
        validate_parameters(loc_dims=loc_dims)
        if format != self.Liou_type:
            self.convert_liouvillian(format)
        # Save local dimensions
        self.loc_dims = loc_dims
        # Determine number of eigenvalues to compute
        if isinstance(n_eigs, int):
            if n_eigs > self.shape[0]:
                msg = f"n_eigs must be < L.shape[0]={self.shape[0]}, got {n_eigs}"
                raise ValueError(msg)
            self.n_eigs = n_eigs
        elif isinstance(n_eigs, str) and n_eigs == "full":
            self.n_eigs = self.shape[0]
        else:
            raise ValueError(f"n_eigs must be int or 'full', not {n_eigs}.")
        # Select diagonalization format
        logger.info(f"----------------------------------------------------")
        if self.Liou_type == "dense" or self.n_eigs == "full":
            logger.info("DIAGONALIZE (dense) LIOUVILLIAN")
            eigVals, eigVecs = array_eig(self.Liou)
        elif (
            self.Liou_type == "sparse"
            or self.Liou_type == "linear"
            and isinstance(n_eigs, int)
        ):
            logger.info(f"DIAGONALIZE {self.Liou_type} LIOUVILLIAN")
            # We care about the eigvals with real parts near 0. All eigvals will have a negative real part
            print("Diagonalizing...")
            eigVals, eigVecs = sparse_eig(
                self.Liou,
                k=self.n_eigs,
                which="LR",
                ncv=8 * self.n_eigs,
                maxiter=10000,
                tol=1e-12,
            )
        # Sort and save eigenvalues/eigenvectors
        if not is_sorted_by_real_desc(eigVals):
            order = np.argsort(-eigVals.real)
            eigVals, eigVecs = eigVals[order], eigVecs[:, order]
        self.eigVals = eigVals
        self.eigVecs = eigVecs

        # The eigenvectors of the Liouvillian with 0 eigVal are steadyStates of the system
        self.steadyStates = []
        steadyInd = np.where(np.abs(eigVals) < 1e-12)[0]

        # Fail safe so if it failed to find steady state, it still runs. This will be detected later and deleted.
        if steadyInd.size == 0:
            steadyInd = [0]

        for ind in steadyInd:
            print("Saving Steady State...")
            temp_state = QMB_densityMatrix(
                rho=eigVecs[:, ind],
                rho_form="vec",
                lvals=self.lvals,
                loc_dims=self.loc_dims,
            )
            temp_state.normalize()
            self.steadyStates.append(temp_state)

    # Calculate the steady state without diagonalizing by enforcing unit trace of the steady state. Works well for medium sized
    # problems but is no longer feasible for really big problems
    def get_steadyState(self):
        hilbertDim = int(np.sqrt(self.shape[0]))

        Lred = self.Liou[:-1, :-1]
        vec = self.Liou[:-1, -1:]

        indices = np.arange(hilbertDim)
        trace_vec = sp.csc_matrix(
            (np.ones(hilbertDim), (np.zeros(hilbertDim), indices * (hilbertDim + 1))),
            shape=(1, hilbertDim * hilbertDim),
        )

        Lred += -vec @ trace_vec[:, :-1]

        rho_ss = sp.linalg.spsolve(Lred, -vec, permc_spec="MMD_AT_PLUS_A")

        # Save result
        rho_ss = QMB_densityMatrix(
            rho_vec=rho_ss, lvals=self.lvals, loc_dims=self.loc_dims
        )
        rho_ss.normalize()
        self.steadyStates.append(rho_ss)

    def time_evolution(
        self,
        initial_state: np.ndarray,
        time_line: np.ndarray,
        loc_dims: np.ndarray,
        sparse_ev_method="RK4",
    ):
        """
        Perform time evolution using sparse method (for sparse / linear operators).
        """
        # Save local dimensions
        self.loc_dims = loc_dims
        # Compute the time spacing and the time line
        delta_t = time_line[1] - time_line[0]
        msg = f"------------ TIME EVOLUTION: DELTA {round(delta_t,4)} ------------"
        logger.info(msg)
        # Check Liouvillian type and compute the time evolution
        if self.Liou_type == "dense":
            raise NotImplementedError("Dense Evolution Not Tested Yet")
            # Check if Liouvillian is already diagonalized
            if not hasattr(self, "eigVals") or not hasattr(self, "eigVecs"):
                self.diagonalize(n_eigs="full", format="dense", loc_dims=loc_dims)
            # Run the exact time evolution
            rho_vec_time = exact_time_evolution(
                time_line,
                self.eigVals,
                self.eigVecs,
                initial_state.astype(np.complex128),
            )
        elif self.Liou_type == "sparse":
            A = self.Liou.tocsr()
            if sparse_ev_method == "expm":
                rho_vec_time = expm_multiply(
                    A=A,
                    B=initial_state,
                    start=time_line[0],
                    stop=time_line[-1],
                    num=len(time_line),
                    endpoint=True,
                )
            elif sparse_ev_method == "RK4":
                rho_vec_time = runge_kutta4_time_evolution(
                    initial_state=initial_state,
                    indptr=A.indptr,
                    indices=A.indices,
                    data=A.data,
                    time_line=time_line,
                )
            elif sparse_ev_method == "RK4_2":
                sol = solve_ivp(
                    rhodot,
                    (time_line[0], time_line[-1]),
                    initial_state,
                    t_eval=time_line,
                    args=[A],
                    atol=1e-12,
                    rtol=1e-6,
                )
                rho_vec_time = sol.y.T

            elif sparse_ev_method == "RK4_linear":
                self.convert_liouvillian("linear")

                def matvec_lin(psi):
                    return self.Liou.matvec(psi)

                def matmat_lin(psis):
                    return self.Liou.matmat(psis)

                def rmatvec_lin(psi):
                    return self.Liou.matvec(psi)

                def rmatmat_lin(psis):
                    return self.Liou.matmat(psis)

                A_lin = LinearOperator(
                    shape=self.shape,
                    matvec=matvec_lin,
                    matmat=matmat_lin,
                    rmatvec=rmatvec_lin,
                    rmatmat=rmatmat_lin,
                    dtype=np.complex128,
                )
                sol = solve_ivp(
                    rhodot,
                    (time_line[0], time_line[-1]),
                    initial_state,
                    t_eval=time_line,
                    args=[A_lin],
                )
                rho_vec_time = sol.y.T

        elif self.Liou_type == "linear":

            def matvec_lin(psi):
                return self.Liou.matvec(psi)

            def matmat_lin(psis):
                return self.Liou.matmat(psis)

            def rmatvec_lin(psi):
                return self.Liou.matvec(psi)

            def rmatmat_lin(psis):
                return self.Liou.matmat(psis)

            A_lin = LinearOperator(
                shape=self.shape,
                matvec=matvec_lin,
                matmat=matmat_lin,
                rmatvec=rmatvec_lin,
                rmatmat=rmatmat_lin,
                dtype=np.complex128,
            )
            traceA = compute_trace_L(self.row_list, self.col_list, self.value_list)
            rho_vec_time = expm_multiply(
                A=A_lin,
                B=initial_state,
                start=time_line[0],
                stop=time_line[-1],
                num=len(time_line),
                endpoint=True,
                traceA=traceA,
            )
        logger.info("Saving states")
        # Save them in vector formalism
        self.rho_time = [
            QMB_densityMatrix(
                rho=rho_vec_time[ii, :],
                rho_form="vec",
                lvals=self.lvals,
                loc_dims=self.loc_dims,
            )
            for ii in range(len(time_line))
        ]

    def get_environment_correlator(self, corr_type, D0_coef, sigma=0.0):
        """
        Args:
            corr_type: Type of environment correlator = (Delta, Constant, Gaussian)
            D0_coef: Self-correlation magnitude = D_{n,n}
            sigma: Standard deviation if gaussian distribution
        """
        N_matter = self.lvals[0]
        if corr_type == "Delta":
            env_corr = D0_coef * np.eye(N_matter)
        elif corr_type == "Constant":
            env_corr = D0_coef * np.ones([N_matter, N_matter])
        elif corr_type == "Gaussian":
            env_corr = np.zeros([N_matter, N_matter])
            for n in range(N_matter):
                env_corr += np.diag(
                    D0_coef
                    * np.exp(-(n**2) / (2 * (sigma**2)))
                    * np.ones(N_matter - n),
                    n,
                )
            env_corr += env_corr.T - np.diag(np.diag(env_corr))
        else:
            raise ValueError("corr_type must be 'Delta', 'Constant' or 'Gaussian'")
        return env_corr

    def get_sparsity(self):
        sparsity = len(self.row_list) / (self.shape[0] ** 2)
        logger.info(f"SPARSITY: {round(sparsity,16)}")

    # There is an even stronger check (CPTP) which might be useful for sanity
    def check_valid_Liou(self, atol):
        """Check if the Liouvillian preserves the trace"""
        liouDim = self.shape[0]
        hilbertDim = int(np.sqrt(liouDim))
        diag_indices = np.arange(0, liouDim, hilbertDim + 1)
        if self.Liou_type == "Linear":
            self.convert_liouvillian("Sparse")
        trace_check = np.asarray(self.Liou[diag_indices, :].sum(axis=0)).ravel()

        if not np.allclose(trace_check, 0.0, atol=atol):
            raise ValueError("The Liouvillian doesn't preserve the trace")

    def check_dimension(self):
        liouDim = self.shape[0]
        root = int(np.sqrt(liouDim))
        if not root * root == liouDim:
            raise ValueError(
                f"Invalid dimenionality of Liouvillian. Matrix Liouvillian must be a perfect square, {liouDim} is not"
            )


def is_sorted_by_real_desc(complex_array1D):
    real_parts = complex_array1D.real
    return np.all(real_parts[:-1] >= real_parts[1:])  # each element >= next


@njit(cache=True)
def build_csr_numba(N, row_list, col_list, data_list):
    nnz = row_list.shape[0]
    indptr = np.zeros(N + 1, np.int32)
    # 1) count nonzeros per row
    for idx in range(nnz):
        indptr[row_list[idx] + 1] += 1
    # 2) prefix-sum
    for ii in range(1, N + 1):
        indptr[ii] += indptr[ii - 1]
    # 3) scatter into CSR
    indices = np.empty(nnz, np.int32)
    data = np.empty(nnz, data_list.dtype)
    nextpos = indptr[:-1].copy()
    for idx in range(nnz):
        r = row_list[idx]
        p = nextpos[r]
        indices[p] = col_list[idx]
        data[p] = data_list[idx]
        nextpos[r] += 1

    return indptr, indices, data


@njit(cache=True, parallel=True)
def csr_spmv(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, x: np.ndarray
) -> np.ndarray:
    N = indptr.shape[0] - 1
    Hpsi = np.zeros(N, dtype=np.complex128)
    for ii in prange(N):
        value = 0.0 + 0j
        for pp in range(indptr[ii], indptr[ii + 1]):
            value += data[pp] * x[indices[pp]]
        Hpsi[ii] = value
    return Hpsi


@njit(cache=True, parallel=True)
def csr_matmat(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """
    Compute Y = H @ X where H is in CSR form (indptr, indices, data)
    and X is (N, nvec).  Returns Y of shape (N, nvec).

    Parallel over rows of H (and therefore rows of Y).
    """
    N = indptr.shape[0] - 1
    nvec = X.shape[1]
    Y = np.zeros((N, nvec), dtype=np.complex128)

    # outer loop parallelized over the N rows
    for ii in prange(N):
        # for each nonzero in row i
        for pp in range(indptr[ii], indptr[ii + 1]):
            jj = indices[pp]  # column index
            value = data[pp]  # value H[i,j]
            # accumulate v * X[j, :] into Y[i, :]
            for kk in range(nvec):
                Y[ii, kk] += value * X[jj, kk]
    return Y


@njit(cache=True)
def compute_trace_L(
    row_list: np.ndarray, col_list: np.ndarray, data_list: np.ndarray
) -> complex:
    """
    Sum data_list[ii] whenever row_list[ii] == col_list[ii].
    """
    tr = 0.0 + 0.0j
    nnz = row_list.shape[0]
    for idx in range(nnz):
        if row_list[idx] == col_list[idx]:
            tr += data_list[idx]
    return tr


@njit(parallel=True, cache=True)
def exact_time_evolution(  # NOTE: This is a different method than changing to the Liouvillian diagonal basis.
    tline: np.ndarray,  # I am not sure it is still correct since the eigenvectors of a non-hermitian matrix may not be
    eigenvalues: np.ndarray,  # orthogonal. The method where we change basis does work but want to test this method first.
    eigenvectors: np.ndarray,
    initial_state: np.ndarray,
):
    """
    Perform time evolution of a quantum state assuming the knowledge of the whole energy spectrum

    Args:
        tline (ndarray): set of time steps where to measure the time evolution
        eigenvalues (ndarray): Array of eigenvalues of the Liouvillian.
        eigenvectors (ndarray): 2D array of eigenvectors of the Liouvillian (columns are eigenvectors).
        initial_state (ndarray): Initial state vector.

    Returns:
        ndarray: Array of time-evolved states with shape (n_steps, len(initial_state)).
    """
    psi_dim = len(initial_state)
    n_eigs = len(eigenvalues)
    # To store time-evolved states
    n_steps = len(tline)
    psi_time = np.empty((n_steps, psi_dim), np.complex128)
    # Precompute overlaps <E_i|psi(0)> for all i
    overlaps = compute_overlap_with_eigenstates(eigenvectors, initial_state)
    # Loop over time steps in parallel
    for t_idx in prange(n_steps):
        t = tline[t_idx]
        evolved_state = np.zeros(psi_dim, dtype=np.complex128)
        # Loop over eigenstates
        for ii in range(n_eigs):
            # Compute phase factor
            phase = np.exp(-1j * eigenvalues[ii] * t)
            # Accumulate contributions
            evolved_state += phase * overlaps[ii] * eigenvectors[:, ii]
        psi_time[t_idx, :] = evolved_state
    return psi_time


@njit(cache=True, parallel=True)
def compute_overlap_with_eigenstates(eigenvectors: np.ndarray, state: np.ndarray):
    """
    overlaps[i] = <E_i | psi(0)> = sum_j conj(evecs[j,i]) * initial_state[j]
    Parallelized over i.
    """
    psi_dim = eigenvectors.shape[0]
    n_eigs = eigenvectors.shape[1]
    overlaps = np.empty(n_eigs, np.complex128)
    for ii in prange(n_eigs):
        acc = 0.0 + 0.0j
        for jj in range(psi_dim):
            acc += np.conj(eigenvectors[jj, ii]) * state[jj]
        overlaps[ii] = acc
    return overlaps


@njit(parallel=True)
def csr_spmv_lin(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    x: np.ndarray,
    out: np.ndarray,
):
    """
    Parallel CSR sparse-matrix-vector multiply for A = Liou.
    indptr, indices, data define the CSR structure of H.
    The result out = Liou @ x.
    """
    N = indptr.shape[0] - 1
    for ii in prange(N):
        value = 0.0 + 0.0j
        for pp in range(indptr[ii], indptr[ii + 1]):
            value += data[pp] * x[indices[pp]]
        out[ii] = value


@njit(parallel=True, cache=True)
def runge_kutta4_time_evolution(
    initial_state: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    time_line: np.ndarray,
) -> np.ndarray:
    """
    Fully-parallel RK4 integrator for drho/dt = Liourho.
    Uses csr_spmv_lin to compute Liou@rho in parallel.
    Returns array of shape (len(time_line), N) containing rho at each time.
    """
    N = initial_state.shape[0]
    steps = len(time_line)
    dt = time_line[1] - time_line[0]
    # allocate output and temporaries
    out = np.empty((steps, N), np.complex128)
    # ——— ensure rho is complex128 ———
    rho = np.empty(N, np.complex128)
    for ii in prange(N):
        rho[ii] = initial_state[ii]  # copies and *converts* into a complex slot
        out[0, ii] = initial_state[ii]
    k1 = np.empty(N, np.complex128)
    k2 = np.empty(N, np.complex128)
    k3 = np.empty(N, np.complex128)
    k4 = np.empty(N, np.complex128)
    tmp = np.empty(N, np.complex128)
    for s in range(1, steps):
        # k1 = Liou @ rho
        csr_spmv_lin(indptr, indices, data, rho, k1)
        # k2 = Liou @ (rho + dt/2*k1)
        for ii in prange(N):
            tmp[ii] = rho[ii] + 0.5 * dt * k1[ii]
        csr_spmv_lin(indptr, indices, data, tmp, k2)
        # k3 = Liou @ (rho + dt/2*k2)
        for ii in prange(N):
            tmp[ii] = rho[ii] + 0.5 * dt * k2[ii]
        csr_spmv_lin(indptr, indices, data, tmp, k3)
        # k4 = Liou @ (rho + dt*k3)
        for ii in prange(N):
            tmp[ii] = rho[ii] + dt * k3[ii]
        csr_spmv_lin(indptr, indices, data, tmp, k4)
        # combine to advance rho
        for ii in prange(N):
            rho[ii] += dt * (k1[ii] + 2 * k2[ii] + 2 * k3[ii] + k4[ii]) / 6.0
        # renormalize to unit norm
        nrm = get_trace_norm_vec(rho)
        for ii in prange(N):
            rho[ii] /= nrm
        out[s] = rho
    return out


# Differential equation function used for the solve_ivp (which internally uses RK4) method
def rhodot(t, rho, L):
    return L.dot(rho)
