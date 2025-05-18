import numpy as np
from numba import njit, prange
from scipy.linalg import eigh as array_eigh
from scipy.sparse.linalg import eigsh as sparse_eigh, expm_multiply, expm
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csc_matrix, csr_matrix, isspmatrix, coo_matrix
from ed_lgt.tools import validate_parameters, check_hermitian
from .qmb_state import QMB_state
import logging

logger = logging.getLogger(__name__)

__all__ = ["QMB_hamiltonian", "get_entropy_partition"]


class QMB_hamiltonian:
    def __init__(self, lvals, size):
        """
        Initialize the Hamiltonian class.

        Args:
            Ham: The Hamiltonian, which can be a dense NumPy array, a sparse matrix (csc_matrix),
                 or a LinearOperator.
            lvals: Additional lattice values or parameters.
        """
        validate_parameters(lvals=lvals)
        self.lvals = lvals
        self.shape = (size, size)
        # Initialize row_list, col_list, value_list as empty arrays
        self.row_list = np.array([], dtype=int)
        self.col_list = np.array([], dtype=int)
        self.value_list = np.array([], dtype=np.complex128)

    def add_term(self, term):
        """
        Add a term to the Hamiltonian.

        Args:
            term: The term to add, which can be:
                  - A dense NumPy array (np.ndarray).
                  - A sparse matrix (scipy.sparse.csc_matrix or csr_matrix).
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
        Construct the Hamiltonian as a sparse matrix or LinearOperator.

        Args:
            format (str): Target type ('dense', 'sparse', or 'linear').
        """

        # ==========================================================================
        def matvec(psi):
            """Function to compute H @ v."""
            result = hamiltonian_vector_product(
                psi, self.row_list, self.col_list, self.value_list
            )
            if result is None:
                raise ValueError("matvec returned None.")
            if not isinstance(result, np.ndarray):
                raise TypeError(f"matvec must return np.ndarray, got {type(result)}.")
            return result

        # ==========================================================================
        def matmat(psis):
            """Function to compute H @ matrix (batch of vectors)."""
            result = np.column_stack(
                [
                    hamiltonian_vector_product(
                        psi, self.row_list, self.col_list, self.value_list
                    )
                    for psi in psis.T
                ]
            )
            if result is None:
                raise ValueError("matmat returned None.")
            if not isinstance(result, np.ndarray):
                raise TypeError(f"matmat must return np.ndarray, got {type(result)}.")
            return result

        # ==========================================================================
        if format == "sparse":
            self.Ham = csc_matrix(
                (self.value_list, (self.row_list, self.col_list)), shape=self.shape
            )
            # Check if the matrix is Hermitian
            check_hermitian(self.Ham)
            # Measure sparsity
            self.get_sparsity(self.Ham)
        # ==========================================================================
        elif format == "linear":
            self.Ham = LinearOperator(shape=self.shape, matvec=matvec, matmat=matmat)
        # ==========================================================================
        elif format == "dense":
            self.Ham = csc_matrix(
                (self.value_list, (self.row_list, self.col_list)), shape=self.shape
            )
            self.Ham = self.Ham.toarray()
        self.Ham_type = format

    def convert_hamiltonian(self, format):
        """
        Convert the Hamiltonian to the specified representation type.

        Args:
            format (str): Target type ('dense', 'sparse', or 'linear').
        """
        logger.info(f"CONVERT HAMILTONIAN from {self.Ham_type} to {format}")
        if format == self.Ham_type:
            return  # Already in the desired format
        elif format == "dense":
            if self.Ham_type == "sparse":
                self.Ham = self.Ham.toarray()
            elif self.Ham_type == "linear":
                self.build("dense")
        elif format == "sparse":
            if self.Ham_type == "dense":
                self.Ham = csc_matrix(self.Ham)
            elif self.Ham_type == "linear":
                self.build("sparse")
        elif format == "linear":
            self.build("linear")
        else:
            msg = f"Invalid format: {format}. Use 'dense', 'sparse', or 'linear'."
            raise ValueError(msg)
        self.Ham_type = format

    def diagonalize(self, n_eigs, format, loc_dims):
        # Ensure Hamiltonian is Hermitian and check sparsity
        # check_hermitian(self.Ham)
        validate_parameters(loc_dims=loc_dims)
        if format != self.Ham_type:
            self.convert_hamiltonian(format)
        # Save local dimensions
        self.loc_dims = loc_dims
        # Determine number of eigenvalues to compute
        if isinstance(n_eigs, int):
            if n_eigs > self.shape[0]:
                msg = f"n_eigs must be smaller than H.shape[0]={self.shape[0]}, got {n_eigs}"
                raise ValueError(msg)
            self.n_eigs = n_eigs
        elif isinstance(n_eigs, str) and n_eigs == "full":
            self.n_eigs = self.shape[0]
        else:
            raise ValueError(f"n_eigs must be an integer or 'full', not {n_eigs}.")
        # Select diagonalization format
        if self.Ham_type == "dense" or self.n_eigs == "full":
            logger.info("DIAGONALIZE (dense) HAMILTONIAN")
            Nenergies, Npsi = array_eigh(self.Ham)
        elif (
            self.Ham_type == "sparse"
            or self.Ham_type == "linear"
            and isinstance(n_eigs, int)
        ):
            logger.info(f"DIAGONALIZE {self.Ham_type} HAMILTONIAN")
            Nenergies, Npsi = sparse_eigh(self.Ham, k=self.n_eigs, which="SA")
        # Sort and save eigenvalues/eigenstates
        if not is_sorted(Nenergies):
            order = np.argsort(Nenergies)
            Nenergies, Npsi = Nenergies[order], Npsi[:, order]
        # Save the eigenstates as QMB_states
        self.Nenergies = Nenergies
        self.Npsi = [
            QMB_state(Npsi[:, ii], self.lvals, self.loc_dims)
            for ii in range(self.n_eigs)
        ]
        # Save GROUND STATE PROPERTIES
        self.GSenergy = self.Nenergies[0]
        self.GSpsi = self.Npsi[0]

    def time_evolution(
        self,
        initial_state: np.ndarray,
        time_line: np.ndarray,
        loc_dims: np.ndarray,
    ):
        """
        Perform time evolution using sparse method (for sparse / linear operators).
        """
        # Save local dimensions
        self.loc_dims = loc_dims
        # Compute the time spacing and the time line
        delta_t = time_line[1] - time_line[0]
        msg = f"------------- TIME EVOLUTION: DELTA {round(delta_t,2)} ------------"
        logger.info(msg)
        # Check Hamiltonian type and compute the time evolution
        if self.Ham_type == "dense":
            # Check if Hamiltonian is already diagonalized
            if not hasattr(self, "Nenergies") or not hasattr(self, "Npsi"):
                self.diagonalize(n_eigs="full", format="dense", loc_dims=loc_dims)
            # Run the time evolution with numba
            psi_time, self.Deff = time_evolve_numba(
                time_line,
                self.Nenergies,
                np.array([A.psi for A in self.Npsi]).T,
                initial_state.astype(np.complex128),
            )
            self.Deff
            logger.info(f"Eff. Hilbert Space dim: {self.Deff}")
        # Compute the trace if Ham is a LinearOperator
        elif self.Ham_type in ["linear", "sparse"]:
            if self.Ham_type == "linear":
                logger.warning("Falling back to 'sparse' format for time evolution.")
                self.convert_hamiltonian("sparse")
                """
                self.Ham = scalar_multiply_linear_operator(-1j, self.Ham)
                # Identify diagonal elements
                diagonal_mask = self.row_list == self.col_list
                # Sum the diagonal elements
                traceA = np.sum(-1j * self.value_list[diagonal_mask])
                A = self.Ham
                """
            # Sparse matrices have their trace natively
            traceA = None
            A = complex(0, -1) * self.Ham
            # Compute the evolved psi at each time step
            psi_time = expm_multiply(
                A=A,
                B=initial_state,
                start=time_line[0],
                stop=time_line[-1],
                num=len(time_line),
                endpoint=True,
                traceA=traceA,
            )
        # Save them as QMB_states
        self.psi_time = [
            QMB_state(psi_time[ii, :], self.lvals, self.loc_dims)
            for ii in range(len(time_line))
        ]

    def partition_function(self, beta):
        """
        Computes the partition function Z for a given inverse temperature beta.

        Args:
            beta (float): Inverse temperature.

        Returns:
            float: The computed partition function.
        """
        if self.n_eigs == self.Ham.shape[0]:
            Z = np.sum(np.exp(-beta * self.Nenergies))
        else:
            Z = np.real(csc_matrix(expm(-beta * self.Ham)).trace())
        if Z <= 0:
            raise ValueError(f"Z must be positive, not {Z}.")
        return Z

    def free_energy(self, beta):
        """
        Calculates the free energy F of the system at a specified inverse temperature beta.

        Args:
            beta (float): Inverse temperature.

        Returns:
            float: The computed free energy.
        """
        Z = self.partition_function(beta)
        F = -1 / beta * np.log(Z)
        return F

    def thermal_average(self, beta):
        """
        Calculates the thermal average of the Hamiltonian at a given inverse temperature beta.

        Args:
            beta (float): Inverse temperature.

        Returns:
            float: The thermal average of the energy.
        """
        Z = self.partition_function(beta)
        if self.n_eigs == self.Ham.shape[0]:
            return self.Nenergies.dot(np.exp(-beta * self.Nenergies))
        else:
            return np.real(self.Ham.dot(expm(-beta * self.Ham)).trace()) / Z

    def F_prime(self, beta):
        Z = self.partition_function(beta)
        if self.n_eigs == self.Ham.shape[0]:
            exph2 = np.square(self.Nenergies).dot(np.exp(-beta * self.Nenergies)) / Z
        else:
            exph2 = np.real((self.Ham**2).dot(expm(-beta * self.Ham)).trace()) / Z
        return -exph2 - self.thermal_average(beta) ** 2

    def get_beta(self, state, threshold=1e-10, max_iter=1000):
        """
        Uses the Newton-Raphson method to estimate the inverse temperature beta that minimizes
        the free energy difference for a given quantum state.
        The function iteratively adjusts beta based on the gradient of the free energy
        difference until it converges to a minimum.

        Args:
            state (np.ndarray): The quantum state for which to optimize beta.

            threshold (float, optional): Convergence threshold for the Newton-Raphson iteration. Defaults to 1e-10.

            max_iter (int, optional): Maximum number of iterations to prevent infinite loops. Defaults to 1000.

        Returns:
            float: The estimated beta that minimizes the free energy difference for the given state.

        Raises:
            ValueError: If the Newton-Raphson method fails to converge within the maximum number of iterations.

        Notes:
            This method relies on the derivative of the free energy, computed as F',
            It adjusts beta using the formula:
            beta_new = beta_old - (F(beta_old) / F'(beta))
            where F is the difference in free energy and F' is its derivative with respect to beta.
        """
        iter_count = 0
        accuracy = 1
        beta = 1e-7
        logger.info(f"=========== GET BETA ===============")
        # Get the reference energy value for the chosen state
        state_energy = QMB_state(state).expectation_value(self.Ham)
        while accuracy > threshold and iter_count < max_iter:
            iter_count += 1
            if self.n_eigs == self.Ham.shape[0]:
                Z = self.partition_function(beta)
                E = self.Nenergies.dot(np.exp(-beta * self.Nenergies)) / Z
                F = E - state_energy
                E2 = np.square(self.Nenergies).dot(np.exp(-beta * self.Nenergies)) / Z
                Fp = -E2 - (E**2)
            else:
                expH = csc_matrix(expm(-beta * self.Ham))
                Z = np.real(expH.trace())
                E = np.real(self.Ham.dot(expH).trace()) / Z
                F = E - state_energy
                Fp = -np.real((self.Ham**2).dot(expH).trace()) / Z - E**2
            # ==================================================
            prevVal = beta
            beta = beta - F / Fp
            accuracy = abs(beta - prevVal)
            logger.info(f"------------------------------------")
            logger.info(f"F={F}")
            logger.info(f"Fp={Fp}")
            logger.info(f"beta={beta}")
            logger.info(f"accuracy={accuracy}")
        if iter_count >= max_iter:
            logger.warning("Maximum iterations reached without convergence.")
        logger.info(f"BETA {beta}")
        return beta

    def get_r_value(self):
        """
        Compute the r-value for a list of energies.

        Args:
            energies (np.ndarray): Array of eigenvalues.

        Returns:
            r_value: average r value
        """
        # Check Hamiltonian type and compute the time evolution
        if self.Ham_type == "dense":
            # Check if Hamiltonian is already diagonalized
            if not hasattr(self, "Nenergies") or not hasattr(self, "Npsi"):
                return ValueError("The hamiltonian has to be fully diagonalized")
        else:
            return ValueError("The hamiltonian has to be fully diagonalized")
        # Compute level spacings
        delta_E = np.diff(np.sort(self.Nenergies))
        # Compute r-values
        r_array = np.array(
            [
                min(delta_E[ii], delta_E[ii + 1]) / max(delta_E[ii], delta_E[ii + 1])
                for ii in range(len(delta_E) - 1)
            ]
        )
        # Define the bulk range
        bulk_start = int(0.1 * len(r_array))
        bulk_end = int(0.9 * len(r_array))
        # Compute the average r-value for the bulk
        self.r_value = np.mean(r_array[bulk_start:bulk_end])
        logger.info(f"R value {self.r_value}")
        return r_array

    def print_energy(self, en_state):
        logger.info("====================================================")
        logger.info(f"{en_state} ENERGY: {round(self.Nenergies[en_state],9)}")

    @staticmethod
    def get_sparsity(array: csr_matrix):
        # MEASURE SPARSITY
        num_nonzero = array.nnz
        # Total number of elements
        total_elements = array.shape[0] * array.shape[1]
        # Calculate sparsity
        sparsity = 1 - (num_nonzero / total_elements)
        logger.info(f"SPARSITY: {round(sparsity*100,5)}%")


def is_sorted(array1D):
    return np.all(array1D[:-1] <= array1D[1:])


def get_sorted_indices(data):
    abs_data = np.abs(data)
    real_data = np.real(data)
    # Lexsort by real part first (secondary key), then by absolute value (primary key)
    sorted_indices = np.lexsort((real_data, abs_data))
    return sorted_indices[::-1]  # Descending order


def get_entropy_partition(lvals, option="half"):
    if option == "half":
        if len(lvals) == 1:
            partition_indices = list(np.arange(0, int(lvals[0] / 2), 1))
        else:
            partition_indices = list(np.arange(0, int(lvals[0] / 2), 1)) + list(
                np.arange(lvals[0], lvals[0] + int(lvals[0] / 2), 1)
            )
    else:
        raise NotImplementedError("for the moment it works only for option=half")
    return partition_indices


@njit(cache=True)
def hamiltonian_vector_product(psi, row_list, col_list, value_list):
    """
    Compute the Hamiltonian-vector product H @ psi directly from the nonzero elements
    of the Hamiltonian without constructing the full matrix.

    Args:
        psi (np.ndarray): The vector to multiply with the Hamiltonian.
        row_list (np.ndarray): Row indices of nonzero elements in the operator.
        col_list (np.ndarray): Column indices of nonzero elements in the operator.
        value_list (np.ndarray): Nonzero values of the operator.

    Returns:
        np.ndarray: The resulting vector H @ psi.
    """
    Hpsi = np.zeros(len(psi), dtype=np.complex128)
    for idx in range(len(value_list)):
        row = row_list[idx]
        col = col_list[idx]
        value = value_list[idx]
        Hpsi[row] += value * psi[col]
    return Hpsi


def scalar_multiply_linear_operator(scalar, linear_operator: LinearOperator):
    """
    Multiply a LinearOperator by a scalar.

    Args:
        scalar (complex): The scalar to multiply.
        linear_operator (LinearOperator): The operator to multiply.

    Returns:
        LinearOperator: The scaled operator.
    """

    def matvec(psi):
        return scalar * linear_operator.matvec(psi)

    def matmat(psis):
        return scalar * linear_operator.matmat(psis)

    return LinearOperator(
        shape=linear_operator.shape,
        matvec=matvec,
        matmat=matmat,
    )


@njit(parallel=True, cache=True)
def time_evolve_numba(
    tline: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    initial_state: np.ndarray,
):
    """
    Perform time evolution of a quantum state using Numba for optimization.

    Args:
        tline (ndarray): set of time steps where to measure the time evolution
        eigenvalues (ndarray): Array of eigenvalues of the Hamiltonian.
        eigenvectors (ndarray): 2D array of eigenvectors of the Hamiltonian (columns are eigenvectors).
        initial_state (ndarray): Initial state vector.

    Returns:
        ndarray: Array of time-evolved states with shape (n_steps, len(initial_state)).
    """
    psi_dim = len(initial_state)
    # To store time-evolved states
    n_steps = len(tline)
    psi_time = np.zeros((n_steps, psi_dim), dtype=np.complex128)
    # Precompute overlaps <E_i|psi(0)>
    overlaps = np.dot(eigenvectors.T.conj(), initial_state)
    # Loop over time steps in parallel
    for t_idx in prange(n_steps):
        t = tline[t_idx]
        evolved_state = np.zeros(psi_dim, dtype=np.complex128)
        # Loop over eigenstates
        for ii in range(len(eigenvalues)):
            # Compute phase factor
            phase = np.exp(-1j * eigenvalues[ii] * t)
            # Accumulate contributions
            evolved_state += phase * overlaps[ii] * eigenvectors[:, ii]
        psi_time[t_idx, :] = evolved_state
    # Measure the effective Hilbert space dimension
    Deff = np.sum(np.abs(overlaps) ** 4)
    return psi_time, Deff
