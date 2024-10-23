import numpy as np
from scipy.linalg import eigh as array_eigh
from scipy.sparse.linalg import eigsh as sparse_eigh, expm_multiply, expm
from scipy.sparse import csc_matrix
from ed_lgt.tools import validate_parameters, check_hermitian
from .qmb_state import QMB_state
import logging

logger = logging.getLogger(__name__)

__all__ = ["QMB_hamiltonian"]


class QMB_hamiltonian:
    def __init__(self, Ham, lvals):
        validate_parameters(lvals=lvals)
        self.Ham = Ham
        self.lvals = lvals

    def convert_to_csc(self):
        # Converts the Hamiltonian matrix to compressed sparse column (CSC) format.
        self.Ham = csc_matrix(self.Ham)

    def diagonalize(self, n_eigs, format, loc_dims):
        validate_parameters(op_list=[self.Ham], int_list=[n_eigs], loc_dims=loc_dims)
        self.convert_to_csc()
        # Save local dimensions
        self.loc_dims = loc_dims
        # Save the number or eigenvalues
        self.n_eigs = n_eigs
        # COMPUTE THE LOWEST n_eigs ENERGY VALUES AND THE 1ST EIGENSTATE
        check_hermitian(self.Ham)
        # CHECK HAMILTONIAN SPARSITY
        self.get_sparsity(self.Ham)
        # Diagonalize it
        if format == "sparse":
            logger.info("DIAGONALIZE (sparse) HAMILTONIAN")
            Nenergies, Npsi = sparse_eigh(self.Ham, k=n_eigs, which="SA")
        else:
            logger.info("DIAGONALIZE (standard) HAMILTONIAN")
            Nenergies, Npsi = array_eigh(self.Ham.toarray(), eigvals=(0, n_eigs - 1))
        # Check and sort energies and corresponding eigenstates if necessary
        if not is_sorted(Nenergies):
            order = np.argsort(Nenergies)
            Nenergies = Nenergies[order]
            Npsi = Npsi[:, order]
        # Save the eigenstates as QMB_states
        self.Nenergies = Nenergies
        self.Npsi = [
            QMB_state(Npsi[:, ii], self.lvals, self.loc_dims) for ii in range(n_eigs)
        ]
        # Save GROUND STATE PROPERTIES
        self.GSenergy = self.Nenergies[0]
        self.GSpsi = self.Npsi[0]

    def print_energy(self, en_state):
        logger.info("====================================================")
        logger.info(f"{en_state} ENERGY: {round(self.Nenergies[en_state],9)}")

    def time_evolution(self, initial_state, start, stop, n_steps, loc_dims):
        self.convert_to_csc()
        # Save local dimensions
        self.loc_dims = loc_dims
        # Compute the time spacing
        delta_n = (stop - start) / n_steps
        logger.info(f"------- TIME EVOLUTION: DELTA {round(delta_n,2)} -----------")
        # Compute the evolved psi at each time step
        psi_time = expm_multiply(
            A=complex(0, -1) * self.Ham,
            B=initial_state,
            start=start,
            stop=stop,
            num=n_steps,
            endpoint=True,
        )
        # Save them as QMB_states
        self.psi_time = [
            QMB_state(psi_time[ii, :], self.lvals, self.loc_dims)
            for ii in range(n_steps)
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
            self.convert_to_csc()
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

            threshold (float, optional): The convergence threshold for the Newton-Raphson iteration. Defaults to 1e-10.

            max_iter (int, optional): Maximum number of iterations to prevent infinite loops. Defaults to 1000.

        Returns:
            float: The estimated beta that minimizes the free energy difference for the given state.

        Raises:
            ValueError: If the Newton-Raphson method fails to converge within the maximum number of iterations.

        Notes:
            This method relies on the derivative of the free energy, computed as F', and adjusts beta using the formula:
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

    @staticmethod
    def get_sparsity(array):
        # MEASURE SPARSITY
        num_nonzero = array.nnz
        # Total number of elements
        total_elements = array.shape[0] * array.shape[1]
        # Calculate sparsity
        sparsity = 1 - (num_nonzero / total_elements)
        logger.info(f"SPARSITY: {round(sparsity*100,3)}%")


def is_sorted(array1D):
    return np.all(array1D[:-1] <= array1D[1:])


def get_sorted_indices(data):
    abs_data = np.abs(data)
    real_data = np.real(data)
    # Lexsort by real part first (secondary key), then by absolute value (primary key)
    sorted_indices = np.lexsort((real_data, abs_data))
    return sorted_indices[::-1]  # Descending order
