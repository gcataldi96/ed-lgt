import numpy as np
from math import prod
from scipy.linalg import eigh as array_eigh, svd
from scipy.sparse.linalg import eigsh as sparse_eigh, expm_multiply, expm
from scipy.sparse import csr_matrix, isspmatrix, csr_array, csc_matrix
from ed_lgt.tools import validate_parameters, check_hermitian
from ed_lgt.symmetries import (
    separate_configs,
    config_to_index_binarysearch,
    index_to_config,
)
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "QMB_hamiltonian",
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

    def expectation_value(self, operator):
        """
        Calculates the expectation value of the given operator with the current quantum state.

        Args:
            operator (np.ndarray or sparse_matrix): The operator to apply.

        Returns:
            float: The real part of the expectation value.
        """
        validate_parameters(op_list=[operator])
        return np.real(np.dot(np.conjugate(self.psi), (operator.dot(self.psi))))

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
            subsystem_configs, environment_configs = separate_configs(
                sector_configs, keep_indices
            )
            # Find unique environment configurations
            unique_env_configs = np.unique(environment_configs, axis=0)
            # Initialize the RDM whose dimension= # Unique subsystem configurations
            unique_subsys_configs = np.unique(subsystem_configs, axis=0)
            subsystem_dim = unique_subsys_configs.shape[0]
            RDM = np.zeros((subsystem_dim, subsystem_dim), dtype=np.complex128)
            # Iterate over unique environment configurations
            for env_config in unique_env_configs:
                # Find indices where the environment matches env_config
                matching_indices = (environment_configs == env_config).all(axis=1)
                # Create the subsystem wavefunction slice
                subsystem_psi = np.zeros(subsystem_dim, dtype=np.complex128)
                for idx, match in enumerate(matching_indices):
                    if match:
                        subsys_config = tuple(subsystem_configs[idx])
                        subsys_index = config_to_index_binarysearch(
                            subsys_config, unique_subsys_configs
                        )
                        subsystem_psi[subsys_index] = self.psi[idx]
                # Compute the RDM for this slice and add to the final RDM
                RDM += np.tensordot(subsystem_psi, subsystem_psi.conj(), axes=0)
            return RDM
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
        logger.info("----------------------------------------------------")


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
