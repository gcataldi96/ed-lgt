import numpy as np
from numba import njit
from .generate_configs import get_translated_state_indices

__all__ = [
    "check_normalization",
    "check_orthogonality",
    "momentum_basis_k0",
    "momentum_basis",
]


@njit
def check_normalization(basis):
    for ii in range(basis.shape[1]):
        if not np.isclose(np.linalg.norm(basis[:, ii]), 1):
            return False
    return True


@njit
def check_orthogonality(basis):
    for ii in range(basis.shape[1]):
        for jj in range(ii + 1, basis.shape[1]):
            if not np.isclose(np.vdot(basis[:, ii], basis[:, jj]), 0, atol=1e-10):
                return False
    return True


@njit(cache=True)
def momentum_basis_k0(sector_configs, logical_unit_size):
    sector_dim = sector_configs.shape[0]
    normalization = np.zeros(sector_dim, dtype=np.int32)
    independent_indices = np.zeros(sector_dim, dtype=np.bool_)

    for ii in range(sector_dim):
        config = sector_configs[ii]
        # Compute all the set of translated configurations in terms of indices
        trans_indices = get_translated_state_indices(
            config, sector_configs, logical_unit_size
        )
        is_independent = True
        # Check this configuration against all previously marked independent configurations
        for jj in range(ii):
            if independent_indices[jj] and jj in trans_indices:
                is_independent = False
                break
        if is_independent:
            independent_indices[ii] = True
            # The norm for the state is the number of unique translations
            normalization[ii] = len(np.unique(trans_indices))
    # Obtain the reference configurations (their indices and norms) to build the momentum basis
    ref_indices = np.flatnonzero(independent_indices)
    norm = normalization[ref_indices]
    # Define the basis
    basis = np.zeros((sector_dim, len(ref_indices)), dtype=np.float64)

    for ii in range(len(ref_indices)):
        ind_index = ref_indices[ii]
        trans_indices = get_translated_state_indices(
            sector_configs[ind_index], sector_configs, logical_unit_size
        )
        for jj in range(norm[ii]):
            basis[trans_indices[jj], ii] = 1 / np.sqrt(norm[ii])
    # if not check_normalization(basis) or not check_orthogonality(basis):
    #    raise ValueError("Basis normalization or orthogonality failed.")
    return basis


def momentum_basis(sector_configs, k=0):
    sector_dim = len(sector_configs)
    sector_indices = np.arange(sector_dim, dtype=int)
    normalization = np.zeros(sector_dim, dtype=int)
    independent_indices = np.zeros(sector_dim, dtype=bool)
    for ii, config in enumerate(sector_configs):
        # Compute all the set of translated configurations in terms of indices
        trans_indices = get_translated_state_indices(config, sector_configs)
        # Bool variable to check if config is an independent configuration
        is_independent = True
        if ii > 0:
            """
            Run over all the already found independent configurations and look at their corresponding indices.
            If any of them is included in the new candidate's translated configurations,
            then the candidate cannot be independent, and I want to go on and check the next candidate.
            On the contrary, if this if statement is never satisfied (that is for all the already independent configurations),
            after the for loop, I will save the candidate as a new independent configuration.
            """
            # pause(f"----------{ii}-------------------", True)
            # print("Sector indices", sector_indices[independent_indices])
            for ind_index in sector_indices[independent_indices]:
                # print("Check independence")
                # print(ind_index, trans_indices)
                if ind_index in trans_indices:
                    is_independent = False
                    break
        if is_independent:
            independent_indices[ii] = True
            # Acquire the periodicity R of the configuration under translation
            R = len(np.unique(trans_indices))
            # Get the norm of the momentum state
            normalization[ii] = R

    # Obtain the reference configurations (their indices and norms) to build the momentum basis
    ref_indices = sector_indices[independent_indices]
    ref_configs = sector_configs[independent_indices, :]
    norm = normalization[independent_indices]
    momentum_basis_dim = len(ref_indices)
    # Once we have the set of reference configs, construct the momentum basis
    basis_dtype = np.complex128 if k != 0 else np.float64
    basis = np.zeros((sector_dim, momentum_basis_dim), dtype=basis_dtype)
    for ii, ind_index in enumerate(ref_indices):
        # For any reference configs, get all its translated versions (and indices)
        trans_indices = get_translated_state_indices(ref_configs[ii], sector_configs)
        # Create the momentum state associated with this reference state
        if k != 0:
            for jj in range(norm[ii]):
                phase_factor = np.exp(-1j * 2 * np.pi * k * jj / norm[ii])
                basis[trans_indices[jj], ii] = phase_factor / np.sqrt(norm[ii])
        else:
            for jj in range(norm[ii]):
                basis[trans_indices[jj], ii] = 1 / np.sqrt(norm[ii])

    check_normalization(basis)
    check_orthogonality(basis)
    return basis


@njit
def nbody_operator_data(op_list, op_sites_list, sector_configs, momentum_basis, k):
    sector_dim = sector_configs.shape[0]
    # Calculate matrix dimensions based on the momentum basis
    basis_dim = momentum_basis.shape[1]
    row_list = []
    col_list = []
    value_list = []

    for row in range(basis_dim):
        for col in range(basis_dim):
            element = 0
            for ii, site in enumerate(op_sites_list):
                op = op_list[ii, site]
                # Calculate phase difference due to position 'site' in the momentum basis
                phase_diff = np.exp(-1j * 2 * np.pi * k * site / sector_dim)

                # Applying operator with phase consideration
                for config_index in range(sector_dim):
                    if (
                        sector_configs[config_index, site] == site
                    ):  # condition to apply operator
                        transition_amplitude = op[
                            sector_configs[row, site], sector_configs[col, site]
                        ]
                        # Include phase factor in transition amplitude
                        element += (
                            momentum_basis[config_index, row].conj()
                            * transition_amplitude
                            * phase_diff
                            * momentum_basis[config_index, col]
                        )

            if not np.isclose(element, 0, atol=1e-10):
                row_list.append(np.int32(row))
                col_list.append(np.int32(col))
                value_list.append(element)

    return np.array(row_list), np.array(col_list), np.array(value_list)
