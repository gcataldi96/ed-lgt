import numpy as np
from numba import njit, prange
from .generate_configs import get_translated_state_indices

__all__ = [
    "check_normalization",
    "check_orthogonality",
    "momentum_basis_k0",
    "momentum_basis",
    "momentum_basis_k0_par",
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


@njit(cache=True, parallel=True)
def momentum_basis_k0_par(sector_configs: np.ndarray, logical_unit_size: int):
    """
    Compute the momentum basis (k=0) in parallel for a given symmetry sector.

    Args:
        sector_configs (np.ndarray): Array of unique sector configurations.
        logical_unit_size (int): Size of the logical unit for translation symmetry.

    Returns:
        basis (np.ndarray): Momentum basis matrix for k=0.
    """
    # Initialize dimensions and arrays
    sector_dim = sector_configs.shape[0]
    num_translations = sector_configs.shape[1] // logical_unit_size
    normalization = np.zeros(sector_dim, dtype=np.int32)
    independent_indices = np.zeros(sector_dim, dtype=np.bool_)
    all_trans_indices = np.zeros((sector_dim, num_translations), dtype=np.int32)

    # Step 1: Precompute all translations in parallel
    for ii in prange(sector_dim):
        config = sector_configs[ii]
        all_trans_indices[ii] = get_translated_state_indices(
            config, sector_configs, logical_unit_size
        )

    # Step 2: Mark independent indices sequentially
    for ii in range(sector_dim):
        trans_indices = all_trans_indices[ii]
        is_independent = True
        for jj in range(ii):  # Sequential check for independence
            if independent_indices[jj] and jj in trans_indices:
                is_independent = False
                break
        if is_independent:
            independent_indices[ii] = True
            # The norm for the state is the number of unique translations
            normalization[ii] = len(np.unique(trans_indices))

    # Step 3: Parallelize basis construction
    ref_indices = np.flatnonzero(independent_indices)  # Extract independent indices
    norm = normalization[ref_indices]
    basis = np.zeros((sector_dim, len(ref_indices)), dtype=np.float64)

    for ii in prange(len(ref_indices)):  # Parallelized
        ind_index = ref_indices[ii]
        trans_indices = all_trans_indices[ind_index]
        for jj in range(norm[ii]):
            basis[trans_indices[jj], ii] = 1 / np.sqrt(norm[ii])

    # Optional: Uncomment for debugging
    # if not check_normalization(basis) or not check_orthogonality(basis):
    #     raise ValueError("Basis normalization or orthogonality failed.")

    return basis


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


@njit(cache=True, parallel=True)
def momentum_basis_par(sector_configs: np.ndarray, logical_unit_size: int, k: int):
    """
    Compute the momentum basis for a general momentum value k
    using translational symmetry with a given logical unit size.
    This version is parallelized with Numba.

    Args:
        sector_configs (np.ndarray): Array of configurations (shape = [hilbert_dim, N]).
        logical_unit_size (int): The number of sites (logical unit) over which the translational symmetry applies.
        k (int): Momentum index. k=0 produces a real basis, while k != 0 yields a complex basis.

    Returns:
        basis (np.ndarray): Momentum basis matrix.
            -- For k==0, the dtype is np.float64.
            -- For k!=0, the dtype is np.complex128.
    """
    sector_dim = sector_configs.shape[0]
    # Determine how many translations there are (number of logical units in the chain)
    num_translations = sector_configs.shape[1] // logical_unit_size
    # Store the number of unique translations per config
    normalization = np.zeros(sector_dim, dtype=np.int32)
    # Mark which configs are independent
    independent_indices = np.zeros(sector_dim, dtype=np.bool_)
    # Store translation indices for each config
    all_trans_indices = np.zeros((sector_dim, num_translations), dtype=np.int32)

    # Step 1: Precompute all translation indices in parallel.
    # For each configuration, obtain its set of translation indices.
    for ii in prange(sector_dim):
        # Here we call the helper that uses np.roll to compute the translated configuration
        all_trans_indices[ii] = get_translated_state_indices(
            sector_configs[ii], sector_configs, logical_unit_size
        )

    # Step 2: Sequentially mark independent configurations.
    # A configuration is independent if none of the previously marked independent configs
    # appears in its set of translation indices.
    for ii in range(sector_dim):
        trans_indices = all_trans_indices[ii]
        is_independent = True
        for jj in range(ii):
            if independent_indices[jj]:
                # Check (manually) if jj is among the translation indices for config ii.
                for tt in range(num_translations):
                    if trans_indices[tt] == jj:
                        is_independent = False
                        break
                if not is_independent:
                    break
        if is_independent:
            independent_indices[ii] = True
            # Compute the number of unique translations.
            count = 0
            for j in range(num_translations):
                unique = True
                for prev in range(j):
                    if trans_indices[j] == trans_indices[prev]:
                        unique = False
                        break
                if unique:
                    count += 1
            normalization[ii] = count

    # Step 3: Build the list of independent (reference) indices.
    # (Since Numba’s support for np.flatnonzero is limited, we build it manually.)
    num_refs = 0
    for i in range(sector_dim):
        if independent_indices[i]:
            num_refs += 1
    ref_indices = np.empty(num_refs, dtype=np.int32)
    ptr = 0
    for i in range(sector_dim):
        if independent_indices[i]:
            ref_indices[ptr] = i
            ptr += 1

    # Determine the dtype of the basis based on k.
    if k == 0:
        basis = np.zeros((sector_dim, num_refs), dtype=np.float64)
    else:
        basis = np.zeros((sector_dim, num_refs), dtype=np.complex128)

    # Step 4: Construct the momentum basis for each independent state in parallel.
    # For each reference configuration, its "orbit" under translation (with R = normalization[ref])
    # gives the corresponding momentum state. For k != 0 add a phase factor for each translated copy.
    for ref_ptr in prange(num_refs):
        ref = ref_indices[ref_ptr]
        R = normalization[ref]  # number of unique translations for this state
        norm_val = np.sqrt(R)
        # It is assumed that in get_translated_state_indices the first R elements
        # correspond to the unique translated configurations.
        for j in range(R):
            idx = all_trans_indices[ref, j]
            if k == 0:
                basis[idx, ref_ptr] = 1.0 / norm_val
            else:
                phase = np.exp(-1j * 2 * np.pi * k * j / R) / norm_val
                basis[idx, ref_ptr] = phase

    return basis
