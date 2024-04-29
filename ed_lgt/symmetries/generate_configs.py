from ed_lgt.tools import get_time
import numpy as np
from numba import njit, prange


__all__ = [
    "index_to_config",
    "config_to_index",
    "get_state_configs",
    "compare_configs",
    "config_to_index_linearsearch",
    "config_to_index_binarysearch",
    "separate_configs",
    "get_translated_state_indices",
    "get_reference_indices",
]


@njit
def index_to_config(qmb_index, loc_dims):
    """
    Convert a linear QMB index to a configuration based on local dimensions.
    """
    num_sites = len(loc_dims)
    config = np.zeros(num_sites, dtype=np.uint8)
    for site_index in range(num_sites):
        dim = loc_dims[site_index]
        config[site_index] = qmb_index % dim
        qmb_index //= dim
    return config


@njit
def config_to_index(config, loc_dims):
    """
    This function generate the QMB index out the the indices of the single lattices sites.
    The latter ones can display local Hilbert space with different dimension.
    The order of the sites must match the order of the dimensionality of the local basis
    Args:
        loc_states (list of ints): list of numbered state of the lattice sites

        loc_dims (list of ints, np.ndarray of ints, or int): list of lattice site dimensions
            (in the same order as they are stored in the loc_states!)

    Returns:
        int: QMB index
    """
    qmb_index = 0
    multiplier = 1
    for site_index in reversed(range(len(config))):
        qmb_index += config[site_index] * multiplier
        multiplier *= loc_dims[site_index]
    return qmb_index


@get_time
@njit(parallel=True)
def get_state_configs(loc_dims):
    """
    This function creates all the possible QMB state configurations of a system made
    by L units/sites, each one living in a Hilbert space of dimension loc_dim.

    Args:
        loc_dims (np.array): 1D array of single-site local dimensions.
            For Exact Diagonlization (ED) purposes, each local dimension is always smaller that 2^{8}-1
            For this reason, loc_dims.dtype = np.uint8

    Returns:
        np.array: matrix configs, with shape=(prod(loc_dims), len(loc_dims)) and dtype = np.uint8
            Each row is a possible configuration of the QMB state.
    """
    # Total number of configs
    total_configs = 1
    for dim in loc_dims:
        total_configs *= dim
    # Len of each config
    num_dims = len(loc_dims)
    configs = np.zeros((total_configs, num_dims), dtype=np.uint8)
    # Iterate over all the possible configs
    for ii in prange(total_configs):
        tmp = ii
        for dim_index in range(num_dims):
            divisor = (
                np.prod(loc_dims[dim_index + 1 :]) if dim_index + 1 < num_dims else 1
            )
            configs[ii, dim_index] = (tmp // divisor) % loc_dims[dim_index]
    return configs


def separate_configs(sector_configs, keep_indices):
    # Indices for the environment
    env_indices = [i for i in range(sector_configs.shape[1]) if i not in keep_indices]
    # Separate the subsystem and environment configurations
    subsystem_configs = sector_configs[:, keep_indices]
    environment_configs = sector_configs[:, env_indices]
    return subsystem_configs, environment_configs


@get_time
@njit
def config_to_index_linearsearch(config, unique_configs):
    # Linear search (not the most efficient case)
    for idx in range(unique_configs.shape[0]):
        # Comparing each element; break early if any mismatch found
        match = True
        for i in range(len(config)):  # Iterate through elements of the configuration
            if config[i] != unique_configs[idx, i]:
                match = False
                break  # Break as soon as any element doesn't match
        if match:
            return idx


@njit
def config_to_index_binarysearch(config, unique_configs):
    low = 0
    high = len(unique_configs) - 1
    while low <= high:
        idx = (low + high) // 2
        comp_result = compare_configs(
            unique_configs[idx], config
        )  # Custom comparison function
        if comp_result == 0:
            return idx
        elif comp_result < 0:
            low = idx + 1
        else:
            high = idx - 1
    return -1  # Configuration not found


@njit
def compare_configs(config1, config2):
    # Custom function to compare configurations element-wise
    for i in range(len(config1)):
        if config1[i] < config2[i]:
            return -1
        elif config1[i] > config2[i]:
            return 1
    return 0


@njit
def get_translated_state_indices(config, sector_configs, logical_unit_size):
    """Generate all translations of a given configuration of a 1d QMB system,
    considering logical units in translation."""
    # Get the size of the QMB system
    N = len(config)
    if N % logical_unit_size != 0:
        raise ValueError("Number of sites is not a multiple of the logical unit size.")
    if N != sector_configs.shape[1]:
        raise ValueError(
            f"config.shape[0]={N} must be equal to sector_configs.shape[1]={sector_configs.shape[1]}"
        )
    # Get the number of translations of the logical_unit_size
    num_translations = N // logical_unit_size
    trans_indices = np.zeros(num_translations, dtype=np.int32)
    # Run over the number of possible translations
    for ii in range(num_translations):
        # Perform roll operation by logical_unit_size steps
        roll_steps = ii * logical_unit_size
        rolled_config = np.roll(config, -roll_steps)
        trans_indices[ii] = config_to_index_binarysearch(rolled_config, sector_configs)
    return trans_indices


@get_time
@njit
def get_reference_indices(sector_configs):
    sector_dim = sector_configs.shape[0]
    normalization = np.zeros(sector_dim, dtype=np.int32)
    independent_indices = np.zeros(sector_dim, dtype=np.bool_)

    for ii in range(sector_dim):
        config = sector_configs[ii]
        # Compute all the set of translated configurations in terms of indices
        trans_indices = get_translated_state_indices(config, sector_configs)
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
    return ref_indices, norm
