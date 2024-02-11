from ed_lgt.tools import get_time
import numpy as np
from numba import njit, prange


__all__ = ["index_to_config", "get_state_configs"]


@njit
def index_to_config(ii, loc_dims):
    """
    Convert a linear QMB index to a configuration based on local dimensions.
    """
    num_sites = len(loc_dims)
    config = np.zeros(num_sites, dtype=np.uint8)
    for site_index in range(num_sites):
        dim = loc_dims[site_index]
        config[site_index] = ii % dim
        ii //= dim
    return config


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
