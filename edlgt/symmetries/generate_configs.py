"""Configuration encoding and symmetry-sector basis expansion helpers.

This module provides low-level utilities to convert between linear many-body
basis indices and site-resolved configurations, generate all configurations for
a product Hilbert space, and build projectors between symmetry-reduced and full
bases.
"""

import numpy as np
from numba import njit, prange
from scipy.sparse import csc_matrix
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "index_to_config",
    "config_to_index",
    "get_state_configs",
    "config_to_index_linsearch",
    "config_to_index_binarysearch",
    "compare_configs",
    "get_translated_state_indices",
    "get_reference_indices",
    "subenv_map_to_unique_indices",
    "build_sector_expansion_projector",
]


@njit
def index_to_config(qmb_index, loc_dims):
    """Convert a linear many-body basis index to a site configuration.

    Parameters
    ----------
    qmb_index : int
        Linear basis index.
    loc_dims : ndarray
        Local Hilbert-space dimensions, one per site.

    Returns
    -------
    ndarray
        Configuration array with one local basis label per site.
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
    """Convert a site configuration into a linear many-body basis index.

    Parameters
    ----------
    config : numpy.ndarray
        One-dimensional array of local basis labels, one per site.
    loc_dims : numpy.ndarray
        One-dimensional array of local Hilbert-space dimensions in the same site
        order as ``config``.

    Returns
    -------
    int
        Linear QMB basis index.
    """
    qmb_index = 0
    multiplier = 1
    n_sites = len(config)
    for site_index in range(n_sites - 1, -1, -1):
        qmb_index += config[site_index] * multiplier
        multiplier *= loc_dims[site_index]
    return qmb_index


@njit(parallel=True, cache=True)
def get_state_configs(loc_dims):
    """Enumerate all product-basis configurations for a set of local dimensions.

    Parameters
    ----------
    loc_dims : ndarray
        One-dimensional array of local Hilbert-space dimensions.

    Returns
    -------
    ndarray
        Array of shape ``(prod(loc_dims), len(loc_dims))`` with dtype
        ``np.uint8``. Each row is one many-body configuration.
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


@njit
def config_to_index_linsearch(config, unique_configs):
    """Find a configuration index by linear search in a sorted/unsorted table.

    Parameters
    ----------
    config : ndarray
        Configuration to search for.
    unique_configs : ndarray
        Candidate configuration table, one configuration per row.

    Returns
    -------
    int
        Row index if found, otherwise ``-1``.
    """
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
    return -1  # Configuration not found


@njit(cache=True)
def config_to_index_binarysearch(config, unique_configs):
    """Find a configuration index by binary search in a sorted table.

    Parameters
    ----------
    config : ndarray
        Configuration to search for.
    unique_configs : ndarray
        Lexicographically sorted configuration table, one row per config.

    Returns
    -------
    int
        Row index if found, otherwise ``-1``.
    """
    low = 0
    high = len(unique_configs) - 1
    while low <= high:
        idx = (low + high) // 2
        # Custom comparison function
        comp_result = compare_configs(unique_configs[idx], config)
        if comp_result == 0:
            return idx
        elif comp_result < 0:
            low = idx + 1
        else:
            high = idx - 1
    return -1  # Configuration not found


@njit(cache=True, parallel=True)
def subenv_map_to_unique_indices(
    subsystem_configs: np.ndarray,  # (sector_dim, |S|)
    environment_configs: np.ndarray,  # (sector_dim, |E|)
    unique_subsys_configs: np.ndarray,  # (subsys_dim, |S|)
    unique_env_configs: np.ndarray,  # (env_dim, |E|)
):
    """Map subsystem/environment configurations to indices in unique tables.

    Parameters
    ----------
    subsystem_configs : ndarray
        Subsystem configurations, one row per sector basis state.
    environment_configs : ndarray
        Environment configurations, one row per sector basis state.
    unique_subsys_configs : ndarray
        Unique subsystem configurations used as lookup table.
    unique_env_configs : ndarray
        Unique environment configurations used as lookup table.

    Returns
    -------
    tuple
        ``(subsys_map, env_map)`` integer arrays with lookup indices for each
        sector basis state.
    """
    sector_dim = subsystem_configs.shape[0]
    subsys_map = np.empty(sector_dim, dtype=np.int64)
    env_map = np.empty(sector_dim, dtype=np.int64)
    for idx in prange(sector_dim):
        env_map[idx] = config_to_index_linsearch(
            environment_configs[idx], unique_env_configs
        )
        subsys_map[idx] = config_to_index_linsearch(
            subsystem_configs[idx], unique_subsys_configs
        )
    return subsys_map, env_map


@njit(cache=True)
def compare_configs(config1, config2):
    """Lexicographically compare two configurations.

    Parameters
    ----------
    config1, config2 : ndarray
        Configuration arrays of equal length.

    Returns
    -------
    int
        ``-1`` if ``config1 < config2``, ``1`` if ``config1 > config2``, and
        ``0`` if they are equal.
    """
    # Custom function to compare configurations element-wise
    for i in range(len(config1)):
        if config1[i] < config2[i]:
            return -1
        elif config1[i] > config2[i]:
            return 1
    return 0


@njit
def get_translated_state_indices(config, sector_configs, logical_unit_size=1):
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


@njit
def get_reference_indices(sector_configs):
    """Select translation-inequivalent reference configurations in 1D.

    Parameters
    ----------
    sector_configs : ndarray
        Sorted sector configurations, one row per basis state.

    Returns
    -------
    tuple
        ``(ref_indices, norm)`` where ``ref_indices`` are independent
        configuration indices and ``norm`` contains the number of unique
        translations for each reference.
    """
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


def build_sector_expansion_projector_old(
    sector_configs: np.ndarray, local_dims: np.ndarray
) -> csc_matrix:
    """Build the full-space expansion projector from sector configurations.

    Parameters
    ----------
    sector_configs : ndarray
        Allowed configurations in the reduced sector, one row per basis state.
    local_dims : ndarray
        Local Hilbert-space dimensions in the same site order as
        ``sector_configs`` columns.

    Returns
    -------
    scipy.sparse.csc_matrix
        Projector of shape ``(prod(local_dims), sector_dim)`` with one nonzero
        entry per column.
    """
    logger.info("----------------------------------------------------")
    logger.info("Projector from symmetry-sector to full space")
    sector_dim, n_sites = sector_configs.shape
    local_dims = np.ascontiguousarray(local_dims, dtype=np.int32)
    assert n_sites == len(local_dims)
    D_full = int(np.prod(local_dims))
    rows = np.empty(sector_dim, dtype=np.int64)
    cols = np.arange(sector_dim, dtype=np.int64)
    data = np.ones(sector_dim, dtype=np.float64)
    for idx in range(sector_dim):
        rows[idx] = config_to_index(sector_configs[idx], local_dims)
    projector = csc_matrix((data, (rows, cols)), shape=(D_full, sector_dim))
    return projector.toarray()


@njit(cache=True, parallel=True)
def build_sector_expansion_projector(
    sector_configs: np.ndarray, local_dims: np.ndarray
) -> np.ndarray:
    """Build a dense expansion projector from sector to full basis.

    Parameters
    ----------
    sector_configs : ndarray
        Allowed configurations in the reduced sector, one row per basis state.
    local_dims : ndarray
        Local Hilbert-space dimensions in the same site order.

    Returns
    -------
    ndarray
        Dense binary projector of shape ``(prod(local_dims), sector_dim)``.
    """
    sector_dim = sector_configs.shape[0]
    D_full = np.prod(local_dims)
    projector = np.zeros((D_full, sector_dim), dtype=np.uint8)
    for idx in prange(sector_dim):
        row_idx = config_to_index(sector_configs[idx], local_dims)
        projector[row_idx, idx] = 1
    return projector
