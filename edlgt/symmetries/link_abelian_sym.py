"""Local/link Abelian symmetry filters for many-body configurations.

This module provides checks and configuration generators for symmetry sectors
defined by link-local two-site constraints, as commonly used in lattice gauge
models with open or periodic geometries.
"""

from edlgt.tools import get_time
import numpy as np
import logging
from numba import njit, prange

logger = logging.getLogger(__name__)

__all__ = [
    "check_link_sym",
    "check_link_sym_sitebased",
    "link_abelian_sector",
    "link_sector_configs",
]


@njit(cache=True)
def check_link_sym(config, sym_op_diags, sym_sectors, pair_list):
    """Check whether a configuration satisfies link Abelian symmetry sectors.

    Parameters
    ----------
    config : ndarray
        Single many-body configuration.
    sym_op_diags : ndarray
        Link-symmetry generator diagonals with shape
        ``(n_dirs, 2, loc_dim)`` for uniform local spaces.
    sym_sectors : ndarray
        Target sector value for each lattice direction.
    pair_list : sequence
        For each lattice direction, a 2-column integer array listing the site
        pairs constrained by that symmetry.

    Returns
    -------
    bool
        ``True`` if ``config`` belongs to the chosen sector.
    """
    check = True
    num_lattice_directions = sym_op_diags.shape[0]
    for idx in range(num_lattice_directions):
        # Iterate through each pair for this direction
        for pair_idx in range(pair_list[idx].shape[0]):
            # Get the pair of site indices for this pair
            site_indices = pair_list[idx][pair_idx]
            sum = 0.0
            for op_idx in range(2):
                op_diag = sym_op_diags[idx, op_idx]
                site_index = site_indices[op_idx]
                sum += op_diag[config[site_index]]
            if not np.isclose(sum, sym_sectors[idx], atol=1e-10):
                check = False
                # Early exit on first failure
                return check
    return check


@njit(cache=True)
def check_link_sym_sitebased(config, sym_op_diags, sym_sectors, pair_list):
    """Site-based version of :func:`check_link_sym` for nonuniform local bases.

    Parameters
    ----------
    config : ndarray
        Single many-body configuration.
    sym_op_diags : ndarray
        Site-resolved diagonals with shape
        ``(n_dirs, 2, n_sites, max(loc_dims))``.
    sym_sectors : ndarray
        Target sector value for each lattice direction.
    pair_list : sequence
        For each lattice direction, a 2-column integer array listing the site
        pairs constrained by that symmetry.

    Returns
    -------
    bool
        ``True`` if ``config`` belongs to the chosen sector.
    """
    check = True
    num_lattice_directions = len(pair_list)
    for idx in range(num_lattice_directions):
        # Access the 2D array for this direction
        pairs_for_direction = pair_list[idx]
        num_pairs = pairs_for_direction.shape[0]
        for pair_idx in range(num_pairs):
            site_indices = pairs_for_direction[pair_idx]
            sum = 0.0
            # Assuming two operators per direction
            for op_idx in range(2):
                site_index = site_indices[op_idx]
                op_diag = sym_op_diags[idx, op_idx, site_index, :]
                sum += op_diag[config[site_index]]
            if not np.isclose(sum, sym_sectors[idx], atol=1e-10):
                check = False
                return check
    return check


@get_time
def link_abelian_sector(loc_dims, sym_op_diags, sym_sectors, pair_list):
    """Generate configurations belonging to a link Abelian symmetry sector.

    Parameters
    ----------
    loc_dims : ndarray
        Local Hilbert-space dimensions.
    sym_op_diags : ndarray
        Link-symmetry generator diagonals (uniform or site-based representation).
    sym_sectors : ndarray or sequence
        Target sector values for the link generators.
    pair_list : sequence
        Pair list describing which site pairs are checked in each direction.

    Returns
    -------
    tuple
        ``(sector_indices, sector_configs)`` with linear basis indices and the
        corresponding site configurations.
    """
    if not isinstance(sym_sectors, np.ndarray):
        sym_sectors = np.array(sym_sectors, dtype=float)
    # Acquire Sector dimension
    sector_dim = np.prod(loc_dims)
    logger.info(f"TOT DIM: {sector_dim}, 2^{round(np.log2(sector_dim),3)}")
    # Compute the sector configs satisfying the symmetry sectors
    sector_configs = link_sector_configs(loc_dims, sym_op_diags, sym_sectors, pair_list)
    sector_indices = np.ravel_multi_index(sector_configs.T, loc_dims)
    # Acquire dimension of the new sector
    sector_dim = len(sector_configs)
    logger.info(f"SEC DIM: {sector_dim}, 2^{round(np.log2(sector_dim),3)}")
    return sector_indices, sector_configs


@njit(parallel=True, cache=True)
def link_sector_configs(loc_dims, link_op_diags, link_sectors, pair_list):
    """Enumerate configurations satisfying site-based link symmetry constraints.

    Parameters
    ----------
    loc_dims : ndarray
        Local Hilbert-space dimensions.
    link_op_diags : ndarray
        Site-resolved link-symmetry generator diagonals.
    link_sectors : ndarray
        Target sector values for the link generators.
    pair_list : sequence
        Pair list describing which site pairs are checked in each direction.

    Returns
    -------
    ndarray
        Filtered configuration table.
    """
    # Total number of configs
    sector_dim = 1
    for dim in loc_dims:
        sector_dim *= dim
    # Len of each config
    num_dims = len(loc_dims)
    configs = np.zeros((sector_dim, num_dims), dtype=np.uint8)
    # Use an auxiliary array to mark valid configurations
    checks = np.zeros(sector_dim, dtype=np.bool_)
    # Iterate over all the possible configs
    for ii in prange(sector_dim):
        tmp = ii
        for dim_index in range(num_dims):
            divisor = (
                np.prod(loc_dims[dim_index + 1 :]) if dim_index + 1 < num_dims else 1
            )
            configs[ii, dim_index] = (tmp // divisor) % loc_dims[dim_index]
        # Check if the config satisfied the symmetries
        if check_link_sym_sitebased(
            configs[ii], link_op_diags, link_sectors, pair_list
        ):
            checks[ii] = True
    # Filter configs based on checks
    return configs[checks]
