from ed_lgt.tools import get_time
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


@njit
def check_link_sym(config, sym_op_diags, sym_sectors, pair_list):
    """
    This function checks if a given QMB state configuration concurrently belongs
    to a set of twobody abelian symmetry sectors of Zn.

    Args:
        config (np.array): 1D array corresponding to a single QMB state configuration

        sym_op_diags (np.array): 3D array with shape (number_of_lattice_directions, 2, loc_dim),
            where each "lattice direction" contains 2 operators,
            and each operator is represented by its diagonal
            of length equal to the local dimension = loc_dim

        sym_sectors (np.array): 1D array with shape (number_of_lattice_directions,)
            containing the sector values for each lattice direction.

        pair_list (typed.List() of np.arrays): for each lattice direction the 2D array has
            shape (number_of_site_pairs_per_direction, 2),
            specifying the pair of site indices.

    Returns:
        bool: True if the config belongs to the chosen sector, False otherwise
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


@njit
def check_link_sym_sitebased(config, sym_op_diags, sym_sectors, pair_list):
    """
    Checks if a given QMB state configuration belongs to a set of two-body symmetry sectors,
    NOTE: In this case, operators acting differently on different lattice sites
    (as in Lattice Gauge Theories within the dressed site formalism).

    Args:
        config (np.array of np.uint8): 1D array for a single QMB state configuration.

        sym_op_diags (np.array of floats): 4D array with
            shape=(num_directions, 2, n_sites, max(loc_dims))
            where each "lattice direction" contains 2 operators, and each
            operator is represented by its diagonal on each lattice site.

        sym_sectors (np.array of floats): 1D array with sector values for each lattice direction.

        pair_list (list of np.array of np.uint8): Each 2D array specifies the pair
            of site indices along the corresponding lattice direction.

    Returns:
        bool: True if the config belongs to the chosen sector, False otherwise.
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
    """
    This function returns the QMB state configurations (and the corresponding 1D indices)
    that belongs to the intersection of multiple link symmetry sectors

    Args:
        loc_dims (np.array): 1D array of single-site local dimensions.
            For Exact Diagonlization (ED) purposes, each local dimension is always smaller that 2^{8}-1
            For this reason, loc_dims.dtype = np.uint8

        sym_op_diags (np.array of floats): Array with (diagonals of) operators generating the link
            symmetries of the model.
            If len(shape)=3, then it handles the case of different local Hilbert spaces,
            and, for each operator, its diagonal is evaluated on each site Hilbert basis

        sym_sectors (np.array of floats): 1D array with sector values for each operator.
            NOTE: sym_sectors.shape[0] = sym_op_diags.shape[0]


    Returns:
        (np.array of ints, np.array of ints): 1D array of indices and 2D array of QMB state configurations
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


@njit(parallel=True)
def link_sector_configs(loc_dims, link_op_diags, link_sectors, pair_list):
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
