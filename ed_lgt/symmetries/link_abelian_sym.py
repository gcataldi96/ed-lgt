from ed_lgt.tools import get_time
import numpy as np
import logging
from numba import njit, prange
from .generate_configs import get_state_configs

logger = logging.getLogger(__name__)

__all__ = [
    "check_link_sym",
    "check_link_sym_sitebased",
    "check_link_sym_configs_sitebased",
    "link_abelian_sector",
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


@njit(parallel=True)
def check_link_sym_configs_sitebased(
    config_batch, sym_op_diags, sym_sectors, pair_list
):
    num_configs = config_batch.shape[0]
    checks = np.zeros(num_configs, dtype=np.bool_)
    for ii in prange(num_configs):
        checks[ii] = check_link_sym_sitebased(
            config_batch[ii], sym_op_diags, sym_sectors, pair_list
        )
    return checks


@get_time
def link_abelian_sector(loc_dims, sym_op_diags, sym_sectors, pair_list, configs=None):
    if configs is None:
        # Get QMB state configurations
        configs = get_state_configs(loc_dims)
    # Acquire Sector dimension
    sector_dim = len(configs)
    logger.info(f"TOT DIM: {sector_dim}, 2^{round(np.log2(sector_dim),3)}")
    sym_sectors = np.array(sym_sectors, dtype=float)
    checks = check_link_sym_configs_sitebased(
        configs, sym_op_diags, sym_sectors, pair_list
    )
    # Filter configs based on checks
    sector_configs = configs[checks]
    sector_indices = np.ravel_multi_index(sector_configs.T, loc_dims)
    # Acquire dimension of the new sector
    sector_dim = len(sector_configs)
    logger.info(f"SEC DIM: {sector_dim}, 2^{round(np.log2(sector_dim),3)}")
    return (sector_indices, sector_configs)
