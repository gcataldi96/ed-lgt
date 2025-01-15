from ed_lgt.tools import get_time
import numpy as np
import logging
from scipy.sparse import csr_matrix
from numba import njit, prange
from .global_abelian_sym import check_global_sym_sitebased, check_string_sym_sitebased
from .link_abelian_sym import check_link_sym_sitebased

logger = logging.getLogger(__name__)

__all__ = ["symmetry_sector_configs", "get_symmetry_sector_generators"]


@njit(parallel=True, cache=True)
def sitebased_sym_sector_configs(
    loc_dims,
    glob_op_diags,
    glob_sectors,
    sym_type_flag,
    link_op_diags,
    link_sectors,
    pair_list,
):
    # =============================================================================
    # Get all the possible QMB state configurations
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
        if check_global_sym_sitebased(
            configs[ii], glob_op_diags, glob_sectors, sym_type_flag
        ) and check_link_sym_sitebased(
            configs[ii], link_op_diags, link_sectors, pair_list
        ):
            checks[ii] = True
    # =============================================================================
    # Filter configs based on checks
    return configs[checks]


@get_time
def symmetry_sector_configs(
    loc_dims,
    glob_op_diags,
    glob_sectors,
    sym_type_flag,
    link_op_diags,
    link_sectors,
    pair_list,
    string_op_diags=None,
    string_sectors=None,
):
    if not isinstance(link_sectors, np.ndarray):
        link_sectors = np.array(link_sectors, dtype=float)
    if not isinstance(glob_sectors, np.ndarray):
        glob_sectors = np.array(glob_sectors, dtype=float)
    # Acquire Sector dimension
    sector_dim = np.prod(loc_dims)
    logger.info(f"TOT DIM: {sector_dim}, 2^{round(np.log2(sector_dim),3)}")
    if string_op_diags is not None:
        sector_configs = iterative_sitebased_sym_sector_configs1(
            loc_dims,
            glob_op_diags,
            glob_sectors,
            0 if sym_type_flag == "U" else 1,
            link_op_diags,
            link_sectors,
            pair_list,
            string_op_diags,
            string_sectors,
        )
    else:
        sector_configs = iterative_sitebased_sym_sector_configs(
            # sector_configs = sitebased_sym_sector_configs(
            loc_dims,
            glob_op_diags,
            glob_sectors,
            0 if sym_type_flag == "U" else 1,
            link_op_diags,
            link_sectors,
            pair_list,
        )
    sector_indices = np.ravel_multi_index(sector_configs.T, loc_dims)
    # Acquire dimension of the new sector
    sector_dim = len(sector_configs)
    logger.info(f"SEC DIM: {sector_dim}, 2^{round(np.log2(sector_dim),3)}")
    return sector_indices, sector_configs


def get_symmetry_sector_generators(
    op_list: list[csr_matrix],
    loc_dims: np.ndarray[int],
    action: str = "global",
    gauge_basis: dict = None,
    lattice_labels=None,
):
    def project_on_basis(
        op: csr_matrix, basis_label: str, gauge_basis: dict
    ) -> csr_matrix:
        return gauge_basis[basis_label].transpose() @ op @ gauge_basis[basis_label]

    n_sites = len(loc_dims)
    # Generators of Global Abelian Symmetry sector
    if action == "global":
        op_diagonals = np.zeros((len(op_list), n_sites, max(loc_dims)), dtype=float)
        for ii, op in enumerate(op_list):
            for jj, loc_dim in enumerate(loc_dims):
                if gauge_basis is not None:
                    op_diag = project_on_basis(op, lattice_labels[jj], gauge_basis)
                    op_diagonals[ii, jj, :loc_dim] = op_diag.diagonal()
                else:
                    op_diagonals[ii, jj, :loc_dim] = op.diagonal()

    else:
        # Generators of Link Abelian symmetry sector
        lattice_dim = len(op_list)
        op_diagonals = np.zeros((lattice_dim, 2, n_sites, max(loc_dims)), dtype=float)
        for ii in range(lattice_dim):
            for jj in range(2):
                for kk, loc_dim in enumerate(loc_dims):
                    if gauge_basis is not None:
                        op_diag = project_on_basis(
                            op_list[ii][jj], lattice_labels[kk], gauge_basis
                        )
                        op_diagonals[ii, jj, kk, :loc_dim] = op_diag.diagonal()
                    else:
                        op_diagonals[ii, jj, kk, :loc_dim] = op_list[ii][jj].diagonal()
    return op_diagonals


@njit(cache=True)
def check_link_sym_partial(config, sym_op_diags, sym_sectors, pair_list):
    """
    Partial check of link symmetries for a configuration.
    The function does not remove already checked pairs, allowing
    a simpler implementation without redundancy management.

    Args:
        config (np.array of np.uint8): Partial QMB configuration array.
        sym_op_diags (np.array of floats): 4D array with shape=(lattice_dim, 2, n_sites, max(loc_dims)).
            Each "lattice direction" contains 2 operators, and each operator
            is represented by its diagonal on each lattice site.
        sym_sectors (np.array of floats): 1D array with sector values for each lattice direction.
        pair_list (list of np.array of np.uint8): List of 2D arrays specifying pairs of site indices
            along the corresponding lattice direction.

    Returns:
        bool: True if the partial config satisfies the link symmetries, False otherwise.
    """
    num_sites = config.shape[0]
    num_lattice_directions = len(pair_list)
    # Iterate over all lattice directions
    for idx in range(num_lattice_directions):
        pairs_for_direction = pair_list[idx]
        num_pairs = pairs_for_direction.shape[0]
        # Iterate over all pairs in the current direction
        for pair_idx in range(num_pairs):
            site_indices = pairs_for_direction[pair_idx]
            # Skip pairs where one or both indices are out of the current configuration length
            if site_indices[0] >= num_sites or site_indices[1] >= num_sites:
                continue
            # Compute the sum for the symmetry condition
            sum = 0.0
            for op_idx in range(2):  # Two operators per link
                site_index = site_indices[op_idx]
                op_diag = sym_op_diags[idx, op_idx, site_index, :]
                sum += op_diag[config[site_index]]

            # Check if the sum violates the symmetry sector
            if not np.isclose(sum, sym_sectors[idx], atol=1e-10):
                return False

    return True


@njit(cache=True)
def check_glob_sym_partial(config, sym_op_diags, sym_sectors, sym_type_flag):
    """
    Check if a (partial or complete) QMB state configuration belongs to a global abelian symmetry sector.
    For U(1) symmetry, partial configurations are checked to ensure the quantum number does not exceed the target sector.
    For Z2 symmetry, the check is only performed for complete configurations.

    Args:
        config (np.array of np.uint8): 1D array with the state of each lattice site.

        sym_op_diags (np.array of floats): 3D array of shape=(num_operators, n_sites, max(loc_dims)).
            Each operator's diagonal is expressed in the proper basis of each lattice site.

        sym_sectors (np.array of floats): 1D array with sector values for each operator.
            NOTE: sym_sectors.shape[0] = sym_op_diags.shape[0].

        sym_type_flag (int): Flag indicating the symmetry type (0 = U(1), 1 = Z2).

    Returns:
        bool: True if the (partial or complete) configuration is compatible with the sector, False otherwise.
    """
    # Number of sites in the current configuration
    num_sites = config.shape[0]
    num_operators = sym_op_diags.shape[0]
    # Total number of lattice sites
    max_sites = sym_op_diags.shape[1]
    for jj in range(num_operators):
        # U(1) case
        if sym_type_flag == 0:
            operation_result = 0
            for kk in range(num_sites):
                operation_result += sym_op_diags[jj, kk, config[kk]]
            # Check if partial configuration is valid
            if num_sites < max_sites:
                if operation_result - sym_sectors[jj] > 1e-10:
                    return False
            # Full configuration must match exactly
            else:
                if not np.isclose(operation_result, sym_sectors[jj], atol=1e-10):
                    return False
        # Z2 case
        elif sym_type_flag == 1:
            # Skip Z2 check for partial configurations
            if num_sites < max_sites:
                continue
            # Full configuration must match exactly
            else:
                operation_result = 1
                for kk in range(num_sites):
                    operation_result *= sym_op_diags[jj, kk, config[kk]]
                if not np.isclose(operation_result, sym_sectors[jj], atol=1e-10):
                    return False
    return True


@njit(parallel=True, cache=True)
def iterative_sitebased_sym_sector_configs(
    loc_dims,
    glob_op_diags,
    glob_sectors,
    sym_type_flag,
    link_op_diags,
    link_sectors,
    pair_list,
):
    """
    Iteratively compute the configurations belonging to a symmetry sector,
    refining the configurations one site at a time.

    Args:
        loc_dims (np.ndarray): 1D array of single-site local dimensions.
        glob_op_diags (np.ndarray): 3D array of diagonals of global symmetry operators.
        glob_sectors (np.ndarray): 1D array of global symmetry sector values.
        sym_type_flag (int): Flag indicating symmetry type (0 = U(1), 1 = Z2, 2 = string).
        link_op_diags (np.ndarray): 3D array of diagonals of link symmetry operators.
        link_sectors (np.ndarray): 1D array of link symmetry sector values.
        pair_list (np.ndarray): List of site pairs for link symmetries.

    Returns:
        np.ndarray: Array of configurations belonging to the symmetry sector.
    """
    # Start with the first two sites
    num_sites = len(loc_dims)
    configs_prev = np.zeros((loc_dims[0] * loc_dims[1], 2), dtype=np.uint8)
    checks_prev = np.zeros(configs_prev.shape[0], dtype=np.bool_)

    # Initialize configurations for the first two sites
    for i in prange(configs_prev.shape[0]):
        configs_prev[i, 0] = i // loc_dims[1]  # First site index
        configs_prev[i, 1] = i % loc_dims[1]  # Second site index
        # Perform the checks for global and link symmetries
        checks_prev[i] = check_glob_sym_partial(
            configs_prev[i], glob_op_diags, glob_sectors, sym_type_flag
        ) and check_link_sym_partial(
            configs_prev[i], link_op_diags, link_sectors, pair_list
        )

    # Filter configurations for the first two sites
    configs_prev = configs_prev[checks_prev]
    num_configs_prev = configs_prev.shape[0]

    # Iteratively add one site at a time
    for site_idx in range(2, num_sites):
        loc_dim_next = loc_dims[site_idx]
        num_configs_next = num_configs_prev * loc_dim_next
        # Allocate new configurations and checks
        configs_next = np.zeros((num_configs_next, site_idx + 1), dtype=np.uint8)
        checks_next = np.zeros(num_configs_next, dtype=np.bool_)

        # Build new configurations
        for i in prange(num_configs_next):
            prev_config_idx = i // loc_dim_next
            new_site_state = i % loc_dim_next

            # Copy the previous configuration and add the new site's state
            configs_next[i, :site_idx] = configs_prev[prev_config_idx]
            configs_next[i, site_idx] = new_site_state

            # Perform the checks for global and link symmetries
            checks_next[i] = check_glob_sym_partial(
                configs_next[i], glob_op_diags, glob_sectors, sym_type_flag
            ) and check_link_sym_partial(
                configs_next[i], link_op_diags, link_sectors, pair_list
            )
        # Filter configurations for the current site
        configs_prev = configs_next[checks_next]
        num_configs_prev = configs_prev.shape[0]

    return configs_prev


@njit(parallel=True, cache=True)
def iterative_sitebased_sym_sector_configs1(
    loc_dims,
    glob_op_diags,
    glob_sectors,
    sym_type_flag,
    link_op_diags,
    link_sectors,
    pair_list,
    string_op_diags,
    string_sectors,
):
    """
    Iteratively compute the configurations belonging to a symmetry sector,
    refining the configurations one site at a time.

    Args:
        loc_dims (np.ndarray): 1D array of single-site local dimensions.
        glob_op_diags (np.ndarray): 3D array of diagonals of global symmetry operators.
        glob_sectors (np.ndarray): 1D array of global symmetry sector values.
        sym_type_flag (int): Flag indicating symmetry type (0 = U(1), 1 = Z2, 2 = string).
        link_op_diags (np.ndarray): 3D array of diagonals of link symmetry operators.
        link_sectors (np.ndarray): 1D array of link symmetry sector values.
        pair_list (np.ndarray): List of site pairs for link symmetries.

    Returns:
        np.ndarray: Array of configurations belonging to the symmetry sector.
    """
    # Start with the first two sites
    num_sites = len(loc_dims)
    configs_prev = np.zeros((loc_dims[0] * loc_dims[1], 2), dtype=np.uint8)
    checks_prev = np.zeros(configs_prev.shape[0], dtype=np.bool_)

    # Initialize configurations for the first two sites
    for i in prange(configs_prev.shape[0]):
        configs_prev[i, 0] = i // loc_dims[1]  # First site index
        configs_prev[i, 1] = i % loc_dims[1]  # Second site index
        # Perform the checks for global and link symmetries
        checks_prev[i] = (
            check_glob_sym_partial(
                configs_prev[i], glob_op_diags, glob_sectors, sym_type_flag
            )
            and check_link_sym_partial(
                configs_prev[i], link_op_diags, link_sectors, pair_list
            )
            and check_string_sym_sitebased(
                configs_prev[i], string_op_diags, string_sectors
            )
        )

    # Filter configurations for the first two sites
    configs_prev = configs_prev[checks_prev]
    num_configs_prev = configs_prev.shape[0]

    # Iteratively add one site at a time
    for site_idx in range(2, num_sites):
        loc_dim_next = loc_dims[site_idx]
        num_configs_next = num_configs_prev * loc_dim_next
        # Allocate new configurations and checks
        configs_next = np.zeros((num_configs_next, site_idx + 1), dtype=np.uint8)
        checks_next = np.zeros(num_configs_next, dtype=np.bool_)

        # Build new configurations
        for i in prange(num_configs_next):
            prev_config_idx = i // loc_dim_next
            new_site_state = i % loc_dim_next

            # Copy the previous configuration and add the new site's state
            configs_next[i, :site_idx] = configs_prev[prev_config_idx]
            configs_next[i, site_idx] = new_site_state

            # Perform the checks for global and link symmetries
            checks_next[i] = (
                check_glob_sym_partial(
                    configs_next[i], glob_op_diags, glob_sectors, sym_type_flag
                )
                and check_link_sym_partial(
                    configs_next[i], link_op_diags, link_sectors, pair_list
                )
                and check_string_sym_sitebased(
                    configs_next[i], string_op_diags, string_sectors
                )
            )
        # Filter configurations for the current site
        configs_prev = configs_next[checks_next]
        num_configs_prev = configs_prev.shape[0]

    return configs_prev
