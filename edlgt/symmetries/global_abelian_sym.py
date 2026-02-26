"""Global Abelian symmetry filters for many-body configurations.

The functions in this module test and generate configuration subsets satisfying
global U(1), Z2, or string-like diagonal symmetry constraints.
"""

import numpy as np
import logging
import math
from numba import njit, prange
from edlgt.tools import get_time, arrays_equal

logger = logging.getLogger(__name__)

__all__ = [
    "check_global_sym",
    "check_global_sym_sitebased",
    "global_abelian_sector",
    "global_sector_configs",
    "check_string_sym_sitebased",
]


@njit(cache=True)
def check_global_sym(config, sym_op_diags, sym_sectors, sym_type_flag):
    """Check whether a configuration satisfies global Abelian symmetry sectors.

    Parameters
    ----------
    config : ndarray
        Single many-body configuration (one local state label per site).
    sym_op_diags : ndarray
        Operator diagonals defining the symmetries. Shape is typically
        ``(n_sym, loc_dim)`` for uniform local spaces.
    sym_sectors : ndarray
        Target sector value for each symmetry generator.
    sym_type_flag : int
        Symmetry type selector: ``0`` for additive U(1), ``1`` for multiplicative
        Z2, and values ``>1`` for element-wise string constraints.

    Returns
    -------
    bool
        ``True`` if ``config`` belongs to the chosen sector.
    """
    num_operators = sym_op_diags.shape[0]
    check = True
    # Run over all the number of symmetries
    for jj in range(num_operators):
        # Perform sum or product based on sym_type_flag
        if sym_type_flag == 0:
            # "U" for sum
            operation_result = np.sum(sym_op_diags[jj][config])
            check = np.isclose(operation_result, sym_sectors[jj], atol=1e-10)
        elif sym_type_flag == 1:
            # "Z2" for product
            operation_result = np.prod(sym_op_diags[jj][config])
            check = np.isclose(operation_result, sym_sectors[jj], atol=1e-10)
        else:
            # "string"
            check = arrays_equal(sym_op_diags[jj][config], sym_sectors[jj])
        if not check:
            # Early exit on first failure
            return check
    return check


@njit(cache=True)
def check_global_sym_sitebased(config, sym_op_diags, sym_sectors, sym_type_flag):
    """Site-based version of :func:`check_global_sym` for nonuniform local bases.

    Parameters
    ----------
    config : ndarray
        Single many-body configuration.
    sym_op_diags : ndarray
        Site-resolved operator diagonals of shape
        ``(n_sym, n_sites, max(loc_dims))``.
    sym_sectors : ndarray
        Target sector value for each symmetry generator.
    sym_type_flag : int
        Symmetry type selector: ``0`` for U(1), ``1`` for Z2.

    Returns
    -------
    bool
        ``True`` if ``config`` belongs to the chosen sector.
    """
    num_sites = config.shape[0]
    num_operators = sym_op_diags.shape[0]

    for jj in range(num_operators):
        # Initialize for sum (U)
        if sym_type_flag == 0:
            operation_result = 0
            for kk in range(num_sites):
                # Actual dimension for the site
                operation_result += sym_op_diags[jj, kk, config[kk]]
            check = np.isclose(operation_result, sym_sectors[jj], atol=1e-10)
        # Initialize for product (Z2)
        elif sym_type_flag == 1:
            operation_result = 1
            for kk in range(num_sites):
                # Actual dimension for the site
                operation_result *= sym_op_diags[jj, kk, config[kk]]
            check = np.isclose(operation_result, sym_sectors[jj], atol=1e-10)
        if not check:
            # Early exit on first failure
            return check
    return check


@njit(cache=True)
def check_string_sym_sitebased(config, sym_op_diags, sym_sectors):
    """Check site-resolved string constraints for one configuration.

    Parameters
    ----------
    config : ndarray
        Single many-body configuration.
    sym_op_diags : ndarray
        Site-resolved diagonals for the string constraints.
    sym_sectors : ndarray
        Target values for the string constraints.

    Returns
    -------
    bool
        ``True`` if the configuration satisfies all string constraints.
    """
    num_sites = config.shape[0]
    num_operators = sym_op_diags.shape[0]
    for jj in range(num_operators):
        for kk in range(num_sites):
            # Actual dimension for the site
            check = np.isclose(
                sym_op_diags[jj, kk, config[kk]], sym_sectors[jj, kk], atol=1e-10
            )
            if not check:
                return check
        if not check:
            return check
    return check


@get_time
def global_abelian_sector(loc_dims, sym_op_diags, sym_sectors, sym_type, configs=None):
    """Filter configurations by global Abelian symmetry constraints.

    Parameters
    ----------
    loc_dims : ndarray
        Local Hilbert-space dimensions.
    sym_op_diags : ndarray
        Diagonals of symmetry generators. Can be site-independent or site-based.
    sym_sectors : ndarray or sequence
        Target sector values for the generators.
    sym_type : str
        Symmetry type code: ``"U"`` (U(1)), ``"Z"`` (Z2), or other values for
        string constraints.
    configs : ndarray, optional
        If provided, filter this existing configuration table instead of
        generating the full product basis.

    Returns
    -------
    ndarray
        Configurations belonging to the requested global symmetry sector.
    """
    if not isinstance(sym_sectors, np.ndarray):
        sym_sectors = np.array(sym_sectors, dtype=float)
    # Convert sym_type to a flag: there are 3 options: U(1), Z2, and String
    if sym_type == "U":
        sym_type_flag = 0
    elif sym_type == "Z":
        sym_type_flag = 1
    else:
        sym_type_flag = 2

    if configs is not None:
        # Acquire Sector dimension
        sector_dim = configs.shape[0]
        logger.info(f"TOT DIM: {sector_dim}, 2^{round(np.log2(sector_dim),3)}")
        sector_configs = global_sector_configs_from_sector(
            sym_op_diags, sym_sectors, sym_type_flag, configs
        )
    else:
        # Acquire Sector dimension
        sector_dim = math.prod(int(d) for d in loc_dims)
        bits = sum(math.log2(d) for d in loc_dims)
        logger.info(f"TOT DIM: {sector_dim}, 2^{round(np.log2(bits),3)}")
        sector_configs = global_sector_configs(
            loc_dims, sym_op_diags, sym_sectors, sym_type_flag
        )
    # Acquire dimension of the new sector
    sector_dim = len(sector_configs)
    logger.info(f"SEC DIM: {sector_dim}, 2^{round(np.log2(sector_dim),3)}")
    return sector_configs


@njit(parallel=True, cache=True)
def global_sector_configs(loc_dims, glob_op_diags, glob_sectors, sym_type_flag):
    """Enumerate site-based global-symmetry configurations from the full basis.

    Parameters
    ----------
    loc_dims : ndarray
        Local Hilbert-space dimensions.
    glob_op_diags : ndarray
        Site-resolved diagonals of global symmetry generators.
    glob_sectors : ndarray
        Target sector values.
    sym_type_flag : int
        Symmetry type selector used by :func:`check_global_sym_sitebased`.

    Returns
    -------
    ndarray
        Filtered configuration table.
    """
    # =============================================================================
    # Get all the possible QMB state configurations
    sector_dim = np.prod(loc_dims)
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
        ):
            checks[ii] = True
    # =============================================================================
    # Filter configs based on checks
    return configs[checks]


@njit(parallel=True, cache=True)
def global_sector_configs_from_sector(
    glob_op_diags, glob_sectors, sym_type_flag, configs
):
    """Filter an existing configuration table by global/string constraints.

    Parameters
    ----------
    glob_op_diags : ndarray
        Site-resolved diagonals of global symmetry generators.
    glob_sectors : ndarray
        Target sector values.
    sym_type_flag : int
        Symmetry type selector.
    configs : ndarray
        Candidate configurations to filter.

    Returns
    -------
    ndarray
        Filtered configuration table.
    """
    # =============================================================================
    # Total number of configs
    sector_dim = configs.shape[0]
    # Use an auxiliary array to mark valid configurations
    checks = np.zeros(sector_dim, dtype=np.bool_)
    if sym_type_flag > 1:
        for ii in prange(sector_dim):
            if check_string_sym_sitebased(configs[ii], glob_op_diags, glob_sectors):
                checks[ii] = True
    else:
        for ii in prange(sector_dim):
            # Check if the config satisfied the symmetries
            if check_global_sym_sitebased(
                configs[ii], glob_op_diags, glob_sectors, sym_type_flag
            ):
                checks[ii] = True
    # =============================================================================
    # Filter configs based on checks
    return configs[checks]
