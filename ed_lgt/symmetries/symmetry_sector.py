from ed_lgt.tools import get_time
import numpy as np
import logging
from numba import njit, prange
from .global_abelian_sym import check_global_sym_sitebased
from .link_abelian_sym import check_link_sym_sitebased

logger = logging.getLogger(__name__)

__all__ = ["symmetry_sector_configs", "get_symmetry_sector_generators"]


@get_time
@njit(parallel=True)
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
):
    if not isinstance(link_sectors, np.ndarray):
        link_sectors = np.array(link_sectors, dtype=float)
    if not isinstance(glob_sectors, np.ndarray):
        glob_sectors = np.array(glob_sectors, dtype=float)
    # Acquire Sector dimension
    sector_dim = np.prod(loc_dims)
    logger.info(f"TOT DIM: {sector_dim}, 2^{round(np.log2(sector_dim),3)}")
    sector_configs = sitebased_sym_sector_configs(
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
    op_list, loc_dims, action="global", gauge_basis=None, lattice_labels=None
):
    def apply_basis_projection(op, basis_label, gauge_basis):
        return gauge_basis[basis_label].transpose() @ op @ gauge_basis[basis_label]

    n_sites = len(loc_dims)
    if action == "global":
        # Generators of Global Abelian Symmetry sector
        if gauge_basis is not None:
            op_diagonals = np.zeros((len(op_list), n_sites, max(loc_dims)), dtype=float)
            for ii, op in enumerate(op_list):
                for jj, loc_dim in enumerate(loc_dims):
                    op_diag = apply_basis_projection(
                        op, lattice_labels[jj], gauge_basis
                    ).diagonal()
                    op_diagonals[ii, jj, :loc_dim] = op_diag
        else:
            op_diagonals = np.array([op.diagonal() for op in op_list], dtype=float)
    else:
        # Generators of Link Abelian symmetry sector
        num_directions = len(op_list)

        if gauge_basis is not None:
            op_diagonals = np.zeros(
                (num_directions, 2, n_sites, max(loc_dims)), dtype=float
            )
            for ii in range(num_directions):
                for jj in range(2):
                    for kk, loc_dim in enumerate(loc_dims):
                        op_diag = apply_basis_projection(
                            op_list[ii][jj], lattice_labels[kk], gauge_basis
                        ).diagonal()
                        op_diagonals[ii, jj, kk, :loc_dim] = op_diag
        else:
            op_diagonals = np.array([op.diagonal() for op in op_list], dtype=float)
    return op_diagonals
