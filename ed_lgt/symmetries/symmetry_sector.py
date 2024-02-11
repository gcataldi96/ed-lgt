from ed_lgt.tools import get_time
import numpy as np
import logging
from numba import njit, prange
from .generate_configs import index_to_config
from .global_sym_checks import check_global_sym_sitebased
from .link_sym_checks import check_link_sym_sitebased

logger = logging.getLogger(__name__)

__all__ = ["sitebased_sym_sector_configs", "get_symmetry_sector_generators"]


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
    total_configs = np.prod(loc_dims)
    print("TOTAL DIM", total_configs, np.log2(total_configs))
    # Pre-allocate an array large enough to hold all configurations
    all_configs = np.zeros((total_configs, len(loc_dims)), dtype=np.uint8)
    # Use an auxiliary array to mark valid configurations
    valid_marks = np.zeros(total_configs, dtype=np.bool_)
    for ii in prange(total_configs):
        config = index_to_config(ii, loc_dims)
        if check_global_sym_sitebased(
            config, glob_op_diags, glob_sectors, sym_type_flag
        ) and check_link_sym_sitebased(config, link_op_diags, link_sectors, pair_list):
            all_configs[ii] = config
            valid_marks[ii] = True
    # Filter to keep only valid configurations
    valid_configs = all_configs[valid_marks]
    print("SECTOR DIM", len(valid_configs), np.log2(len(valid_configs)))
    return valid_configs


def get_symmetry_sector_generators(
    op_list, loc_dims, action="global", site_basis=None, lattice_labels=None
):
    def apply_basis_projection(op, basis_label, site_basis):
        return site_basis[basis_label].transpose() @ op @ site_basis[basis_label]

    n_sites = len(loc_dims)
    if action == "global":
        # Generators of Global Abelian Symmetry sector
        if site_basis is not None:
            op_diagonals = np.zeros((len(op_list), n_sites, max(loc_dims)), dtype=float)
            for ii, op in enumerate(op_list):
                for jj, loc_dim in enumerate(loc_dims):
                    op_diag = apply_basis_projection(
                        op, lattice_labels[jj], site_basis
                    ).diagonal()
                    op_diagonals[ii, jj, :loc_dim] = op_diag
        else:
            op_diagonals = np.array([op.diagonal() for op in op_list], dtype=float)
    else:
        # Generators of Link Abelian symmetry sector
        num_directions = len(op_list)

        if site_basis is not None:
            op_diagonals = np.zeros(
                (num_directions, 2, n_sites, max(loc_dims)), dtype=float
            )
            for ii in range(num_directions):
                for jj in range(2):
                    for kk, loc_dim in enumerate(loc_dims):
                        op_diag = apply_basis_projection(
                            op_list[ii][jj], lattice_labels[kk], site_basis
                        ).diagonal()
                        op_diagonals[ii, jj, kk, :loc_dim] = op_diag
        else:
            op_diagonals = np.array([op.diagonal() for op in op_list], dtype=float)
    return op_diagonals
