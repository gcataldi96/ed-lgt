import logging

import numpy as np

from edlgt.modeling.lattice_geometry import get_lattice_link_site_pairs
from edlgt.symmetries.global_abelian_sym import check_string_sym_sitebased
from edlgt.symmetries.link_abelian_sym import link_sector_configs
from edlgt.symmetries.symmetry_sector import (
    check_Z2_nbody_sym_partial,
    check_nbody_sym_partial,
    iterative_link_sector_configs,
    iterative_link_sector_configs_plus,
    iterative_sitebased_sym_sector_configs,
    iterative_sitebased_sym_sector_configs1,
    sitebased_sym_sector_configs,
)

logger = logging.getLogger(__name__)


def _sort_configs(configs: np.ndarray) -> np.ndarray:
    if configs.shape[0] == 0:
        return configs
    sort_keys = []
    for site_idx in range(configs.shape[1] - 1, -1, -1):
        sort_keys.append(configs[:, site_idx])
    sort_perm = np.lexsort(tuple(sort_keys))
    return np.ascontiguousarray(configs[sort_perm], dtype=configs.dtype)


def _assert_same_configs(
    check_name: str, expected: np.ndarray, observed: np.ndarray
) -> None:
    exp_sorted = _sort_configs(expected)
    obs_sorted = _sort_configs(observed)
    if exp_sorted.shape != obs_sorted.shape:
        raise AssertionError(
            f"{check_name}: shape mismatch expected={exp_sorted.shape}, observed={obs_sorted.shape}"
        )
    if not np.array_equal(exp_sorted, obs_sorted):
        raise AssertionError(f"{check_name}: configuration mismatch")
    logger.info(f"{check_name}: PASS (dim={exp_sorted.shape[0]})")


def _build_base_inputs(n_sites: int):
    loc_dims = np.full(n_sites, 2, dtype=np.int32)

    # Global U(1): sum of local occupancies.
    glob_op_diags = np.zeros((1, n_sites, 2), dtype=np.float64)
    for site_idx in range(n_sites):
        glob_op_diags[0, site_idx, 0] = 0.0
        glob_op_diags[0, site_idx, 1] = 1.0
    glob_sectors = np.array([n_sites / 2], dtype=np.float64)

    # Link constraint: state(site0) - state(site1) = 0, i.e. neighboring states equal.
    link_op_diags = np.zeros((1, 2, n_sites, 2), dtype=np.float64)
    for site_idx in range(n_sites):
        link_op_diags[0, 0, site_idx, 0] = 0.0
        link_op_diags[0, 0, site_idx, 1] = 1.0
        link_op_diags[0, 1, site_idx, 0] = 0.0
        link_op_diags[0, 1, site_idx, 1] = -1.0
    link_sectors = np.array([0.0], dtype=np.float64)
    pair_list = get_lattice_link_site_pairs([n_sites], [False])
    return loc_dims, glob_op_diags, glob_sectors, link_op_diags, link_sectors, pair_list


def _filter_with_string_constraints(
    configs: np.ndarray, string_op_diags: np.ndarray, string_sectors: np.ndarray
) -> np.ndarray:
    mask = np.zeros(configs.shape[0], dtype=np.bool_)
    for cfg_idx in range(configs.shape[0]):
        mask[cfg_idx] = check_string_sym_sitebased(
            configs[cfg_idx], string_op_diags, string_sectors
        )
    return configs[mask]


def _filter_with_nbody_constraints(
    configs: np.ndarray,
    nbody_op_diags: np.ndarray,
    nbody_sectors: np.ndarray,
    nbody_sites_list,
    nbody_sym_value: int,
) -> np.ndarray:
    mask = np.zeros(configs.shape[0], dtype=np.bool_)
    for cfg_idx in range(configs.shape[0]):
        if nbody_sym_value == 0:
            mask[cfg_idx] = check_nbody_sym_partial(
                configs[cfg_idx], nbody_op_diags, nbody_sectors, nbody_sites_list
            )
        else:
            mask[cfg_idx] = check_Z2_nbody_sym_partial(
                configs[cfg_idx], nbody_op_diags, nbody_sectors, nbody_sites_list
            )
    return configs[mask]


def main():
    n_sites = 8
    (
        loc_dims,
        glob_op_diags,
        glob_sectors,
        link_op_diags,
        link_sectors,
        pair_list,
    ) = _build_base_inputs(n_sites=n_sites)

    logger.info("----------------------------------------------------")
    logger.info("Checking global+link iterative builder (U1)")
    brute_global_link = sitebased_sym_sector_configs(
        loc_dims,
        glob_op_diags,
        glob_sectors,
        0,
        link_op_diags,
        link_sectors,
        pair_list,
    )
    fast_global_link = iterative_sitebased_sym_sector_configs(
        loc_dims,
        glob_op_diags,
        glob_sectors,
        0,
        link_op_diags,
        link_sectors,
        pair_list,
    )
    _assert_same_configs(
        "global+link (U1): iterative == exhaustive", brute_global_link, fast_global_link
    )

    logger.info("----------------------------------------------------")
    logger.info("Checking global+link+string iterative builder")
    string_op_diags = np.zeros((1, n_sites, 2), dtype=np.float64)
    string_sectors = np.zeros((1, n_sites), dtype=np.float64)
    for site_idx in range(n_sites):
        string_op_diags[0, site_idx, 0] = 0.0
        string_op_diags[0, site_idx, 1] = 1.0
        string_sectors[0, site_idx] = 0.0 if site_idx % 2 == 0 else 1.0
    brute_string = _filter_with_string_constraints(
        brute_global_link, string_op_diags, string_sectors
    )
    fast_string = iterative_sitebased_sym_sector_configs1(
        loc_dims,
        glob_op_diags,
        glob_sectors,
        0,
        link_op_diags,
        link_sectors,
        pair_list,
        string_op_diags,
        string_sectors,
    )
    _assert_same_configs(
        "global+link+string: iterative == exhaustive filter", brute_string, fast_string
    )

    logger.info("----------------------------------------------------")
    logger.info("Checking link-only iterative builder")
    brute_link = link_sector_configs(loc_dims, link_op_diags, link_sectors, pair_list)
    fast_link = iterative_link_sector_configs(
        loc_dims, link_op_diags, link_sectors, pair_list
    )
    _assert_same_configs("link-only: iterative == exhaustive", brute_link, fast_link)

    logger.info("----------------------------------------------------")
    logger.info("Checking link+nbody iterative builder (U1)")
    nbody_op_diags_u1 = np.zeros((1, n_sites, 2), dtype=np.float64)
    for site_idx in range(n_sites):
        nbody_op_diags_u1[0, site_idx, 0] = 0.0
        nbody_op_diags_u1[0, site_idx, 1] = 1.0
    nbody_sites_list_u1 = [np.array([0, 2, 4, 6], dtype=np.uint16)]
    nbody_sectors_u1 = np.array([2.0], dtype=np.float64)
    brute_link_nbody_u1 = _filter_with_nbody_constraints(
        brute_link,
        nbody_op_diags_u1,
        nbody_sectors_u1,
        nbody_sites_list_u1,
        nbody_sym_value=0,
    )
    fast_link_nbody_u1 = iterative_link_sector_configs_plus(
        loc_dims,
        link_op_diags,
        link_sectors,
        pair_list,
        nbody_op_diags_u1,
        nbody_sectors_u1,
        nbody_sites_list_u1,
        0,
    )
    _assert_same_configs(
        "link+nbody (U1): iterative == exhaustive filter",
        brute_link_nbody_u1,
        fast_link_nbody_u1,
    )

    logger.info("----------------------------------------------------")
    logger.info("Checking link+nbody iterative builder (Z2)")
    nbody_op_diags_z2 = np.zeros((1, n_sites, 2), dtype=np.float64)
    for site_idx in range(n_sites):
        nbody_op_diags_z2[0, site_idx, 0] = 1.0
        nbody_op_diags_z2[0, site_idx, 1] = -1.0
    nbody_sites_list_z2 = [np.array([1, 3, 5, 7], dtype=np.uint16)]
    nbody_sectors_z2 = np.array([1.0], dtype=np.float64)
    brute_link_nbody_z2 = _filter_with_nbody_constraints(
        brute_link,
        nbody_op_diags_z2,
        nbody_sectors_z2,
        nbody_sites_list_z2,
        nbody_sym_value=1,
    )
    fast_link_nbody_z2 = iterative_link_sector_configs_plus(
        loc_dims,
        link_op_diags,
        link_sectors,
        pair_list,
        nbody_op_diags_z2,
        nbody_sectors_z2,
        nbody_sites_list_z2,
        1,
    )
    _assert_same_configs(
        "link+nbody (Z2): iterative == exhaustive filter",
        brute_link_nbody_z2,
        fast_link_nbody_z2,
    )

    logger.info("====================================================")
    logger.info("symmetry_sector iterative consistency: PASS")
    logger.info("====================================================")


if __name__ == "__main__":
    main()
