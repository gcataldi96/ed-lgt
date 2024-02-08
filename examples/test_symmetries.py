# %%
from ed_lgt.operators import get_Pauli_operators
from ed_lgt.modeling import lattice_base_configs
from ed_lgt.tools import get_time
from math import prod
from itertools import product
from functools import partial
import numpy as np
from ed_lgt.operators import (
    Z2_FermiHubbard_dressed_site_operators,
    Z2_FermiHubbard_gauge_invariant_states,
)
from ed_lgt.modeling import zig_zag, get_neighbor_sites
import os
from concurrent.futures import ProcessPoolExecutor as pool
from concurrent.futures import ThreadPoolExecutor as pool_thread


def apply_basis_projection(op, basis_label, site_basis):
    op = site_basis[basis_label].transpose() @ op @ site_basis[basis_label]
    return op.toarray()


def get_projected_diagonals(op_list, lattice_labels, site_basis):
    projected_diagonals = []
    for op in op_list:
        op_diagonals_per_site = []
        for label in lattice_labels:
            projected_op = apply_basis_projection(op, label, site_basis)
            op_diagonals_per_site.append(np.diag(projected_op))
        projected_diagonals.append(op_diagonals_per_site)
    return projected_diagonals


def get_configs(loc_dims):
    # Precompute the ranges for each dimension
    ranges = [np.arange(dim, dtype=np.uint8) for dim in loc_dims]
    # Create configurations using meshgrid with 'ij' indexing
    configs = np.transpose(np.meshgrid(*ranges, indexing="ij")).reshape(
        -1, len(loc_dims)
    )
    return configs


def link_op_diagonals(op_list):
    # Define the list of opertors
    op_diagonals = []
    for op_pair in op_list:
        op_diagonals.append([])
        for op in op_pair:
            op_diagonals[-1].append(op.diagonal())
    return op_diagonals


def check_config(config, op_diagonals, pair_list, op_sectors_list):
    check = all(
        np.isclose(prod(op_diag[c] for c in config[site]), sector, atol=1e-10)
        for op_diag_pair, site_pair, sector in zip(
            op_diagonals, pair_list, op_sectors_list
        )
        for site in site_pair
        for op_diag in op_diag_pair
    )
    return check


def check_config_batch(config_batch, op_diagonals, pair_list, op_sectors_list):
    checks_list = []
    for config in config_batch:
        check = check_config(config, op_diagonals, pair_list, op_sectors_list)
        checks_list.append(check)
    return checks_list


@get_time
def abelian_sector_indices(
    loc_dims,
    op_list,
    op_sectors_list,
    sym_type="U",
    site_basis=None,
    lattice_labels=None,
    configs=None,
):
    print("TOT DIM", prod(loc_dims), np.log2(prod(loc_dims)))
    if configs is None:
        configs = get_configs(loc_dims)

    print("ACTUAL DIM", configs.shape[0], np.log2(configs.shape[0]))
    # Precompute the diagonals of the operators
    if site_basis is not None:
        # For each operator, get a list of its action on every lattice site (with different local basis)
        op_diagonals = get_projected_diagonals(op_list, lattice_labels, site_basis)
        sector_indices = []
        sector_configs = []
        for n, config in enumerate(configs):
            if sym_type == "U":
                check = all(
                    sum(op_diag[i][c] for i, c in enumerate(config)) == sector
                    for op_diag, sector in zip(op_diagonals, op_sectors_list)
                )
            elif sym_type == "P":
                check = all(
                    prod(op_diag[i][c] for i, c in enumerate(config)) == sector
                    for op_diag, sector in zip(op_diagonals, op_sectors_list)
                )
            else:
                raise ValueError(
                    f"For now sym_type can only be P (parity) or U(1), not {sym_type}"
                )
            if check:
                sector_indices.append(n)
                sector_configs.append(list(config))
        sector_indices = np.array(sector_indices, dtype=int)
        sector_configs = np.array(sector_configs, dtype=int)
    else:
        # Precompute the diagonals of the operators
        op_diagonals = [np.diag(op) for op in op_list]
        # Vectorize the symmetry sector check
        if sym_type == "U":
            # Sum the diagonal elements for each operator and check against sector values
            checks = np.all(
                [
                    np.sum(op_diag[configs], axis=1) == sector
                    for op_diag, sector in zip(op_diagonals, op_sectors_list)
                ],
                axis=0,
            )
        elif sym_type == "P":
            # Multiply the diagonal elements for each operator and check against sector values
            checks = np.all(
                [
                    np.prod(op_diag[configs], axis=1) == sector
                    for op_diag, sector in zip(op_diagonals, op_sectors_list)
                ],
                axis=0,
            )
        else:
            raise ValueError(
                f"For now sym_type can only be P (parity) or U(1), not {sym_type}"
            )
        # Filter configs based on checks
        sector_configs = configs[checks]
        sector_indices = np.ravel_multi_index(sector_configs.T, loc_dims)
        # Use the sorting indices to reorder both sector_indices and sector_configs
        sector_configs = sector_configs[np.argsort(sector_indices)]
        sector_indices = sector_indices[np.argsort(sector_indices)]

    print("SECTOR DIM", len(sector_indices), np.log2(len(sector_indices)))
    return sector_indices, sector_configs


def abelian_sector_indices_par(
    loc_dims,
    op_list,
    op_sectors_list,
    configs=None,
    pair_list=None,
):
    if configs is None:
        configs = get_configs(loc_dims)
    # Get link op diagonals
    op_diagonals = link_op_diagonals(op_list)
    # Get size of batch according to the number of cores
    num_cores = os.cpu_count()
    # Define a batch of configs
    initial_batches_per_core = 2  # Or more, based on profiling
    total_batches = num_cores * initial_batches_per_core
    batch_size = max(1, configs.shape[0] // total_batches)
    batches = [
        configs[i : i + batch_size, :] for i in range(0, configs.shape[0], batch_size)
    ]
    # create partial for check_config_batch
    partial_check_config_batch = partial(
        check_config_batch,
        op_diagonals=op_diagonals,
        pair_list=pair_list,
        op_sectors_list=op_sectors_list,
    )
    # parallelize the computation on the number of cores
    with pool(max_workers=num_cores) as executor:
        list_of_checks_list = executor.map(partial_check_config_batch, batches)
    checks = []
    for check in list_of_checks_list:
        checks.extend(list(check))
    # Filter configs based on checks
    sector_configs = configs[np.array(checks)]
    sector_indices = np.ravel_multi_index(sector_configs.T, loc_dims)
    # Use the sorting indices to reorder both sector_indices and sector_configs
    print("SECTOR DIM", len(sector_indices), np.log2(len(sector_indices)))
    return (
        sector_indices[np.argsort(sector_indices)],
        sector_configs[np.argsort(sector_indices)],
    )


@get_time
def main():
    # LATTICE GEOMETRY
    lvals = [2, 2]
    dim = len(lvals)
    directions = "xyz"[:dim]
    n_sites = prod(lvals)
    has_obc = [False, False]
    # ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
    ops = Z2_FermiHubbard_dressed_site_operators(lattice_dim=dim)
    # ACQUIRE SU2 BASIS and GAUGE INVARIANT STATES
    M, _ = Z2_FermiHubbard_gauge_invariant_states(lattice_dim=dim)
    # GET THE SITES OF PBC
    for op in ops.keys():
        ops[op] = M["site"].transpose() * ops[op] * M["site"]
        ops[op] = ops[op].toarray()
    # ACQUIRE LOCAL DIMENSION OF EVERY SINGLE SITE
    lattice_base, loc_dims = lattice_base_configs(M, lvals, has_obc, staggered=False)
    loc_dims = loc_dims.transpose().reshape(n_sites)
    lattice_base = lattice_base.transpose().reshape(n_sites)
    # Acquire the twobody link symmetry operators
    pair_list = []
    for d in directions:
        pair_list.append([])
        for ii in range(prod(lvals)):
            # Compute the corresponding coords
            coords = zig_zag(lvals, ii)
            # Check if it admits a twobody term according to the lattice geometry
            _, sites_list = get_neighbor_sites(coords, lvals, d, has_obc)
            if sites_list is not None:
                pair_list[-1].append(sites_list)
    # Select global symmetry sector
    op_list = [ops["N_tot"], ops["N_up"]]
    op_sectors_list = [n_sites, int(n_sites / 2)]
    ind, basis = abelian_sector_indices(
        loc_dims,
        op_list,
        op_sectors_list,
        sym_type="U",
    )
    # Select sector of link symmetries
    link_ops = [[ops["P_px"], ops["P_mx"]], [ops["P_py"], ops["P_my"]]]
    link_sectors = [1, 1]
    ind1, basis1 = abelian_sector_indices_par(
        loc_dims,
        op_list=link_ops,
        op_sectors_list=link_sectors,
        pair_list=pair_list,
        configs=basis,
    )


if __name__ == "__main__":
    # The freeze_support() call is recommended for scripts that will be frozen to produce a Windows executable.
    # If you're not freezing your script, you can omit this call.
    # multiprocessing.freeze_support()
    main()

# %%
