"""Sparse operator-application kernels inside symmetry-reduced bases.

This module builds triplet-form sparse data (row, column, value) for local and
few-body operators acting on symmetry-sector configuration tables. It also
supports optional momentum-basis projection through precomputed sparse factors.
"""

import numpy as np
from numba import njit, prange
from edlgt.dtype_config import coerce_numeric_array
from .generate_configs import config_to_index_binarysearch
from .translational_sym import (
    nbody_data_momentum,
    nbody_data_momentum_4sites,
    nbody_data_momentum_2sites,
    nbody_data_momentum_1site,
)
import logging

logger = logging.getLogger(__name__)


__all__ = [
    "nbody_term",
    "nbody_data",
    "nbody_data_2sites",
    "localbody_data_par",
]


def nbody_term(
    op_list: np.ndarray,
    op_sites_list: np.ndarray,
    sector_configs: np.ndarray,
    momentum_basis=None,  # dict with CSC/CSR arrays OR old dense ndarray (back-compat)
):
    """
    Build triplets (row, col, value) for an n-body operator, optionally projected
    into a momentum sector.

    Parameters
    ----------
    op_list : ndarray
        Shape (n_ops, n_sites, d_loc, d_loc).
    op_sites_list : ndarray
        Shape (n_ops,), int32 — the lattice sites the operator acts on.
    sector_configs : ndarray
        Shape (n_states, n_sites), int32 — basis configurations in the (symmetry) sector.
    momentum_basis : object, optional
        Momentum-projection data.
        If a dictionary is provided, it must contain the sparse left/right
        projection arrays with keys ``"L_col_ptr"``, ``"L_row_idx"``,
        ``"L_data"``, ``"R_row_ptr"``, ``"R_col_idx"``, and ``"R_data"``
        (``"n_rows"``/``"n_cols"`` may also be present).
        If an ndarray is provided (legacy path), it is interpreted as a dense
        projection basis of shape ``(n_states, basis_dim)``.
        If ``None``, the operator is built in the real-space symmetry sector.

    Returns
    -------
    tuple
        ``(row_list, col_list, value_list)`` triplet arrays for the projected
        operator.
    """
    # normalize site-count & sanity-check
    n_ops = int(len(op_sites_list))
    if n_ops < 1:
        raise ValueError(f"nbody operator must act on at least one site, got {n_ops}")
    # === No momentum projection: original real-space path ===
    if momentum_basis is None:
        if n_ops == 1:
            row_list, col_list, value_list = localbody_data_par(
                op_list[0], op_sites_list[0], sector_configs
            )
        else:
            row_list, col_list, value_list = nbody_data(
                op_list, op_sites_list, sector_configs
            )
    else:
        # Required keys (we do not use n_rows/n_cols here, but keep them checked)
        required = (
            "L_col_ptr",
            "L_row_idx",
            "L_data",
            "R_row_ptr",
            "R_col_idx",
            "R_data",
        )
        for key_name in required:
            if key_name not in momentum_basis:
                raise KeyError(
                    f"momentum_basis dict missing required key '{key_name}'. "
                    f"Present keys: {tuple(momentum_basis.keys())}"
                )
        L_col_ptr = momentum_basis["L_col_ptr"]
        L_row_idx = momentum_basis["L_row_idx"]
        L_data = momentum_basis["L_data"]
        R_row_ptr = momentum_basis["R_row_ptr"]
        R_col_idx = momentum_basis["R_col_idx"]
        R_data = momentum_basis["R_data"]
        if n_ops == 1:
            row_list, col_list, value_list = nbody_data_momentum_1site(
                op_list,
                op_sites_list,
                sector_configs,
                L_col_ptr,
                L_row_idx,
                L_data,
                R_row_ptr,
                R_col_idx,
                R_data,
            )
        elif n_ops == 2:
            row_list, col_list, value_list = nbody_data_momentum_2sites(
                op_list,
                op_sites_list,
                sector_configs,
                L_col_ptr,
                L_row_idx,
                L_data,
                R_row_ptr,
                R_col_idx,
                R_data,
            )
        elif n_ops == 4:
            row_list, col_list, value_list = nbody_data_momentum_4sites(
                op_list,
                op_sites_list,
                sector_configs,
                L_col_ptr,
                L_row_idx,
                L_data,
                R_row_ptr,
                R_col_idx,
                R_data,
            )
        else:
            row_list, col_list, value_list = nbody_data_momentum(
                op_list,
                op_sites_list,
                sector_configs,
                L_col_ptr,
                L_row_idx,
                L_data,
                R_row_ptr,
                R_col_idx,
                R_data,
            )
    value_list = coerce_numeric_array(value_list, name="symmetry nbody values")
    return row_list, col_list, value_list


@njit(parallel=True, cache=True)
def nbody_data(
    op_list: np.ndarray, op_sites_list: np.ndarray, sector_configs: np.ndarray
):
    """Build sparse triplets for a generic n-body operator in a symmetry basis.

    Parameters
    ----------
    op_list : ndarray
        Site-resolved operator matrices.
    op_sites_list : ndarray
        Site indices on which the operator acts.
    sector_configs : ndarray
        Symmetry-sector configurations, one row per basis state.

    Returns
    -------
    tuple
        ``(row_list, col_list, value_list)`` sparse triplet arrays.
    """
    n_states = sector_configs.shape[0]
    n_sites = sector_configs.shape[1]
    local_dim = np.int32(op_list[0].shape[-1])
    n_ops = len(op_sites_list)
    # Number of nonzero columns generated per row
    nnz_counts = np.zeros(n_states, dtype=np.int32)
    # Estimate the number of nonzero elements per row
    for state_idx in prange(n_states):
        row_nnz = 1
        for op_idx in range(n_ops):
            site_idx = op_sites_list[op_idx]
            site_op = op_list[op_idx, site_idx]
            bra_loc = sector_configs[state_idx, site_idx]
            n_transitions = 0
            for ket_loc in range(local_dim):
                if np.abs(site_op[bra_loc, ket_loc]) > 1e-10:
                    n_transitions += 1
            row_nnz *= n_transitions
        nnz_counts[state_idx] = row_nnz
    # Row offsets in the flattened triplet arrays
    nnz_offsets = np.empty(n_states, dtype=np.int32)
    # Allocate the output arrays
    total_nnz = 0
    for state_idx in range(n_states):
        nnz_offsets[state_idx] = total_nnz
        total_nnz += nnz_counts[state_idx]
    # Preallocate the output arrays
    row_list = np.empty(total_nnz, dtype=np.int32)
    col_list = np.empty(total_nnz, dtype=np.int32)
    value_list = np.empty(total_nnz, dtype=np.complex128)
    # PASS 2: enumerate the cartesian product of local nonzero transitions
    for bra_idx in prange(n_states):
        # 1) grab the bra's full config and where to write
        bra_config = sector_configs[bra_idx]
        if nnz_counts[bra_idx] == 0:
            continue
        write_idx = nnz_offsets[bra_idx]
        # 2) build per-site transitions
        ket_local_states = np.empty((n_ops, local_dim), np.int32)
        transition_values = np.empty((n_ops, local_dim), np.complex128)
        transition_counts = np.empty(n_ops, np.int32)
        for op_idx in range(n_ops):
            site_idx = op_sites_list[op_idx]
            site_op = op_list[op_idx, site_idx]
            bra_loc = bra_config[site_idx]
            n_transitions = 0
            for ket_loc in range(local_dim):
                elem = site_op[bra_loc, ket_loc]
                if np.abs(elem) > 1e-10:
                    ket_local_states[op_idx, n_transitions] = ket_loc
                    transition_values[op_idx, n_transitions] = elem
                    n_transitions += 1
            transition_counts[op_idx] = n_transitions
        # 3) enumerate all combinations with a mixed-radix counter
        transition_counters = np.zeros(n_ops, np.int32)
        ket_config = np.empty(n_sites, np.int32)
        finished = False
        while not finished:
            # start from the bra each time
            for site_idx in range(n_sites):
                ket_config[site_idx] = bra_config[site_idx]
            amplitude = 1.0 + 0.0j
            for op_idx in range(n_ops):
                trans_idx = transition_counters[op_idx]
                amplitude *= transition_values[op_idx, trans_idx]
                ket_config[op_sites_list[op_idx]] = ket_local_states[
                    op_idx, trans_idx
                ]
            ket_idx = config_to_index_binarysearch(ket_config, sector_configs)
            row_list[write_idx] = bra_idx
            col_list[write_idx] = ket_idx
            value_list[write_idx] = amplitude
            write_idx += 1
            # increment mixed-radix counter from last digit
            for op_idx in range(n_ops - 1, -1, -1):
                transition_counters[op_idx] += 1
                if transition_counters[op_idx] < transition_counts[op_idx]:
                    break
                transition_counters[op_idx] = 0
                if op_idx == 0:
                    finished = True
    return row_list, col_list, value_list


@njit(parallel=True, cache=True)
def nbody_data_4sites(
    op_list: np.ndarray, op_sites_list: np.ndarray, sector_configs: np.ndarray
):
    """Build sparse triplets for a four-site operator in a symmetry basis.

    Parameters
    ----------
    op_list : ndarray
        Site-resolved operator matrices.
    op_sites_list : ndarray
        Four site indices on which the operator acts.
    sector_configs : ndarray
        Symmetry-sector configurations, one row per basis state.

    Returns
    -------
    tuple
        ``(row_list, col_list, value_list)`` sparse triplet arrays.
    """
    n_states = sector_configs.shape[0]
    n_sites = sector_configs.shape[1]
    local_dim = np.int32(op_list[0].shape[-1])
    n_ops = len(op_sites_list)
    # Array to hold the number of nonzero columns per row
    nnz_cols_per_row = np.zeros(n_states, dtype=np.int32)
    # Estimate the number of nonzero elements per row
    for state_idx in prange(n_states):
        row_nnz = 1
        for op_idx in range(n_ops):
            site_idx = op_sites_list[op_idx]
            site_op = op_list[op_idx, site_idx]
            bra_loc = sector_configs[state_idx, site_idx]
            n_transitions = 0
            for ket_loc in range(local_dim):
                if np.abs(site_op[bra_loc, ket_loc]) > 1e-10:
                    n_transitions += 1
            row_nnz *= n_transitions
        nnz_cols_per_row[state_idx] = row_nnz
    # Allocate the output arrays
    total_nnz = 0
    for state_idx in range(n_states):
        tmp_nnz = nnz_cols_per_row[state_idx]
        nnz_cols_per_row[state_idx] = total_nnz
        total_nnz += tmp_nnz
    # Preallocate the output arrays
    row_list = np.empty(total_nnz, dtype=np.int32)
    col_list = np.empty(total_nnz, dtype=np.int32)
    value_list = np.empty(total_nnz, dtype=np.complex128)
    # --- PASS 2: explicit 4-nested-loops version for n_ops=4  ---------------
    for bra_idx in prange(n_states):
        # 1) grab the bra’s full config and where to write
        bra_config = sector_configs[bra_idx]
        ptr = nnz_cols_per_row[bra_idx]
        # 2) build the per-site transitions from the one-site operators
        ket_local_states = np.empty((n_ops, local_dim), np.int32)
        transition_values = np.empty((n_ops, local_dim), np.complex128)
        transition_counts = np.empty(n_ops, np.int32)
        for op_idx in range(n_ops):
            site_idx = op_sites_list[op_idx]
            site_op = op_list[op_idx, site_idx]
            bra_loc = bra_config[site_idx]
            n_transitions = 0
            for ket_loc in range(local_dim):
                elem = site_op[bra_loc, ket_loc]
                if np.abs(elem) > 1e-10:
                    ket_local_states[op_idx, n_transitions] = ket_loc
                    transition_values[op_idx, n_transitions] = elem
                    n_transitions += 1
            transition_counts[op_idx] = n_transitions
        # 3) a scratch array for the ket
        ket_config = np.empty(n_sites, np.int32)
        # start from the bra each time
        for site_idx in range(n_sites):
            ket_config[site_idx] = bra_config[site_idx]
        # 4) four fully‐nested loops
        site0 = op_sites_list[0]
        site1 = op_sites_list[1]
        site2 = op_sites_list[2]
        site3 = op_sites_list[3]
        for ii0 in range(transition_counts[0]):
            ket0 = ket_local_states[0, ii0]
            val0 = transition_values[0, ii0]
            ket_config[site0] = ket0
            for ii1 in range(transition_counts[1]):
                ket1 = ket_local_states[1, ii1]
                val1 = transition_values[1, ii1]
                ket_config[site1] = ket1
                for ii2 in range(transition_counts[2]):
                    ket2 = ket_local_states[2, ii2]
                    val2 = transition_values[2, ii2]
                    ket_config[site2] = ket2
                    for ii3 in range(transition_counts[3]):
                        ket3 = ket_local_states[3, ii3]
                        val3 = transition_values[3, ii3]
                        ket_config[site3] = ket3
                        # compute the full amplitude
                        amp = val0 * val1 * val2 * val3
                        # lookup that ket in your sorted sector_configs
                        ket_idx = config_to_index_binarysearch(ket_config, sector_configs)
                        # write out the entry
                        row_list[ptr] = bra_idx
                        col_list[ptr] = ket_idx
                        value_list[ptr] = amp
                        ptr += 1
    return row_list, col_list, value_list


@njit(parallel=True, cache=True)
def nbody_data_2sites(
    op_list: np.ndarray, op_sites_list: np.ndarray, sector_configs: np.ndarray
):
    """Build sparse triplets for a two-site operator in a symmetry basis.

    Parameters
    ----------
    op_list : ndarray
        Site-resolved operator matrices.
    op_sites_list : ndarray
        Two site indices on which the operator acts.
    sector_configs : ndarray
        Symmetry-sector configurations, one row per basis state.

    Returns
    -------
    tuple
        ``(row_list, col_list, value_list)`` sparse triplet arrays.
    """
    n_states = sector_configs.shape[0]
    n_sites = sector_configs.shape[1]
    local_dim = np.int32(op_list[0].shape[-1])
    n_ops = len(op_sites_list)
    # Array to hold the number of nonzero columns per row
    nnz_cols_per_row = np.zeros(n_states, dtype=np.int32)
    # Estimate the number of nonzero elements per row
    for state_idx in prange(n_states):
        row_nnz = 1
        for op_idx in range(n_ops):
            site_idx = op_sites_list[op_idx]
            site_op = op_list[op_idx, site_idx]
            bra_loc = sector_configs[state_idx, site_idx]
            n_transitions = 0
            for ket_loc in range(local_dim):
                if np.abs(site_op[bra_loc, ket_loc]) > 1e-10:
                    n_transitions += 1
            row_nnz *= n_transitions
        nnz_cols_per_row[state_idx] = row_nnz
    # Allocate the output arrays
    total_nnz = 0
    for state_idx in range(n_states):
        tmp_nnz = nnz_cols_per_row[state_idx]
        nnz_cols_per_row[state_idx] = total_nnz
        total_nnz += tmp_nnz
    # Preallocate the output arrays
    row_list = np.empty(total_nnz, dtype=np.int32)
    col_list = np.empty(total_nnz, dtype=np.int32)
    value_list = np.empty(total_nnz, dtype=np.complex128)
    # --- PASS 2: explicit 2-nested-loops version for n_ops=2  ---------------
    for bra_idx in prange(n_states):
        # 1) grab the bra’s full config and where to write
        bra_config = sector_configs[bra_idx]
        ptr = nnz_cols_per_row[bra_idx]
        # 2) build the per-site transitions from the one-site operators
        ket_local_states = np.empty((n_ops, local_dim), np.int32)
        transition_values = np.empty((n_ops, local_dim), np.complex128)
        transition_counts = np.empty(n_ops, np.int32)
        for op_idx in range(n_ops):
            site_idx = op_sites_list[op_idx]
            site_op = op_list[op_idx, site_idx]
            bra_loc = bra_config[site_idx]
            n_transitions = 0
            for ket_loc in range(local_dim):
                elem = site_op[bra_loc, ket_loc]
                if np.abs(elem) > 1e-10:
                    ket_local_states[op_idx, n_transitions] = ket_loc
                    transition_values[op_idx, n_transitions] = elem
                    n_transitions += 1
            transition_counts[op_idx] = n_transitions
        # 3) a scratch array for the ket
        ket_config = np.empty(n_sites, np.int32)
        # 4) four fully‐nested loops
        site0 = op_sites_list[0]
        site1 = op_sites_list[1]
        for ii0 in range(transition_counts[0]):
            # start from the bra each time
            for site_idx in range(n_sites):
                ket_config[site_idx] = bra_config[site_idx]
            ket0 = ket_local_states[0, ii0]
            val0 = transition_values[0, ii0]
            ket_config[site0] = ket0
            for ii1 in range(transition_counts[1]):
                ket1 = ket_local_states[1, ii1]
                val1 = transition_values[1, ii1]
                ket_config[site1] = ket1
                # compute the full amplitude
                amp = val0 * val1
                # lookup that ket in your sorted sector_configs
                ket_idx = config_to_index_binarysearch(ket_config, sector_configs)
                # write out the entry
                row_list[ptr] = bra_idx
                col_list[ptr] = ket_idx
                value_list[ptr] = amp
                ptr += 1
    return row_list, col_list, value_list


@njit(parallel=True, cache=True)
def localbody_data_par2(
    op: np.ndarray, op_site_list: list[int], sector_configs: np.ndarray
):
    """Build sparse triplets for a diagonal operator summed over several sites.

    Parameters
    ----------
    op : ndarray
        Site-resolved diagonal operator matrices.
    op_site_list : list
        Site indices on which the operator contributes.
    sector_configs : ndarray
        Symmetry-sector configurations, one row per basis state.

    Returns
    -------
    tuple
        ``(row_list, col_list, value_list)`` sparse triplet arrays for the
        diagonal operator restricted to nonzero entries.
    """
    sector_dim = sector_configs.shape[0]
    # Start with every configuration.
    row_list = np.arange(sector_dim, dtype=np.int32)
    # We'll use a boolean mask to filter out rows with zero contribution.
    check_rows = np.zeros(sector_dim, dtype=np.bool_)
    # Preallocate an array for the computed operator values.
    value_list = np.zeros(sector_dim, dtype=np.complex128)
    # Process each configuration in parallel.
    for row in prange(sector_dim):
        # Accumulate the contribution from each site in op_site_list.
        for ii in range(len(op_site_list)):
            op_site = int(op_site_list[ii])
            op_diag = op[op_site]  # Extract the diagonal matrix for this site.
            value_list[row] += op_diag[
                sector_configs[row, op_site], sector_configs[row, op_site]
            ]
        # Check that the element is nonzero
        if np.abs(value_list[row]) >= 1e-10:
            # Mark the row as having at least one nonzero element
            check_rows[row] = True
    # Filter out zero elements
    row_list = row_list[check_rows]
    value_list = value_list[check_rows]
    return row_list, row_list, value_list


@njit(parallel=True, cache=True)
def localbody_data_par(op: np.ndarray, op_site: int, sector_configs: np.ndarray):
    """Build sparse triplets for a diagonal single-site operator.

    Parameters
    ----------
    op : ndarray
        Site-resolved diagonal operator matrices.
    op_site : int
        Site index where the operator acts.
    sector_configs : ndarray
        Symmetry-sector configurations, one row per basis state.

    Returns
    -------
    tuple
        ``(row_list, col_list, value_list)`` sparse triplet arrays for the
        diagonal operator restricted to nonzero entries.
    """
    sector_dim = sector_configs.shape[0]
    # Initialize row_list and col_list as the diagonal indices
    row_list = np.arange(sector_dim, dtype=np.int32)
    check_rows = np.zeros(sector_dim, dtype=np.bool_)
    value_list = np.zeros(sector_dim, dtype=np.complex128)
    # Isolate the action of the operator on the site
    op_diag = op[op_site]
    for row in prange(len(row_list)):
        value_list[row] = op_diag[
            sector_configs[row, op_site], sector_configs[row, op_site]
        ]
        # Check that the element is nonzero
        if np.abs(value_list[row]) >= 1e-10:
            # Mark the row as having at least one nonzero element
            check_rows[row] = True
    # Filter out zero elements
    row_list = row_list[check_rows]
    value_list = value_list[check_rows]
    return row_list, row_list, value_list
