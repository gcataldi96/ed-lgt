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
OP_TOL = 1e-10


__all__ = [
    "nbody_term",
    "nbody_data",
    "nbody_data_2sites",
    "localbody_data_par",
]


@njit(cache=True)
def _prepare_local_transition_data(
    op_list: np.ndarray, op_sites_list: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute all nonzero local transitions for the operator support.

    Parameters
    ----------
    op_list : ndarray
        Site-resolved operator matrices.
    op_sites_list : ndarray
        Site indices on which the operator acts.

    Returns
    -------
    tuple
        ``(transition_counts, ket_local_states, transition_values)`` where
        ``transition_counts[op_idx, bra_loc]`` stores how many nonzero outgoing
        transitions the local operator has from ``bra_loc``. The other two arrays
        store the corresponding destination local states and matrix elements.

    Notes
    -----
    ``local_dim`` is the common padded local dimension carried by the rectangular
    operator tensor ``op_list``. Some lattice sites may have a smaller physical
    local Hilbert space, but in that case their operator rows/columns are padded
    with zeros. Therefore scanning up to ``local_dim`` is safe, and only the
    physically allowed local states actually appear in ``sector_configs``.
    """
    n_ops = len(op_sites_list)
    # Define the local_dim as the largest local dimension in the lattice
    # even if some sites will have a smaller one this does not break the scheme
    local_dim = np.int32(op_list[0].shape[-1])
    # transition_counts[op_idx, bra_loc] says how many allowed ket local states
    # exist for operator factor op_idx when the local bra state is bra_loc.
    transition_counts = np.zeros((n_ops, local_dim), dtype=np.int32)
    # ket_local_states[op_idx, bra_loc, k] stores the k-th reachable local ket state.
    ket_local_states = np.empty((n_ops, local_dim, local_dim), dtype=np.int32)
    # transition_values[op_idx, bra_loc, k] stores the corresponding matrix element.
    transition_values = np.empty((n_ops, local_dim, local_dim), dtype=np.complex128)
    # Each local operator row is scanned exactly once here and then reused by all
    # bra states in both pass 1 (row nnz counting) and pass 2 (triplet generation).
    for op_idx in range(n_ops):
        # Select the actual lattice site where this operator acts.
        site_idx = op_sites_list[op_idx]
        # Get the shape of the operator on that lattice site (a matrix)
        site_op = op_list[op_idx, site_idx]
        # Run over all possible local bra states of that one-site matrix.
        for bra_loc in range(local_dim):
            n_transitions = 0
            # Scan the whole local row once and keep only the nonzero moves.
            for ket_loc in range(local_dim):
                elem = site_op[bra_loc, ket_loc]
                if np.abs(elem) > OP_TOL:
                    # Store both the reachable ket state and the corresponding amplitude.
                    ket_local_states[op_idx, bra_loc, n_transitions] = ket_loc
                    transition_values[op_idx, bra_loc, n_transitions] = elem
                    n_transitions += 1
            # Save how many entries of the previous two tables are actually valid.
            transition_counts[op_idx, bra_loc] = n_transitions
    return transition_counts, ket_local_states, transition_values


@njit(parallel=True, cache=True)
def _count_row_nnz_from_transition_data(
    sector_configs: np.ndarray,
    op_sites_list: np.ndarray,
    transition_counts: np.ndarray,
) -> np.ndarray:
    """Count how many nonzero matrix elements each bra row generates.

    Parameters
    ----------
    sector_configs : ndarray
        Symmetry-sector configurations, one row per basis state.
    op_sites_list : ndarray
        Site indices on which the operator acts.
    transition_counts : ndarray
        Precomputed local transition counts from
        :func:`_prepare_local_transition_data`.

    Returns
    -------
    ndarray
        Number of generated triplets for each bra configuration.
    """
    n_states = sector_configs.shape[0]
    n_ops = len(op_sites_list)
    # nnz_counts[row] will contain how many ket configurations are produced when
    # the full n-body operator acts on bra configuration sector_configs[row].
    nnz_counts = np.zeros(n_states, dtype=np.int32)
    for state_idx in prange(n_states):
        # Start from 1 because the total number of emitted matrix elements is the
        # Cartesian-product size of the local transition sets.
        row_nnz = 1
        for op_idx in range(n_ops):
            # Select the actual lattice site where this operator acts.
            site_idx = op_sites_list[op_idx]
            # Read the actual local bra state of the site in the current state config.
            bra_loc = sector_configs[state_idx, site_idx]
            # Multiply by the number of local transitions available from that state.
            row_nnz *= transition_counts[op_idx, bra_loc]
        # Save the total number of triplets generated by this bra row.
        nnz_counts[state_idx] = row_nnz
    return nnz_counts


@njit(cache=True)
def _get_nnz_offsets(nnz_counts: np.ndarray) -> tuple[np.ndarray, int]:
    """Convert per-row nnz counts into exclusive offsets.

    Parameters
    ----------
    nnz_counts : ndarray
        Number of generated triplets for each bra configuration.

    Returns
    -------
    tuple
        ``(nnz_offsets, total_nnz)`` where ``nnz_offsets[row]`` is the first
        write position of that row in the flattened triplet arrays.
    """
    n_states = nnz_counts.shape[0]
    # nnz_offsets[row] is the first position in the flat COO arrays where the
    # contributions generated from bra row "row" must be written.
    nnz_offsets = np.empty(n_states, dtype=np.int32)
    total_nnz = 0
    for state_idx in range(n_states):
        # Store the current write pointer before advancing it.
        nnz_offsets[state_idx] = total_nnz
        # Advance the write pointer by the number of entries emitted by this row.
        total_nnz += nnz_counts[state_idx]
    return nnz_offsets, total_nnz


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
        elif n_ops == 2:
            row_list, col_list, value_list = nbody_data_2sites(
                op_list, op_sites_list, sector_configs
            )
        elif n_ops == 4:
            row_list, col_list, value_list = nbody_data_4sites(
                op_list, op_sites_list, sector_configs
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
    n_ops = len(op_sites_list)
    # Precompute all one-site transitions once and reuse them in both passes.
    transition_counts, ket_local_states, transition_values = (
        _prepare_local_transition_data(op_list, op_sites_list)
    )
    # Count how many nonzero columns each bra state will generate.
    nnz_counts = _count_row_nnz_from_transition_data(
        sector_configs, op_sites_list, transition_counts
    )
    # Convert row counts to write offsets in the flattened triplet arrays.
    nnz_offsets, total_nnz = _get_nnz_offsets(nnz_counts)
    # Preallocate the output arrays
    row_list = np.empty(total_nnz, dtype=np.int32)
    col_list = np.empty(total_nnz, dtype=np.int32)
    value_list = np.empty(total_nnz, dtype=np.complex128)
    # PASS 2: enumerate the cartesian product of local nonzero transitions
    for bra_idx in prange(n_states):
        # 1) Grab the bra config and the start of its write window inside
        bra_config = sector_configs[bra_idx]
        if nnz_counts[bra_idx] == 0:
            # skip bra_idx with zero outgoing local transitions (no matrix elements).
            continue
        write_idx = nnz_offsets[bra_idx]
        # 2) For bra_config, get precomp. local transition rows applying on each acted site.
        bra_local_states = np.empty(n_ops, dtype=np.int32)
        active_transition_counts = np.empty(n_ops, dtype=np.int32)
        for op_idx in range(n_ops):
            site_idx = op_sites_list[op_idx]
            # Get (site_idx)-site state of the bra_config
            bra_loc = bra_config[site_idx]
            # Store the site-state inside the mixed-radix
            # enumeration without reading sector_configs again.
            bra_local_states[op_idx] = bra_loc
            # Store how many local transitions are available for that state.
            active_transition_counts[op_idx] = transition_counts[op_idx, bra_loc]
        # 3) Enumerate all combos of loc transitions with a mixed-radix counter
        # Each digit of the counter chooses one loc transition for one operator factor.
        transition_counters = np.zeros(n_ops, np.int32)
        ket_config = np.empty(n_sites, np.int32)
        # Initialize the ket_config as the bra (unaffected sites stay unchanged)
        # where acted sites are overwritten at every mixed-radix step.
        for site_idx in range(n_sites):
            ket_config[site_idx] = bra_config[site_idx]
        finished = False
        while not finished:
            # Reset total amplitude for this specific combination of loc transitions.
            amplitude = 1.0 + 0.0j
            for op_idx in range(n_ops):
                bra_loc = bra_local_states[op_idx]
                trans_idx = transition_counters[op_idx]
                # Multiply by the local matrix element selected by the current
                # mixed-radix digit.
                amplitude *= transition_values[op_idx, bra_loc, trans_idx]
                # Overwrite only the acted site with the corresponding local ket
                # state. All untouched sites remain equal to the bra.
                ket_config[op_sites_list[op_idx]] = ket_local_states[
                    op_idx, bra_loc, trans_idx
                ]
            # Look up the full ket configuration inside the sorted symmetry basis.
            ket_idx = config_to_index_binarysearch(ket_config, sector_configs)
            # Store the matrix element <bra|O|ket> in COO format.
            row_list[write_idx] = bra_idx
            col_list[write_idx] = ket_idx
            value_list[write_idx] = amplitude
            write_idx += 1
            # increment mixed-radix counter from last digit
            for op_idx in range(n_ops - 1, -1, -1):
                transition_counters[op_idx] += 1
                if transition_counters[op_idx] < active_transition_counts[op_idx]:
                    break
                # This digit overflowed, so reset it and carry to the digit on
                # the left, exactly as in a positional numeral system.
                transition_counters[op_idx] = 0
                if op_idx == 0:
                    # Overflow of the leftmost digit means all combinations have
                    # been enumerated.
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
    n_ops = len(op_sites_list)
    # Precompute all one-site transitions once and reuse them in both passes.
    transition_counts, ket_local_states, transition_values = (
        _prepare_local_transition_data(op_list, op_sites_list)
    )
    # Count how many nonzero columns each bra state will generate.
    nnz_cols_per_row = _count_row_nnz_from_transition_data(
        sector_configs, op_sites_list, transition_counts
    )
    # Convert row counts to write offsets in the flattened triplet arrays.
    nnz_cols_per_row, total_nnz = _get_nnz_offsets(nnz_cols_per_row)
    # Preallocate the output arrays
    row_list = np.empty(total_nnz, dtype=np.int32)
    col_list = np.empty(total_nnz, dtype=np.int32)
    value_list = np.empty(total_nnz, dtype=np.complex128)
    # --- PASS 2: explicit 4-nested-loops version for n_ops=4  ---------------
    for bra_idx in prange(n_states):
        # 1) grab the bra’s full config and where to write
        bra_config = sector_configs[bra_idx]
        ptr = nnz_cols_per_row[bra_idx]
        # 2) pick the already-precomputed local transition rows matching this bra.
        bra_loc0 = bra_config[op_sites_list[0]]
        bra_loc1 = bra_config[op_sites_list[1]]
        bra_loc2 = bra_config[op_sites_list[2]]
        bra_loc3 = bra_config[op_sites_list[3]]
        count0 = transition_counts[0, bra_loc0]
        count1 = transition_counts[1, bra_loc1]
        count2 = transition_counts[2, bra_loc2]
        count3 = transition_counts[3, bra_loc3]
        # 3) a scratch array for the ket
        ket_config = np.empty(n_sites, np.int32)
        # start from the bra each time
        for site_idx in range(n_sites):
            ket_config[site_idx] = bra_config[site_idx]
        # 4) explicit 2-nested-loops evaluation
        site0 = op_sites_list[0]
        site1 = op_sites_list[1]
        site2 = op_sites_list[2]
        site3 = op_sites_list[3]
        for ii0 in range(count0):
            ket0 = ket_local_states[0, bra_loc0, ii0]
            val0 = transition_values[0, bra_loc0, ii0]
            ket_config[site0] = ket0
            for ii1 in range(count1):
                ket1 = ket_local_states[1, bra_loc1, ii1]
                val1 = transition_values[1, bra_loc1, ii1]
                ket_config[site1] = ket1
                for ii2 in range(count2):
                    ket2 = ket_local_states[2, bra_loc2, ii2]
                    val2 = transition_values[2, bra_loc2, ii2]
                    ket_config[site2] = ket2
                    for ii3 in range(count3):
                        ket3 = ket_local_states[3, bra_loc3, ii3]
                        val3 = transition_values[3, bra_loc3, ii3]
                        ket_config[site3] = ket3
                        # compute the full amplitude
                        amp = val0 * val1 * val2 * val3
                        # lookup that ket in your sorted sector_configs
                        ket_idx = config_to_index_binarysearch(
                            ket_config, sector_configs
                        )
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
    n_ops = len(op_sites_list)
    # Precompute all one-site transitions once and reuse them in both passes.
    transition_counts, ket_local_states, transition_values = (
        _prepare_local_transition_data(op_list, op_sites_list)
    )
    # Count how many nonzero columns each bra state will generate.
    nnz_cols_per_row = _count_row_nnz_from_transition_data(
        sector_configs, op_sites_list, transition_counts
    )
    # Convert row counts to write offsets in the flattened triplet arrays.
    nnz_cols_per_row, total_nnz = _get_nnz_offsets(nnz_cols_per_row)
    # Preallocate the output arrays
    row_list = np.empty(total_nnz, dtype=np.int32)
    col_list = np.empty(total_nnz, dtype=np.int32)
    value_list = np.empty(total_nnz, dtype=np.complex128)
    # --- PASS 2: explicit 2-nested-loops version for n_ops=2  ---------------
    for bra_idx in prange(n_states):
        # 1) grab the bra’s full config and where to write
        bra_config = sector_configs[bra_idx]
        ptr = nnz_cols_per_row[bra_idx]
        # 2) pick the already-precomputed local transition rows matching this bra.
        bra_loc0 = bra_config[op_sites_list[0]]
        bra_loc1 = bra_config[op_sites_list[1]]
        count0 = transition_counts[0, bra_loc0]
        count1 = transition_counts[1, bra_loc1]
        # 3) a scratch array for the ket
        ket_config = np.empty(n_sites, np.int32)
        # start from the bra once. Unaffected sites stay unchanged forever.
        for site_idx in range(n_sites):
            ket_config[site_idx] = bra_config[site_idx]
        # 4) four fully‐nested loops
        site0 = op_sites_list[0]
        site1 = op_sites_list[1]
        for ii0 in range(count0):
            ket0 = ket_local_states[0, bra_loc0, ii0]
            val0 = transition_values[0, bra_loc0, ii0]
            ket_config[site0] = ket0
            for ii1 in range(count1):
                ket1 = ket_local_states[1, bra_loc1, ii1]
                val1 = transition_values[1, bra_loc1, ii1]
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
        if np.abs(value_list[row]) >= OP_TOL:
            # Mark the row as having at least one nonzero element
            check_rows[row] = True
    # Filter out zero elements
    row_list = row_list[check_rows]
    value_list = value_list[check_rows]
    return row_list, row_list, value_list
