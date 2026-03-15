"""Combined symmetry-sector configuration builders.

This module combines global, link, and optional string/n-body constraints to
construct symmetry-reduced configuration tables used by the Hamiltonian and
observable builders.
"""

from edlgt.tools import get_time
import math
import numpy as np
import logging
from numba import njit, prange
from .global_abelian_sym import check_global_sym_sitebased, check_string_sym_sitebased
from .link_abelian_sym import check_link_sym_sitebased

logger = logging.getLogger(__name__)
SYM_TOL = 1e-13

__all__ = [
    "symmetry_sector_configs",
    "get_symmetry_sector_generators",
    "check_link_sym_partial",
    "get_link_sector_configs",
]


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
    configs = np.zeros((sector_dim, num_dims), dtype=np.uint16)
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
    loc_dims: np.ndarray,
    glob_op_diags: np.ndarray,
    glob_sectors,
    sym_type_flag,
    link_op_diags: np.ndarray,
    link_sectors,
    pair_list,
    string_op_diags: np.ndarray | None = None,
    string_sectors: np.ndarray | None = None,
) -> np.ndarray:
    """Build configurations satisfying combined global and link symmetries.

    Parameters
    ----------
    loc_dims : ndarray
        Local Hilbert-space dimensions.
    glob_op_diags : ndarray
        Site-resolved diagonals of global symmetry generators.
    glob_sectors : ndarray or sequence
        Target sector values for global symmetries.
    sym_type_flag : str
        Global symmetry type (typically ``"U"`` or ``"Z"``).
    link_op_diags : ndarray
        Site-resolved diagonals of link symmetry generators.
    link_sectors : ndarray or sequence
        Target sector values for link symmetries.
    pair_list : sequence
        Link pair definitions, grouped by lattice direction.
    string_op_diags : ndarray, optional
        Site-resolved diagonals for additional string constraints.
    string_sectors : ndarray, optional
        Target values for the string constraints.

    Returns
    -------
    ndarray
        Configurations satisfying all requested constraints.
    """
    # Normalize user-provided sector targets to arrays.
    if not isinstance(link_sectors, np.ndarray):
        link_sectors = np.array(link_sectors, dtype=float)
    if not isinstance(glob_sectors, np.ndarray):
        glob_sectors = np.array(glob_sectors, dtype=float)
    # Acquire Sector dimension
    sector_dim = math.prod(int(d) for d in loc_dims)
    bits = sum(math.log2(d) for d in loc_dims)
    logger.info(f"----------------------------------------------------")
    logger.info(f"TOT DIM: {sector_dim}, 2^{round(bits,3)}")
    # Dispatch to the string-aware or string-free iterative builder.
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
    # Report the final reduced sector dimension.
    sector_dim = len(sector_configs)
    logger.info(f"SEC DIM: {sector_dim}, 2^{round(np.log2(sector_dim),3)}")
    return sector_configs


def get_symmetry_sector_generators(op_list: list, action: str):
    """Extract site-resolved diagonals of symmetry generators.

    Parameters
    ----------
    op_list : list
        Operators grouped according to ``action``.
    action : str
        One of ``"global"``, ``"link"``, or ``"nbody"``.

    Returns
    -------
    ndarray
        Array of real diagonals formatted for the corresponding symmetry
        routines.
    """
    if action == "global":
        # Generators of Global Abelian Symmetry sector
        n_sites = op_list[0].shape[0]
        max_loc_dim = op_list[0].shape[1]
        logger.debug(f"GLOBAL: nsites: {n_sites}, max_loc_dim: {max_loc_dim}")
        op_diagonals = np.zeros((len(op_list), n_sites, max_loc_dim), dtype=float)
        for ii, operator in enumerate(op_list):
            for jj in range(n_sites):
                op_diagonals[ii, jj, :] = np.real(np.diagonal(operator[jj]))
    elif action == "link":
        # Generators of Link Abelian symmetry sector
        lattice_dim = len(op_list)
        n_sites = op_list[0][0].shape[0]
        max_loc_dim = op_list[0][0].shape[1]
        logger.debug(f"LINK: nsites: {n_sites}, max_loc_dim: {max_loc_dim}")
        op_diagonals = np.zeros((lattice_dim, 2, n_sites, max_loc_dim), dtype=float)
        for ii in range(lattice_dim):
            for jj in range(2):
                for kk in range(n_sites):
                    op_diagonals[ii, jj, kk, :] = np.real(
                        np.diagonal(op_list[ii][jj][kk])
                    )
    elif action == "nbody":
        # Generators of N-body Abelian symmetry sector
        n_symmetries = len(op_list)
        n_sites = op_list[0].shape[0]
        max_loc_dim = op_list[0].shape[1]
        logger.debug(f"nBODY: nsites: {n_sites}, max_loc_dim: {max_loc_dim}")
        op_diagonals = np.zeros((n_symmetries, n_sites, max_loc_dim), dtype=float)
        for ii, operator in enumerate(op_list):
            for jj in range(n_sites):
                op_diagonals[ii, jj, :] = np.real(np.diagonal(operator[jj]))
    return op_diagonals


@njit(cache=True)
def _prepare_global_u1_bounds(
    glob_op_diags: np.ndarray, loc_dims: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute remaining min/max U(1) contributions for branch-and-bound checks.

    Parameters
    ----------
    glob_op_diags : ndarray
        Site-resolved diagonals of global generators, shape
        ``(n_sym, n_sites, max_loc_dim)``.
    loc_dims : ndarray
        Local Hilbert-space dimensions.

    Returns
    -------
    tuple
        ``(remaining_min, remaining_max)`` arrays with shape
        ``(n_sym, n_sites + 1)``. Entry ``[:, site_idx]`` stores the minimum/maximum
        contribution reachable from sites ``site_idx..n_sites-1``.
    """
    # Global dimensions of the precomputation tables.
    n_symmetries = glob_op_diags.shape[0]
    n_sites = len(loc_dims)
    remaining_min = np.zeros((n_symmetries, n_sites + 1), dtype=np.float64)
    remaining_max = np.zeros((n_symmetries, n_sites + 1), dtype=np.float64)
    # Backward sweep: at each depth, accumulate the min/max contribution
    # still available from the yet-unassigned sites.
    for sym_idx in range(n_symmetries):
        for site_idx in range(n_sites - 1, -1, -1):
            loc_dim = int(loc_dims[site_idx])
            min_val = glob_op_diags[sym_idx, site_idx, 0]
            max_val = min_val
            for state in range(1, loc_dim):
                value = glob_op_diags[sym_idx, site_idx, state]
                if value < min_val:
                    min_val = value
                if value > max_val:
                    max_val = value
            remaining_min[sym_idx, site_idx] = (
                remaining_min[sym_idx, site_idx + 1] + min_val
            )
            remaining_max[sym_idx, site_idx] = (
                remaining_max[sym_idx, site_idx + 1] + max_val
            )
    return remaining_min, remaining_max


@njit(inline="always")
def _check_u1_reachability(
    partial_value: float, target_value: float, rem_min: float, rem_max: float
) -> bool:
    """Check whether a U(1) target can still be reached from a partial value.

    Parameters
    ----------
    partial_value : float
        Current value of the symmetry generator on the assigned prefix.
    target_value : float
        Target sector value.
    rem_min : float
        Minimum possible contribution from unassigned sites.
    rem_max : float
        Maximum possible contribution from unassigned sites.

    Returns
    -------
    bool
        ``True`` if the target is inside the reachable interval.
    """
    lower_bound = partial_value + rem_min
    upper_bound = partial_value + rem_max
    if target_value < lower_bound - SYM_TOL:
        return False
    if target_value > upper_bound + SYM_TOL:
        return False
    return True


@njit(cache=True)
def _prepare_link_activation_data(
    pair_list, n_sites: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert link-pair data into activation-indexed arrays for fast checks.

    Each link pair is checked exactly when its highest site index is reached
    during iterative configuration growth.

    Parameters
    ----------
    pair_list : sequence
        Per-direction arrays of site pairs.
    n_sites : int
        Number of sites in the full lattice configuration.

    Returns
    -------
    tuple
        ``(pair_site0, pair_site1, active_pair_ids, active_pair_counts)`` where
        ``active_pair_ids[direction, site, :]`` lists pair ids that become fully
        checkable when ``site`` is assigned.
    """
    # "Activation" rule:
    # a pair (site0, site1) can be checked only after both sites are assigned.
    # Therefore, it activates at max(site0, site1) in the iterative build.
    # PASS 1: gather pair counts and allocate compact pair tables.
    # We first build dense endpoint tables so later kernels can avoid Python lists.
    n_directions = len(pair_list)
    pair_counts = np.zeros(n_directions, dtype=np.int32)
    max_pairs = 0
    for direction_idx in range(n_directions):
        n_pairs = pair_list[direction_idx].shape[0]
        pair_counts[direction_idx] = n_pairs
        if n_pairs > max_pairs:
            max_pairs = n_pairs
    # Keep at least one column to avoid zero-sized dimensions in Numba arrays.
    if max_pairs == 0:
        max_pairs = 1
    # pair_site0/1[direction, pair_id] -> endpoints of that pair.
    pair_site0 = np.full((n_directions, max_pairs), -1, dtype=np.int32)
    pair_site1 = np.full((n_directions, max_pairs), -1, dtype=np.int32)
    # active_pair_counts[direction, site] -> how many pairs activate at this site.
    active_pair_counts = np.zeros((n_directions, n_sites), dtype=np.int32)
    # PASS 2: store pair endpoints and count how many pairs activate per site.
    for direction_idx in range(n_directions):
        n_pairs = pair_counts[direction_idx]
        for pair_idx in range(n_pairs):
            site0 = int(pair_list[direction_idx][pair_idx, 0])
            site1 = int(pair_list[direction_idx][pair_idx, 1])
            pair_site0[direction_idx, pair_idx] = site0
            pair_site1[direction_idx, pair_idx] = site1
            # A pair becomes checkable when the later endpoint is reached.
            active_site = site0 if site0 >= site1 else site1
            if 0 <= active_site < n_sites:
                active_pair_counts[direction_idx, active_site] += 1
    # Determine the largest activation bucket to allocate a padded 3D table.
    max_active_pairs = 0
    for direction_idx in range(n_directions):
        for site_idx in range(n_sites):
            if active_pair_counts[direction_idx, site_idx] > max_active_pairs:
                max_active_pairs = active_pair_counts[direction_idx, site_idx]
    # Same padding safeguard as above (Numba-friendly fixed rank arrays).
    if max_active_pairs == 0:
        max_active_pairs = 1
    # PASS 3: materialize activation lists (pair ids per site/direction).
    # Goal of active_pair_ids:
    # for each (direction, active_site) bucket, store *which original pair ids*
    # must be checked when that site is reached.
    # This is therefore a reverse lookup table:
    #     (direction, active_site, local_bucket_index) -> pair_idx
    # Example:
    # if in the x direction the pairs [0,1] and [2,3] activate at sites 1 and 3,
    # then the table stores
    #     active_pair_ids[x, 1, 0] = 0
    #     active_pair_ids[x, 3, 0] = 1
    # We store pair ids, not endpoints, because the endpoints are already stored in
    # pair_site0/pair_site1. At runtime we first select which pairs are relevant at
    # the current depth, and only then recover their endpoints from those tables.
    # The third axis is a padded bucket index. It allows multiple pairs to activate
    # at the same site in the same direction on more general lattices. Any unused
    # padded entries remain at -1.
    active_pair_ids = np.full(
        (n_directions, n_sites, max_active_pairs), -1, dtype=np.int32
    )
    # write_counts[direction, site] is the fill pointer of that bucket:
    # it tells us where the next activating pair id must be written.
    write_counts = np.zeros((n_directions, n_sites), dtype=np.int32)
    for direction_idx in range(n_directions):
        n_pairs = pair_counts[direction_idx]
        for pair_idx in range(n_pairs):
            site0 = pair_site0[direction_idx, pair_idx]
            site1 = pair_site1[direction_idx, pair_idx]
            active_site = site0 if site0 >= site1 else site1
            if 0 <= active_site < n_sites:
                # Pick the next free slot inside the (direction, active_site) bucket.
                write_pos = write_counts[direction_idx, active_site]
                # Store the original pair id in that slot.
                active_pair_ids[direction_idx, active_site, write_pos] = pair_idx
                # Advance the fill pointer so the next pair for the same bucket
                # will be written into the following slot.
                write_counts[direction_idx, active_site] += 1
    # These four arrays are the complete activation lookup package used at runtime.
    return pair_site0, pair_site1, active_pair_ids, active_pair_counts


@njit(inline="always")
def _check_link_constraints_activated(
    config: np.ndarray,
    site_idx: int,
    link_op_diags: np.ndarray,
    link_sectors: np.ndarray,
    pair_site0: np.ndarray,
    pair_site1: np.ndarray,
    active_pair_ids: np.ndarray,
    active_pair_counts: np.ndarray,
) -> bool:
    """Check only link constraints that become fully defined at ``site_idx``.

    Parameters
    ----------
    config : ndarray
        Current partial/full configuration.
    site_idx : int
        Newly assigned site index.
    link_op_diags : ndarray
        Site-resolved link diagonals.
    link_sectors : ndarray
        Target link-sector values.
    pair_site0, pair_site1 : ndarray
        Pair endpoint lookup tables.
    active_pair_ids, active_pair_counts : ndarray
        Activation-indexed pair ids and counts from
        :func:`_prepare_link_activation_data`.

    Returns
    -------
    bool
        ``True`` if all newly activated link constraints are satisfied.
    """
    # Only scan constraints activated at this depth.
    # This is the key optimization: we do not re-check old pairs.
    n_directions = active_pair_counts.shape[0]
    for direction_idx in range(n_directions):
        # Number of pairs in this direction that become fully defined at site_idx.
        n_active = active_pair_counts[direction_idx, site_idx]
        for active_idx in range(n_active):
            # Recover the original pair id from the (direction, site_idx) bucket.
            # active_idx is only the local position inside that bucket.
            pair_idx = active_pair_ids[direction_idx, site_idx, active_idx]
            # Use that pair id to recover the actual lattice endpoints of the link.
            site0 = pair_site0[direction_idx, pair_idx]
            site1 = pair_site1[direction_idx, pair_idx]
            # Evaluate the link generator on the two endpoints using local states.
            link_value = link_op_diags[direction_idx, 0, site0, config[site0]]
            link_value += link_op_diags[direction_idx, 1, site1, config[site1]]
            # Early reject as soon as one activated pair violates its sector.
            if np.abs(link_value - link_sectors[direction_idx]) > SYM_TOL:
                return False
    # If all activated pairs passed, this prefix remains admissible.
    return True


@njit(inline="always")
def _check_string_constraints_on_site(
    site_state: int,
    site_idx: int,
    string_op_diags: np.ndarray,
    string_sectors: np.ndarray,
) -> bool:
    """Check site-local string constraints only on the newly assigned site.

    Parameters
    ----------
    site_state : int
        Local basis state assigned to ``site_idx``.
    site_idx : int
        Newly assigned site index.
    string_op_diags : ndarray
        Site-resolved diagonals of string generators.
    string_sectors : ndarray
        Site-resolved target values for string generators.

    Returns
    -------
    bool
        ``True`` if all string constraints on ``site_idx`` are satisfied.
    """
    n_string_symmetries = string_op_diags.shape[0]
    for sym_idx in range(n_string_symmetries):
        if (
            np.abs(
                string_op_diags[sym_idx, site_idx, site_state]
                - string_sectors[sym_idx, site_idx]
            )
            > SYM_TOL
        ):
            return False
    return True


@njit(cache=True)
def _prepare_nbody_activation_data(
    nbody_sites_list, n_sites: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert n-body site lists into activation-indexed lookup arrays.

    Parameters
    ----------
    nbody_sites_list : sequence
        List of site-index arrays, one per n-body symmetry.
    n_sites : int
        Number of sites in the full lattice configuration.

    Returns
    -------
    tuple
        ``(nbody_sites_table, nbody_site_counts, active_sym_ids, active_sym_counts)``.
    """
    # PASS 1: determine padded table sizes.
    n_symmetries = len(nbody_sites_list)
    max_sites_per_symmetry = 0
    for sym_idx in range(n_symmetries):
        n_sites_sym = len(nbody_sites_list[sym_idx])
        if n_sites_sym > max_sites_per_symmetry:
            max_sites_per_symmetry = n_sites_sym
    if max_sites_per_symmetry == 0:
        max_sites_per_symmetry = 1
    nbody_sites_table = np.full(
        (n_symmetries, max_sites_per_symmetry), -1, dtype=np.int32
    )
    nbody_site_counts = np.zeros(n_symmetries, dtype=np.int32)
    activation_site_per_sym = np.full(n_symmetries, -1, dtype=np.int32)
    active_sym_counts = np.zeros(n_sites, dtype=np.int32)
    # PASS 2: flatten each symmetry's site list and record its activation depth.
    for sym_idx in range(n_symmetries):
        site_indices = nbody_sites_list[sym_idx]
        n_sites_sym = len(site_indices)
        nbody_site_counts[sym_idx] = n_sites_sym
        max_site_idx = -1
        for local_idx in range(n_sites_sym):
            site_idx = int(site_indices[local_idx])
            nbody_sites_table[sym_idx, local_idx] = site_idx
            if site_idx > max_site_idx:
                max_site_idx = site_idx
        activation_site_per_sym[sym_idx] = max_site_idx
        if 0 <= max_site_idx < n_sites:
            active_sym_counts[max_site_idx] += 1
    max_active_sym = 0
    for site_idx in range(n_sites):
        if active_sym_counts[site_idx] > max_active_sym:
            max_active_sym = active_sym_counts[site_idx]
    if max_active_sym == 0:
        max_active_sym = 1

    # PASS 3: materialize activation-indexed symmetry ids.
    active_sym_ids = np.full((n_sites, max_active_sym), -1, dtype=np.int32)
    write_counts = np.zeros(n_sites, dtype=np.int32)
    for sym_idx in range(n_symmetries):
        activation_site = activation_site_per_sym[sym_idx]
        if 0 <= activation_site < n_sites:
            write_pos = write_counts[activation_site]
            active_sym_ids[activation_site, write_pos] = sym_idx
            write_counts[activation_site] += 1
    return nbody_sites_table, nbody_site_counts, active_sym_ids, active_sym_counts


@njit(inline="always")
def _check_nbody_constraints_activated(
    config: np.ndarray,
    site_idx: int,
    nbody_op_diags: np.ndarray,
    nbody_sectors: np.ndarray,
    nbody_sym_value: int,
    nbody_sites_table: np.ndarray,
    nbody_site_counts: np.ndarray,
    active_sym_ids: np.ndarray,
    active_sym_counts: np.ndarray,
) -> bool:
    """Check only n-body constraints activated by the newly assigned site.

    Parameters
    ----------
    config : ndarray
        Current partial/full configuration.
    site_idx : int
        Newly assigned site index.
    nbody_op_diags : ndarray
        Site-resolved n-body diagonals.
    nbody_sectors : ndarray
        Target n-body sector values.
    nbody_sym_value : int
        ``0`` for additive (U1-like) and ``1`` for multiplicative (Z2-like)
        n-body checks.
    nbody_sites_table, nbody_site_counts, active_sym_ids, active_sym_counts : ndarray
        Activation data produced by :func:`_prepare_nbody_activation_data`.

    Returns
    -------
    bool
        ``True`` if all newly activated n-body constraints are satisfied.
    """
    # Evaluate only symmetries that become fully specified at this depth.
    n_active = active_sym_counts[site_idx]
    for active_idx in range(n_active):
        sym_idx = active_sym_ids[site_idx, active_idx]
        n_sites_sym = nbody_site_counts[sym_idx]
        if nbody_sym_value == 0:
            sym_value = 0.0
            for local_idx in range(n_sites_sym):
                site = nbody_sites_table[sym_idx, local_idx]
                sym_value += nbody_op_diags[sym_idx, site, config[site]]
        else:
            sym_value = 1.0
            for local_idx in range(n_sites_sym):
                site = nbody_sites_table[sym_idx, local_idx]
                sym_value *= nbody_op_diags[sym_idx, site, config[site]]
        if np.abs(sym_value - nbody_sectors[sym_idx]) > SYM_TOL:
            return False
    return True


@njit(cache=True)
def check_link_sym_partial(config, sym_op_diags, sym_sectors, pair_list):
    """
    Partial check of link symmetries for a configuration.
    The function does not remove already checked pairs, allowing
    a simpler implementation without redundancy management.

    Parameters
    ----------
    config : ndarray
        Partial configuration (prefix of a full many-body configuration).
    sym_op_diags : ndarray
        Site-resolved diagonals of link symmetry generators.
    sym_sectors : ndarray
        Target sector values for the link generators.
    pair_list : sequence
        Per-direction arrays of site-index pairs.

    Returns
    -------
    bool
        ``True`` if the partial configuration does not violate any fully
        specified link constraint.
    """
    n_sites = config.shape[0]
    num_lattice_directions = len(pair_list)
    # Iterate over all lattice directions
    for idx in range(num_lattice_directions):
        pairs_for_direction = pair_list[idx]
        num_pairs = pairs_for_direction.shape[0]
        # Iterate over all pairs in the current direction
        for pair_idx in range(num_pairs):
            site_indices = pairs_for_direction[pair_idx]
            # Skip pairs where one or both indices are out of the current configuration length
            if site_indices[0] >= n_sites or site_indices[1] >= n_sites:
                continue
            # Compute the sum for the symmetry condition
            sum = 0.0
            for op_idx in range(2):  # Two operators per link
                site_index = site_indices[op_idx]
                op_diag = sym_op_diags[idx, op_idx, site_index, :]
                sum += op_diag[config[site_index]]

            # Check if the sum violates the symmetry sector
            if not np.isclose(sum, sym_sectors[idx], atol=1e-13):
                return False

    return True


@njit(cache=True)
def check_nbody_sym_partial(config, sym_op_diags, sym_sectors, nsites_list):
    """
    Partial check of nbody symmetries for a configuration.
    NOTE: All the nbody symmetries are supposed to be of type U(1) and to involve the same number of sites.

    Args:
        config (np.array of np.uint16): Partial QMB configuration array.
        sym_op_diags (np.array of floats): 3D array with shape=(n_symmetries, n_sites, max(loc_dims)).
            Each symmetry applies an operator to n_sites number of sites (the same operator on all the given sites).
            Each operator is represented by its diagonal on each lattice site.
        sym_sectors (np.array of floats): 1D array of shape=(n_symmetries) with the sector value of each symmetry.
        nsites_list (list of np.array of np.uint16): List of 1D arrays.
            Each array specifies the sites on which the n_ops operators (of each symmetry) act.

    Returns:
        bool: True if the partial config satisfies the nbody symmetries, False otherwise.
    """
    n_sites = config.shape[0]
    n_symmetries = sym_op_diags.shape[0]
    # Iterate over all nbody symmetries
    for sym_idx in range(n_symmetries):
        site_indices = nsites_list[sym_idx]
        nsites_per_symmetry = len(site_indices)
        # Skip symmetries where one or more indices
        # are out of the current configuration length
        if np.any(site_indices >= n_sites):
            continue
        # Compute the sum for the symmetry condition
        sum = 0.0
        for ii in range(nsites_per_symmetry):
            site_index = site_indices[ii]
            op_diag = sym_op_diags[sym_idx, site_index, :]
            sum += op_diag[config[site_index]]
        # Check if the sum violates the symmetry sector
        if not np.isclose(sum, sym_sectors[sym_idx], atol=1e-13):
            return False
    return True


@njit(cache=True)
def check_Z2_nbody_sym_partial(config, sym_op_diags, sym_sectors, nsites_list):
    """
    Partial check of Z2 nbody symmetries for a configuration.
    NOTE: All the nbody symmetries are supposed to be of type U(1) and to involve the same number of sites.

    Args:
        config (np.array of np.uint16): Partial QMB configuration array.
        sym_op_diags (np.array of floats): 3D array with shape=(n_symmetries, n_sites, max(loc_dims)).
            Each symmetry applies an operator to n_sites number of sites (the same operator on all the given sites).
            Each operator is represented by its diagonal on each lattice site.
        sym_sectors (np.array of floats): 1D array of shape=(n_symmetries) with the sector value of each symmetry.
        nsites_list (list of np.array of np.uint16): List of 1D arrays.
            Each array specifies the sites on which the n_ops operators (of each symmetry) act.

    Returns:
        bool: True if the partial config satisfies the nbody symmetries, False otherwise.
    """
    n_sites = config.shape[0]
    n_symmetries = sym_op_diags.shape[0]
    # Iterate over all nbody symmetries
    for sym_idx in range(n_symmetries):
        site_indices = nsites_list[sym_idx]
        nsites_per_symmetry = len(site_indices)
        # Skip symmetries where one or more indices
        # are out of the current configuration length
        if np.any(site_indices >= n_sites):
            continue
        # Compute the sum for the symmetry condition
        sum = 1.0
        for ii in range(nsites_per_symmetry):
            site_index = site_indices[ii]
            op_diag = sym_op_diags[sym_idx, site_index, :]
            sum *= op_diag[config[site_index]]
        # Check if the sum violates the symmetry sector
        if not np.isclose(sum, sym_sectors[sym_idx], atol=1e-13):
            return False
    return True


@njit(cache=True)
def check_glob_sym_partial(config, sym_op_diags, sym_sectors, sym_type_flag):
    """
    Check if a (partial or complete) QMB state configuration belongs to a global abelian symmetry sector.
    For U(1) symmetry, partial configurations are checked to ensure the quantum number does not exceed the target sector.
    For Z2 symmetry, the check is only performed for complete configurations.

    Args:
        config (np.array of np.uint16): 1D array with the state of each lattice site.

        sym_op_diags (np.array of floats): 3D array of shape=(num_operators, n_sites, max(loc_dims)).
            Each operator's diagonal is expressed in the proper basis of each lattice site.

        sym_sectors (np.array of floats): 1D array with sector values for each operator.
            NOTE: sym_sectors.shape[0] = sym_op_diags.shape[0].

        sym_type_flag (int): Flag indicating the symmetry type (0 = U(1), 1 = Z2).

    Returns:
        bool: True if the (partial or complete) configuration is compatible with the sector, False otherwise.
    """
    # Number of sites in the current configuration
    n_sites = config.shape[0]
    num_operators = sym_op_diags.shape[0]
    # Total number of lattice sites
    max_sites = sym_op_diags.shape[1]
    for jj in range(num_operators):
        # U(1) case
        if sym_type_flag == 0:
            operation_result = 0
            for kk in range(n_sites):
                operation_result += sym_op_diags[jj, kk, config[kk]]
            # Check if partial configuration is valid
            if n_sites < max_sites:
                if operation_result - sym_sectors[jj] > 1e-13:
                    return False
            # Full configuration must match exactly
            else:
                if not np.isclose(operation_result, sym_sectors[jj], atol=1e-13):
                    return False
        # Z2 case
        elif sym_type_flag == 1:
            # Skip Z2 check for partial configurations
            if n_sites < max_sites:
                continue
            # Full configuration must match exactly
            else:
                operation_result = 1
                for kk in range(n_sites):
                    operation_result *= sym_op_diags[jj, kk, config[kk]]
                if not np.isclose(operation_result, sym_sectors[jj], atol=1e-13):
                    return False
    return True


@njit(parallel=True, cache=True)
def iterative_sitebased_sym_sector_configs(
    loc_dims: np.ndarray,
    glob_op_diags: np.ndarray,
    glob_sectors: np.ndarray,
    sym_type_flag: int,
    link_op_diags: np.ndarray,
    link_sectors: np.ndarray,
    pair_list,
) -> np.ndarray:
    """Iteratively build the global+link sector with incremental constraint checks.

    The routine grows configurations one site at a time and checks only the
    newly activated constraints at each depth:

    - global U(1): update running sums and prune branches using remaining
      min/max reachable values;
    - global Z2: update running products and validate exactly only at full depth;
    - link constraints: validate only pairs whose largest site index equals the
      newly assigned site.

    Parameters
    ----------
    loc_dims : ndarray
        Local Hilbert-space dimensions.
    glob_op_diags : ndarray
        Site-resolved diagonals of global symmetry generators.
    glob_sectors : ndarray
        Target sector values for global generators.
    sym_type_flag : int
        Symmetry flag for global constraints (``0`` for U1, ``1`` for Z2).
    link_op_diags : ndarray
        Site-resolved diagonals of link symmetry generators.
    link_sectors : ndarray
        Target sector values for link generators.
    pair_list : sequence
        Per-direction arrays of link site pairs.

    Returns
    -------
    ndarray
        Configurations belonging to the requested global+link sector.
    """
    n_sites = len(loc_dims)
    n_glob_syms = glob_op_diags.shape[0]
    # Precompute branch-and-bound helpers for global U(1) checks.
    remaining_min, remaining_max = _prepare_global_u1_bounds(glob_op_diags, loc_dims)
    # Precompute link-pair activation data.
    (pair_site0, pair_site1, active_pair_ids, active_pair_counts) = (
        _prepare_link_activation_data(pair_list, n_sites)
    )
    # STEP 1: initialize depth-0 configurations (single-site prefixes).
    loc_dim0 = int(loc_dims[0])
    # Each row of configs_prev is a one-site candidate config
    configs_prev = np.empty((loc_dim0, 1), dtype=np.uint16)
    # global_vals_prev[ii] stores the already accumulated value
    # of each global symmetry for prefix configs_prev[ii]
    global_vals_prev = np.empty((loc_dim0, n_glob_syms), dtype=np.float64)
    # Boolean mask saying which candidate survives
    checks_prev = np.zeros(loc_dim0, dtype=np.bool_)
    for config_idx in prange(loc_dim0):
        site_state = config_idx
        # Store 1-site config candidate
        configs_prev[config_idx, 0] = site_state
        is_valid = True
        # Run over global symmetries
        for sym_idx in range(n_glob_syms):
            # take the value of the global_sym_op in that site state
            site_value = glob_op_diags[sym_idx, 0, site_state]
            # store that value
            global_vals_prev[config_idx, sym_idx] = site_value
            if sym_type_flag == 0:
                # U(1): check reachability of the target with remaining sites.
                if n_sites == 1:
                    if np.abs(site_value - glob_sectors[sym_idx]) > SYM_TOL:
                        is_valid = False
                        break
                # check if glob_sectors[sym_idx] \in [site_value + rem_min, site_value + rem_max]
                elif not _check_u1_reachability(
                    site_value,
                    glob_sectors[sym_idx],
                    remaining_min[sym_idx, 1],
                    remaining_max[sym_idx, 1],
                ):
                    is_valid = False
                    break
            else:
                # Z2: only full configurations are constrained.
                if n_sites == 1:
                    if np.abs(site_value - glob_sectors[sym_idx]) > SYM_TOL:
                        is_valid = False
                        break
        if is_valid and not _check_link_constraints_activated(
            configs_prev[config_idx],
            0,
            link_op_diags,
            link_sectors,
            pair_site0,
            pair_site1,
            active_pair_ids,
            active_pair_counts,
        ):
            is_valid = False
        # Save if that config is valid or not
        checks_prev[config_idx] = is_valid
    # Filter the configurations keeping only the valid ones
    configs_prev = configs_prev[checks_prev]
    # Filter the global_prevalues keeping only the valid ones
    global_vals_prev = global_vals_prev[checks_prev]
    n_configs_prev = configs_prev.shape[0]
    # STEP 2: grow prefixes site-by-site and prune invalid branches.
    for site_idx in range(1, n_sites):
        loc_dim_next = int(loc_dims[site_idx])
        # update number of possible configs (the valid ones * all states of next site)
        n_configs_next = n_configs_prev * loc_dim_next
        configs_next = np.empty((n_configs_next, site_idx + 1), dtype=np.uint16)
        global_vals_next = np.empty((n_configs_next, n_glob_syms), dtype=np.float64)
        checks_next = np.zeros(n_configs_next, dtype=np.bool_)
        for config_idx in prange(n_configs_next):
            # get the index of the previous config up to site_dix -1
            prev_config_idx = config_idx // loc_dim_next
            # get the state of the current site_dix
            site_state = config_idx % loc_dim_next
            # store the state up to the previous site
            configs_next[config_idx, :site_idx] = configs_prev[prev_config_idx]
            # store the state of the current site
            configs_next[config_idx, site_idx] = site_state
            is_valid = True
            # run over the different symmetries
            for sym_idx in range(n_glob_syms):
                # check the previous global value of the symmetry
                prev_value = global_vals_prev[prev_config_idx, sym_idx]
                # get the value added by the current site
                site_value = glob_op_diags[sym_idx, site_idx, site_state]
                if sym_type_flag == 0:
                    # update the symmetry value
                    new_value = prev_value + site_value
                    # update the global_prevalues
                    global_vals_next[config_idx, sym_idx] = new_value
                    if site_idx < n_sites - 1:
                        # check if target is still reachable in the next site
                        if not _check_u1_reachability(
                            new_value,
                            glob_sectors[sym_idx],
                            remaining_min[sym_idx, site_idx + 1],
                            remaining_max[sym_idx, site_idx + 1],
                        ):
                            is_valid = False
                            break
                    # check the target value
                    elif np.abs(new_value - glob_sectors[sym_idx]) > SYM_TOL:
                        is_valid = False
                        break
                else:
                    # update the symmetry value
                    new_value = prev_value * site_value
                    global_vals_next[config_idx, sym_idx] = new_value
                    if site_idx == n_sites - 1:
                        if np.abs(new_value - glob_sectors[sym_idx]) > SYM_TOL:
                            is_valid = False
                            break
            # check the new available link symmetries
            if is_valid and not _check_link_constraints_activated(
                configs_next[config_idx],
                site_idx,
                link_op_diags,
                link_sectors,
                pair_site0,
                pair_site1,
                active_pair_ids,
                active_pair_counts,
            ):
                is_valid = False
            # save if the config is valid
            checks_next[config_idx] = is_valid
        # restrict the configs to the feasible ones
        configs_prev = configs_next[checks_next]
        # get the global_prevalues
        global_vals_prev = global_vals_next[checks_next]
        # measure the number of current configs
        n_configs_prev = configs_prev.shape[0]
    # STEP 3: final survivors are full configurations in the target sector.
    return configs_prev


@njit(parallel=True, cache=True)
def iterative_sitebased_sym_sector_configs1(
    loc_dims: np.ndarray,
    glob_op_diags: np.ndarray,
    glob_sectors: np.ndarray,
    sym_type_flag: int,
    link_op_diags: np.ndarray,
    link_sectors: np.ndarray,
    pair_list,
    string_op_diags: np.ndarray,
    string_sectors: np.ndarray,
) -> np.ndarray:
    """Iteratively build the global+link+string sector with incremental checks.

    Compared to :func:`iterative_sitebased_sym_sector_configs`, this variant also
    enforces site-local string constraints immediately when each new site is
    assigned.

    Parameters
    ----------
    loc_dims : ndarray
        Local Hilbert-space dimensions.
    glob_op_diags : ndarray
        Site-resolved diagonals of global symmetry generators.
    glob_sectors : ndarray
        Target sector values for global generators.
    sym_type_flag : int
        Symmetry flag for global constraints (``0`` for U1, ``1`` for Z2).
    link_op_diags : ndarray
        Site-resolved diagonals of link symmetry generators.
    link_sectors : ndarray
        Target sector values for link generators.
    pair_list : sequence
        Per-direction arrays of link site pairs.
    string_op_diags : ndarray
        Site-resolved diagonals of string-like constraints.
    string_sectors : ndarray
        Site-resolved target values for string-like constraints.

    Returns
    -------
    ndarray
        Configurations belonging to the requested global+link+string sector.
    """
    n_sites = len(loc_dims)
    n_glob_syms = glob_op_diags.shape[0]
    # Precompute branch-and-bound helpers for global U(1) checks.
    remaining_min, remaining_max = _prepare_global_u1_bounds(glob_op_diags, loc_dims)
    # Precompute link-pair activation data.
    (pair_site0, pair_site1, active_pair_ids, active_pair_counts) = (
        _prepare_link_activation_data(pair_list, n_sites)
    )
    # STEP 1: initialize depth-0 configurations.
    loc_dim0 = int(loc_dims[0])
    # Each row of configs_prev is a one-site candidate config
    configs_prev = np.empty((loc_dim0, 1), dtype=np.uint16)
    # global_vals_prev[ii] stores the already accumulated value
    # of each global symmetry for prefix configs_prev[ii]
    global_vals_prev = np.empty((loc_dim0, n_glob_syms), dtype=np.float64)
    # Boolean mask saying which candidate survives
    checks_prev = np.zeros(loc_dim0, dtype=np.bool_)
    for config_idx in prange(loc_dim0):
        site_state = config_idx
        # Store 1-site config candidate
        configs_prev[config_idx, 0] = site_state
        # Immediately check the string-like constraint on the new site
        is_valid = _check_string_constraints_on_site(
            site_state, 0, string_op_diags, string_sectors
        )
        if is_valid:
            # Run over global symmetries
            for sym_idx in range(n_glob_syms):
                # take the value of the global_sym_op in that site state
                site_value = glob_op_diags[sym_idx, 0, site_state]
                # store that value
                global_vals_prev[config_idx, sym_idx] = site_value
                if sym_type_flag == 0:
                    if n_sites == 1:
                        if np.abs(site_value - glob_sectors[sym_idx]) > SYM_TOL:
                            is_valid = False
                            break
                    # check if glob_sectors[sym_idx] \in [site_value + rem_min, site_value + rem_max]
                    elif not _check_u1_reachability(
                        site_value,
                        glob_sectors[sym_idx],
                        remaining_min[sym_idx, 1],
                        remaining_max[sym_idx, 1],
                    ):
                        is_valid = False
                        break
                else:
                    if (
                        n_sites == 1
                        and np.abs(site_value - glob_sectors[sym_idx]) > SYM_TOL
                    ):
                        is_valid = False
                        break
        # Check the links that become available after assigning site 0
        if is_valid and not _check_link_constraints_activated(
            configs_prev[config_idx],
            0,
            link_op_diags,
            link_sectors,
            pair_site0,
            pair_site1,
            active_pair_ids,
            active_pair_counts,
        ):
            is_valid = False
        # Save if that config is valid or not
        checks_prev[config_idx] = is_valid
    # Filter the configurations keeping only the valid ones
    configs_prev = configs_prev[checks_prev]
    # Filter the global_prevalues keeping only the valid ones
    global_vals_prev = global_vals_prev[checks_prev]
    n_configs_prev = configs_prev.shape[0]
    # STEP 2: extend prefixes and apply string/global/link checks incrementally.
    for site_idx in range(1, n_sites):
        loc_dim_next = int(loc_dims[site_idx])
        # update number of possible configs (the valid ones * all states of next site)
        n_configs_next = n_configs_prev * loc_dim_next
        configs_next = np.empty((n_configs_next, site_idx + 1), dtype=np.uint16)
        global_vals_next = np.empty((n_configs_next, n_glob_syms), dtype=np.float64)
        checks_next = np.zeros(n_configs_next, dtype=np.bool_)
        for config_idx in prange(n_configs_next):
            # get the index of the previous config up to site_dix -1
            prev_config_idx = config_idx // loc_dim_next
            # get the state of the current site_dix
            site_state = config_idx % loc_dim_next
            # store the state up to the previous site
            configs_next[config_idx, :site_idx] = configs_prev[prev_config_idx]
            # store the state of the current site
            configs_next[config_idx, site_idx] = site_state
            # First check the string-like constraint on the new site only
            is_valid = _check_string_constraints_on_site(
                site_state, site_idx, string_op_diags, string_sectors
            )
            if is_valid:
                # run over the different symmetries
                for sym_idx in range(n_glob_syms):
                    # check the previous global value of the symmetry
                    prev_value = global_vals_prev[prev_config_idx, sym_idx]
                    # get the value added by the current site
                    site_value = glob_op_diags[sym_idx, site_idx, site_state]
                    if sym_type_flag == 0:
                        # update the symmetry value
                        new_value = prev_value + site_value
                        # update the global_prevalues
                        global_vals_next[config_idx, sym_idx] = new_value
                        if site_idx < n_sites - 1:
                            # check if target is still reachable in the next site
                            if not _check_u1_reachability(
                                new_value,
                                glob_sectors[sym_idx],
                                remaining_min[sym_idx, site_idx + 1],
                                remaining_max[sym_idx, site_idx + 1],
                            ):
                                is_valid = False
                                break
                        # check the target value
                        elif np.abs(new_value - glob_sectors[sym_idx]) > SYM_TOL:
                            is_valid = False
                            break
                    else:
                        # update the symmetry value
                        new_value = prev_value * site_value
                        global_vals_next[config_idx, sym_idx] = new_value
                        if (
                            site_idx == n_sites - 1
                            and np.abs(new_value - glob_sectors[sym_idx]) > SYM_TOL
                        ):
                            is_valid = False
                            break
            # check the new available link symmetries
            if is_valid and not _check_link_constraints_activated(
                configs_next[config_idx],
                site_idx,
                link_op_diags,
                link_sectors,
                pair_site0,
                pair_site1,
                active_pair_ids,
                active_pair_counts,
            ):
                is_valid = False
            # save if the config is valid
            checks_next[config_idx] = is_valid
        # restrict the configs to the feasible ones
        configs_prev = configs_next[checks_next]
        # get the global_prevalues
        global_vals_prev = global_vals_next[checks_next]
        # measure the number of current configs
        n_configs_prev = configs_prev.shape[0]
    # STEP 3: final survivors are full configurations in the target sector.
    return configs_prev


@get_time
def get_link_sector_configs(
    loc_dims: np.ndarray,
    link_op_diags: np.ndarray,
    link_sectors,
    pair_list,
    nbody_op_diags: np.ndarray | None = None,
    nbody_sectors: np.ndarray | None = None,
    nbody_sites_list=None,
    nbody_sym_type: str | None = "U",
) -> np.ndarray:
    """Build configurations satisfying link symmetries and optional n-body constraints.

    Parameters
    ----------
    loc_dims : ndarray
        Local Hilbert-space dimensions.
    link_op_diags : ndarray
        Site-resolved diagonals of link symmetry generators.
    link_sectors : ndarray or sequence
        Target sector values for link symmetries.
    pair_list : sequence
        Per-direction arrays of site-index pairs.
    nbody_op_diags : ndarray, optional
        Site-resolved diagonals of additional n-body generators.
    nbody_sectors : ndarray or sequence, optional
        Target sector values for the n-body generators.
    nbody_sites_list : sequence, optional
        Site-index lists for each n-body symmetry generator.
    nbody_sym_type : str or None, optional
        ``"U"`` or ``"Z"`` for additive/multiplicative n-body constraints.

    Returns
    -------
    ndarray
        Configurations satisfying the requested constraints.
    """
    logger.debug("GETTING LINK SECTOR CONFIGURATIONS")
    # Normalize target sectors before entering JIT kernels.
    if not isinstance(link_sectors, np.ndarray):
        link_sectors = np.array(link_sectors, dtype=float)
    # Acquire Sector dimension
    sector_dim = math.prod(int(d) for d in loc_dims)
    bits = sum(math.log2(d) for d in loc_dims)
    logger.info(f"TOT DIM: {sector_dim}, 2^{bits:3f}")
    # Choose between link-only and link+nbody iterative constructors.
    if nbody_op_diags is not None:
        if not isinstance(nbody_sectors, np.ndarray):
            nbody_sectors = np.array(nbody_sectors, dtype=float)
        nbody_sym_value = 0 if nbody_sym_type == "U" else 1
        sector_configs = iterative_link_sector_configs_plus(
            loc_dims,
            link_op_diags,
            link_sectors,
            pair_list,
            nbody_op_diags,
            nbody_sectors,
            nbody_sites_list,
            nbody_sym_value,
        )
    else:
        sector_configs = iterative_link_sector_configs(
            loc_dims, link_op_diags, link_sectors, pair_list
        )
    # Acquire dimension of the new sector
    sector_dim = len(sector_configs)
    logger.info(f"SEC DIM: {sector_dim}, 2^{np.log2(sector_dim):3f}")
    return sector_configs


@njit(parallel=True, cache=True)
def iterative_link_sector_configs(
    loc_dims: np.ndarray,
    link_op_diags: np.ndarray,
    link_sectors: np.ndarray,
    pair_list,
) -> np.ndarray:
    """Iteratively build the link-only symmetry sector with activated checks.

    Parameters
    ----------
    loc_dims : ndarray
        Local Hilbert-space dimensions.
    link_op_diags : ndarray
        Site-resolved diagonals of link symmetry generators.
    link_sectors : ndarray
        Target link-sector values.
    pair_list : sequence
        Per-direction arrays of link site pairs.

    Returns
    -------
    ndarray
        Configurations satisfying all link constraints.
    """
    n_sites = len(loc_dims)
    # Precompute link-pair activation data.
    (
        pair_site0,
        pair_site1,
        active_pair_ids,
        active_pair_counts,
    ) = _prepare_link_activation_data(pair_list, n_sites)
    # STEP 1: initialize depth-0 configurations.
    loc_dim0 = int(loc_dims[0])
    # Each row of configs_prev is a one-site candidate config
    configs_prev = np.empty((loc_dim0, 1), dtype=np.uint16)
    # Boolean mask saying which candidate survives
    checks_prev = np.zeros(loc_dim0, dtype=np.bool_)
    for config_idx in prange(loc_dim0):
        site_state = config_idx
        # Store 1-site config candidate
        configs_prev[config_idx, 0] = site_state
        # Check the links that become available after assigning site 0
        checks_prev[config_idx] = _check_link_constraints_activated(
            configs_prev[config_idx],
            0,
            link_op_diags,
            link_sectors,
            pair_site0,
            pair_site1,
            active_pair_ids,
            active_pair_counts,
        )
    # Filter the configurations keeping only the valid ones
    configs_prev = configs_prev[checks_prev]
    n_configs_prev = configs_prev.shape[0]
    # STEP 2: extend prefixes and evaluate only activated link constraints.
    for site_idx in range(1, n_sites):
        loc_dim_next = int(loc_dims[site_idx])
        # update number of possible configs (the valid ones * all states of next site)
        n_configs_next = n_configs_prev * loc_dim_next
        configs_next = np.empty((n_configs_next, site_idx + 1), dtype=np.uint16)
        checks_next = np.zeros(n_configs_next, dtype=np.bool_)
        for config_idx in prange(n_configs_next):
            # get the index of the previous config up to site_dix -1
            prev_config_idx = config_idx // loc_dim_next
            # get the state of the current site_dix
            site_state = config_idx % loc_dim_next
            # store the state up to the previous site
            configs_next[config_idx, :site_idx] = configs_prev[prev_config_idx]
            # store the state of the current site
            configs_next[config_idx, site_idx] = site_state
            # check the new available link symmetries
            checks_next[config_idx] = _check_link_constraints_activated(
                configs_next[config_idx],
                site_idx,
                link_op_diags,
                link_sectors,
                pair_site0,
                pair_site1,
                active_pair_ids,
                active_pair_counts,
            )
        # restrict the configs to the feasible ones
        configs_prev = configs_next[checks_next]
        # measure the number of current configs
        n_configs_prev = configs_prev.shape[0]
    # STEP 3: return full configurations that survived all checks.
    return configs_prev


@njit(parallel=True, cache=True)
def iterative_link_sector_configs_plus(
    loc_dims: np.ndarray,
    link_op_diags: np.ndarray,
    link_sectors: np.ndarray,
    pair_list,
    nbody_op_diags: np.ndarray,
    nbody_sectors: np.ndarray,
    nbody_sites_list,
    nbody_sym_value: int,
) -> np.ndarray:
    """Iteratively build the link+nbody sector with activated constraints.

    Parameters
    ----------
    loc_dims : ndarray
        Local Hilbert-space dimensions.
    link_op_diags : ndarray
        Site-resolved diagonals of link symmetry generators.
    link_sectors : ndarray
        Target link-sector values.
    pair_list : sequence
        Per-direction arrays of link site pairs.
    nbody_op_diags : ndarray
        Site-resolved diagonals of additional n-body generators.
    nbody_sectors : ndarray
        Target n-body sector values.
    nbody_sites_list : sequence
        Site-index lists defining each n-body generator support.
    nbody_sym_value : int
        ``0`` for additive (U1-like) and ``1`` for multiplicative (Z2-like)
        n-body checks.

    Returns
    -------
    ndarray
        Configurations satisfying link and n-body constraints.
    """
    n_sites = len(loc_dims)
    # Precompute link-pair activation data.
    (
        pair_site0,
        pair_site1,
        active_pair_ids,
        active_pair_counts,
    ) = _prepare_link_activation_data(pair_list, n_sites)
    # Precompute nbody activation data.
    (
        nbody_sites_table,
        nbody_site_counts,
        active_sym_ids,
        active_sym_counts,
    ) = _prepare_nbody_activation_data(nbody_sites_list, n_sites)

    # STEP 1: initialize depth-0 configurations.
    loc_dim0 = int(loc_dims[0])
    # Each row of configs_prev is a one-site candidate config
    configs_prev = np.empty((loc_dim0, 1), dtype=np.uint16)
    # Boolean mask saying which candidate survives
    checks_prev = np.zeros(loc_dim0, dtype=np.bool_)
    for config_idx in prange(loc_dim0):
        site_state = config_idx
        # Store 1-site config candidate
        configs_prev[config_idx, 0] = site_state
        # Check the links that become available after assigning site 0
        is_valid = _check_link_constraints_activated(
            configs_prev[config_idx],
            0,
            link_op_diags,
            link_sectors,
            pair_site0,
            pair_site1,
            active_pair_ids,
            active_pair_counts,
        )
        if is_valid:
            # Check the nbody symmetries that become available after assigning site 0
            is_valid = _check_nbody_constraints_activated(
                configs_prev[config_idx],
                0,
                nbody_op_diags,
                nbody_sectors,
                nbody_sym_value,
                nbody_sites_table,
                nbody_site_counts,
                active_sym_ids,
                active_sym_counts,
            )
        # Save if that config is valid or not
        checks_prev[config_idx] = is_valid
    # Filter the configurations keeping only the valid ones
    configs_prev = configs_prev[checks_prev]
    n_configs_prev = configs_prev.shape[0]
    # STEP 2: extend prefixes and apply activated link+nbody checks.
    for site_idx in range(1, n_sites):
        loc_dim_next = int(loc_dims[site_idx])
        # update number of possible configs (the valid ones * all states of next site)
        n_configs_next = n_configs_prev * loc_dim_next
        configs_next = np.empty((n_configs_next, site_idx + 1), dtype=np.uint16)
        checks_next = np.zeros(n_configs_next, dtype=np.bool_)
        for config_idx in prange(n_configs_next):
            # get the index of the previous config up to site_dix -1
            prev_config_idx = config_idx // loc_dim_next
            # get the state of the current site_dix
            site_state = config_idx % loc_dim_next
            # store the state up to the previous site
            configs_next[config_idx, :site_idx] = configs_prev[prev_config_idx]
            # store the state of the current site
            configs_next[config_idx, site_idx] = site_state
            # check the new available link symmetries
            is_valid = _check_link_constraints_activated(
                configs_next[config_idx],
                site_idx,
                link_op_diags,
                link_sectors,
                pair_site0,
                pair_site1,
                active_pair_ids,
                active_pair_counts,
            )
            if is_valid:
                # check the new available nbody symmetries
                is_valid = _check_nbody_constraints_activated(
                    configs_next[config_idx],
                    site_idx,
                    nbody_op_diags,
                    nbody_sectors,
                    nbody_sym_value,
                    nbody_sites_table,
                    nbody_site_counts,
                    active_sym_ids,
                    active_sym_counts,
                )
            # save if the config is valid
            checks_next[config_idx] = is_valid
        # restrict the configs to the feasible ones
        configs_prev = configs_next[checks_next]
        # measure the number of current configs
        n_configs_prev = configs_prev.shape[0]
    # STEP 3: return full configurations that survived all constraints.
    return configs_prev
