"""Translation symmetry and momentum-basis construction utilities.

This module builds translation orbits and sparse momentum-basis projectors for
symmetry-sector configuration tables, and provides momentum-projected sparse
operator kernels for one-, two-, and four-site factorized operators.
"""

import numpy as np
from numba import njit, prange
from .generate_configs import config_to_index_binarysearch

OP_EPS = 1e-12  # small cutoff for local operator entries

__all__ = [
    "check_normalization",
    "check_orthogonality",
    "get_momentum_basis",
    "nbody_data_momentum_4sites",
    "nbody_data_momentum_2sites",
    "nbody_data_momentum_1site",
]


@njit(cache=True)
def _prefix_sum_counts(nnz_per_col: np.ndarray) -> np.ndarray:
    """
    Turn column nnz counts into CSC col_ptr with a prefix sum.
    col_ptr has length (n_cols + 1) and col_ptr[-1] = total_nnz.
    """
    n_cols = nnz_per_col.size
    col_ptr = np.empty(n_cols + 1, np.int32)
    s = 0
    col_ptr[0] = 0
    for j in range(n_cols):
        s += nnz_per_col[j]
        col_ptr[j + 1] = s
    return col_ptr


@njit(inline="always")
def _insertion_sort_by_row(rows: np.ndarray, vals: np.ndarray, length: int) -> None:
    """
    Small, stable, numba-friendly in-place sort by 'rows' for the first 'length' items.
    Keeps CSC column rows in ascending order (nice to have, often expected).
    """
    for i in range(1, length):
        r_key = rows[i]
        v_key = vals[i]
        j = i - 1
        while j >= 0 and rows[j] > r_key:
            rows[j + 1] = rows[j]
            vals[j + 1] = vals[j]
            j -= 1
        rows[j + 1] = r_key
        vals[j + 1] = v_key


@njit(cache=True, parallel=True)
def precompute_C_sign_per_config(
    sector_configs: np.ndarray,  # (N, L) int32
    C_label_sign: np.ndarray,  # (d_loc,) float64, entries ∈ {+1.0, -1.0}
) -> np.ndarray:
    """
    Returns S[i] = ∏_j C_label_sign[ sector_configs[i, j] ] as float64 ±1.
    Works for both k=0 and finite-k paths.
    """
    N, L = sector_configs.shape
    out = np.ones(N, np.float64)
    for i in prange(N):
        s = 1.0
        for j in range(L):
            s *= C_label_sign[sector_configs[i, j]]
        out[i] = s
    return out


# ---------- linear index <-> coords (ROW-MAJOR)----------
@njit(inline="always")
def linear_to_coords_rowmajor(
    site_index: int, lvals: np.ndarray, out_coords: np.ndarray
) -> None:
    """
    Convert a linear site index [0..prod(L)-1] into D coords in row-major order.
    out_coords is preallocated (length D).
    """
    lattice_dim = lvals.size
    for ax in range(lattice_dim - 1, -1, -1):
        L = lvals[ax]
        out_coords[ax] = site_index % L
        site_index //= L


@njit(inline="always")
def coords_to_linear_rowmajor(coords: np.ndarray, lvals: np.ndarray) -> int:
    """
    Convert D coords -> linear site index in row-major order.
    """
    lattice_dim = lvals.size
    idx = 0
    for ax in range(lattice_dim):
        idx = idx * lvals[ax] + coords[ax]
    return idx


# ---------- mixed-radix encode/decode of per-axis block-shifts ----------
@njit(inline="always")
def encode_shift(axis_shifts: np.ndarray, shifts_per_dir: np.ndarray) -> int:
    """
    Map a D-vector of per-axis block shifts t=(t0,...,t_{D-1})
    with ranges [0..R_d-1] to a single flat index in [0..prod(R)-1], using row-major.
    """
    flat = 0
    lattice_dim = shifts_per_dir.size
    for ax in range(lattice_dim):
        flat = flat * shifts_per_dir[ax] + (axis_shifts[ax] % shifts_per_dir[ax])
    return flat


@njit(inline="always")
def decode_shift(
    flat_index: int, shifts_per_dir: np.ndarray, out_axis_shifts: np.ndarray
) -> None:
    """
    The inverse of encode_shift: flat -> D-vector of per-axis shifts.
    """
    lattice_dim = shifts_per_dir.size
    for ax in range(lattice_dim - 1, -1, -1):
        out_axis_shifts[ax] = flat_index % shifts_per_dir[ax]
        flat_index //= shifts_per_dir[ax]


@njit(inline="always")
def decode_mixed_index(idx: int, bases: np.ndarray, out: np.ndarray) -> None:
    """
    Mixed-radix decode matching encode_shift's row-major convention:
    axis 0 is most significant, axis D-1 least significant.
    """
    lattice_dim = bases.size
    for ax in range(lattice_dim - 1, -1, -1):
        out[ax] = idx % bases[ax]
        idx //= bases[ax]


@njit(cache=True)
def check_normalization(basis: np.ndarray) -> bool:
    """Check whether all columns of a basis matrix are normalized.

    Parameters
    ----------
    basis : ndarray
        Basis matrix with basis vectors stored as columns.

    Returns
    -------
    bool
        ``True`` if every column has unit norm.
    """
    for ii in range(basis.shape[1]):
        if not np.isclose(np.linalg.norm(basis[:, ii]), 1):
            return False
    return True


@njit(cache=True)
def check_orthogonality(basis: np.ndarray) -> bool:
    """Check whether the columns of a basis matrix are mutually orthogonal.

    Parameters
    ----------
    basis : ndarray
        Basis matrix with basis vectors stored as columns.

    Returns
    -------
    bool
        ``True`` if all distinct column pairs are orthogonal.
    """
    for ii in range(basis.shape[1]):
        for jj in range(ii + 1, basis.shape[1]):
            if not np.isclose(np.vdot(basis[:, ii], basis[:, jj]), 0, atol=1e-10):
                return False
    return True


@njit(cache=True, parallel=True)
def build_TC_translations(
    sector_configs: np.ndarray,  # (N, L) dressed local-state ids
    C_map: np.ndarray,  # (d_loc,) local map for even sites
):
    """
    Build the orbit table of the combined generator X = T ∘ C on a 1D ring.
    IMPORTANT: At each step apply C (sitewise, even/odd possibly different),
    then translate by +1. Repeat this t times to get X^t.

    Returns:
        Ttab           : (N, L) int32, Ttab[i, t] = index of (X^t)|config_i>
        shifts_per_dir : (1,)   int32, with shifts_per_dir[0] = L
    """
    N, L = sector_configs.shape
    R = L
    Ttab = np.empty((N, R), np.int32)

    for cfg_idx in prange(N):
        base = sector_configs[cfg_idx]

        # current config after t applications (start at t=0)
        curr = np.empty(L, np.int32)
        for j in range(L):
            curr[j] = base[j]
        Ttab[cfg_idx, 0] = cfg_idx  # X^0 = identity

        work = np.empty(L, np.int32)  # scratch for C action

        for t in range(1, R):
            # 1) apply C at this step in the LAB frame (before translating)
            for j in range(L):
                a = curr[j]
                if (j & 1) == 0:  # even site index
                    work[j] = C_map[a]
                else:  # odd site index
                    work[j] = C_map[a]

            # 2) translate by +1: dest[(j+1) % L] = work[j]
            for j in range(L):
                curr[(j + 1) % L] = work[j]

            # 3) lookup
            Ttab[cfg_idx, t] = config_to_index_binarysearch(curr, sector_configs)

    shifts_per_dir = np.empty(1, np.int32)
    shifts_per_dir[0] = R
    return Ttab, shifts_per_dir


# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, parallel=True)
def build_all_translations(
    sector_configs: np.ndarray,
    lvals: np.ndarray,
    unit_cell_size: np.ndarray,
):
    """
    Build the table of all allowed block-translations for every configuration in sector_config.
    For each direction ax the number of possible translations or shifts are
    shifts_per_dir[ax] = lvals[ax] // unit_cell_size[ax]

    Given a fixed config, roll it along ax by multiples of unit_cell_size[ax]
    and record the index within sector_configs of each translation.

    Theory:
    -------
    Allowed translations are t_d * s_d along axis d, with t_d in [0..R_d-1],
    where R_d = L_d / s_d. The full translation group is the product of those
    cyclic groups. For each config c, its orbit points are T(t) c for all t.

    Returns:
        translations_array: int32 array of shape (N_configs, prod(R_d))
            translations_array[i, s] gives the index (row in sector_configs)
            of the configuration obtained by applying the block-shift decoded
            from 's' to configuration i.
        shifts_per_dir: int32 array (D,) with R_d = L_d / s_d.
    """
    n_configs, n_sites = sector_configs.shape
    lattice_dim = lvals.size
    shifts_per_dir = lvals // unit_cell_size
    total_shifts = np.prod(shifts_per_dir)
    # Allocate the memory for the array with all the translations
    translations_array = np.empty((n_configs, total_shifts), np.int32)
    # Run over all the configs
    for cfg_idx in prange(n_configs):
        cfg = sector_configs[cfg_idx]
        # per-thread scratch (PRIVATE per cfg_idx)
        coords = np.empty(lattice_dim, np.int32)
        new_coords = np.empty(lattice_dim, np.int32)
        # Consider all the possible combined shift (t0,...,t_{D-1}) along each direction
        # For each flat shift, decode the per-axis shifts and build the rolled config
        for flat_shift in range(total_shifts):
            axis_shifts = np.empty(lattice_dim, np.int32)
            decode_shift(flat_shift, shifts_per_dir, axis_shifts)
            # Allocate the memory for the corresponding rolled configuration
            rolled_cfg = np.empty(n_sites, sector_configs.dtype)
            # Apply translation: move each site by (t_d * s_d) on axis d
            for site_idx in range(n_sites):
                # Find the corresponding volume coordinates
                linear_to_coords_rowmajor(site_idx, lvals, coords)
                # Obtain shifted coordinates
                for ax in range(lattice_dim):
                    new_coords[ax] = (
                        coords[ax] + axis_shifts[ax] * unit_cell_size[ax]
                    ) % lvals[ax]
                # Get back to the corresponding flat index
                new_site_idx = coords_to_linear_rowmajor(new_coords, lvals)
                # Implement the shift on the new flat index
                rolled_cfg[new_site_idx] = cfg[site_idx]
            # Find the index of the rolled config in sector_configs
            translations_array[cfg_idx, flat_shift] = config_to_index_binarysearch(
                rolled_cfg, sector_configs
            )
    return translations_array, shifts_per_dir


@njit(cache=True)
def select_references(
    translations_array: np.ndarray,  # shape (n_configs, total_shifts). Each row: orbit via all block-shifts
    shifts_per_dir: np.ndarray,  # shape (D,). R_d = L_d // s_d (number of block positions per axis)
    k_vals: np.ndarray,  # shape (D,), momentum labels k_d in [0..R_d-1]
):
    """
    Pick one reference per orbit, compute its axis-period vector p (minimal block
    shifts along each axis returning to itself), and keep the orbit iff
    (k_d * p_d) % R_d == 0 for ALL axes d.

    Theory:
    -------
    - The orbit of i is the set { translations_array[i, s] for s in 0..total_shifts-1 }.
    - Axis period p_d is the smallest p>0 such that shifting by p blocks on axis d
      returns i to itself. In the table, that means:
        translations_array[i, encode_shift(p * e_d)] == i.
    - Momentum sector k is consistent with the orbit iff the character is trivial
      on the stabilizer: exp(-2π i k_d p_d / R_d) == 1, i.e. (k_d * p_d) % R_d == 0.
    """
    n_configs, total_shifts = translations_array.shape
    lattice_dim = shifts_per_dir.size
    # Output buffers (over-allocated, trimmed at the end)
    references = np.empty(n_configs, np.int32)
    period_vectors = np.empty((n_configs, lattice_dim), np.int32)
    # Bookkeeping: which config indices already belong to an orbit we've processed?
    assigned = np.zeros(n_configs, np.uint8)
    # Scratch for “axis-only” shift (p * e_d)
    axis_only = np.zeros(lattice_dim, np.int32)
    n_refs = 0
    # loop in index order
    for cfg_idx in range(n_configs):
        if assigned[cfg_idx]:
            # Already included in a previously discovered orbit
            continue
        # Row listing all T(t)|cfg_idx>, for all combined block shifts t
        orbit_row = translations_array[cfg_idx]
        # ---------- 1) Compute the axis period vector p (minimal periods) ----------
        pvec = np.empty(lattice_dim, np.int32)
        for ax in range(lattice_dim):
            found = False
            # Scan p = 1..R_d until the shift along axis 'ax' returns to the reference
            for p in range(1, shifts_per_dir[ax] + 1):
                axis_only[:] = 0
                axis_only[ax] = p  # p * e_d
                flat = encode_shift(axis_only, shifts_per_dir)
                if orbit_row[flat] == cfg_idx:  # returned to itself
                    pvec[ax] = p
                    found = True
                    break
            if not found:
                # Should not occur for well-formed tables; fall back to the full cycle
                pvec[ax] = shifts_per_dir[ax]
        # ---------- 2) Mark the ENTIRE orbit as assigned ----------
        # Important: do this before the momentum filter, so we never duplicate orbits
        for s in range(total_shifts):
            assigned[orbit_row[s]] = 1
        # ---------- 3) Momentum-compatibility filter ----------
        # Keep the orbit iff (k_d * p_d) % R_d == 0 for all axes d
        keep = True
        for ax in range(lattice_dim):
            if (k_vals[ax] * pvec[ax]) % shifts_per_dir[ax] != 0:
                keep = False
                break
        if not keep:
            # Orbit incompatible with the requested momentum; skip it
            continue
        # ---------- 4) Accept this orbit representative ----------
        references[n_refs] = cfg_idx
        for ax in range(lattice_dim):
            period_vectors[n_refs, ax] = pvec[ax]
        n_refs += 1
    # Trim the over-allocated outputs
    return references[:n_refs], period_vectors[:n_refs, :]


@njit(cache=True, parallel=True)
def momentum_basis_zero_k(
    sector_configs: np.ndarray,  # (n_configs, n_sites) int32
    lvals: np.ndarray,  # (D,) int32 lattice lengths
    unit_cell_size: np.ndarray,  # (D,) int32 block sizes s_d (must divide L_d)
):
    """
    Build the Γ (k=0) momentum projector B in **sparse form**:
      - CSC arrays: (L_col_ptr, L_row_idx, L_data)  — float64
      - CSR arrays: (R_row_ptr, R_col_idx, R_data)  — float64

    Mathematical content (Γ sector):
    --------------------------------
    For each translation orbit (represented by a 'reference' config with axis
    period-vector p), the Γ vector is:
        |Γ; ref> = (1 / sqrt(∏_d p_d)) * sum_{0 <= t_d < p_d} T(t) |ref>.
    All phases are 1, so the basis is real.

    Implementation overview:
    ------------------------
    1) Precompute the translation table and per-axis block counts R_d.
    2) Choose one orbit representative per orbit (select_references with k=0).
    3) Two-pass CSC build:
       PASS 1: for each column (reference), deduplicate images over the “period box”
               and count how many unique rows it will write (nnz_per_col).
       PASS 2: repeat, but write the normalized values into CSC arrays.
               (We also sort row indices within each column for canonical CSC.)
    4) Build a CSR view from CSC (row-wise prefix sum + scatter).

    Returns
    -------
    L_col_ptr : (n_cols+1,) int32
    L_row_idx : (nnz,)      int32
    L_data    : (nnz,)      float64
    R_row_ptr : (n_rows+1,) int32
    R_col_idx : (nnz,)      int32
    R_data    : (nnz,)      float64

    Note
    ----
    You can get:
      n_rows = sector_configs.shape[0]
      n_cols = L_col_ptr.shape[0] - 1
    """
    # ---------- 0) Sizes & translations ----------
    n_rows = sector_configs.shape[0]
    translations_array, shifts_per_dir = build_all_translations(
        sector_configs, lvals, unit_cell_size
    )
    lattice_dim = shifts_per_dir.size
    # ---------- 1) Orbit representatives and their period-vectors ----------
    k_zero = np.zeros(lattice_dim, np.int32)
    references, period_vectors = select_references(
        translations_array, shifts_per_dir, k_zero
    )
    n_cols = references.shape[0]
    # ---------- 2) PASS 1: count per-column nonzeros (after dedup) ----------
    nnz_per_col = np.zeros(n_cols, np.int32)
    for col_idx in prange(n_cols):
        ref_cfg_index = references[col_idx]
        pvec = period_vectors[col_idx, :]  # (D,)
        # Upper bound for distinct images when scanning the period box
        orbit_size = np.prod(pvec)
        # Per-column dedup accumulator (indices + counts)
        used_indices = np.empty(orbit_size, np.int32)
        used_values = np.zeros(orbit_size, np.float64)  # counts (phase=1)
        used_len = 0
        # local mixed-radix index in the period box
        t_local = np.zeros(lattice_dim, np.int32)
        # Enumerate all t in 0..p_d-1 and map via the precomputed table
        for flat_local in range(orbit_size):
            decode_mixed_index(flat_local, pvec, t_local)
            flat_full = encode_shift(t_local, shifts_per_dir)  # base R_d
            cfg_row = translations_array[ref_cfg_index, flat_full]
            # -----------------------------------------------------
            # deduplicate: linear scan OK (orbit boxes are small)
            pos = -1
            for u in range(used_len):
                if used_indices[u] == cfg_row:
                    pos = u
                    break
            if pos == -1:
                pos = used_len
                used_indices[pos] = cfg_row
                used_len += 1
            used_values[pos] += 1
        nnz_per_col[col_idx] = used_len
    # CSC structure
    L_col_ptr = _prefix_sum_counts(nnz_per_col)
    total_nnz = L_col_ptr[-1]
    L_row_idx = np.empty(total_nnz, np.int32)
    L_data = np.empty(total_nnz, np.float64)
    # ---------- 3) PASS 2: fill CSC (dedup + normalize + (optional) sort) ----------
    for col_idx in prange(n_cols):
        ref_cfg_index = references[col_idx]
        pvec = period_vectors[col_idx, :]
        # same bound
        orbit_size = np.prod(pvec)
        used_indices = np.empty(orbit_size, np.int32)
        used_values = np.zeros(orbit_size, np.float64)
        used_len = 0
        t_local = np.zeros(lattice_dim, np.int32)
        for flat_local in range(orbit_size):
            decode_mixed_index(flat_local, pvec, t_local)
            flat_full = encode_shift(t_local, shifts_per_dir)
            cfg_row = translations_array[ref_cfg_index, flat_full]
            # -----------------------------------------------------
            pos = -1
            for u in range(used_len):
                if used_indices[u] == cfg_row:
                    pos = u
                    break
            if pos == -1:
                pos = used_len
                used_indices[pos] = cfg_row
                used_len += 1
            used_values[pos] += 1
        # Normalize the column by its 2-norm (counts are real here)
        norm_sq = 0.0
        for u in range(used_len):
            norm_sq += used_values[u] * used_values[u]
        if np.isclose(norm_sq, 0.0):
            continue
        inv_norm = 1.0 / np.sqrt(norm_sq)
        # (Optional but nice): sort by row index for canonical CSC
        _insertion_sort_by_row(used_indices, used_values, used_len)
        # Write this column’s entries into CSC arrays
        write = L_col_ptr[col_idx]
        for u in range(used_len):
            L_row_idx[write] = used_indices[u]
            L_data[write] = used_values[u] * inv_norm
            write += 1
    # ---------- 4) Build CSR view from CSC (no SciPy) ----------
    R_row_ptr = np.zeros(n_rows + 1, np.int32)
    # row counts
    for p in range(total_nnz):
        r = L_row_idx[p]
        R_row_ptr[r + 1] += 1
    # prefix sum
    for r in range(n_rows):
        R_row_ptr[r + 1] += R_row_ptr[r]
    # scatter
    R_col_idx = np.empty(total_nnz, np.int32)
    R_data = np.empty(total_nnz, np.float64)
    # work heads (copy)
    heads = np.empty(n_rows, np.int32)
    for r in range(n_rows):
        heads[r] = R_row_ptr[r]
    for col in range(n_cols):
        start = L_col_ptr[col]
        stop = L_col_ptr[col + 1]
        for p in range(start, stop):
            r = L_row_idx[p]
            q = heads[r]
            R_col_idx[q] = col
            R_data[q] = L_data[p]
            heads[r] += 1
    return (L_col_ptr, L_row_idx, L_data, R_row_ptr, R_col_idx, R_data)


@njit(cache=True, parallel=True)
def momentum_basis_finite_k(
    sector_configs: np.ndarray,  # (n_configs, n_sites) int32
    lvals: np.ndarray,  # (D,) int32
    unit_cell_size: np.ndarray,  # (D,) int32 (must divide L_d)
    k_vals: np.ndarray,  # (D,) int32   momenta mod R_d
):
    """
    Build the finite-k momentum projector B in sparse form:

      - CSC arrays: (L_col_ptr, L_row_idx, L_data)    — complex128
      - CSR arrays: (R_row_ptr, R_col_idx, R_data)    — complex128

    Math:
      R_d = L_d / s_d, group G = Z_{R0} × ... × Z_{R_{D-1}}.
      For an orbit representative `ref` with axis‐periods p_d, the finite-k Bloch sum is
        |k; ref> = (1/√N) ∑_{0≤t_d<p_d} exp[-2πi Σ_d (k_d t_d / R_d)] T(t) |ref>,
      where N = ∑_{distinct images j} |amplitude_j|^2 after deduplication.
      If the orbit is incompatible with k, all amplitudes cancel → empty column.

    Determinism:
      Each column is built independently (per-iteration scratch), then we sort
      its (row, value) pairs by row index before writing CSC.

    Returns
    -------
    L_col_ptr : (n_cols+1,) int32
    L_row_idx : (nnz,)      int32
    L_data    : (nnz,)      complex128
    R_row_ptr : (n_rows+1,) int32
    R_col_idx : (nnz,)      int32
    R_data    : (nnz,)      complex128

    (As usual: n_rows = sector_configs.shape[0], n_cols = L_col_ptr.size - 1.)
    """
    # Tolerances for pruning true zeros / near-incompatible columns
    TOL_ZERO = 1e-14  # per-entry amplitude threshold
    TOL_COLNORM = 1e-30  # column norm^2 threshold
    # ---------- 0) Sizes & translations ----------
    n_rows = sector_configs.shape[0]
    translations_array, shifts_per_dir = build_all_translations(
        sector_configs, lvals, unit_cell_size
    )
    lattice_dim = shifts_per_dir.size
    # ---------- 1) Orbit representatives & periods (filtered by k_vals) ----------
    references, period_vectors = select_references(
        translations_array, shifts_per_dir, k_vals
    )
    n_cols = references.shape[0]
    # ---------- 2) PASS 1: count nonzeros per column (after cancellations) ----------
    nnz_per_col = np.zeros(n_cols, np.int32)
    for col_idx in prange(n_cols):
        ref_cfg_index = references[col_idx]
        pvec = period_vectors[col_idx, :]  # (D,)
        # Orbit-box upper bound
        orbit_size = 1
        for ax in range(lattice_dim):
            orbit_size *= pvec[ax]
        # Per-column accumulators (dedup by image row)
        used_indices = np.empty(orbit_size, np.int32)
        used_values = np.zeros(orbit_size, np.complex128)
        used_len = 0
        t_local = np.zeros(lattice_dim, np.int32)
        # Enumerate period box, accumulate complex phase per distinct image
        for flat_local in range(orbit_size):
            decode_mixed_index(flat_local, pvec, t_local)
            # phase = exp(-2πi Σ_d (k_d * t_d / R_d))
            phase_arg = 0.0
            for ax in range(lattice_dim):
                kd = k_vals[ax] % shifts_per_dir[ax]
                phase_arg += (kd * t_local[ax]) / float(shifts_per_dir[ax])
            phase = np.exp(-1j * 2.0 * np.pi * phase_arg)
            flat_full = encode_shift(t_local, shifts_per_dir)  # base R_d
            cfg_row = translations_array[ref_cfg_index, flat_full]
            # -----------------------------------------------------
            # deduplicate by row
            pos = -1
            for u in range(used_len):
                if used_indices[u] == cfg_row:
                    pos = u
                    break
            if pos == -1:
                pos = used_len
                used_indices[pos] = cfg_row
                used_len += 1
            used_values[pos] += phase
        # Count only truly non-zero amplitudes after cancellation
        cnt = 0
        for u in range(used_len):
            if np.abs(used_values[u]) > TOL_ZERO:
                cnt += 1
        nnz_per_col[col_idx] = cnt
    # Allocate CSC
    L_col_ptr = _prefix_sum_counts(nnz_per_col)
    total_nnz = L_col_ptr[-1]
    L_row_idx = np.empty(total_nnz, np.int32)
    L_data = np.empty(total_nnz, np.complex128)
    # ---------- 3) PASS 2: fill CSC (dedup + prune zeros + normalize + sort) ----------
    for col_idx in prange(n_cols):
        ref_cfg_index = references[col_idx]
        pvec = period_vectors[col_idx, :]
        orbit_size = np.prod(pvec)
        used_indices = np.empty(orbit_size, np.int32)
        used_values = np.zeros(orbit_size, np.complex128)
        used_len = 0
        t_local = np.zeros(lattice_dim, np.int32)
        # Deduplicate + accumulate complex phases
        for flat_local in range(orbit_size):
            decode_mixed_index(flat_local, pvec, t_local)
            phase_arg = 0.0
            for ax in range(lattice_dim):
                kd = k_vals[ax] % shifts_per_dir[ax]
                phase_arg += (kd * t_local[ax]) / float(shifts_per_dir[ax])
            phase = np.exp(-1j * 2.0 * np.pi * phase_arg)
            flat_full = encode_shift(t_local, shifts_per_dir)
            cfg_row = translations_array[ref_cfg_index, flat_full]
            # -----------------------------------------------------
            # deduplicate by row
            pos = -1
            for u in range(used_len):
                if used_indices[u] == cfg_row:
                    pos = u
                    break
            if pos == -1:
                pos = used_len
                used_indices[pos] = cfg_row
                used_len += 1
            used_values[pos] += phase
        # In-place prune entries that cancelled to ~0, and compute column norm
        kept_len = 0
        norm_sq = 0.0
        for u in range(used_len):
            a = used_values[u]
            if np.abs(a) > TOL_ZERO:
                used_indices[kept_len] = used_indices[u]
                used_values[kept_len] = a
                norm_sq += a.real * a.real + a.imag * a.imag
                kept_len += 1
        # If the column fully cancels, write nothing
        if norm_sq <= TOL_COLNORM or kept_len == 0:
            continue
        inv_norm = 1.0 / np.sqrt(norm_sq)
        # Sort by row index for canonical CSC
        _insertion_sort_by_row(used_indices, used_values, kept_len)
        # Materialize
        write = L_col_ptr[col_idx]
        for u in range(kept_len):
            L_row_idx[write] = used_indices[u]
            L_data[write] = used_values[u] * inv_norm
            write += 1
    # ---------- 4) Build CSR from CSC ----------
    R_row_ptr = np.zeros(n_rows + 1, np.int32)
    for p in range(total_nnz):
        r = L_row_idx[p]
        R_row_ptr[r + 1] += 1
    for r in range(n_rows):
        R_row_ptr[r + 1] += R_row_ptr[r]
    R_col_idx = np.empty(total_nnz, np.int32)
    R_data = np.empty(total_nnz, np.complex128)
    heads = np.empty(n_rows, np.int32)
    for r in range(n_rows):
        heads[r] = R_row_ptr[r]
    for col in range(n_cols):
        start = L_col_ptr[col]
        stop = L_col_ptr[col + 1]
        for p in range(start, stop):
            r = L_row_idx[p]
            q = heads[r]
            R_col_idx[q] = col
            R_data[q] = L_data[p]
            heads[r] += 1
    return (L_col_ptr, L_row_idx, L_data, R_row_ptr, R_col_idx, R_data)


@njit(cache=True, parallel=True)
def momentum_basis_zero_k_TC(sector_configs: np.ndarray):
    """
    Build the Γ (k=0) momentum projector B in **sparse form**:
      - CSC arrays: (L_col_ptr, L_row_idx, L_data)  — float64
      - CSR arrays: (R_row_ptr, R_col_idx, R_data)  — float64

    Mathematical content (Γ sector):
    --------------------------------
    For each translation orbit (represented by a 'reference' config with axis
    period-vector p), the Γ vector is:
        |Γ; ref> = (1 / sqrt(∏_d p_d)) * sum_{0 <= t_d < p_d} T(t) |ref>.
    All phases are 1, so the basis is real.

    Implementation overview:
    ------------------------
    1) Precompute the translation table and per-axis block counts R_d.
    2) Choose one orbit representative per orbit (select_references with k=0).
    3) Two-pass CSC build:
       PASS 1: for each column (reference), deduplicate images over the “period box”
               and count how many unique rows it will write (nnz_per_col).
       PASS 2: repeat, but write the normalized values into CSC arrays.
               (We also sort row indices within each column for canonical CSC.)
    4) Build a CSR view from CSC (row-wise prefix sum + scatter).

    Returns
    -------
    L_col_ptr : (n_cols+1,) int32
    L_row_idx : (nnz,)      int32
    L_data    : (nnz,)      float64
    R_row_ptr : (n_rows+1,) int32
    R_col_idx : (nnz,)      int32
    R_data    : (nnz,)      float64

    Note
    ----
    You can get:
      n_rows = sector_configs.shape[0]
      n_cols = L_col_ptr.shape[0] - 1
    """
    # ---------- 0) Sizes & translations ----------
    n_rows = sector_configs.shape[0]
    # Special 1D case with TC+inversion symmetry
    C_map = np.array([4, 5, 2, 3, 0, 1], dtype=np.int32)
    C_phase_label = np.array([+1, +1, +1, -1, +1, +1], dtype=np.float64)
    C_phase_per_config = precompute_C_sign_per_config(sector_configs, C_phase_label)
    translations_array, shifts_per_dir = build_TC_translations(sector_configs, C_map)
    lattice_dim = shifts_per_dir.size
    # ---------- 1) Orbit representatives and their period-vectors ----------
    k_zero = np.zeros(lattice_dim, np.int32)
    references, period_vectors = select_references(
        translations_array, shifts_per_dir, k_zero
    )
    n_cols = references.shape[0]
    # ---------- 2) PASS 1: count per-column nonzeros (after dedup) ----------
    nnz_per_col = np.zeros(n_cols, np.int32)
    for col_idx in prange(n_cols):
        ref_cfg_index = references[col_idx]
        pvec = period_vectors[col_idx, :]  # (D,)
        # Upper bound for distinct images when scanning the period box
        orbit_size = np.prod(pvec)
        # Per-column dedup accumulator (indices + counts)
        used_indices = np.empty(orbit_size, np.int32)
        used_values = np.zeros(orbit_size, np.float64)  # counts (phase=1)
        used_len = 0
        # local mixed-radix index in the period box
        t_local = np.zeros(lattice_dim, np.int32)
        # Enumerate all t in 0..p_d-1 and map via the precomputed table
        for flat_local in range(orbit_size):
            decode_mixed_index(flat_local, pvec, t_local)
            flat_full = encode_shift(t_local, shifts_per_dir)  # base R_d
            cfg_row = translations_array[ref_cfg_index, flat_full]
            # -----------------------------------------------------
            incr = 1.0
            # Special 1D case with TC symmetry
            ref_sign = C_phase_per_config[ref_cfg_index]
            if (flat_full & 1) == 1:  # t odd?
                incr = ref_sign
            # deduplicate: linear scan OK (orbit boxes are small)
            pos = -1
            for u in range(used_len):
                if used_indices[u] == cfg_row:
                    pos = u
                    break
            if pos == -1:
                pos = used_len
                used_indices[pos] = cfg_row
                used_len += 1
            # change here, before was +1
            used_values[pos] += incr
        nnz_per_col[col_idx] = used_len
    # CSC structure
    L_col_ptr = _prefix_sum_counts(nnz_per_col)
    total_nnz = L_col_ptr[-1]
    L_row_idx = np.empty(total_nnz, np.int32)
    L_data = np.empty(total_nnz, np.float64)
    # ---------- 3) PASS 2: fill CSC (dedup + normalize + (optional) sort) ----------
    for col_idx in prange(n_cols):
        ref_cfg_index = references[col_idx]
        pvec = period_vectors[col_idx, :]
        # same bound
        orbit_size = np.prod(pvec)
        used_indices = np.empty(orbit_size, np.int32)
        used_values = np.zeros(orbit_size, np.float64)
        used_len = 0
        t_local = np.zeros(lattice_dim, np.int32)
        for flat_local in range(orbit_size):
            decode_mixed_index(flat_local, pvec, t_local)
            flat_full = encode_shift(t_local, shifts_per_dir)
            cfg_row = translations_array[ref_cfg_index, flat_full]
            # -----------------------------------------------------
            incr = 1.0
            # Special 1D case with TC symmetry
            ref_sign = C_phase_per_config[ref_cfg_index]
            if (flat_full & 1) == 1:  # t odd?
                incr = ref_sign
            pos = -1
            for u in range(used_len):
                if used_indices[u] == cfg_row:
                    pos = u
                    break
            if pos == -1:
                pos = used_len
                used_indices[pos] = cfg_row
                used_len += 1
            used_values[pos] += incr
        # Normalize the column by its 2-norm (counts are real here)
        norm_sq = 0.0
        for u in range(used_len):
            norm_sq += used_values[u] * used_values[u]
        if np.isclose(norm_sq, 0.0):
            continue
        inv_norm = 1.0 / np.sqrt(norm_sq)
        # (Optional but nice): sort by row index for canonical CSC
        _insertion_sort_by_row(used_indices, used_values, used_len)
        # Write this column’s entries into CSC arrays
        write = L_col_ptr[col_idx]
        for u in range(used_len):
            L_row_idx[write] = used_indices[u]
            L_data[write] = used_values[u] * inv_norm
            write += 1
    # ---------- 4) Build CSR view from CSC (no SciPy) ----------
    R_row_ptr = np.zeros(n_rows + 1, np.int32)
    # row counts
    for p in range(total_nnz):
        r = L_row_idx[p]
        R_row_ptr[r + 1] += 1
    # prefix sum
    for r in range(n_rows):
        R_row_ptr[r + 1] += R_row_ptr[r]
    # scatter
    R_col_idx = np.empty(total_nnz, np.int32)
    R_data = np.empty(total_nnz, np.float64)
    # work heads (copy)
    heads = np.empty(n_rows, np.int32)
    for r in range(n_rows):
        heads[r] = R_row_ptr[r]
    for col in range(n_cols):
        start = L_col_ptr[col]
        stop = L_col_ptr[col + 1]
        for p in range(start, stop):
            r = L_row_idx[p]
            q = heads[r]
            R_col_idx[q] = col
            R_data[q] = L_data[p]
            heads[r] += 1
    return (L_col_ptr, L_row_idx, L_data, R_row_ptr, R_col_idx, R_data)


@njit(cache=True, parallel=True)
def momentum_basis_finite_k_TC(
    sector_configs: np.ndarray,  # (n_configs, n_sites) int32
    k_vals: np.ndarray,  # (D,) int32   momenta mod R_d
):
    """
    Build the finite-k momentum projector B in sparse form:

      - CSC arrays: (L_col_ptr, L_row_idx, L_data)    — complex128
      - CSR arrays: (R_row_ptr, R_col_idx, R_data)    — complex128

    Math:
      R_d = L_d / s_d, group G = Z_{R0} × ... × Z_{R_{D-1}}.
      For an orbit representative `ref` with axis‐periods p_d, the finite-k Bloch sum is
        |k; ref> = (1/√N) ∑_{0≤t_d<p_d} exp[-2πi Σ_d (k_d t_d / R_d)] T(t) |ref>,
      where N = ∑_{distinct images j} |amplitude_j|^2 after deduplication.
      If the orbit is incompatible with k, all amplitudes cancel → empty column.

    Determinism:
      Each column is built independently (per-iteration scratch), then we sort
      its (row, value) pairs by row index before writing CSC.

    Returns
    -------
    L_col_ptr : (n_cols+1,) int32
    L_row_idx : (nnz,)      int32
    L_data    : (nnz,)      complex128
    R_row_ptr : (n_rows+1,) int32
    R_col_idx : (nnz,)      int32
    R_data    : (nnz,)      complex128

    (As usual: n_rows = sector_configs.shape[0], n_cols = L_col_ptr.size - 1.)
    """
    # Tolerances for pruning true zeros / near-incompatible columns
    TOL_ZERO = 1e-14  # per-entry amplitude threshold
    TOL_COLNORM = 1e-30  # column norm^2 threshold
    # ---------- 0) Sizes & translations ----------
    n_rows = sector_configs.shape[0]
    # Special 1D case with TC+inversion symmetry
    C_map = np.array([4, 5, 2, 3, 0, 1], dtype=np.int32)
    C_phase_label = np.array([+1, +1, +1, -1, +1, +1], dtype=np.float64)
    C_phase_per_config = precompute_C_sign_per_config(sector_configs, C_phase_label)
    translations_array, shifts_per_dir = build_TC_translations(sector_configs, C_map)
    lattice_dim = shifts_per_dir.size
    # ---------- 1) Orbit representatives & periods (filtered by k_vals) ----------
    references, period_vectors = select_references(
        translations_array, shifts_per_dir, k_vals
    )
    n_cols = references.shape[0]
    # ---------- 2) PASS 1: count nonzeros per column (after cancellations) ----------
    nnz_per_col = np.zeros(n_cols, np.int32)
    for col_idx in prange(n_cols):
        ref_cfg_index = references[col_idx]
        pvec = period_vectors[col_idx, :]  # (D,)
        # Orbit-box upper bound
        orbit_size = 1
        for ax in range(lattice_dim):
            orbit_size *= pvec[ax]
        # Per-column accumulators (dedup by image row)
        used_indices = np.empty(orbit_size, np.int32)
        used_values = np.zeros(orbit_size, np.complex128)
        used_len = 0
        t_local = np.zeros(lattice_dim, np.int32)
        # Enumerate period box, accumulate complex phase per distinct image
        for flat_local in range(orbit_size):
            decode_mixed_index(flat_local, pvec, t_local)
            # phase = exp(-2πi Σ_d (k_d * t_d / R_d))
            phase_arg = 0.0
            for ax in range(lattice_dim):
                kd = k_vals[ax] % shifts_per_dir[ax]
                phase_arg += (kd * t_local[ax]) / float(shifts_per_dir[ax])
            phase = np.exp(-1j * 2.0 * np.pi * phase_arg)
            flat_full = encode_shift(t_local, shifts_per_dir)  # base R_d
            cfg_row = translations_array[ref_cfg_index, flat_full]
            # -----------------------------------------------------
            incr = 1.0
            # In 1D SU(2) we can implement the TC symmetry
            ref_sign = C_phase_per_config[ref_cfg_index]
            if (flat_full & 1) == 1:  # t odd?
                incr = ref_sign
            # deduplicate by row
            pos = -1
            for u in range(used_len):
                if used_indices[u] == cfg_row:
                    pos = u
                    break
            if pos == -1:
                pos = used_len
                used_indices[pos] = cfg_row
                used_len += 1
            used_values[pos] += incr * phase
        # Count only truly non-zero amplitudes after cancellation
        cnt = 0
        for u in range(used_len):
            if np.abs(used_values[u]) > TOL_ZERO:
                cnt += 1
        nnz_per_col[col_idx] = cnt
    # Allocate CSC
    L_col_ptr = _prefix_sum_counts(nnz_per_col)
    total_nnz = L_col_ptr[-1]
    L_row_idx = np.empty(total_nnz, np.int32)
    L_data = np.empty(total_nnz, np.complex128)
    # ---------- 3) PASS 2: fill CSC (dedup + prune zeros + normalize + sort) ----------
    for col_idx in prange(n_cols):
        ref_cfg_index = references[col_idx]
        pvec = period_vectors[col_idx, :]
        orbit_size = np.prod(pvec)
        used_indices = np.empty(orbit_size, np.int32)
        used_values = np.zeros(orbit_size, np.complex128)
        used_len = 0
        t_local = np.zeros(lattice_dim, np.int32)
        # Deduplicate + accumulate complex phases
        for flat_local in range(orbit_size):
            decode_mixed_index(flat_local, pvec, t_local)
            phase_arg = 0.0
            for ax in range(lattice_dim):
                kd = k_vals[ax] % shifts_per_dir[ax]
                phase_arg += (kd * t_local[ax]) / float(shifts_per_dir[ax])
            phase = np.exp(-1j * 2.0 * np.pi * phase_arg)
            flat_full = encode_shift(t_local, shifts_per_dir)
            cfg_row = translations_array[ref_cfg_index, flat_full]
            # -----------------------------------------------------
            incr = 1.0
            # In 1D SU(2) we can implement the TC symmetry
            ref_sign = C_phase_per_config[ref_cfg_index]
            if (flat_full & 1) == 1:  # t odd?
                incr = ref_sign
            # deduplicate by row
            pos = -1
            for u in range(used_len):
                if used_indices[u] == cfg_row:
                    pos = u
                    break
            if pos == -1:
                pos = used_len
                used_indices[pos] = cfg_row
                used_len += 1
            used_values[pos] += incr * phase
        # In-place prune entries that cancelled to ~0, and compute column norm
        kept_len = 0
        norm_sq = 0.0
        for u in range(used_len):
            a = used_values[u]
            if np.abs(a) > TOL_ZERO:
                used_indices[kept_len] = used_indices[u]
                used_values[kept_len] = a
                norm_sq += a.real * a.real + a.imag * a.imag
                kept_len += 1
        # If the column fully cancels, write nothing
        if norm_sq <= TOL_COLNORM or kept_len == 0:
            continue
        inv_norm = 1.0 / np.sqrt(norm_sq)
        # Sort by row index for canonical CSC
        _insertion_sort_by_row(used_indices, used_values, kept_len)
        # Materialize
        write = L_col_ptr[col_idx]
        for u in range(kept_len):
            L_row_idx[write] = used_indices[u]
            L_data[write] = used_values[u] * inv_norm
            write += 1
    # ---------- 4) Build CSR from CSC ----------
    R_row_ptr = np.zeros(n_rows + 1, np.int32)
    for p in range(total_nnz):
        r = L_row_idx[p]
        R_row_ptr[r + 1] += 1
    for r in range(n_rows):
        R_row_ptr[r + 1] += R_row_ptr[r]
    R_col_idx = np.empty(total_nnz, np.int32)
    R_data = np.empty(total_nnz, np.complex128)
    heads = np.empty(n_rows, np.int32)
    for r in range(n_rows):
        heads[r] = R_row_ptr[r]
    for col in range(n_cols):
        start = L_col_ptr[col]
        stop = L_col_ptr[col + 1]
        for p in range(start, stop):
            r = L_row_idx[p]
            q = heads[r]
            R_col_idx[q] = col
            R_data[q] = L_data[p]
            heads[r] += 1
    return (L_col_ptr, L_row_idx, L_data, R_row_ptr, R_col_idx, R_data)


# ─────────────────────────────────────────────────────────────────────────────
def get_momentum_basis(
    sector_configs: np.ndarray,
    lvals: list[int],
    unit_cell_size: np.ndarray,
    k_vals: np.ndarray,
    TC_symmetry: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct a sparse momentum-basis projector representation.

    Parameters
    ----------
    sector_configs : ndarray
        Symmetry-sector configurations, one row per basis state.
    lvals : list
        Lattice lengths along each spatial direction.
    unit_cell_size : ndarray
        Translation step (logical unit-cell size) per spatial direction.
    k_vals : ndarray
        Momentum quantum numbers (one per spatial direction, or the effective
        translation-combined symmetry momentum in ``TC_symmetry`` mode).
    TC_symmetry : bool, optional
        If ``True``, use the translation-combined (TC) symmetry construction.

    Returns
    -------
    tuple
        Sparse left/right representations of the momentum-basis projector
        ``B``:
        ``(L_col_ptr, L_row_idx, L_data, R_row_ptr, R_col_idx, R_data)``,
        where the first three arrays encode the CSC representation of ``B`` and
        the last three arrays encode the CSR representation of ``B``.
    """
    lvals = np.ascontiguousarray(lvals, dtype=np.int32)
    unit_cell_size = np.ascontiguousarray(unit_cell_size, dtype=np.int32)
    k_vals = np.ascontiguousarray(k_vals, dtype=np.int32)
    if np.any(k_vals != 0):
        if TC_symmetry:
            return momentum_basis_finite_k_TC(sector_configs, k_vals)
        else:
            return momentum_basis_finite_k(
                sector_configs, lvals, unit_cell_size, k_vals
            )
    else:
        if TC_symmetry:
            return momentum_basis_zero_k_TC(sector_configs)
        else:
            return momentum_basis_zero_k(sector_configs, lvals, unit_cell_size)
    # Optional: Uncomment for debugging
    # if not check_normalization(basis) or not check_orthogonality(basis):
    #     raise ValueError("Basis normalization or orthogonality failed.")


# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, parallel=True)
def precompute_nonzero_csr(matrix: np.ndarray):
    """
    Build CSR pointers for nonzeros of each column of `mat`.
    Returns (indptr, indices) so that
      indices[indptr[p]:indptr[p+1]] are the rows c where mat[c,p]!=0.
    """
    Nx, Ny = matrix.shape
    col_counts = np.zeros(Ny, np.int32)
    for jj in prange(Ny):
        for ii in range(Nx):
            if np.abs(matrix[ii, jj]) > 1e-10:
                col_counts[jj] += 1

    indptr = np.empty(Ny + 1, np.int32)
    indptr[0] = 0
    for jj in range(Ny):
        indptr[jj + 1] = indptr[jj] + col_counts[jj]

    total_nz = indptr[Ny]
    indices = np.empty(total_nz, np.int32)

    for jj in prange(Ny):
        start = indptr[jj]
        pos = start
        for ii in range(Nx):
            if np.abs(matrix[ii, jj]) > 1e-10:
                indices[pos] = ii
                pos += 1
    return indptr, indices


@njit(cache=True, parallel=True)
def nbody_data_momentum_1site(
    op_list: np.ndarray,  # (1, n_sites, d_loc, d_loc)
    op_sites_list: np.ndarray,  # (1,), int32
    sector_configs: np.ndarray,  # (N, n_sites), int32
    # --- sparse momentum basis B arrays ---
    L_col_ptr: np.ndarray,  # (Ldim+1,) int32   -- columns of B
    L_row_idx: np.ndarray,  # (nnz_B,)  int32   -- rows for each CSC entry
    L_data: np.ndarray,  # (nnz_B,)  float64 or complex128 -- B[row, col]
    R_row_ptr: np.ndarray,  # (N+1,)    int32   -- rows of B
    R_col_idx: np.ndarray,  # (nnz_B,)  int32   -- cols for each CSR entry
    R_data: np.ndarray,  # (nnz_B,)  float64 or complex128 -- B[row, col]
):
    """Build sparse triplets for a one-site operator in the momentum basis.

    Parameters
    ----------
    op_list : ndarray
        One-site factorized operator data (shape ``(1, n_sites, d_loc, d_loc)``).
    op_sites_list : ndarray
        Site index of the operator action (shape ``(1,)``).
    sector_configs : ndarray
        Symmetry-sector configurations, one row per basis state.
    L_col_ptr, L_row_idx, L_data : ndarray
        CSC representation of the momentum-basis projector ``B``.
    R_row_ptr, R_col_idx, R_data : ndarray
        CSR representation of the same projector ``B``.

    Returns
    -------
    tuple
        ``(row_list, col_list, value_list)`` triplets for the projected
        operator ``B^† H B``.
    """
    n_sites = sector_configs.shape[1]
    Ldim = L_col_ptr.size - 1  # number of momentum columns (dim of projected space)
    d_loc = op_list.shape[2]
    # Selected site/operator
    site = op_sites_list[0]
    Op = op_list[0, site]
    # ----------------------------
    # PASS 1: count nnz per momentum-row (prow)
    # ----------------------------
    nnz_per_row = np.zeros(Ldim, np.int32)
    for prow in prange(Ldim):
        cnt = 0
        # (CHANGED) iterate all real-space rows j1 with B[j1, prow] != 0 via CSC
        start1 = L_col_ptr[prow]
        stop1 = L_col_ptr[prow + 1]
        for p1 in range(start1, stop1):
            j1 = L_row_idx[p1]  # row index in real space
            # bra config
            bra_cfg = sector_configs[j1]
            # build list of target local states b where Op[a,b] != 0
            a = bra_cfg[site]
            idxs = np.empty(d_loc, np.int32)
            cnt1 = 0
            for b in range(d_loc):
                if np.abs(Op[a, b]) > 1e-10:
                    idxs[cnt1] = b
                    cnt1 += 1
            lens = cnt1
            # scratch ket (copy bra → then edit one site)
            ket_cfg = np.empty(n_sites, np.int32)
            for jj in range(n_sites):
                ket_cfg[jj] = bra_cfg[jj]
            # for each allowed local change a→b0, find the ket index j2
            for i0 in range(lens):
                b0 = idxs[i0]
                ket_cfg[site] = b0
                j2 = config_to_index_binarysearch(ket_cfg, sector_configs)
                if j2 < 0:
                    continue
                # (CHANGED) number of nonzero momentum-cols for row j2 via CSR
                cnt += R_row_ptr[j2 + 1] - R_row_ptr[j2]
        nnz_per_row[prow] = cnt
    # prefix-sum offsets
    offset = 0
    for prow in range(Ldim):
        tmp = nnz_per_row[prow]
        nnz_per_row[prow] = offset
        offset += tmp
    total_nnz = offset
    # ----------------------------
    # PASS 2: fill triplets
    # ----------------------------
    row_list = np.empty(total_nnz, np.int32)
    col_list = np.empty(total_nnz, np.int32)
    value_list = np.empty(total_nnz, np.complex128)
    for prow in prange(Ldim):
        ptr = nnz_per_row[prow]
        # (CHANGED) iterate B[:, prow] via CSC
        start1 = L_col_ptr[prow]
        stop1 = L_col_ptr[prow + 1]
        for p1 in range(start1, stop1):
            j1 = L_row_idx[p1]  # real-space row
            B1 = L_data[p1]  # value B[j1, prow] (float64 or complex128)
            bra_cfg = sector_configs[j1]
            # rebuild idxs, vs for Op[a, :]
            a = bra_cfg[site]
            idxs = np.empty(d_loc, np.int32)
            vs = np.empty(d_loc, np.complex128)
            cnt2 = 0
            for b in range(d_loc):
                v = Op[a, b]
                if np.abs(v) > 1e-10:
                    idxs[cnt2] = b
                    # ensure complex128 for downstream products
                    vs[cnt2] = v if np.iscomplexobj(Op) else (v + 0.0j)
                    cnt2 += 1
            lens = cnt2
            # scratch ket
            ket_cfg = np.empty(n_sites, np.int32)
            for jj in range(n_sites):
                ket_cfg[jj] = bra_cfg[jj]
            # explicit 1-nested loop + projection
            for i0 in range(lens):
                b0 = idxs[i0]
                v0 = vs[i0]
                ket_cfg[site] = b0
                j2 = config_to_index_binarysearch(ket_cfg, sector_configs)
                if j2 < 0:
                    continue
                # (CHANGED) project into momentum columns of row j2 via CSR
                start2 = R_row_ptr[j2]
                stop2 = R_row_ptr[j2 + 1]
                for p2 in range(start2, stop2):
                    pcol = R_col_idx[p2]
                    B2 = R_data[p2]  # value B[j2, pcol]
                    # (CHANGED) use sparse B entries
                    val = np.conj(B1) * v0 * B2
                    row_list[ptr] = prow
                    col_list[ptr] = pcol
                    value_list[ptr] = val
                    ptr += 1
    return row_list, col_list, value_list


@njit(cache=True, parallel=True)
def nbody_data_momentum_2sites(
    op_list: np.ndarray,  # shape (2, n_sites, d_loc, d_loc)
    op_sites_list: np.ndarray,  # shape (2,), int32
    sector_configs: np.ndarray,  # shape (N, n_sites), int32
    # ---- momentum basis B in sparse form ----
    L_col_ptr: np.ndarray,  # (Ldim+1,), int32   -- columns of B
    L_row_idx: np.ndarray,  # (nnz_B,),  int32   -- real-space rows j with B[j, prow] != 0
    L_data: np.ndarray,  # (nnz_B,),  complex128/float64 -- B[j, prow]
    R_row_ptr: np.ndarray,  # (N+1,),    int32   -- rows of B
    R_col_idx: np.ndarray,  # (nnz_B,),  int32   -- projected cols pcol with B[j, pcol] != 0
    R_data: np.ndarray,  # (nnz_B,),  complex128/float64 -- B[j, pcol]
):
    """Build sparse triplets for a two-site operator in the momentum basis.

    Parameters
    ----------
    op_list : ndarray
        Two-site factorized operator data (shape ``(2, n_sites, d_loc, d_loc)``).
    op_sites_list : ndarray
        Two site indices where the operator acts.
    sector_configs : ndarray
        Symmetry-sector configurations, one row per basis state.
    L_col_ptr, L_row_idx, L_data : ndarray
        CSC representation of the momentum-basis projector ``B``.
    R_row_ptr, R_col_idx, R_data : ndarray
        CSR representation of the same projector ``B``.

    Returns
    -------
    tuple
        ``(row_list, col_list, value_list)`` triplets for the projected
        operator ``B^H H B``.
    """
    N, n_sites = sector_configs.shape
    Ldim = L_col_ptr.shape[0] - 1
    d_loc = op_list.shape[2]
    M = len(op_sites_list)
    # -------------------------------
    # PASS 1: count nonzeros per projected row
    # -------------------------------
    nnz_per_row = np.zeros(Ldim, np.int32)
    for prow in prange(Ldim):
        cnt = 0
        # all real-space rows j1 with B[j1, prow] != 0  (CSC of B)
        start1 = L_col_ptr[prow]
        stop1 = L_col_ptr[prow + 1]
        for idx1 in range(start1, stop1):
            j1 = L_row_idx[idx1]
            bra_cfg = sector_configs[j1]
            # per-site allowed target indices and their counts
            idxs = np.empty((M, d_loc), np.int32)
            lens = np.empty(M, np.int32)
            for kk in range(M):
                site = op_sites_list[kk]
                Op = op_list[kk, site]
                a = bra_cfg[site]
                cnt1 = 0
                for b in range(d_loc):
                    if np.abs(Op[a, b]) > OP_EPS:
                        idxs[kk, cnt1] = b
                        cnt1 += 1
                lens[kk] = cnt1
            # scratch ket config (start from bra each time)
            ket_cfg = np.empty(n_sites, np.int32)
            for jj in range(n_sites):
                ket_cfg[jj] = bra_cfg[jj]
            site0, site1 = op_sites_list[0], op_sites_list[1]
            # explicit 2-nested loops → real-space j2
            for i0 in range(lens[0]):
                b0 = idxs[0, i0]
                ket_cfg[site0] = b0
                for i1 in range(lens[1]):
                    b1 = idxs[1, i1]
                    ket_cfg[site1] = b1
                    # find j2 in the sector
                    j2 = config_to_index_binarysearch(ket_cfg, sector_configs)
                    if j2 < 0:
                        continue
                    # number of projected columns reachable from j2 (CSR of B)
                    cnt += R_row_ptr[j2 + 1] - R_row_ptr[j2]
        nnz_per_row[prow] = cnt
    # prefix-sum → row offsets
    offset = 0
    for prow in range(Ldim):
        tmp = nnz_per_row[prow]
        nnz_per_row[prow] = offset
        offset += tmp
    total_nnz = offset
    # -------------------------------
    # PASS 2: fill triplets
    # -------------------------------
    row_list = np.empty(total_nnz, np.int32)
    col_list = np.empty(total_nnz, np.int32)
    value_list = np.empty(total_nnz, np.complex128)
    for prow in prange(Ldim):
        ptr = nnz_per_row[prow]
        start1 = L_col_ptr[prow]
        stop1 = L_col_ptr[prow + 1]
        for idx1 in range(start1, stop1):
            j1 = L_row_idx[idx1]
            bra_cfg = sector_configs[j1]
            amp_L = np.conj(L_data[idx1])  # conj(B[j1, prow])
            # rebuild idxs, vs, lens
            idxs = np.empty((M, d_loc), np.int32)
            vs = np.empty((M, d_loc), np.complex128)
            lens = np.empty(M, np.int32)
            for kk in range(M):
                site = op_sites_list[kk]
                Op = op_list[kk, site]
                a = bra_cfg[site]
                cnt2 = 0
                for b in range(d_loc):
                    v = Op[a, b]
                    if np.abs(v) > OP_EPS:
                        idxs[kk, cnt2] = b
                        vs[kk, cnt2] = v
                        cnt2 += 1
                lens[kk] = cnt2
            ket_cfg = np.empty(n_sites, np.int32)
            for jj in range(n_sites):
                ket_cfg[jj] = bra_cfg[jj]
            site0, site1 = op_sites_list[0], op_sites_list[1]
            for i0 in range(lens[0]):
                b0 = idxs[0, i0]
                v0 = vs[0, i0]
                ket_cfg[site0] = b0
                for i1 in range(lens[1]):
                    b1 = idxs[1, i1]
                    v1 = vs[1, i1]
                    ket_cfg[site1] = b1
                    j2 = config_to_index_binarysearch(ket_cfg, sector_configs)
                    if j2 < 0:
                        continue
                    amp_M = v0 * v1
                    # project into momentum columns (CSR of B)
                    start2 = R_row_ptr[j2]
                    stop2 = R_row_ptr[j2 + 1]
                    for tt in range(start2, stop2):
                        pcol = R_col_idx[tt]
                        amp_R = R_data[tt]  # B[j2, pcol]
                        val = amp_L * amp_M * amp_R
                        row_list[ptr] = prow
                        col_list[ptr] = pcol
                        value_list[ptr] = val
                        ptr += 1
    return row_list, col_list, value_list


@njit(cache=True, parallel=True)
def nbody_data_momentum_4sites(
    op_list: np.ndarray,  # shape (4, n_sites, d_loc, d_loc)
    op_sites_list: np.ndarray,  # shape (4,), int32
    sector_configs: np.ndarray,  # shape (N, n_sites), int32
    # ---- momentum basis B in sparse form ----
    L_col_ptr: np.ndarray,  # (Ldim+1,), int32
    L_row_idx: np.ndarray,  # (nnz_B,),  int32
    L_data: np.ndarray,  # (nnz_B,),  complex128/float64
    R_row_ptr: np.ndarray,  # (N+1,),    int32
    R_col_idx: np.ndarray,  # (nnz_B,),  int32
    R_data: np.ndarray,  # (nnz_B,),  complex128/float64
):
    """Build sparse triplets for a four-site operator in the momentum basis.

    Parameters
    ----------
    op_list : ndarray
        Four-site factorized operator data (shape ``(4, n_sites, d_loc, d_loc)``).
    op_sites_list : ndarray
        Four site indices where the operator acts.
    sector_configs : ndarray
        Symmetry-sector configurations, one row per basis state.
    L_col_ptr, L_row_idx, L_data : ndarray
        CSC representation of the momentum-basis projector ``B``.
    R_row_ptr, R_col_idx, R_data : ndarray
        CSR representation of the same projector ``B``.

    Returns
    -------
    tuple
        ``(row_list, col_list, value_list)`` triplets for the projected
        operator ``B^H H B``.
    """
    N, n_sites = sector_configs.shape
    Ldim = L_col_ptr.shape[0] - 1
    d_loc = op_list.shape[2]
    M = len(op_sites_list)
    # -------------------------------
    # PASS 1: count nonzeros per projected row
    # -------------------------------
    nnz_per_row = np.zeros(Ldim, np.int32)
    for prow in prange(Ldim):
        cnt = 0
        # all real-space rows j1 with B[j1, prow] != 0  (CSC of B)
        start1 = L_col_ptr[prow]
        stop1 = L_col_ptr[prow + 1]
        for idx1 in range(start1, stop1):
            j1 = L_row_idx[idx1]
            bra_cfg = sector_configs[j1]
            # per-site allowed target indices and counts
            idxs = np.empty((M, d_loc), np.int32)
            lens = np.empty(M, np.int32)
            for kk in range(M):
                site = op_sites_list[kk]
                Op = op_list[kk, site]
                a = bra_cfg[site]
                cnt1 = 0
                for b in range(d_loc):
                    if np.abs(Op[a, b]) > OP_EPS:
                        idxs[kk, cnt1] = b
                        cnt1 += 1
                lens[kk] = cnt1
            ket_cfg = np.empty(n_sites, np.int32)
            for jj in range(n_sites):
                ket_cfg[jj] = bra_cfg[jj]
            site0 = op_sites_list[0]
            site1 = op_sites_list[1]
            site2 = op_sites_list[2]
            site3 = op_sites_list[3]
            # explicit 4-nested loops → real-space j2
            for i0 in range(lens[0]):
                b0 = idxs[0, i0]
                ket_cfg[site0] = b0
                for i1 in range(lens[1]):
                    b1 = idxs[1, i1]
                    ket_cfg[site1] = b1
                    for i2 in range(lens[2]):
                        b2 = idxs[2, i2]
                        ket_cfg[site2] = b2
                        for i3 in range(lens[3]):
                            b3 = idxs[3, i3]
                            ket_cfg[site3] = b3
                            j2 = config_to_index_binarysearch(ket_cfg, sector_configs)
                            if j2 < 0:
                                continue
                            # number of projected columns from j2 (CSR of B)
                            cnt += R_row_ptr[j2 + 1] - R_row_ptr[j2]
        nnz_per_row[prow] = cnt
    # prefix-sum → row offsets
    offset = 0
    for prow in range(Ldim):
        tmp = nnz_per_row[prow]
        nnz_per_row[prow] = offset
        offset += tmp
    total_nnz = offset
    # -------------------------------
    # PASS 2: fill triplets
    # -------------------------------
    row_list = np.empty(total_nnz, np.int32)
    col_list = np.empty(total_nnz, np.int32)
    value_list = np.empty(total_nnz, np.complex128)
    for prow in prange(Ldim):
        ptr = nnz_per_row[prow]
        start1 = L_col_ptr[prow]
        stop1 = L_col_ptr[prow + 1]
        for idx1 in range(start1, stop1):
            j1 = L_row_idx[idx1]
            bra_cfg = sector_configs[j1]
            amp_L = np.conj(L_data[idx1])  # conj(B[j1, prow])
            # rebuild idxs, vs, lens
            idxs = np.empty((M, d_loc), np.int32)
            vs = np.empty((M, d_loc), np.complex128)
            lens = np.empty(M, np.int32)
            for kk in range(M):
                site = op_sites_list[kk]
                Op = op_list[kk, site]
                a = bra_cfg[site]
                cnt2 = 0
                for b in range(d_loc):
                    v = Op[a, b]
                    if np.abs(v) > OP_EPS:
                        idxs[kk, cnt2] = b
                        vs[kk, cnt2] = v
                        cnt2 += 1
                lens[kk] = cnt2
            ket_cfg = np.empty(n_sites, np.int32)
            for jj in range(n_sites):
                ket_cfg[jj] = bra_cfg[jj]
            site0 = op_sites_list[0]
            site1 = op_sites_list[1]
            site2 = op_sites_list[2]
            site3 = op_sites_list[3]
            # explicit 4-loops + projection
            for i0 in range(lens[0]):
                b0 = idxs[0, i0]
                v0 = vs[0, i0]
                ket_cfg[site0] = b0
                for i1 in range(lens[1]):
                    b1 = idxs[1, i1]
                    v1 = vs[1, i1]
                    ket_cfg[site1] = b1
                    for i2 in range(lens[2]):
                        b2 = idxs[2, i2]
                        v2 = vs[2, i2]
                        ket_cfg[site2] = b2
                        for i3 in range(lens[3]):
                            b3 = idxs[3, i3]
                            v3 = vs[3, i3]
                            ket_cfg[site3] = b3
                            j2 = config_to_index_binarysearch(ket_cfg, sector_configs)
                            if j2 < 0:
                                continue
                            amp_M = v0 * v1 * v2 * v3
                            # project into momentum columns (CSR of B)
                            start2 = R_row_ptr[j2]
                            stop2 = R_row_ptr[j2 + 1]
                            for tt in range(start2, stop2):
                                pcol = R_col_idx[tt]
                                amp_R = R_data[tt]  # B[j2, pcol]
                                val = amp_L * amp_M * amp_R
                                row_list[ptr] = prow
                                col_list[ptr] = pcol
                                value_list[ptr] = val
                                ptr += 1
    return row_list, col_list, value_list
