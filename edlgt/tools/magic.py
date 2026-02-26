"""Low-level encoded-configuration utilities used in stabilizer calculations.

This module provides Numba-accelerated helpers for working with basis
configurations encoded as integer keys using a mixed-radix convention.

Main use cases
--------------
- Encode and decode many-body configurations using per-site local dimensions.
- Generate and deduplicate X-string shift keys active on a truncated support.
- Evaluate the stabilizer Rényi-2 sum on a truncated support of a state.

Conventions
-----------
- The rightmost site is the fastest-varying digit.
- Keys are built from ``loc_dims`` and the corresponding ``strides`` returned by
  :func:`compute_strides`.
- Most functions here are low-level kernels and expect consistent, already
  validated arrays.
"""

import numpy as np
from numba import njit, prange
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "compute_strides",
    "encode_config",
    "encode_all_configs",
    "decode_key_to_config",
    "binary_search_sorted",
    "decode_Xstrings",
    "unique_sorted_int64",
    "all_pairwise_pkeys_support",
    "stabilizer_renyi_sum",
]


@njit(cache=True)
def compute_strides(loc_dims: np.ndarray) -> np.ndarray:
    """Compute mixed-radix strides for encoding site configurations into int64 keys.

    This defines how to map an N-site configuration vector (digits with different bases)
    into a unique integer key:

        key = sum_{site=0..N-1} config[site] * strides[site]

    Stride convention used throughout this project:
    - Rightmost site (index N-1) is the fastest digit.
    - Leftmost site (index 0) is the slowest digit.

    Concretely:
    - strides[N-1] = 1
    - strides[k] = product of loc_dims[k+1], ..., loc_dims[N-1]

    Parameters
    ----------
    loc_dims : numpy.ndarray
        Array of shape (n_sites,) containing local dimensions per site.

    Returns
    -------
    strides : numpy.ndarray
        int64 array of shape (n_sites,) encoding weights for each site.

    Notes
    -----
    - If all local dimensions are equal to d, then this is equivalent to base-d encoding
      with the last site as the least significant digit.
    - This encoding is only valid if each config[site] satisfies:
      0 <= config[site] < loc_dims[site].
    """
    n_sites = loc_dims.shape[0]
    strides = np.empty(n_sites, dtype=np.int64)
    running_stride = np.int64(1)
    for kk in range(n_sites - 1, -1, -1):
        strides[kk] = running_stride
        running_stride *= np.int64(loc_dims[kk])
    return strides


@njit(cache=True)
def encode_config(config: np.ndarray, strides: np.ndarray) -> np.int64:
    """Encode a single configuration into an int64 key using precomputed strides.

    Parameters
    ----------
    config : numpy.ndarray
        Array of shape (n_sites,) with local basis indices at each site.
    strides : numpy.ndarray
        Array of shape (n_sites,) produced by :func:`compute_strides`.

    Returns
    -------
    key : numpy.int64
        int64 encoding of the configuration.

    Notes
    -----
    - Assumes `config` entries are within the allowed range for each site.
    - Uses the "rightmost-fastest" convention embedded in `strides`.
    """
    key = np.int64(0)
    for kidx in range(config.shape[0]):
        key += np.int64(config[kidx]) * strides[kidx]
    return key


@njit(parallel=True, cache=True)
def encode_all_configs(configs: np.ndarray, strides: np.ndarray) -> np.ndarray:
    """Encode many configurations into int64 keys using precomputed strides.

    Parameters
    ----------
    configs : numpy.ndarray
        Array of shape (n_configs, n_sites) with local basis indices.
    strides : numpy.ndarray
        Array of shape (n_sites,) produced by :func:`compute_strides`.

    Returns
    -------
    keys : numpy.ndarray
        int64 array of shape (n_configs,) containing the encoded keys.

    Notes
    -----
    - Parallelized over configurations.
    - `keys` are not sorted by default; sort them if you want binary search membership.
    """
    n_configs, n_sites = configs.shape
    keys = np.empty(n_configs, dtype=np.int64)
    for ii in prange(n_configs):
        s = np.int64(0)
        for kk in range(n_sites):
            s += np.int64(configs[ii, kk]) * strides[kk]
        keys[ii] = s
    return keys


@njit(cache=True)
def decode_key_to_config(key: np.int64, loc_dims: np.ndarray) -> np.ndarray:
    """Decode an int64 key back into a configuration vector.

    This is the inverse of `encode_config` when using `compute_strides(loc_dims)`
    with the "rightmost-fastest" convention.

    Parameters
    ----------
    key : numpy.int64
        Non-negative int64 key produced by `encode_config` / `encode_all_configs`.
    loc_dims : numpy.ndarray
        Array of shape (n_sites,) containing local dimensions per site.

    Returns
    -------
    config : numpy.ndarray
        uint16 array of shape (n_sites,) reconstructing the site digits.

    Notes
    -----
    - This assumes the key is within range, i.e. 0 <= key < product(loc_dims).
    - Decoding proceeds from right to left because the rightmost site is fastest.
    """
    n_sites = loc_dims.shape[0]
    config = np.empty(n_sites, dtype=np.uint16)
    remainder = np.int64(key)
    for kk in range(n_sites - 1, -1, -1):
        dim_site = np.int64(loc_dims[kk])
        config[kk] = np.uint16(remainder % dim_site)
        remainder //= dim_site
    return config


@njit(cache=True)
def binary_search_sorted(keys_sorted: np.ndarray, target: np.int64) -> int:
    """Binary search on a sorted int64 array.

    Parameters
    ----------
    keys_sorted : numpy.ndarray
        1D int64 array sorted in non-decreasing order.
    target : numpy.int64
        int64 value to search.

    Returns
    -------
    index : int
        Index of `target` in `keys_sorted` if found, otherwise -1.

    Notes
    -----
    - If there are duplicates, this returns some matching index (not guaranteed first).
    - Requires `keys_sorted` to be sorted; no checks are performed.
    """
    lo = 0
    hi = keys_sorted.shape[0] - 1
    while lo <= hi:
        mid = (lo + hi) >> 1
        v = keys_sorted[mid]
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid
    return -1


@njit(parallel=True, cache=True)
def decode_Xstrings(Xp_keys: np.ndarray, loc_dims: np.ndarray) -> np.ndarray:
    """Decode encoded X-string keys into per-site shift (power) vectors.

    Parameters
    ----------
    Xp_keys : numpy.ndarray
        1D int64 array of encoded X-string keys (each key encodes a power vector).
    loc_dims : numpy.ndarray
        Array of shape (n_sites,) with local dimensions per site.

    Returns
    -------
    x_strings : numpy.ndarray
        ``uint16`` array of shape ``(n_strings, n_sites)``. Each row is one
        decoded per-site shift vector.

    Notes
    -----
    - Uses :func:`decode_key_to_config`, so it follows the same stride convention.
    """
    n_strings = Xp_keys.shape[0]
    n_sites = loc_dims.shape[0]
    x_strings = np.empty((n_strings, n_sites), dtype=np.uint16)
    for ii in prange(n_strings):
        x_strings[ii, :] = decode_key_to_config(Xp_keys[ii], loc_dims)
    return x_strings


@njit(cache=True)
def unique_sorted_int64(arr_sorted: np.ndarray) -> np.ndarray:
    """Return unique values from a sorted int64 array.

    Parameters
    ----------
    arr_sorted : numpy.ndarray
        1D int64 array sorted in non-decreasing order.

    Returns
    -------
    unique_values : numpy.ndarray
        1D int64 array containing the unique values in `arr_sorted`, in sorted order.

    Notes
    -----
    - Requires `arr_sorted` to be sorted; no checks are performed.
    - If `arr_sorted` is empty, it is returned unchanged.
    """
    n = arr_sorted.shape[0]
    if n == 0:
        return arr_sorted
    # count uniques
    cnt = 1
    prev = arr_sorted[0]
    for i in range(1, n):
        v = arr_sorted[i]
        if v != prev:
            cnt += 1
            prev = v
    out = np.empty(cnt, dtype=np.int64)
    out[0] = arr_sorted[0]
    j = 1
    prev = arr_sorted[0]
    for i in range(1, n):
        v = arr_sorted[i]
        if v != prev:
            out[j] = v
            j += 1
            prev = v
    return out


@njit(parallel=True, cache=True)
def all_pairwise_pkeys_support(
    support_configs: np.ndarray,  # (K, N) uint16
    loc_dims: np.ndarray,  # (N,) int
    strides: np.ndarray,  # (N,) int64
) -> np.ndarray:
    """Generate X-string keys induced by all ordered pairs in a support.

    For each ordered pair ``(row_config, col_config)`` in ``support_configs``,
    the function computes the modular site-wise difference and encodes it as a
    mixed-radix integer key. The output is typically sorted and deduplicated to
    obtain the set of X-strings active on the support.

    Parameters
    ----------
    support_configs : numpy.ndarray
        Array of shape (n_configs_support, n_sites) with local basis indices.
    loc_dims : numpy.ndarray
        Array of shape (n_sites,) with local dimensions per site.
    strides : numpy.ndarray
        Array of shape (n_sites,) produced by :func:`compute_strides`.
        Uses the convention: rightmost site (n_sites-1) is the fastest digit.

    Returns
    -------
    pkeys_all : numpy.ndarray
        1D int64 array of length (n_configs_support * n_configs_support).
        Entry pkeys_all[row * n_configs_support + col] encodes the X-string that maps
        support_configs[row] to support_configs[col] by modular shifts.

    Notes
    -----
    - The output contains duplicates. A common workflow is:
      ``np.sort(...)`` followed by :func:`unique_sorted_int64`.
    - The identity string (all zero powers) appears when row == col.
    - Complexity: O(n_configs_support^2 * n_sites) time and O(n_configs_support^2) memory.
    """
    # n_cfgs = number of support configs, n_sites = number of sites
    n_cfgs, n_sites = support_configs.shape
    # We will output one key per ordered pair (cfg_idx_row, cfg_idx_col)
    out = np.empty(n_cfgs * n_cfgs, dtype=np.int64)
    # Parallelize over the "row" index (source configuration)
    for cfg_idx_row in prange(n_cfgs):
        # The output for this row occupies a contiguous block of length n_cfgs
        # out[base + cfg_idx_col] corresponds to pair (cfg_idx_row, cfg_idx_col)
        base = cfg_idx_row * n_cfgs
        # Loop over all "col" indices (target configurations)
        for cfg_idx_col in range(n_cfgs):
            # Build the encoded key for p = (col - row) mod loc_dims, site-by-site
            key = np.int64(0)
            for site_idx in range(n_sites):
                # local dimension at this site
                dim_site = np.int64(loc_dims[site_idx])
                # state labels at this site for row and col configs
                a = np.int64(support_configs[cfg_idx_row, site_idx])  # alpha_k
                b = np.int64(support_configs[cfg_idx_col, site_idx])  # beta_k
                # modular difference: p_k = (beta_k - alpha_k) mod d_k
                p = (b - a) % dim_site
                # encode: add digit p_k times its stride weight
                key += p * strides[site_idx]
            out[base + cfg_idx_col] = key
    return out


@njit(parallel=True, cache=True)
def chisq_xstrings_on_support(
    pkeys_uniq: np.ndarray,  # (S,) int64, encoded p vectors
    support_configs: np.ndarray,  # (K,N) uint16
    support_keys: np.ndarray,  # (K,) int64, sorted
    support_coeffs: np.ndarray,  # (K,) complex128 aligned with support_keys
    loc_dims: np.ndarray,  # (N,) int
    strides: np.ndarray,  # (N,) int64
) -> np.ndarray:
    """
    For each X-string p, compute:
      chi_p = sum_{alpha in support} conj(c_{alpha+p}) * c_alpha
    where alpha+p must also be in support (checked via binary search on support_keys).

    Returns chisq[p] = |chi_p|^2.
    """
    n_strings = pkeys_uniq.shape[0]
    n_cfgs, n_sites = support_configs.shape
    chisq = np.empty(n_strings, dtype=np.float64)
    for str_idx in prange(n_strings):
        pvec = decode_key_to_config(pkeys_uniq[str_idx], loc_dims)  # (N,) uint16
        chi_real = 0.0
        chi_imag = 0.0
        for cfg_idx in range(n_cfgs):
            # compute shifted key for this alpha under pvec
            shifted_key = np.int64(0)
            for site_idx in range(n_sites):
                dim_site = np.int64(loc_dims[site_idx])
                s = np.int64(support_configs[cfg_idx, site_idx])
                p = np.int64(pvec[site_idx])
                sp = (s + p) % dim_site
                shifted_key += sp * strides[site_idx]
            j = binary_search_sorted(support_keys, shifted_key)
            if j >= 0:
                a = support_coeffs[cfg_idx]
                b = support_coeffs[j]  # coefficient of shifted config
                # add conj(b) * a
                chi_real += b.real * a.real + b.imag * a.imag
                chi_imag += b.real * a.imag - b.imag * a.real
        chisq[str_idx] = chi_real * chi_real + chi_imag * chi_imag
    return chisq


@njit(parallel=True, cache=True)
def stabilizer_renyi_sum(
    pkeys_uniq: np.ndarray,
    support_configs: np.ndarray,
    support_coeffs: np.ndarray,
    support_keys: np.ndarray,
    loc_dims: np.ndarray,
    strides: np.ndarray,
) -> np.float64:
    """Compute the stabilizer Rényi-2 sum on a truncated support.

    This is the high-level public entry point in this module. It evaluates the
    total sum over the encoded X-strings listed in ``pkeys_uniq`` using the
    support data of a (possibly truncated) state.

    Parameters
    ----------
    pkeys_uniq : numpy.ndarray
        1D int64 array of length (n_strings,). Each entry encodes one X-string
        as a per-site shift vector in mixed-radix form consistent with
        ``loc_dims`` and ``strides``. Typical workflow:
        :func:`all_pairwise_pkeys_support` -> ``np.sort`` ->
        :func:`unique_sorted_int64`.
    support_configs : numpy.ndarray
        2D uint16 array of shape (n_configs_support, n_sites). Each row is a basis
        configuration in the truncated support.
    support_coeffs : numpy.ndarray
        1D complex array of shape (n_configs_support,) with the state
        coefficients aligned with ``support_configs``.
    support_keys : numpy.ndarray
        1D int64 array of shape (n_configs_support,) containing encoded configuration
        keys for `support_configs`, sorted in non-decreasing order. These keys
        must use the same stride convention as ``pkeys_uniq``:
        rightmost site is the fastest digit.
    loc_dims : numpy.ndarray
        1D array of shape (n_sites,) giving the local dimension at each site.
        Must be consistent with the digit ranges used in ``support_configs``.
        Recommended dtype: int64 (Numba-friendly).
    strides : numpy.ndarray
        1D int64 array of shape (n_sites,) produced by
        :func:`compute_strides`,
        using the convention: rightmost site is the fastest digit.

    Returns
    -------
    M2 : numpy.float64
        Stabilizer Rényi-2 sum computed from the provided support and X-string
        set.

    Notes
    -----
    - The loop over X-strings is parallelized with ``prange``.
    - Accuracy depends on the quality of the provided support truncation.
    """
    n_strings = pkeys_uniq.shape[0]
    Tp_array = np.zeros(n_strings, dtype=np.float64)
    for idx in prange(n_strings):
        Tp_array[idx] = exact_Xstring_from_support(
            pkeys_uniq[idx],
            support_configs,
            support_coeffs,
            support_keys,
            loc_dims,
            strides,
        )
    # Final total sum over all X-strings
    M2 = np.float64(0.0)
    for str_idx in range(n_strings):
        M2 += Tp_array[str_idx]
    return M2


@njit(cache=True, inline="always")
def encode_shifted_key(
    config_row: np.ndarray, pvec: np.ndarray, loc_dims: np.ndarray, strides: np.ndarray
):
    """
    Encode the configuration obtained by adding per-site shifts to a configuration.

    Parameters
    ----------
    config_row :
        1D array (n_sites,) representing a basis configuration.
    pvec :
        1D array (n_sites,) with per-site shift values.
    loc_dims :
        1D int array (n_sites,) with local dimension per site.
    strides :
        1D int64 array (n_sites,) encoding strides (rightmost site is fastest).

    Returns
    -------
    shifted_key :
        int64. Encoded key of (config_row + pvec) modulo loc_dims.
    """
    n_sites = loc_dims.shape[0]
    shifted_key = np.int64(0)
    for kk in range(n_sites):
        # acquire local dimension d_k
        d_k = np.int64(loc_dims[kk])
        # acquire state alpha_k of site k
        alpha_k = np.int64(config_row[kk])
        # acquire shift p_k of site k (the X-string power on site k)
        p_k = np.int64(pvec[kk])
        # compute shifted state beta_k = (alpha_k + p_k) mod d_k
        beta_k = (alpha_k + p_k) % d_k
        # accumulate the shifted key using strides
        shifted_key += beta_k * strides[kk]
    return shifted_key


@njit(cache=True, inline="always")
def encode_configs_pair_key(
    cfg1: np.ndarray, cfg2: np.ndarray, loc_dims: np.ndarray, strides: np.ndarray
):
    """
    Encode kappa = cfg1 + cfg2 (componentwise modulo loc_dims).

    Parameters
    ----------
    cfg1, cfg2 :
        1D arrays (n_sites,) representing configurations.
    loc_dims :
        1D int array (n_sites,) local dimensions.
    strides :
        1D int64 array (n_sites,) strides.

    Returns
    -------
    key :
        int64. Encoded key of kappa.
    """
    n_sites = loc_dims.shape[0]
    shifted_key = np.int64(0)
    for kk in range(n_sites):
        # acquire local dimension d_k
        dk = np.int64(loc_dims[kk])
        # acquire state A1_k of site k
        A1_k = np.int64(cfg1[kk])
        # acquire state A2_k of site k
        A2_k = np.int64(cfg2[kk])
        # compute shifted state B_k = (A1_k + A2_k) mod d_k
        B_k = (A1_k + A2_k) % dk
        shifted_key += B_k * strides[kk]
    return shifted_key


@njit(cache=True)
def exact_Xstring_from_support(
    pkey,
    support_configs: np.ndarray,
    support_coeffs: np.ndarray,
    support_keys: np.ndarray,
    loc_dims: np.ndarray,
    strides: np.ndarray,
):
    """
    Compute the exact contribution T_p for a single X-string p on a truncated support.

    This function assumes:
    - support_configs rows correspond to support_coeffs entries.
    - support_keys are the encoded keys of support_configs, sorted ascending.
    - pkey encodes a per-site shift vector p using the same mixed-radix convention.

    The computation proceeds as:

    1) Build a sparse list of nonzero A_alpha(p) values:
       A_alpha(p) = C_alpha * conj(C_{alpha_shifted})
       where alpha_shifted = alpha plus p modulo local dimensions.
       We keep only alpha for which alpha_shifted belongs to the support.

    2) Build V_kappa(p) by accumulating ordered pairs:
       kappa = alpha_i plus alpha_j (modulo loc_dims).
       V_kappa accumulates A_i * conj(A_j) over pairs that yield the same kappa.

    3) Return T_p = sum_kappa |V_kappa|^2.

    Parameters
    ----------
    pkey :
        int64. Encoded X-string shift vector.
    support_configs :
        2D uint array (n_support, n_sites). Support configurations.
    support_coeffs :
        1D complex array (n_support,). Coefficients for support configurations.
        Use normalized coefficients if you interpret the support as an approximate state.
    support_keys :
        1D int64 array (n_support,), sorted. Encoded keys of support_configs.
    loc_dims :
        1D int array (n_sites,). Local dimensions per site.
    strides :
        1D int64 array (n_sites,). Strides from compute_strides(loc_dims).

    Returns
    -------
    Tp :
        float64. The exact C_p contribution for this p on the support.

    Notes
    -----
    - This is an exact O(nA^2) method, where nA is the number of nonzero A_alpha(p).
    - Memory scales as O(nA^2) because we materialize all ordered pairs.
    - For large supports, this is only meant for validation / small systems.
    """
    n_support = support_configs.shape[0]
    # Decode pkey into a per-site shift vector
    pvec = decode_key_to_config(np.int64(pkey), loc_dims)  # (n_sites,) uint16
    # Step 1: build sparse list of nonzero A_alpha(p)
    # Upper bound is n_support
    Avals = np.empty(n_support, dtype=np.complex128)
    Acfg_idx = np.empty(n_support, dtype=np.int64)
    nnzA = 0
    for cfg_idx in range(n_support):
        # compute the shifted configuration
        shifted_key = encode_shifted_key(
            support_configs[cfg_idx], pvec, loc_dims, strides
        )
        shifted_cfg_idx = binary_search_sorted(support_keys, shifted_key)
        if shifted_cfg_idx >= 0:
            # A_alpha = C_alpha * conj(C_shifted)
            Avals[nnzA] = support_coeffs[cfg_idx] * np.conj(
                support_coeffs[shifted_cfg_idx]
            )
            Acfg_idx[nnzA] = cfg_idx
            # Update the number of nonzero shifted configs
            nnzA += 1
    if nnzA == 0:
        return np.float64(0.0)
    # Step 2: build all ordered pair contributions to V_kappa(p)
    n_pairs = nnzA * nnzA
    pair_keys = np.empty(n_pairs, dtype=np.int64)
    pair_vals = np.empty(n_pairs, dtype=np.complex128)
    tmp = 0
    for ia in range(nnzA):
        cfg_i = support_configs[Acfg_idx[ia]]
        ai = Avals[ia]
        for ja in range(nnzA):
            cfg_j = support_configs[Acfg_idx[ja]]
            shifted_pair_key = encode_configs_pair_key(cfg_i, cfg_j, loc_dims, strides)
            pair_keys[tmp] = shifted_pair_key
            pair_vals[tmp] = ai * np.conj(Avals[ja])
            tmp += 1
    # Step 3: sort pairs by pair key and reduce
    order = np.argsort(pair_keys)
    pair_keys = pair_keys[order]
    pair_vals = pair_vals[order]
    Tp = np.float64(0.0)
    current_key = pair_keys[0]
    acc = np.complex128(0.0 + 0.0j)
    for pair_idx in range(n_pairs):
        kk = pair_keys[pair_idx]
        if kk != current_key:
            # flush previous bin
            Tp += acc.real * acc.real + acc.imag * acc.imag
            current_key = kk
            acc = pair_vals[pair_idx]
        else:
            acc += pair_vals[pair_idx]
    # flush last bin
    Tp += acc.real * acc.real + acc.imag * acc.imag
    return Tp
