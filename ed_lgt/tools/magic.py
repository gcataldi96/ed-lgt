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
    loc_dims:
        Array of shape (n_sites,) containing local dimensions per site.

    Returns
    -------
    strides:
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
    config:
        Array of shape (n_sites,) with local basis indices at each site.
    strides:
        Array of shape (n_sites,) produced by `compute_strides`.

    Returns
    -------
    key:
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
    configs:
        Array of shape (n_configs, n_sites) with local basis indices.
    strides:
        Array of shape (n_sites,) produced by `compute_strides`.

    Returns
    -------
    keys:
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
    key:
        Non-negative int64 key produced by `encode_config` / `encode_all_configs`.
    loc_dims:
        Array of shape (n_sites,) containing local dimensions per site.

    Returns
    -------
    config:
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
    keys_sorted:
        1D int64 array sorted in non-decreasing order.
    target:
        int64 value to search.

    Returns
    -------
    index:
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
    """Decode encoded X-string keys into per-site power vectors.

    Parameters
    ----------
    Xp_keys:
        1D int64 array of encoded X-string keys (each key encodes a power vector).
    loc_dims:
        Array of shape (n_sites,) with local dimensions per site.

    Returns
    -------
    x_strings:
        uint16 array of shape (n_strings, n_sites), where each row is a power vector.

    Notes
    -----
    - Uses `decode_key_to_config`, so it follows the same stride convention.
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
    arr_sorted:
        1D int64 array sorted in non-decreasing order.

    Returns
    -------
    unique_values:
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
    """Generate encoded X-string keys from all ordered pairs of support configurations.

    For each ordered pair (row_config, col_config) among the support configurations,
    we define an X-string power vector `pvec` site-by-site as:

        pvec[site] = (col_config[site] - row_config[site]) mod loc_dims[site]

    This `pvec` is then encoded into an int64 key using the provided `strides`:

        pkey = sum_{site} pvec[site] * strides[site]

    Parameters
    ----------
    support_configs:
        Array of shape (n_configs_support, n_sites) with local basis indices.
    loc_dims:
        Array of shape (n_sites,) with local dimensions per site.
    strides:
        Array of shape (n_sites,) produced by `compute_strides(loc_dims)`.
        Uses the convention: rightmost site (n_sites-1) is the fastest digit.

    Returns
    -------
    pkeys_all:
        1D int64 array of length (n_configs_support * n_configs_support).
        Entry pkeys_all[row * n_configs_support + col] encodes the X-string that maps
        support_configs[row] to support_configs[col] by modular shifts.

    Notes
    -----
    - The output contains duplicates; typical usage is:
      sort -> unique_sorted_int64 to obtain the activated set of X-strings.
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
    support_probs: np.ndarray,
    support_keys: np.ndarray,
    loc_dims: np.ndarray,
    strides: np.ndarray,
) -> np.float64:
    """Compute the Rényi-2 stabilizer sum from a truncated support of psi.

    This function evaluates the Rényi-2 "stabilizer sum" using the analytical
    elimination of Z-strings. For each X-string (power vector) p, define:

        S_p = sum over configs alpha in support:
                  P(alpha) * P(alpha_shifted)

    where:
    - P(alpha) is the probability of configuration alpha in the target state,
      P(alpha) = |psi_alpha|^2.
    - alpha_shifted is obtained by shifting each site digit by p at that site:
      alpha_shifted[site] = (alpha[site] + p[site]) mod loc_dims[site].

    The full stabilizer quantity (restricted to the given support) is then:

        M2 = sum over p in pkeys_uniq of S_p.

    Important: this already corresponds to summing over all Z-strings for each
    fixed X-string p (the Z-dependence cancels analytically by character
    orthogonality). Therefore, this function does not enumerate Z powers.

    Parameters
    ----------
    pkeys_uniq:
        1D int64 array of length (n_strings,). Each entry encodes one X-string
        as a per-site power vector p[site] in mixed-radix form consistent with
        `loc_dims` and `strides`. Typically produced by:
        - all_pairwise_pkeys_support(...),
        - sorting, then unique_sorted_int64(...).
    support_configs:
        2D uint16 array of shape (n_configs_support, n_sites). Each row is a basis
        configuration in the truncated support.
    support_probs:
        1D float64 array of shape (n_configs_support,). Probabilities for each support
        configuration: support_probs[row] = |psi_row|^2.
    support_keys:
        1D int64 array of shape (n_configs_support,) containing encoded configuration
        keys for `support_configs`, sorted in non-decreasing order. These keys
        must use the same stride convention as `pkeys_uniq`:
        rightmost site is the fastest digit.
    loc_dims:
        1D array of shape (n_sites,) giving the local dimension at each site.
        Must be consistent with the digit ranges used in support_configs and p.
        Recommended dtype: int64 (Numba-friendly).
    strides:
        1D int64 array of shape (n_sites,) produced by compute_strides(loc_dims),
        using the convention: rightmost site is the fastest digit.

    Returns
    -------
    M2:
        float64 scalar equal to sum_p S_p, restricted to the provided support.

    Notes
    -----
    - Parallelism: the loop over X-strings is parallelized with prange. Each
      string is independent.
    - Membership test: shifted configurations are located by encoding them into
      a key and performing binary search on `support_keys`.
    - Complexity:
        - Time: O(n_strings * n_configs_support * n_sites) plus binary searches.
        - Memory: O(n_strings) for Sp.
    - Correctness requirements:
        - `support_keys` must be sorted.
        - `support_keys` must be computed using the same `strides` convention.
        - `support_probs` must align with `support_configs` rows.
    """
    n_strings = pkeys_uniq.shape[0]
    n_configs_support, n_sites = support_configs.shape
    Sp = np.zeros(n_strings, dtype=np.float64)
    for str_idx in prange(n_strings):
        # Decode the X-string key into a per-site power vector pvec[site].
        pvec = decode_key_to_config(pkeys_uniq[str_idx], loc_dims)
        accum = np.float64(0.0)
        # For each support configuration alpha, compute beta = alpha ⊕ pvec.
        # If beta is also in the support, add P(alpha) * P(beta).
        for row_cfg_idx in range(n_configs_support):
            shifted_key = np.int64(0)
            for site_idx in range(n_sites):
                # acquire local dimension d_k
                dim_site = np.int64(loc_dims[site_idx])
                # acquire state alpha_k of site k
                alpha_site = np.int64(support_configs[row_cfg_idx, site_idx])
                # acquire shift p_k of site k (the X-string power on site k)
                p_site = np.int64(pvec[site_idx])
                # compute shifted state beta_k = (alpha_k + p_k) mod d_k
                beta_site = (alpha_site + p_site) % dim_site
                # accumulate the shifted key using strides
                shifted_key += beta_site * strides[site_idx]
            # Check if shifted_key is in support_keys via binary search
            # If found, the action of that X-string maps in the support
            col_cfg_idx = binary_search_sorted(support_keys, shifted_key)
            if col_cfg_idx >= 0:
                # Sum the contribution P(alpha) * P(alpha ⊕ p)
                accum += support_probs[row_cfg_idx] * support_probs[col_cfg_idx]
        # Store S_p for this X-string
        Sp[str_idx] = accum
    # Final total sum over all X-strings
    M2 = np.float64(0.0)
    for str_idx in range(n_strings):
        M2 += Sp[str_idx]
    return M2
