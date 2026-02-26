"""Parity (inversion) symmetry helpers in symmetry-reduced bases.

The module builds parity permutations and parity operators directly in a
configuration-based sector basis, and provides a lightweight routine to apply
the resulting signed permutation to a state vector.
"""

import numpy as np
from numba import njit, prange
from .generate_configs import config_to_index_binarysearch

__all__ = [
    "apply_parity_to_state",
    "build_parity_operator",
]


@njit(cache=True)
def parity_perm_site(n_sites: int, j0: int) -> np.ndarray:
    """
    Site-centered inversion about site j0:
        j -> (2*j0 - j) mod n_sites

    Parameters
    ----------
    n_sites : int
        Number of lattice sites (L).
    j0 : int
        Site index about which we invert. Can be any integer; reduced mod L.

    Returns
    -------
    ndarray
        Permutation array such that ``perm[j]`` is the new site index of the
        degree of freedom originally at site ``j``.
    """
    L = n_sites
    j0_mod = j0 % L
    perm = np.empty(L, dtype=np.int32)
    center2 = 2 * j0_mod
    for j in range(L):
        perm[j] = (center2 - j) % L
    return perm


@njit(cache=True)
def parity_perm_bond(n_sites: int, j0: int) -> np.ndarray:
    """
    Bond-centered inversion about the bond (j0, j0+1),
    i.e. center at j0 + 0.5:

        j -> (2*(j0 + 0.5) - j) mod n_sites
          = (2*j0 + 1 - j) mod n_sites

    Parameters
    ----------
    n_sites : int
        Number of lattice sites (L).
    j0 : int
        LEFT site index of the bond (j0, j0+1). Can be any integer; reduced mod L.

    Returns
    -------
    ndarray
        Permutation array such that ``perm[j]`` is the new site index of the
        degree of freedom originally at site ``j``.
    """
    L = n_sites
    j0_mod = j0 % L
    perm = np.empty(L, dtype=np.int32)
    center2 = 2 * j0_mod + 1  # 2*(j0 + 0.5)
    for j in range(L):
        perm[j] = (center2 - j) % L
    return perm


@njit(cache=True)
def parity_image_config(
    config: np.ndarray,
    site_perm: np.ndarray,
    loc_perm: np.ndarray,
    loc_phase: np.ndarray,
):
    """
    Apply parity to a single configuration.

    Parameters
    ----------
    config : (n_sites,)
        Local basis indices of original configuration.
    site_perm : (n_sites,)
        site_perm[j] = new site index of DOF originally at j.
    loc_perm : ndarray
        Local basis-label permutation under parity.
    loc_phase : ndarray
        Local parity phase factors associated with the basis labels.

    Returns
    -------
    tuple
        ``(new_config, total_phase)`` with the transformed configuration and
        the accumulated parity sign.
    """
    n_sites = len(config)
    new_config = np.empty(n_sites, dtype=np.int32)
    total_phase = 1
    for j in range(n_sites):
        s = config[j]
        s_p = loc_perm[s]
        total_phase = total_phase * loc_phase[s]
        new_j = site_perm[j]
        new_config[new_j] = s_p
    return new_config, total_phase


@njit(cache=True, parallel=True)
def build_parity_operator(
    sector_configs: np.ndarray,
    loc_perm: np.ndarray,
    loc_phase: np.ndarray,
    wrt_site: np.uint8 = 0,
):
    """Build the parity operator in triplet form for a sector basis.

    Parameters
    ----------
    sector_configs : ndarray
        Symmetry-sector configurations (one row per basis state, lexicographically
        sorted).
    loc_perm : ndarray
        Local basis-label permutation under parity.
    loc_phase : ndarray
        Local parity phase factors associated with the basis labels.
    wrt_site : int, optional
        If ``0``, build a site-centered inversion; otherwise use a bond-centered
        inversion.

    Returns
    -------
    tuple
        ``(row, col, data)`` triplet representation of the parity operator,
        with exactly one nonzero ``±1`` entry per column.
    """
    n_configs, n_sites = sector_configs.shape
    if wrt_site == 0:
        site_perm = parity_perm_site(n_sites, n_sites // 2 - 1)
    else:
        site_perm = parity_perm_bond(n_sites, n_sites // 2 - 1)
    row = np.empty(n_configs, dtype=np.int32)
    col = np.empty(n_configs, dtype=np.int32)
    data = np.empty(n_configs, dtype=np.float64)
    for cidx in prange(n_configs):
        cfg = sector_configs[cidx]
        new_cfg, phase = parity_image_config(cfg, site_perm, loc_perm, loc_phase)
        new_cfg_idx = config_to_index_binarysearch(new_cfg, sector_configs)
        # In a consistent sector construction, new_cfg_idx should never be -1.
        row[cidx] = new_cfg_idx
        col[cidx] = cidx
        data[cidx] = np.float64(phase)
    return row, col, data


@njit(cache=True, parallel=True)
def apply_parity_to_state(
    psi: np.ndarray,  # (n_configs,), complex128 expected
    row: np.ndarray,  # (n_configs,), int32
    col: np.ndarray,  # (n_configs,), int32
    data: np.ndarray,  # (n_configs,), float64 in {+1, -1}
) -> np.ndarray:
    """Apply a parity operator stored as signed-permutation triplets.

    Parameters
    ----------
    psi : ndarray
        State vector in the sector basis.
    row, col, data : ndarray
        Triplet representation returned by :func:`build_parity_operator`.

    Returns
    -------
    ndarray
        Parity-transformed state vector.

    Notes
    -----
    The routine assumes exactly one nonzero entry per column, which holds for a
    permutation-with-signs representation of parity.
    """
    n_configs = psi.size
    psi_out = np.zeros(n_configs, dtype=psi.dtype)
    for cfg_idx in prange(n_configs):
        row_idx = row[cfg_idx]
        col_idx = col[cfg_idx]
        psi_out[row_idx] += psi[col_idx] * np.array(data[cfg_idx], dtype=psi.dtype)
    return psi_out

