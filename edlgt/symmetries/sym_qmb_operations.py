import numpy as np
from numba import njit, prange
from edlgt.dtype_config import coerce_numeric_array
from .generate_configs import config_to_index_binarysearch
from .translational_sym import (
    nbody_data_momentum_4sites,
    nbody_data_momentum_2sites,
    nbody_data_momentum_1site,
)
import logging

logger = logging.getLogger(__name__)


__all__ = [
    "nbody_term",
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
        Shape (M, n_sites, d_loc, d_loc) with M in {1,2,4}.
    op_sites_list : ndarray
        Shape (M,), int32 — the lattice sites the operator acts on.
    sector_configs : ndarray
        Shape (N, n_sites), int32 — basis configurations in the (symmetry) sector.
    momentum_basis : dict | ndarray | None
        Momentum-projection data.
        If a dictionary is provided, it must contain the sparse left/right
        projection arrays with keys ``"L_col_ptr"``, ``"L_row_idx"``,
        ``"L_data"``, ``"R_row_ptr"``, ``"R_col_idx"``, and ``"R_data"``
        (``"n_rows"``/``"n_cols"`` may also be present).
        If an ndarray is provided (legacy path), it is interpreted as a dense
        projection basis of shape ``(N, Bdim)``.
        If ``None``, the operator is built in the real-space symmetry sector.

    Returns
    -------
    row_list, col_list, value_list : ndarrays
        Triplet arrays for the projected operator.
    """
    # normalize site-count & sanity-check
    M = int(len(op_sites_list))
    if M not in [1, 2, 4]:
        msg = f"nbody operators can be only of 1,2,4 sites, got {M}"
        raise NotImplementedError(msg)
    # === No momentum projection → original real-space path ===
    if momentum_basis is None:
        if M == 1:
            row_list, col_list, value_list = localbody_data_par(
                op_list[0], op_sites_list[0], sector_configs
            )
        elif M == 2:
            row_list, col_list, value_list = nbody_data_2sites(
                op_list, op_sites_list, sector_configs
            )
        else:  # M == 4
            row_list, col_list, value_list = nbody_data_4sites(
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
        for k in required:
            if k not in momentum_basis:
                raise KeyError(
                    f"momentum_basis dict missing required key '{k}'. "
                    f"Present keys: {tuple(momentum_basis.keys())}"
                )
        L_col_ptr = momentum_basis["L_col_ptr"]
        L_row_idx = momentum_basis["L_row_idx"]
        L_data = momentum_basis["L_data"]
        R_row_ptr = momentum_basis["R_row_ptr"]
        R_col_idx = momentum_basis["R_col_idx"]
        R_data = momentum_basis["R_data"]
        if M == 1:
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
        elif M == 2:
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
        else:
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
    value_list = coerce_numeric_array(value_list, name="symmetry nbody values")
    return row_list, col_list, value_list


@njit(parallel=True, cache=True)
def nbody_data_4sites(
    op_list: np.ndarray, op_sites_list: np.ndarray, sector_configs: np.ndarray
):
    """
    Compute the nonzero elements of an 4-body-operator.

    Args:
        sector_configs (np.ndarray): Array of sector configurations for lattice sites.
        op_list (np.ndarray): List of 4-operator matrices acting on the lattice sites.
        op_sites_list (list of ints): List of 4 site indices where the operator acts.

    Returns:
        (row_list, col_list, value_list):
        - row_list (np.ndarray of ints): The row indices of nonzero elements.
        - col_list (np.ndarray of ints): The column indices of nonzero elements.
        - value_list (np.ndarray of complex): The nonzero values of the operator elements.
    """
    N = sector_configs.shape[0]
    n_sites = sector_configs.shape[1]
    d_loc = np.int32(op_list[0].shape[-1])
    M = len(op_sites_list)
    # Array to hold the number of nonzero columns per row
    nnz_cols_per_row = np.zeros(N, dtype=np.int32)
    # Estimate the number of nonzero elements per row
    for ii in prange(N):
        prod = 1
        for kk in range(M):
            site = op_sites_list[kk]
            operator = op_list[kk, site]
            row = sector_configs[ii, site]
            count = 0
            for col in range(d_loc):
                if np.abs(operator[row, col]) > 1e-10:
                    count += 1
            prod *= count
        nnz_cols_per_row[ii] = prod
    # Allocate the output arrays
    total_nnz = 0
    for ii in range(N):
        tmp = nnz_cols_per_row[ii]
        nnz_cols_per_row[ii] = total_nnz
        total_nnz += tmp
    # Preallocate the output arrays
    row_list = np.empty(total_nnz, dtype=np.int32)
    col_list = np.empty(total_nnz, dtype=np.int32)
    value_list = np.empty(total_nnz, dtype=np.complex128)
    # --- PASS 2: explicit 4‐nested‐loops version for M=4  ---------------
    for irow in prange(N):
        # 1) grab the bra’s full config and where to write
        row_cfg = sector_configs[irow]
        ptr = nnz_cols_per_row[irow]
        # 2) build the per‐site (idxs,vs,lens) from the one‐site operators
        idxs = np.empty((M, d_loc), np.int32)
        vs = np.empty((M, d_loc), np.complex128)
        lens = np.empty(M, np.int32)
        for kk in range(M):
            site = op_sites_list[kk]
            Op = op_list[kk, site]
            a = row_cfg[site]
            cnt = 0
            for b in range(d_loc):
                v = Op[a, b]  # or Op[b, a], depending on your storage
                if np.abs(v) > 1e-10:
                    idxs[kk, cnt] = b
                    vs[kk, cnt] = v
                    cnt += 1
            lens[kk] = cnt
        # 3) a scratch array for the ket
        ket_config = np.empty(n_sites, np.int32)
        # start from the bra each time
        for jj in range(n_sites):
            ket_config[jj] = row_cfg[jj]
        # 4) four fully‐nested loops
        site0 = op_sites_list[0]
        site1 = op_sites_list[1]
        site2 = op_sites_list[2]
        site3 = op_sites_list[3]
        for i0 in range(lens[0]):
            b0 = idxs[0, i0]
            v0 = vs[0, i0]
            ket_config[site0] = b0
            for i1 in range(lens[1]):
                b1 = idxs[1, i1]
                v1 = vs[1, i1]
                ket_config[site1] = b1
                for i2 in range(lens[2]):
                    b2 = idxs[2, i2]
                    v2 = vs[2, i2]
                    ket_config[site2] = b2
                    for i3 in range(lens[3]):
                        b3 = idxs[3, i3]
                        v3 = vs[3, i3]
                        ket_config[site3] = b3
                        # compute the full amplitude
                        amp = v0 * v1 * v2 * v3
                        # lookup that ket in your sorted sector_configs
                        icol = config_to_index_binarysearch(ket_config, sector_configs)
                        # write out the entry
                        row_list[ptr] = irow
                        col_list[ptr] = icol
                        value_list[ptr] = amp
                        ptr += 1
    return row_list, col_list, value_list


@njit(parallel=True, cache=True)
def nbody_data_2sites(
    op_list: np.ndarray, op_sites_list: np.ndarray, sector_configs: np.ndarray
):
    """
    Compute the nonzero elements of an nbody-operator.

    Args:
        sector_configs (np.ndarray): Array of sector configurations for lattice sites.
        op_list (np.ndarray): List of 2 operator matrices acting on the lattice sites.
        op_sites_list (list of ints): List of 2 site indices where the operator acts.

    Returns:
        (row_list, col_list, value_list):
        - row_list (np.ndarray of ints): The row indices of nonzero elements.
        - col_list (np.ndarray of ints): The column indices of nonzero elements.
        - value_list (np.ndarray of complex): The nonzero values of the operator elements.
    """
    N = sector_configs.shape[0]
    n_sites = sector_configs.shape[1]
    d_loc = np.int32(op_list[0].shape[-1])
    M = len(op_sites_list)
    # Array to hold the number of nonzero columns per row
    nnz_cols_per_row = np.zeros(N, dtype=np.int32)
    # Estimate the number of nonzero elements per row
    for ii in prange(N):
        prod = 1
        for kk in range(M):
            site = op_sites_list[kk]
            operator = op_list[kk, site]
            row = sector_configs[ii, site]
            count = 0
            for col in range(d_loc):
                if np.abs(operator[row, col]) > 1e-10:
                    count += 1
            prod *= count
        nnz_cols_per_row[ii] = prod
    # Allocate the output arrays
    total_nnz = 0
    for ii in range(N):
        tmp = nnz_cols_per_row[ii]
        nnz_cols_per_row[ii] = total_nnz
        total_nnz += tmp
    # Preallocate the output arrays
    row_list = np.empty(total_nnz, dtype=np.int32)
    col_list = np.empty(total_nnz, dtype=np.int32)
    value_list = np.empty(total_nnz, dtype=np.complex128)
    # --- PASS 2: explicit 4‐nested‐loops version for M=4  ---------------
    for irow in prange(N):
        # 1) grab the bra’s full config and where to write
        row_cfg = sector_configs[irow]
        ptr = nnz_cols_per_row[irow]
        # 2) build the per‐site (idxs,vs,lens) from the one‐site operators
        idxs = np.empty((M, d_loc), np.int32)
        vs = np.empty((M, d_loc), np.complex128)
        lens = np.empty(M, np.int32)
        for kk in range(M):
            site = op_sites_list[kk]
            Op = op_list[kk, site]
            a = row_cfg[site]
            cnt = 0
            for b in range(d_loc):
                v = Op[a, b]  # or Op[b, a], depending on your storage
                if np.abs(v) > 1e-10:
                    idxs[kk, cnt] = b
                    vs[kk, cnt] = v
                    cnt += 1
            lens[kk] = cnt

        # 3) a scratch array for the ket
        ket_config = np.empty(n_sites, np.int32)
        # 4) four fully‐nested loops
        site0 = op_sites_list[0]
        site1 = op_sites_list[1]

        for i0 in range(lens[0]):
            # start from the bra each time
            for jj in range(n_sites):
                ket_config[jj] = row_cfg[jj]

            b0 = idxs[0, i0]
            v0 = vs[0, i0]
            ket_config[site0] = b0

            for i1 in range(lens[1]):
                b1 = idxs[1, i1]
                v1 = vs[1, i1]
                ket_config[site1] = b1
                # compute the full amplitude
                amp = v0 * v1
                # lookup that ket in your sorted sector_configs
                icol = config_to_index_binarysearch(ket_config, sector_configs)
                # write out the entry
                row_list[ptr] = irow
                col_list[ptr] = icol
                value_list[ptr] = amp
                ptr += 1
    return row_list, col_list, value_list


@njit(parallel=True, cache=True)
def localbody_data_par2(
    op: np.ndarray, op_site_list: list[int], sector_configs: np.ndarray
):
    """
    Efficiently processes a diagonal operator that acts on several sites at once.
    For each configuration (each row in sector_configs), the function sums the
    contributions from all sites in op_site_list. Only configurations where the
    total contribution is nonzero are kept.

    Args:
        op (np.ndarray): A diagonal operator matrix that is used for each site.
            (It is assumed that op[site] returns the diagonal matrix for that site.)
        op_site_list (list[int]): List of site indices where the operator acts.
        sector_configs (np.ndarray): Array of sector configurations for lattice sites
            (shape (num_configs, num_sites)), with type np.uint8.

    Returns:
        tuple:
            - row_list (np.ndarray of ints): The indices of configurations (rows) with nonzero contribution.
            - col_list (np.ndarray of ints): Identical to row_list (since the operator is diagonal).
            - value_list (np.ndarray of complex): The computed (summed) diagonal values for those configurations.
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
    """
    Efficiently process a diagonal operator on a given sector of configurations.

    Args:
        op (np.ndarray): A single-site diagonal operator matrix.
        op_sites_list (int): site index where the operator acts.
        sector_configs (np.ndarray): Array of sector configurations for lattice sites.

    Returns:
        (row_list, col_list, value_list):
            - row_list (np.ndarray of ints): The row indices of diagonal elements.
            - col_list (np.ndarray of ints): Same as row_list (since diagonal).
            - value_list (np.ndarray of complex): The diagonal elements of the operator.
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
