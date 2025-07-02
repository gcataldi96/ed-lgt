import numpy as np
from numba import njit, prange
from .generate_configs import config_to_index_binarysearch

__all__ = [
    "check_normalization",
    "check_orthogonality",
    "get_momentum_basis",
    "nbody_data_momentum_4sites",
    "nbody_data_momentum_2sites",
    "nbody_data_momentum_1site",
]


# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, parallel=True)
def check_normalization(basis: np.ndarray):
    for ii in prange(basis.shape[1]):
        if not np.isclose(np.linalg.norm(basis[:, ii]), 1):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, parallel=True)
def check_orthogonality(basis: np.ndarray):
    for ii in prange(basis.shape[1]):
        for jj in range(ii + 1, basis.shape[1]):
            if not np.isclose(np.vdot(basis[:, ii], basis[:, jj]), 0, atol=1e-10):
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, parallel=True)
def build_all_translations(sector_configs: np.ndarray, logical_unit_size: int):
    """
    For each of the N configurations, roll it by multiples of logical_unit_size
    and record the index within sector_configs of each translation.
    Returns T of shape (N, R) where R = n_sites//logical_unit_size.
    """
    N, n_sites = sector_configs.shape
    R = n_sites // logical_unit_size
    T = np.empty((N, R), np.int32)
    for i in prange(N):
        cfg = sector_configs[i]
        # manually roll by blocks of size logical_unit_size
        for t in range(R):
            shift = (t * logical_unit_size) % n_sites
            # build rolled in-place
            rolled = np.empty(n_sites, np.int32)
            for j in range(n_sites):
                rolled[j] = cfg[(j + shift) % n_sites]
            T[i, t] = config_to_index_binarysearch(rolled, sector_configs)
    return T


# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True)
def select_references(T: np.ndarray):
    """
    Given T[i,0..R-1] the translation-orbit of config i,
    pick one reference per orbit: the first i that isn't in any previous orbit.
    Also compute norm[i] = number of unique translations for that reference.
    Returns:
      ref_list (length N, with the first N_ref entries filled),
      N_ref (int),
      norm   (length N, only meaningful at ref_list[:N_ref]).
    """
    N, R = T.shape
    ref_list = np.empty(N, np.int32)
    norm = np.zeros(N, np.int32)
    N_ref = 0
    # loop in index order
    for i in range(N):
        orbit = T[i]
        is_ref = True
        # check against previously chosen references
        for rr in range(N_ref):
            r = ref_list[rr]
            # if r appears in orbit, i belongs to that existing orbit
            for t in range(R):
                if orbit[t] == r:
                    is_ref = False
                    break
            if not is_ref:
                break
        if is_ref:
            # accept i as new reference
            ref_list[N_ref] = i
            # count unique translations
            # simple O(R^2) small loop (R <= n_sites)
            unique_count = 0
            for t in range(R):
                x = orbit[t]
                seen = False
                for tt in range(t):
                    if orbit[tt] == x:
                        seen = True
                        break
                if not seen:
                    unique_count += 1
            norm[i] = unique_count
            N_ref += 1
    return ref_list, N_ref, norm


# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, parallel=True)
def momentum_basis_zero_k(
    sector_configs: np.ndarray, logical_unit_size: np.int32
) -> np.ndarray:
    N = sector_configs.shape[0]
    # 1) compute T[i,t] table of all translations
    translations_array = build_all_translations(sector_configs, logical_unit_size)
    # 2) pick one reference per orbit & compute each orbit’s size
    ref_list, N_ref, norm = select_references(translations_array)
    # 3) build the N×N_ref basis matrix with the right phase for k
    k_basis = np.zeros((N, N_ref), np.float64)
    for rr in prange(N_ref):
        i_ref = ref_list[rr]
        R = norm[i_ref]
        for t in range(R):
            i = translations_array[i_ref, t]
            k_basis[i, rr] = 1 / np.sqrt(R)
    return k_basis


# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, parallel=True)
def momentum_basis_finite_k(
    sector_configs: np.ndarray, logical_unit_size: np.int32, k: np.int32
) -> np.ndarray:
    N = sector_configs.shape[0]
    # 1) compute T[i,t] table of all translations
    translations_array = build_all_translations(sector_configs, logical_unit_size)
    # 2) pick one reference per orbit & compute each orbit’s size
    ref_list, N_ref, norm = select_references(translations_array)
    # 3) build the N×N_ref basis matrix with the right phase for k
    k_basis = np.zeros((N, N_ref), np.complex128)
    for r in prange(N_ref):
        i_ref = ref_list[r]
        R = norm[i_ref]
        for t in range(R):
            i = translations_array[i_ref, t]
            phase = np.exp(-1j * 2.0 * np.pi * k * t / R)
            k_basis[i, r] = phase / np.sqrt(R)
    return k_basis


# ─────────────────────────────────────────────────────────────────────────────
def get_momentum_basis(
    sector_configs: np.ndarray, logical_unit_size: np.int32, k: np.int32
) -> np.ndarray:
    if k != 0:
        return momentum_basis_finite_k(sector_configs, logical_unit_size, k)
    else:
        return momentum_basis_zero_k(sector_configs, logical_unit_size)
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


# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, parallel=True)
def nbody_data_momentum_4sites(
    op_list: np.ndarray,  # shape (4, n_sites, d_loc, d_loc)
    op_sites_list: np.ndarray,  # shape (4,), int32
    sector_configs: np.ndarray,  # shape (N, n_sites), int32
    momentum_basis: np.ndarray,  # shape (N, Bdim), complex128 or float64
):
    """
    We want to isolate the nonzero entries r,c,v of
    v=H^(K)_[r,c] = \sum_{i1,i2} B^{*,T}_[r,i1] x H_[i1,i2] x B_[i2,c]
    """
    n_sites = sector_configs.shape[1]
    Bdim = momentum_basis.shape[1]
    d_loc = op_list.shape[2]
    M = len(op_sites_list)
    # 1) get the nonzero col indices of the momentum basis B
    # nb_indptr of shape (Bdim+1) & nb_indices of shape (nnz_B)
    # rows = indices[indptr[c] : indptr[c+1]] are the nonzero rows r such that B[r,c] !=0
    nb_indptr, nb_indices = precompute_nonzero_csr(momentum_basis)
    # 2) get the nonzero row indices of the momentum basis B
    # nbT_indptr of  shape (N+1) & nbT_indices of shape (nnz_B)
    nbT_indptr, nbT_indices = precompute_nonzero_csr(momentum_basis.T)
    # cols = indices[indptr[r] : indptr[r+1]] are the nonzero cols c such that B[r,c] !=0
    # PASS 1: count nonzeros per momentum-row prow
    nnz_per_row = np.zeros(Bdim, np.int32)

    for prow in prange(Bdim):
        cnt = 0
        # all real-space configs j1 with B[j1,prow] != 0
        start1 = nb_indptr[prow]
        stop1 = nb_indptr[prow + 1]
        for idx1 in range(start1, stop1):
            j1 = nb_indices[idx1]
            bra_cfg = sector_configs[j1]
            # build four per-site nonzero lists
            idxs = np.empty((M, d_loc), np.int32)
            lens = np.empty(M, np.int32)
            for kk in range(M):
                site = op_sites_list[kk]
                Op = op_list[kk, site]
                a = sector_configs[j1, site]
                cnt1 = 0
                for b in range(d_loc):
                    if np.abs(Op[a, b]) > 1e-10:
                        idxs[kk, cnt1] = b
                        cnt1 += 1
                lens[kk] = cnt1

            ket_cfg = np.empty(n_sites, np.int32)
            # start from the bra each time
            for jj in range(n_sites):
                ket_cfg[jj] = bra_cfg[jj]
            # 4) four fully‐nested loops
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
                            # number of nonzero momentum-cols for j2
                            cnt += nbT_indptr[j2 + 1] - nbT_indptr[j2]

        nnz_per_row[prow] = cnt

    # prefix-sum offsets
    offset = 0
    for prow in range(Bdim):
        tmp = nnz_per_row[prow]
        nnz_per_row[prow] = offset
        offset += tmp
    total_nnz = offset
    # Preallocate the output arrays
    row_list = np.empty(total_nnz, dtype=np.int32)
    col_list = np.empty(total_nnz, dtype=np.int32)
    value_list = np.empty(total_nnz, np.complex128)

    # PASS 2: fill entries
    for prow in prange(Bdim):
        ptr = nnz_per_row[prow]
        start1 = nb_indptr[prow]
        stop1 = nb_indptr[prow + 1]
        for idx1 in range(start1, stop1):
            j1 = nb_indices[idx1]
            bra_cfg = sector_configs[j1]
            # rebuild idxs, vs, lens
            idxs = np.empty((M, d_loc), np.int32)
            vs = np.empty((M, d_loc), np.complex128)
            lens = np.empty(M, np.int32)
            for kk in range(M):
                site = op_sites_list[kk]
                Op = op_list[kk, site]
                a = sector_configs[j1, site]
                cnt2 = 0
                for b in range(d_loc):
                    v = Op[a, b]
                    if abs(v) > 1e-10:
                        idxs[kk, cnt2] = b
                        vs[kk, cnt2] = v
                        cnt2 += 1
                lens[kk] = cnt2

            # a scratch array for the ket
            ket_cfg = np.empty(n_sites, np.int32)
            # start from the bra each time
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
                            # measure the amplitude
                            amp_M = v0 * v1 * v2 * v3
                            # project into momentum-pcol
                            start2 = nbT_indptr[j2]
                            stop2 = nbT_indptr[j2 + 1]
                            for tt in range(start2, stop2):
                                pcol = nbT_indices[tt]
                                val = (
                                    np.conj(momentum_basis[j1, prow])
                                    * amp_M
                                    * momentum_basis[j2, pcol]
                                )
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
    momentum_basis: np.ndarray,  # shape (N, Bdim), complex128 or float64
):
    """
    We want to isolate the nonzero entries r,c,v of
    v=H^(K)_[r,c] = \sum_{i1,i2} B^{*,T}_[r,i1] x H_[i1,i2] x B_[i2,c]
    """
    n_sites = sector_configs.shape[1]
    Bdim = momentum_basis.shape[1]
    d_loc = op_list.shape[2]
    M = len(op_sites_list)
    # 1) get the nonzero col indices of the momentum basis B
    # nb_indptr of shape (Bdim+1) & nb_indices of shape (nnz_B)
    # rows = indices[indptr[c] : indptr[c+1]] are the nonzero rows r such that B[r,c] !=0
    nb_indptr, nb_indices = precompute_nonzero_csr(momentum_basis)
    # 2) get the nonzero row indices of the momentum basis B
    # nbT_indptr of  shape (N+1) & nbT_indices of shape (nnz_B)
    nbT_indptr, nbT_indices = precompute_nonzero_csr(momentum_basis.T)
    # cols = indices[indptr[r] : indptr[r+1]] are the nonzero cols c such that B[r,c] !=0
    # PASS 1: count nonzeros per momentum-row prow
    nnz_per_row = np.zeros(Bdim, np.int32)

    for prow in prange(Bdim):
        cnt = 0
        # all real-space configs j1 with B[j1,prow] != 0
        start1 = nb_indptr[prow]
        stop1 = nb_indptr[prow + 1]
        for idx1 in range(start1, stop1):
            j1 = nb_indices[idx1]
            bra_cfg = sector_configs[j1]
            # build four per-site nonzero lists
            idxs = np.empty((M, d_loc), np.int32)
            lens = np.empty(M, np.int32)
            for kk in range(M):
                site = op_sites_list[kk]
                Op = op_list[kk, site]
                a = sector_configs[j1, site]
                cnt1 = 0
                for b in range(d_loc):
                    if np.abs(Op[a, b]) > 1e-10:
                        idxs[kk, cnt1] = b
                        cnt1 += 1
                lens[kk] = cnt1

            ket_cfg = np.empty(n_sites, np.int32)
            # start from the bra each time
            for jj in range(n_sites):
                ket_cfg[jj] = bra_cfg[jj]
            # 4) four fully‐nested loops
            site0 = op_sites_list[0]
            site1 = op_sites_list[1]
            # explicit 4-nested loops → real-space j2
            for i0 in range(lens[0]):
                b0 = idxs[0, i0]
                ket_cfg[site0] = b0
                for i1 in range(lens[1]):
                    b1 = idxs[1, i1]
                    ket_cfg[site1] = b1
                    # get the new configuration
                    j2 = config_to_index_binarysearch(ket_cfg, sector_configs)
                    # number of nonzero momentum-cols for j2
                    cnt += nbT_indptr[j2 + 1] - nbT_indptr[j2]

        nnz_per_row[prow] = cnt

    # prefix-sum offsets
    offset = 0
    for prow in range(Bdim):
        tmp = nnz_per_row[prow]
        nnz_per_row[prow] = offset
        offset += tmp
    total_nnz = offset
    # Preallocate the output arrays
    row_list = np.empty(total_nnz, dtype=np.int32)
    col_list = np.empty(total_nnz, dtype=np.int32)
    value_list = np.empty(total_nnz, np.complex128)

    # PASS 2: fill entries
    for prow in prange(Bdim):
        ptr = nnz_per_row[prow]
        start1 = nb_indptr[prow]
        stop1 = nb_indptr[prow + 1]
        for idx1 in range(start1, stop1):
            j1 = nb_indices[idx1]
            bra_cfg = sector_configs[j1]
            # rebuild idxs, vs, lens
            idxs = np.empty((M, d_loc), np.int32)
            vs = np.empty((M, d_loc), np.complex128)
            lens = np.empty(M, np.int32)
            for kk in range(M):
                site = op_sites_list[kk]
                Op = op_list[kk, site]
                a = sector_configs[j1, site]
                cnt2 = 0
                for b in range(d_loc):
                    v = Op[a, b]
                    if abs(v) > 1e-10:
                        idxs[kk, cnt2] = b
                        vs[kk, cnt2] = v
                        cnt2 += 1
                lens[kk] = cnt2

            # a scratch array for the ket
            ket_cfg = np.empty(n_sites, np.int32)
            # start from the bra each time
            for jj in range(n_sites):
                ket_cfg[jj] = bra_cfg[jj]
            site0 = op_sites_list[0]
            site1 = op_sites_list[1]
            # explicit 4-loops + projection
            for i0 in range(lens[0]):
                b0 = idxs[0, i0]
                v0 = vs[0, i0]
                ket_cfg[site0] = b0
                for i1 in range(lens[1]):
                    b1 = idxs[1, i1]
                    v1 = vs[1, i1]
                    ket_cfg[site1] = b1
                    j2 = config_to_index_binarysearch(ket_cfg, sector_configs)
                    # measure the amplitude
                    amp_M = v0 * v1
                    # project into momentum-pcol
                    start2 = nbT_indptr[j2]
                    stop2 = nbT_indptr[j2 + 1]
                    for tt in range(start2, stop2):
                        pcol = nbT_indices[tt]
                        val = (
                            np.conj(momentum_basis[j1, prow])
                            * amp_M
                            * momentum_basis[j2, pcol]
                        )
                        row_list[ptr] = prow
                        col_list[ptr] = pcol
                        value_list[ptr] = val
                        ptr += 1
    return row_list, col_list, value_list


@njit(cache=True, parallel=True)
def nbody_data_momentum_1site(
    op_list: np.ndarray,  # shape (1, n_sites, d_loc, d_loc)
    op_sites_list: np.ndarray,  # shape (1,), int32
    sector_configs: np.ndarray,  # shape (N, n_sites), int32
    momentum_basis: np.ndarray,  # shape (N, Bdim), complex128 or float64
):
    """
    We want to isolate the nonzero entries r,c,v of
    v=H^(K)_[r,c] = \sum_{i1,i2} B^{*,T}_[r,i1] x H_[i1,i2] x B_[i2,c]
    """
    n_sites = sector_configs.shape[1]
    Bdim = momentum_basis.shape[1]
    d_loc = op_list.shape[2]
    # Select the lattice site
    site = op_sites_list[0]
    Op = op_list[0, site]
    # 1) get the nonzero col indices of the momentum basis B
    # nb_indptr of shape (Bdim+1) & nb_indices of shape (nnz_B)
    # rows = indices[indptr[c] : indptr[c+1]] are the nonzero rows r such that B[r,c] !=0
    nb_indptr, nb_indices = precompute_nonzero_csr(momentum_basis)
    # 2) get the nonzero row indices of the momentum basis B
    # nbT_indptr of  shape (N+1) & nbT_indices of shape (nnz_B)
    nbT_indptr, nbT_indices = precompute_nonzero_csr(momentum_basis.T)
    # cols = indices[indptr[r] : indptr[r+1]] are the nonzero cols c such that B[r,c] !=0
    # PASS 1: count nonzeros per momentum-row prow
    nnz_per_row = np.zeros(Bdim, np.int32)
    for prow in prange(Bdim):
        cnt = 0
        # all real-space configs j1 with B[j1,prow] != 0
        start1 = nb_indptr[prow]
        stop1 = nb_indptr[prow + 1]
        for idx1 in range(start1, stop1):
            j1 = nb_indices[idx1]
            bra_cfg = sector_configs[j1]
            # build four per-site nonzero lists
            idxs = np.empty(d_loc, np.int32)
            a = sector_configs[j1, site]
            cnt1 = 0
            for b in range(d_loc):
                if np.abs(Op[a, b]) > 1e-10:
                    idxs[cnt1] = b
                    cnt1 += 1
            lens = cnt1

            ket_cfg = np.empty(n_sites, np.int32)
            # start from the bra each time
            for jj in range(n_sites):
                ket_cfg[jj] = bra_cfg[jj]
            # explicit 1-nested loop → real-space j2
            for i0 in range(lens):
                b0 = idxs[i0]
                ket_cfg[site] = b0
                # get the new configuration
                j2 = config_to_index_binarysearch(ket_cfg, sector_configs)
                # number of nonzero momentum-cols for j2
                cnt += nbT_indptr[j2 + 1] - nbT_indptr[j2]

        nnz_per_row[prow] = cnt

    # prefix-sum offsets
    offset = 0
    for prow in range(Bdim):
        tmp = nnz_per_row[prow]
        nnz_per_row[prow] = offset
        offset += tmp
    total_nnz = offset
    # Preallocate the output arrays
    row_list = np.empty(total_nnz, dtype=np.int32)
    col_list = np.empty(total_nnz, dtype=np.int32)
    value_list = np.empty(total_nnz, np.complex128)
    # PASS 2: fill entries
    for prow in prange(Bdim):
        ptr = nnz_per_row[prow]
        start1 = nb_indptr[prow]
        stop1 = nb_indptr[prow + 1]
        for idx1 in range(start1, stop1):
            j1 = nb_indices[idx1]
            bra_cfg = sector_configs[j1]
            # rebuild idxs, vs, lens
            idxs = np.empty(d_loc, np.int32)
            vs = np.empty(d_loc, np.complex128)
            a = sector_configs[j1, site]
            cnt2 = 0
            for b in range(d_loc):
                v = Op[a, b]
                if abs(v) > 1e-10:
                    idxs[cnt2] = b
                    vs[cnt2] = v
                    cnt2 += 1
            lens = cnt2

            # a scratch array for the ket
            ket_cfg = np.empty(n_sites, np.int32)
            # start from the bra each time
            for jj in range(n_sites):
                ket_cfg[jj] = bra_cfg[jj]
            # explicit 1-nested loop + projection
            for i0 in range(lens):
                b0 = idxs[i0]
                v0 = vs[i0]
                ket_cfg[site] = b0
                j2 = config_to_index_binarysearch(ket_cfg, sector_configs)
                # measure the amplitude
                amp_M = v0
                # project into momentum-pcol
                start2 = nbT_indptr[j2]
                stop2 = nbT_indptr[j2 + 1]
                for tt in range(start2, stop2):
                    pcol = nbT_indices[tt]
                    val = (
                        np.conj(momentum_basis[j1, prow])
                        * amp_M
                        * momentum_basis[j2, pcol]
                    )
                    row_list[ptr] = prow
                    col_list[ptr] = pcol
                    value_list[ptr] = val
                    ptr += 1
    return row_list, col_list, value_list
