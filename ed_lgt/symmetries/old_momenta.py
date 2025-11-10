import numpy as np
from numba import njit, prange
from .generate_configs import config_to_index_binarysearch


@njit(cache=True, parallel=True)
def momentum_basis_zero_k(
    sector_configs: np.ndarray,
    lvals: np.ndarray,
    unit_cell_size: np.ndarray,
) -> np.ndarray:
    """
    Build the Γ-sector (k = 0) projector.

    Theory (Γ sector)
    -----------------
    With k = (0,...,0) every translation-orbit contributes exactly one
    orthonormal basis vector:
        |Γ; ref> = (1/√∏_d p_d) * Σ_{t_d=0}^{p_d-1} T(t) |ref>
    where p_d is the orbit period on axis d. Phases are all 1, so the projector
    can be stored real (float64).

    Inputs
    ------
    sector_configs : (N_configs, N_sites) int
        List of configurations (each row encodes a configuration).
    lvals : (D,) int
        Lattice extents per axis (L_x, L_y, ...).
    unit_cell_size : (D,) int
        Block size s_d per axis used for translations (must divide L_d).

    Returns
    -------
    k_basis : (N_configs, N_refs) float64
        Projector columns spanning the Γ sector (one column per orbit).
    """
    n_configs = sector_configs.shape[0]
    # 1) precompute full translation table and per-axis block counts R_d
    translations_array, shifts_per_dir = build_all_translations(
        sector_configs, lvals, unit_cell_size
    )
    lattice_dim = shifts_per_dir.size
    # 2) choose one representative per orbit and get their period vectors p
    #    Passing k_vals = 0 keeps *all* orbits (compatibility is automatic).
    k_zero = np.zeros(lattice_dim, np.int32)
    references, period_vectors = select_references(
        translations_array, shifts_per_dir, k_zero
    )
    n_refs = references.shape[0]
    # 3) allocate Γ-projector (real is enough since phases are 1)
    k_basis = np.zeros((n_configs, n_refs), np.float64)
    # 4) build each Γ Bloch-sum column
    for ref_col in prange(n_refs):
        ref_idx = references[ref_col]
        pvec = period_vectors[ref_col, :]
        orbit_size = np.prod(pvec)  # ∏_d p_d
        # NEW: per-column sparse accumulator (real-valued here)
        used_indices = np.empty(orbit_size, np.int32)
        used_values = np.zeros(orbit_size, np.float64)
        used_len = 0
        # scratch arrays
        t_local = np.zeros(lattice_dim, np.int32)  # t_d in [0 .. p_d-1]
        # sum over fundamental domain t_d = 0..p_d-1
        for flat_local in range(orbit_size):
            # decode t under base p (periods)
            decode_mixed_index(flat_local, pvec, t_local)
            # map t (base R) into the precomputed translation table
            flat_full = encode_shift(t_local, shifts_per_dir)  # base R_d
            cfg_j = translations_array[ref_idx, flat_full]
            # NEW: deduplicate & accumulate multiplicities (phase = 1)
            pos = -1
            for u in range(used_len):
                if used_indices[u] == cfg_j:
                    pos = u
                    break
            if pos == -1:
                pos = used_len
                used_indices[pos] = cfg_j
                used_len += 1
            used_values[pos] += 1.0
        # NEW: normalize by the actual 2-norm of the accumulated counts
        norm_sq = 0.0
        for u in range(used_len):
            norm_sq += used_values[u] * used_values[u]
        if np.isclose(norm_sq, 0.0):
            continue
        inv_norm = 1.0 / np.sqrt(norm_sq)
        # Write normalized column
        for u in range(used_len):
            k_basis[used_indices[u], ref_col] = used_values[u] * inv_norm
    return k_basis


@njit(cache=True, parallel=True)
def momentum_basis_finite_k(
    sector_configs: np.ndarray,
    lvals: np.ndarray,
    unit_cell_size: np.ndarray,
    k_vals: np.ndarray,
) -> np.ndarray:
    """
    Build the momentum-sector projector for k = (k_x, k_y, k_z, ...).

    Theory summary
    --------------
    Let L_d be the lattice extent along axis d, and s_d = unit_cell_size[d]
    the block size we translate by on that axis. Then R_d = L_d / s_d is the
    number of block-positions along axis d. The allowed translations form
    G = Z_{R0} × ... × Z_{R_{D-1}}.

    For a configuration |c>, its translation orbit size is \prod_d p_d, where p_d is
    the minimal positive integer with T(p_d * e_d)|c> = |c| (i.e. the period on axis d).
    A momentum k is compatible with this orbit iff
        (k_d * p_d) mod R_d == 0  for all axes d.
    For each compatible orbit (represented by a reference index `ref` and periods p),
    the Bloch sum builds one orthonormal basis state:
    |k; ref> = (1/\sqrt{\prod_{d} p_d}) * Σ_{t_d=0}^{p_d-1} exp[-2πi Σ_d (k_d t_d / R_d)] T(t)|ref>
    where t = (t_0,...,t_{D-1}) runs over the “fundamental domain” 0..p_d-1.

    Strategy:
      1) Precompute translations table and per-axis block counts R_d.
      2) Get orbit representatives and their axis periods p_d from select_references.
      3) For each representative, *deduplicate* images:
         - loop over the 'period box' t_d = 0..p_d-1
         - map to an image cfg_j via the precomputed translations
         - accumulate the complex phase for that image
         - after the loop, normalize the column by its 2-norm and write it.
    Inputs
    ------
    sector_configs : (n_configs, n_sites) int
        Your configuration list (each row encodes a configuration).
    lvals : (lattice_dim,) int
        Lattice lengths per axis (L_x, L_y, ...).
    unit_cell_size : (lattice_dim,) int
        Block sizes s_d per axis used for translations (must divide L_d).
    k_vals : (lattice_dim,) int
        Requested momentum labels k_d, understood modulo R_d = L_d / s_d.

    Returns
    -------
    k_basis : (n_configs, n_refs_kept) complex128
        Projector columns spanning the k-sector (one column per kept orbit).
    """
    n_configs = sector_configs.shape[0]
    # 1) precompute all translations and the per-axis block counts R_d
    # shifts_per_dir[d] == R_d
    translations_array, shifts_per_dir = build_all_translations(
        sector_configs, lvals, unit_cell_size
    )
    lattice_dim = shifts_per_dir.size
    # 2) pick orbit representatives compatible with k, and get their period vectors p
    # (select_references also ensures we never duplicate orbits)
    references, period_vectors = select_references(
        translations_array, shifts_per_dir, k_vals
    )
    n_refs = references.shape[0]
    # 3) allocate the projector
    k_basis = np.zeros((n_configs, n_refs), np.complex128)
    # NOTE: to look up into translations_array we still need to encode t in the *full* radix R_d
    # encode_shift(t_local, shifts_per_dir) does that.

    # 4) build each Bloch-sum column
    for ref_col in prange(n_refs):
        ref_idx = references[ref_col]  # representative configuration index
        pvec = period_vectors[ref_col, :]  # periods p_d for this orbit
        # Upper bound on distinct images we’ll see when scanning the period box
        # (may be larger than true orbit size if mixed stabilizers exist).
        orbit_size = np.prod(pvec)  # prod_d p_d
        # NEW: per-column sparse accumulator to deduplicate images
        # We avoid an O(N_configs) mask.
        # Instead, we hold at most 'orbit_size'
        # unique images and their accumulated amplitudes.
        used_indices = np.empty(orbit_size, np.int32)  # image indices
        used_values = np.zeros(orbit_size, np.complex128)  # accumulated amps
        used_len = 0
        # scratch arrays
        t_local = np.zeros(lattice_dim, np.int32)  # t_d in [0 .. p_d-1]
        # sum over the fundamental domain t_d = 0..p_d-1
        for flat_local in range(orbit_size):
            # decode mixed-radix index into per-axis t_d under base p_d
            decode_mixed_index(flat_local, pvec, t_local)
            # phase = exp(-2πi Σ_d (k_d * t_d / R_d))
            phase_arg = 0.0
            for ax in range(lattice_dim):
                # reduce k_d modulo R_d (harmless if already in range)
                kd = k_vals[ax] % shifts_per_dir[ax]
                phase_arg += (kd * t_local[ax]) / float(shifts_per_dir[ax])
            phase = np.exp(-1j * 2.0 * np.pi * phase_arg)
            # find which configuration this combined block-shift produces
            flat_full = encode_shift(t_local, shifts_per_dir)  # base R_d
            # index in sector_configs
            cfg_j = translations_array[ref_idx, flat_full]
            # NEW: deduplicate & accumulate
            # linear search is fine since used_len <= orbit_size (typically small)
            pos = -1
            for u in range(used_len):
                if used_indices[u] == cfg_j:
                    pos = u
                    break
            if pos == -1:
                pos = used_len
                used_indices[pos] = cfg_j
                used_len += 1
            used_values[pos] += phase
        # NEW: column normalization by the actual 2-norm of accumulated amplitudes
        # This is robust even if the selector admitted an incompatible orbit:
        # the sum over a stabilizer with nontrivial character cancels to ~0.
        norm_sq = 0.0
        for u in range(used_len):
            norm_sq += np.abs(used_values[u]) * np.abs(used_values[u])
        if norm_sq <= 1e-30:
            # incompatible orbit at this k → leave this column zero
            # (optional: you could flag and drop these later)
            continue
        inv_norm = 1.0 / np.sqrt(norm_sq)
        # Write normalized column (exactly one entry per unique image)
        for u in range(used_len):
            k_basis[used_indices[u], ref_col] = used_values[u] * inv_norm
    return k_basis


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
    v=H^(K)_[r,c] = sum_{i1,i2} B^{*,T}_[r,i1] x H_[i1,i2] x B_[i2,c]
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
    v=H^(K)_[r,c] = sum_{i1,i2} B^{*,T}_[r,i1] x H_[i1,i2] x B_[i2,c]
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
    momentum_basis: np.ndarray,  # shape (N, n_refs), complex128 or float64
):
    """
    We want to isolate the nonzero entries r,c,v of
    v=H^(K)_[r,c] = sum_{i1,i2} B^{*,T}_[r,i1] x H_[i1,i2] x B_[i2,c]
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
