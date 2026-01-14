import numpy as np
from scipy.sparse import csc_matrix
import logging

logger = logging.getLogger(__name__)

__all__ = ["operator_to_mpo_via_mps2", "choose_rank_by_frobenius", "vector_to_mps"]


def truncate_singular_values(
    sing_vals: np.ndarray, rel_cut: float, chi_max: int | None = None
) -> int:
    """
    Decide how many singular values to keep when compressing an MPS/MPO bond.
    Returns only k_final, but prints concise and relevant diagnostics.
    """
    S = np.asarray(sing_vals)
    if S.ndim != 1:
        raise ValueError("sing_vals must be a 1D array")
    # Frobenius-weight contributions
    S2 = S * S
    total = S2.sum()
    if np.isclose(total, 1e-30) or np.isclose(total, 0.0):
        logger.info("bond truncation: all singular values zero → keeping rank 1.")
        return 1
    cumsum = np.cumsum(S2)
    n_all = len(S)
    # ---------------------------------------------------------
    # Rank dictated ONLY by relative tolerance
    # ---------------------------------------------------------
    target_tail = (rel_cut * rel_cut) * total
    k_rel = n_all
    for r in range(1, n_all + 1):
        discarded = total - cumsum[r - 1]
        if discarded <= target_tail:
            k_rel = r
            break
    # ---------------------------------------------------------
    # Apply chi_max if present
    # ---------------------------------------------------------
    if chi_max is not None:
        k_final = min(k_rel, chi_max)
    else:
        k_final = k_rel
    # ---------------------------------------------------------
    # Compute effective truncation error actually achieved
    # ---------------------------------------------------------
    kept_weight = cumsum[k_final - 1]
    eff_tail = total - kept_weight
    eff_rel_tail = eff_tail / total
    eff_rel_tol = np.sqrt(max(eff_rel_tail, 0.0))
    # ---------------------------------------------------------
    if chi_max is not None and k_final < k_rel:
        logger.info(f" Truncated by χ_max={chi_max}")
    logger.info(
        f"[MPS/MPO trunc] n_sv={n_all}|k_rel={k_rel}|k_final={k_final}|eff_rel_tol={eff_rel_tol:.3e}"
    )
    return k_final


def vector_to_mps(
    state_vector: np.ndarray,
    local_dims: list[int],
    chi_max: int | None = None,
    svd_rel_tol: float = 1e-6,
):
    """
    Decompose a state |psi> on a finite chain into an MPS with controlled bonds.
    Args
    ----
    state_vector:
        1D array of length prod(local_dims).
        State restricted to the support S in the product basis (matching local_dims).
    local_dims:
        [d1, d2, ..., dL] local Hilbert-space dimensions on each site in S,
        in the SAME order as used to flatten state_vector.
    chi_max:
        Optional maximum MPS bond dimension (global cap per bond).
    svd_rel_tol:
        Relative truncation tolerance at each bond:
        Discard singular values so that discarded Frobenius weight
        <= svd_rel_tol^2 of that bond's total.

    Returns
    -------
    mps:
        List of length L.
        mps[i] has shape (bond_left, d_i, bond_right).
        First tensor has bond_left = 1.
        Last tensor  has bond_right = 1.
    """
    L = len(local_dims)
    D = int(np.prod(local_dims))
    assert state_vector.size == D, "state_vector size incompatible with local_dims"
    # Reshape into tensor with one physical leg per site
    psi = state_vector.reshape(local_dims)
    mps: list[np.ndarray] = []
    left_bond_dim = 1
    # Sweep from left to right, stopping before the last site
    for site in range(L - 1):
        d_site = local_dims[site]
        # Group all left legs (current bond + this site) vs all right legs
        psi = psi.reshape(left_bond_dim * d_site, -1)  # (left * d_i) x rest
        # SVD on this bipartition
        U, S, Vh = np.linalg.svd(psi, full_matrices=False)
        # Decide how many singular values to keep using relative criterion
        keep = truncate_singular_values(S, rel_cut=svd_rel_tol, chi_max=chi_max)
        # Truncate
        U = U[:, :keep]  # shape: (left_bond_dim * d_site, keep)
        S = S[:keep]  # shape: (keep,)
        Vh = Vh[:keep, :]  # shape: (keep, rest)
        # Reshape U into the MPS tensor A_site
        #   (left_bond_dim, d_site, new_bond_dim)
        A_site = U.reshape(left_bond_dim, d_site, keep)
        mps.append(A_site)
        # Propagate remaining entanglement to the right:
        # new "psi" has shape (keep, rest)
        psi = S[:, None] * Vh
        left_bond_dim = keep
    # Last site: whatever remains is (left_bond_dim * d_last) components
    d_last = local_dims[-1]
    psi = psi.reshape(left_bond_dim, d_last)
    # Final tensor with right bond dimension fixed to 1
    A_last = psi.reshape(left_bond_dim, d_last, 1)
    mps.append(A_last)
    # Sanity checks
    assert mps[0].shape[0] == 1, "First MPS tensor must have left bond = 1."
    assert mps[-1].shape[2] == 1, "Last MPS tensor must have right bond = 1."
    return mps


def compress_mpo_left_sweep(
    mpo: list[np.ndarray], svd_rel_tol: float = 1e-6, chi_max: int | None = None
):
    L = len(mpo)
    mpo = [W.copy() for W in mpo]
    for ii in range(L - 1):
        W_i = mpo[ii]
        Dl, d_in, d_out, Dr = W_i.shape
        # reshape into (left_index, right_index)
        W_mat = W_i.reshape(Dl * d_in * d_out, Dr)
        # SVD
        U, S, Vh = np.linalg.svd(W_mat, full_matrices=False)
        k = truncate_singular_values(S, rel_cut=svd_rel_tol, chi_max=chi_max)
        _U = U[:, :k]
        _S = S[:k]
        _Vh = Vh[:k, :]
        # update left tensor
        mpo[ii] = _U.reshape(Dl, d_in, d_out, k)
        # absorb into next tensor
        M = _S[:, None] * _Vh  # (k, Dr)
        W_next = mpo[ii + 1]
        Dr_in, d2_in, d2_out, Dr2 = W_next.shape
        W_next_mat = W_next.reshape(Dr_in, d2_in * d2_out * Dr2)
        W_next_new = M @ W_next_mat
        mpo[ii + 1] = W_next_new.reshape(k, d2_in, d2_out, Dr2)
    logger.info("MPO after LEFT compression:")
    for i, W in enumerate(mpo):
        logger.info(f"site {i}: shape {W.shape}")
    return mpo


def compress_mpo_right_sweep(
    mpo: list[np.ndarray], svd_rel_tol: float = 1e-6, chi_max: int | None = None
):
    L = len(mpo)
    mpo = [W.copy() for W in mpo]
    for ii in reversed(range(1, L)):
        W_i = mpo[ii]
        Dl, d_in, d_out, Dr = W_i.shape
        # reshape into (left_index, right_index)
        W_mat = W_i.reshape(Dl, d_in * d_out * Dr)
        # SVD on transpose for right canonicalization
        # We want V to become the right-orthonormal tensor.
        U, S, Vh = np.linalg.svd(W_mat, full_matrices=False)
        k = truncate_singular_values(S, rel_cut=svd_rel_tol, chi_max=chi_max)
        _U = U[:, :k]  # Dl × k
        _S = S[:k]
        _Vh = Vh[:k, :]  # k × (d_in*d_out*Dr)
        # update right tensor: shape (k, d_in, d_out, Dr)
        mpo[ii] = _Vh.reshape(k, d_in, d_out, Dr)
        # absorb U S into previous tensor
        M = _U * _S  # (Dl × k)
        W_prev = mpo[ii - 1]
        Dl_prev, d_prev_in, d_prev_out, Dr_prev = W_prev.shape
        if Dr_prev != Dl:
            raise ValueError(
                f"Incompatible bond dims at site {ii}: "
                f"left bond {Dl} does not match previous right bond {Dr_prev}"
            )
        W_prev_mat = W_prev.reshape(Dl_prev * d_prev_in * d_prev_out, Dr_prev)
        W_prev_new = W_prev_mat @ M  # shape (Dl_prev*d_prev* d_prev, k)
        mpo[ii - 1] = W_prev_new.reshape(Dl_prev, d_prev_in, d_prev_out, k)
    logger.info("MPO after RIGHT compression:")
    for i, W in enumerate(mpo):
        logger.info(f"site {i}: shape {W.shape}")
    return mpo


def compress_mpo(
    mpo: list[np.ndarray],
    svd_rel_tol: float = 1e-6,
    chi_max: int | None = None,
    n_sweeps: int = 1,
):
    mpo_comp = [W.copy() for W in mpo]
    for _ in range(n_sweeps):
        mpo_comp = compress_mpo_left_sweep(
            mpo_comp, svd_rel_tol=svd_rel_tol, chi_max=chi_max
        )
        mpo_comp = compress_mpo_right_sweep(
            mpo_comp, svd_rel_tol=svd_rel_tol, chi_max=chi_max
        )
    return mpo_comp


def operator_to_mpo_via_mps(
    operator: np.ndarray,
    projector: csc_matrix,
    loc_dims: list[int],
    op_svd_rel_tol: float = 1e-6,
    max_rank: int | None = None,
    mps_svd_rel_tol: float = 1e-6,
    mps_chi_max: int = 30,
    mpo_svd_rel_tol: float = 1e-6,
    mpo_chi_max: int = 400,
) -> list[np.ndarray]:
    """
    Build an MPO approximation of the quasi-particle operator:
        O_full ≈ \sigma_r |φ_r><ψ_r|
    starting from its representation `operator` in the symmetry sector.
    Args
    ----
    operator:
        Quasi-particle operator in the reduced (sector) basis on the support.
        Shape: (d_sec, d_sec).
    projector:
        P: (D_full, d_sec) projector from sector basis to full tensor-product basis
        on the support. Built from unique_subsys_configs and loc_dims.
    loc_dims:
        Local dimensions [d1, ..., dL] on the support in the correct order.
    op_svd_rel_tol:
        Relative Frobenius tolerance for truncating the SVD of `operator`.
        Controls error ||O - O_trunc||_F / ||O||_F.
    max_rank:
        Hard cap on number of kept singular components (rank of decomposition).
    mps_chi_max:
        Max MPS bond dimension for each lifted vector.
    mps_svd_rel_tol:
        Relative truncation tolerance inside `vector_to_mps`.
    mpo_chi_max:
        Max MPO bond dimension for each MPO of the sum.
    mpo_svd_rel_tol:
        Relative truncation tolerance inside compress_mpo.
    Returns
    -------
    mpo:
        List of L MPO tensors representing the operator on the support.
        mpo[0].shape[0] == 1
        mpo[-1].shape[3] == 1
        internal bonds moderate (controlled by R and mps_chi_max)
    """
    logger.info("----------------------------------------------------")
    logger.info("TRANSFORM QP OPERATOR TO MPO VIA MPS")
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("operator must be a square 2D array")
    d_sec = operator.shape[0]
    D_full = int(np.prod(loc_dims))
    # Sanity: projector should map sector -> full
    if projector.shape != (D_full, d_sec):
        raise ValueError(f"Pshape {projector.shape}, not ({D_full},{d_sec})")
    # ------------------------------------------------------------------
    # 1. SVD of operator in sector basis
    # ------------------------------------------------------------------
    logger.info("SVD of QP operator (in sector basis)")
    U, S, Vh = np.linalg.svd(operator, full_matrices=False)
    # Choose rank R to control relative Frobenius error
    eff_rank = choose_rank_by_frobenius(S, rel_tol=op_svd_rel_tol, max_rank=max_rank)
    if eff_rank == 0:
        # Completely negligible operator ⇒ raise error
        raise ValueError("Truncated QP operator to 0-rank; increase tol or max_rank.")
    # Truncate the SVD components up to eff_rank
    U = U[:, :eff_rank]
    S = S[:eff_rank]
    Vh = Vh[:eff_rank, :]
    # ------------------------------------------------------------------
    # 2. For each singular triplet, lift to full space and build rank-1 MPO
    # ------------------------------------------------------------------
    mpo_terms: list[list[np.ndarray]] = []
    for sv_idx in range(eff_rank):
        logger.info(f"----------------------------------------------------")
        logger.info(f"Singular value index {sv_idx} / {eff_rank - 1}")
        sigma_r = S[sv_idx]
        u_r = U[:, sv_idx]  # left singular vector in sector basis
        v_r = Vh[sv_idx, :].conj()  # right singular vector in sector basis
        # Symmetric splitting: \sigma_r → √\sigma_r on both sides
        phi_r_full = np.sqrt(sigma_r) * (projector @ u_r)  # (D_full,)
        psi_r_full = np.sqrt(sigma_r) * (projector @ v_r)  # (D_full,)
        # Convert lifted vectors to MPS on support with controlled bonds
        logger.info(f"----------------------------------------------------")
        phi_mps = vector_to_mps(
            state_vector=phi_r_full,
            local_dims=loc_dims,
            chi_max=mps_chi_max,
            svd_rel_tol=mps_svd_rel_tol,
        )
        logger.info(f"----------------------------------------------------")
        psi_mps = vector_to_mps(
            state_vector=psi_r_full,
            local_dims=loc_dims,
            chi_max=mps_chi_max,
            svd_rel_tol=mps_svd_rel_tol,
        )
        # Rank-1 MPO for |φ_r><ψ_r|
        mpo_r = outer_mps_to_mpo(phi_mps, psi_mps)  # \sigma_r|φ_r><ψ_r|
        # 4. Compress the resulting MPO once
        # mpo_r = compress_mpo(mpo_r, mpo_svd_rel_tol, chi_max=mpo_chi_max)
        mpo_terms.append(mpo_r)
    # ------------------------------------------------------------------
    # 3. Sum all rank-1 MPOs into a single MPO
    # ------------------------------------------------------------------
    mpo = sum_mpos(mpo_terms)
    mpo = compress_mpo(mpo, mpo_svd_rel_tol, chi_max=mpo_chi_max)
    # ------------------------------------------------------------------
    logger.info("Final MPO tensors:")
    for i, W in enumerate(mpo):
        logger.info(f"site {i}: shape {W.shape}")
    return mpo


def add_two_mpos(mpoA: list[np.ndarray], mpoB: list[np.ndarray]) -> list[np.ndarray]:
    """
    Return MPO representing (mpoA + mpoB).

    Both MPOs must:
      - have same length L,
      - share the same physical dimensions (d_in, d_out) at each site.

    Construction:
      - Site 0: concatenate on RIGHT bond only → keeps left bond = 1.
      - Site L-1: concatenate on LEFT bond only → keeps right bond = 1.
      - Middle sites: block-diagonal in both bonds.
    """
    L = len(mpoA)
    assert L == len(mpoB), "MPOs must have same length"
    mpo_sum: list[np.ndarray] = []
    # ----- Site 0 -----
    A0 = mpoA[0]  # (1, d, d, chiR_A)
    B0 = mpoB[0]  # (1, d, d, chiR_B)
    assert A0.shape[0] == 1 and B0.shape[0] == 1
    assert A0.shape[1:3] == B0.shape[1:3]
    d = A0.shape[1]
    # Concatenate along right bond: (1, d, d, chiR_A + chiR_B)
    W0 = np.concatenate([A0, B0], axis=3)
    mpo_sum.append(W0)
    # ----- Middle sites -----
    for site in range(1, L - 1):
        A = mpoA[site]  # (chiL_A, d, d, chiR_A)
        B = mpoB[site]  # (chiL_B, d, d, chiR_B)
        assert A.shape[1:3] == B.shape[1:3]
        d = A.shape[1]
        chiL_A, chiR_A = A.shape[0], A.shape[3]
        chiL_B, chiR_B = B.shape[0], B.shape[3]
        # Block-diagonal in bond space
        W = np.zeros((chiL_A + chiL_B, d, d, chiR_A + chiR_B), dtype=np.complex128)
        # Top-left block: A
        W[:chiL_A, :, :, :chiR_A] = A
        # Bottom-right block: B
        W[chiL_A:, :, :, chiR_A:] = B
        mpo_sum.append(W)
    # ----- Last site -----
    AL = mpoA[-1]  # (chiL_A, d, d, 1)
    BL = mpoB[-1]  # (chiL_B, d, d, 1)
    assert AL.shape[3] == 1 and BL.shape[3] == 1
    assert AL.shape[1:3] == BL.shape[1:3]
    d = AL.shape[1]
    # Concatenate along LEFT bond: (chiL_A + chiL_B, d, d, 1)
    WL = np.concatenate([AL, BL], axis=0)
    mpo_sum.append(WL)
    logger.info("************************************:")
    for i, W in enumerate(mpo_sum):
        logger.info(f"site {i}: shape {W.shape}")
    # Boundaries are now canonical:
    assert mpo_sum[0].shape[0] == 1
    assert mpo_sum[-1].shape[3] == 1
    return mpo_sum


def sum_mpos(mpo_list: list[list[np.ndarray]]) -> list[np.ndarray]:
    """
    Sum multiple MPOs using boundary-preserving addition.

    This avoids the insane growth of left/right bonds at the edges.
    Internal bonds still grow (they must), but in a controlled way.

    Assumes:
      - mpo_list is non-empty.
      - All MPOs have same length and same physical dimensions.
    """
    logger.info(f"----------------------------------------------------")
    logger.info(f"SUM ALL MPOs")
    if not mpo_list:
        raise ValueError("sum_mpos called with empty mpo_list")
    # Start from the first MPO
    mpo_sum = [W.copy() for W in mpo_list[0]]
    # Add others one by one
    for ii, mpo in enumerate(mpo_list[1:]):
        logger.info(f"Adding MPO {ii + 1} / {len(mpo_list) - 1}")
        mpo_sum = add_two_mpos(mpo_sum, mpo)
    return mpo_sum


def outer_mps_to_mpo(
    phi_mps: list[np.ndarray],
    psi_mps: list[np.ndarray],
):
    """
    Build an MPO representation of |phi><psi| from their MPS.
    For each site i:
      A[i] : (aL, d, aR)  for |phi>
      B[i] : (bL, d, bR)  for |psi>

    We define:
      W[i]_{(aL,bL), \sigma_in, \sigma_out, (aR,bR)}
        = A[i]_{aL, \sigma_in, aR} * conj(B[i]_{bL, \sigma_out, bR})

    Returns
    -------
    mpo:
        List of length L.
        mpo[i] has shape (bond_left, d_in, d_out, bond_right).
        Boundaries:
          - mpo[0].shape[0] = 1
          - mpo[-1].shape[3] = 1
        provided both input MPS are in standard (1,...,1) boundary form.
    """
    logger.info(f"----------------------------------------------------")
    logger.info(f"OUTER MPS TO MPO")
    L = len(phi_mps)
    assert L == len(psi_mps), "phi_mps and psi_mps must have same length"
    mpo: list[np.ndarray] = []
    for site in range(L):
        A = phi_mps[site]  # (aL, d, aR)
        B = psi_mps[site]  # (bL, d, bR)
        aL, d_in, aR = A.shape
        bL, d_out, bR = B.shape
        assert d_in == d_out
        bond_left_dim = aL * bL
        bond_right_dim = aR * bR
        W = np.zeros((bond_left_dim, d_in, d_out, bond_right_dim), dtype=np.complex128)
        for i_aL in range(aL):
            for i_aR in range(aR):
                A_phys = A[i_aL, :, i_aR]  # (d_in,)
                for i_bL in range(bL):
                    for i_bR in range(bR):
                        left_index = i_aL * bL + i_bL
                        right_index = i_aR * bR + i_bR
                        B_phys = B[i_bL, :, i_bR]  # (d_out,)
                        W[left_index, :, :, right_index] = A_phys[
                            :, None
                        ] * np.conjugate(B_phys[None, :])
        mpo.append(W)
    return mpo


def choose_rank_by_frobenius(
    S: np.ndarray,
    rel_tol: float,
    max_rank: int | None = None,
) -> int:
    """
    Global rank selection for an operator SVD.
    Given singular values S of O = U diag(S) V^† (assumed sorted descending),
    choose a rank R that controls the *relative Frobenius error*:
        ||O - O_R||_F^2 / ||O||_F^2  <=  rel_tol^2,
    where O_R is the best rank-R approximation from the SVD.
    Semantics:
    ----------
    - rel_tol sets the *ideal* rank R_tol.
    - max_rank, if given, is a hard cap: R = min(R_tol, max_rank).
      If the cap is active, the actual error will generally be > rel_tol,
      and we log that explicitly.
    Returns
    -------
    R : int
        Number of singular values to keep. 0 only if total weight is 0.
    """
    S = np.asarray(S)
    if S.ndim != 1:
        raise ValueError("S must be a 1D array of singular values")
    S2 = S * S
    total = S2.sum()
    if np.isclose(total, 0.0):
        logger.info("total Frobenius norm is zero → R=0")
        return 0
    frob_norm_tot = np.sqrt(total)
    logger.info(f"norm ||QP operator||_F = {frob_norm_tot: .6e}")
    # Cumulative sum of kept weight
    cumsum = np.cumsum(S2)
    # Find minimal R_tol such that tail^2 / total <= rel_tol^2
    R_tol = S.size  # default: keep everything
    target_rel_tol = rel_tol
    target_tail = (rel_tol * rel_tol) * total
    for r in range(1, S.size + 1):
        discarded = total - cumsum[r - 1]  # sum_{i>r} S_i^2
        if discarded <= target_tail:
            R_tol = r
            break
    # Apply max_rank cap if provided
    if max_rank is not None:
        R = min(R_tol, max_rank)
    else:
        R = R_tol
    # Enforce min_rank if there is non-zero weight
    R = max(R, 1)
    R = min(R, S.size)  # safety
    # Report actual error for the chosen R
    kept_weight = cumsum[R - 1]
    discarded_weight = total - kept_weight
    eff_rel_tol_sq = discarded_weight / total
    # Guard against tiny negative due to rounding
    if eff_rel_tol_sq < 0.0 and eff_rel_tol_sq > -1e-14:
        eff_rel_tol_sq = 0.0
    elif eff_rel_tol_sq < 0.0:
        logger.warning(f"negative eff_rel_tol_sq={eff_rel_tol_sq:.3e} set to 0.")
        eff_rel_tol_sq = 0.0
    eff_rel_tol = np.sqrt(eff_rel_tol_sq)
    logger.info("----------------------------------------------------")
    logger.info(f"SVD rank selection from FROBENIUS")
    logger.info(f"total_weight={total}, target_rel_tol={target_rel_tol:.1e}")
    logger.info(f"R_tol={R_tol}, max_rank={max_rank}, actual_rel_err={eff_rel_tol:.1e}")
    return R


def operator_to_mpo_via_mps2(
    operator: np.ndarray,
    projector: csc_matrix,
    loc_dims: list[int],
    op_svd_rel_tol: float = 1e-6,
    max_rank: int | None = None,
    mps_svd_rel_tol: float = 1e-6,
    mps_chi_max: int = 30,
    mpo_svd_rel_tol: float = 1e-6,
    mpo_chi_max: int = 400,
    min_vec_norm: float = 1e-14,
) -> list[np.ndarray]:
    """
    Build an MPO approximation of the quasi-particle operator:
        O_full ≈ sum_r |φ_r><ψ_r|
    starting from its representation `operator` in the symmetry sector.

    Pipeline:
      1. SVD in sector basis:
            operator = U diag(S) V^†
         choose R via `choose_rank_by_frobenius(S, op_svd_rel_tol, max_rank)`.

      2. For each r ≤ R:
         - lift sector singular vectors with P:
               φ_raw = P u_r
               ψ_raw = P v_r
         - normalize:
               φ_hat = φ_raw / ||φ_raw||
               ψ_hat = ψ_raw / ||ψ_raw||
               weight = S_r * ||φ_raw|| * ||ψ_raw||
         - convert φ_hat, ψ_hat to MPS on the support with `vector_to_mps`,
           using `mps_svd_rel_tol` and `mps_chi_max`.
         - build rank-1 MPO_r for |φ_hat><ψ_hat|.
         - absorb `weight` into the first MPO tensor of MPO_r.

      3. Incrementally sum:
            MPO_acc <- MPO_0
            MPO_acc <- compress_mpo(MPO_acc)  # to control bond growth
            For each subsequent MPO_r:
               MPO_acc <- add_two_mpos(MPO_acc, MPO_r)
               MPO_acc <- compress_mpo(MPO_acc, mpo_svd_rel_tol, mpo_chi_max)

      4. Final MPO is MPO_acc.

    Args
    ----
    operator:
        Quasi-particle operator in the reduced (sector) basis on the support.
        Shape: (d_sec, d_sec).
    projector:
        P: (D_full, d_sec) projector from sector basis to full tensor-product basis
        on the support. Built from unique_subsys_configs and loc_dims.
    loc_dims:
        Local dimensions [d1, ..., dL] on the support in the correct order.
    op_svd_rel_tol:
        Relative Frobenius tolerance for truncating the SVD of `operator`.
        Controls error ||O - O_trunc||_F / ||O||_F.
    max_rank:
        Hard cap on number of kept singular components (rank of decomposition).
    mps_chi_max:
        Max MPS bond dimension for each lifted vector.
    mps_svd_rel_tol:
        Relative truncation tolerance inside `vector_to_mps`.
    mpo_chi_max:
        Max MPO bond dimension during partial-sum compression.
    mpo_svd_rel_tol:
        Relative truncation tolerance inside `compress_mpo`.
    min_vec_norm:
        If ||P u_r|| or ||P v_r|| < min_vec_norm, that singular component is
        skipped as numerically negligible in the full tensor-product space.

    Returns
    -------
    mpo:
        List of L MPO tensors representing the operator on the support.
        mpo[0].shape[0] == 1
        mpo[-1].shape[3] == 1
    """
    logger.info("----------------------------------------------------")
    logger.info("TRANSFORM QP OPERATOR TO MPO VIA MPS (normalized lifting)")
    # ------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("operator must be a square 2D array")
    d_sec = operator.shape[0]
    D_full = int(np.prod(loc_dims))
    if projector.shape != (D_full, d_sec):
        raise ValueError(
            f"Projector shape {projector.shape}, expected ({D_full},{d_sec})"
        )
    # ------------------------------------------------------
    # 1. SVD of operator in sector basis
    # ------------------------------------------------------
    logger.info("SVD of QP operator (in sector basis)")
    U, S, Vh = np.linalg.svd(operator, full_matrices=False)
    # Choose rank R to control relative Frobenius error
    eff_rank = choose_rank_by_frobenius(S, rel_tol=op_svd_rel_tol, max_rank=max_rank)
    if eff_rank == 0:
        raise ValueError("Truncated QP operator to rank 0; increase tol or max_rank.")
    U = U[:, :eff_rank]
    S = S[:eff_rank]
    Vh = Vh[:eff_rank, :]
    logger.info(f"Using eff_rank = {eff_rank} singular components.")
    # ------------------------------------------------------
    # 2. Incrementally build MPO by adding rank-1 MPOs
    # ------------------------------------------------------
    mpo_acc: list[np.ndarray] | None = None
    K = 15  # compress every 15 MPOs
    for sv_idx in range(eff_rank):
        logger.info("----------------------------------------------------")
        logger.info(f"Singular value index {sv_idx} / {eff_rank - 1}")
        sigma_r = S[sv_idx]
        u_r = U[:, sv_idx]  # sector left singular vector
        v_r = Vh[sv_idx, :].conj()  # sector right singular vector
        # 2a. Lift to full tensor-product space
        phi_raw = projector @ u_r  # shape (D_full,)
        psi_raw = projector @ v_r  # shape (D_full,)
        nphi = np.linalg.norm(phi_raw)
        npsi = np.linalg.norm(psi_raw)
        logger.info(
            f"  ||phi_raw|| = {nphi:.3e}, ||psi_raw|| = {npsi:.3e}, sigma_r = {sigma_r:.3e}"
        )
        # Skip numerically negligible components after lifting
        if nphi < min_vec_norm or npsi < min_vec_norm:
            logger.info("  Skipping component: lifted singular vectors too small.")
            continue
        # 2b. Normalize lifted vectors and keep a single scalar weight
        phi_hat = phi_raw / nphi
        psi_hat = psi_raw / npsi
        weight = sigma_r * nphi * npsi
        logger.info(f"  Effective weight (sigma_r * ||phi|| * ||psi||) = {weight:.3e}")
        # 2c. Convert normalized vectors to MPS
        logger.info("  Converting phi_hat to MPS")
        phi_mps = vector_to_mps(
            state_vector=phi_hat,
            local_dims=loc_dims,
            chi_max=mps_chi_max,
            svd_rel_tol=mps_svd_rel_tol,
        )
        logger.info("  Converting psi_hat to MPS")
        psi_mps = vector_to_mps(
            state_vector=psi_hat,
            local_dims=loc_dims,
            chi_max=mps_chi_max,
            svd_rel_tol=mps_svd_rel_tol,
        )
        # 2d. Rank-1 MPO for |phi_hat><psi_hat|
        mpo_r = outer_mps_to_mpo(phi_mps, psi_mps)
        # 2e. Absorb scalar weight into the first MPO tensor
        # This keeps the MPS/MPO numerically well-conditioned.
        mpo_r[0] = weight * mpo_r[0]
        # --------------------------------------------------
        # 2f. Add this MPO into the accumulator with compression
        # --------------------------------------------------
        if mpo_acc is None:
            # First term: just copy & optionally compress lightly
            mpo_acc = [W.copy() for W in mpo_r]
        else:
            # Add and compress partial sum to control bond growth
            logger.info("  Adding rank-1 MPO to accumulator")
            mpo_acc = add_two_mpos(mpo_acc, mpo_r)
            if (sv_idx + 1) % K == 0:
                logger.info(f"  Compressing accumulated MPO after {sv_idx + 1} terms")
                mpo_acc = compress_mpo(
                    mpo_acc,
                    svd_rel_tol=mpo_svd_rel_tol,
                    chi_max=mpo_chi_max,
                    n_sweeps=1,
                )
    mpo_acc = compress_mpo(
        mpo_acc,
        svd_rel_tol=mpo_svd_rel_tol,
        chi_max=mpo_chi_max,
        n_sweeps=1,
    )
    if mpo_acc is None:
        raise RuntimeError("No non-negligible MPO components were constructed.")
    logger.info("----------------------------------------------------")
    logger.info("Final MPO tensors:")
    for i, W in enumerate(mpo_acc):
        logger.info(f"site {i}: shape {W.shape}")
    return mpo_acc
