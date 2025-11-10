import numpy as np
from simsio import *
from numba import njit, prange
from scipy.optimize import minimize
from numba import njit, prange
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "get_data_from_sim",
    "get_Wannier_support",
    "geodesic_distance",
    "energy_functional",
    "spread_functional",
    "localize_Wannier",
    "operator_to_mpo_via_mps",
]


def get_data_from_sim(sim_filename, obs_name, kindex):
    config_filename = f"scattering/{sim_filename}"
    match = SimsQuery(group_glob=config_filename)
    ugrid, _ = uids_grid(match.uids, ["momentum_k_vals"])
    return get_sim(ugrid[kindex]).res[obs_name]


def get_Wannier_support(
    E_profile: np.ndarray,
    epsilons=(1e-2, 1e-4, 1e-6),
    center: str = "auto",
):
    """
    Determine Wannier support from a peaked energy functional profile.

    Parameters
    ----------
    E_profile : np.ndarray, shape (Nk,)
        Positive (or complex) energy functional values E_R at each site R.
        Only the *shape* matters; it will be made dimensionless and normalized.
    epsilons : iterable of float, optional
        Tail thresholds δ = 1 - sum_{j in S} q_j.
        For each ε, we return the minimal contiguous support S_ε.
    center : {"site", "bond", "auto"}, optional
        - "site": center around the single-site maximum of q_j.
        - "bond": center around the bond (j,j+1) with maximal q_j+q_{j+1}.
        - "auto": choose "bond" if max(q_j+q_{j+1}) > max(q_j), else "site".

    Returns
    -------
    results : dict
        {
            "q_j": float ndarray,          # normalized weights
            "normalization": float,        # sum_j q_j (≈ 1)
            "supports": {ε: np.ndarray},   # site indices for each ε
            "missing_weight": {ε: float},  # δ(ε) = discarded weight
            "mode": str,                   # chosen centering mode
        }

    Notes
    -----
    We interpret
        q_j = |E_j| / Σ_R |E_R|
    as an effective localization profile for the Wannier.
    For each ε, the returned support S_ε is the smallest contiguous interval
    such that Σ_{j∈S_ε} q_j ≥ 1 - ε.
    """
    logger.info(f"----------------------------------------------------")
    logger.info("Get minimal Wannier support")
    E_profile = np.asarray(E_profile)
    Nk = E_profile.size

    # Ensure positivity; robust if numerical noise / tiny imaginary parts exist
    if np.iscomplexobj(E_profile):
        weights = np.abs(E_profile)
    else:
        weights = E_profile.copy()
        # If some tiny negative values appear due to noise, clamp them.
        weights[weights < 0.0] = 0.0

    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("E_profile has zero total weight; cannot define q_j.")

    q_j = weights / total
    norm_check = float(np.sum(q_j))

    # Decide centering mode
    max_site = np.max(q_j)
    max_pair = np.max(q_j[:-1] + q_j[1:]) if Nk > 1 else -np.inf

    if center == "auto":
        mode = "bond" if max_pair > max_site else "site"
    elif center in ("site", "bond"):
        mode = center
    else:
        raise ValueError("center must be 'site', 'bond', or 'auto'.")

    supports = {}
    missing = {}

    for eps in epsilons:
        eps = float(eps)

        if mode == "site":
            # Center on the maximum of q_j
            j0 = int(np.argmax(q_j))
            left = right = j0
            total_in = q_j[j0]

            while total_in < 1.0 - eps and (left > 0 or right < Nk - 1):
                left_w = q_j[left - 1] if left > 0 else -1.0
                right_w = q_j[right + 1] if right < Nk - 1 else -1.0

                if left_w >= right_w and left > 0:
                    left -= 1
                    total_in += q_j[left]
                elif right < Nk - 1:
                    right += 1
                    total_in += q_j[right]
                else:
                    break

        elif mode == "bond":
            if Nk < 2:
                # fallback: only one site available
                j0 = int(np.argmax(q_j))
                left = right = j0
                total_in = q_j[j0]
            else:
                # Center on the pair with maximal combined weight
                pair_weights = q_j[:-1] + q_j[1:]
                j0 = int(np.argmax(pair_weights))
                left, right = j0, j0 + 1
                total_in = q_j[left] + q_j[right]

                while total_in < 1.0 - eps and (left > 0 or right < Nk - 1):
                    expanded = False
                    if left > 0:
                        left -= 1
                        total_in += q_j[left]
                        expanded = True
                    if right < Nk - 1 and total_in < 1.0 - eps:
                        right += 1
                        total_in += q_j[right]
                        expanded = True
                    if not expanded:
                        break

        supports[eps] = np.arange(left, right + 1, dtype=int)
        missing[eps] = float(1.0 - total_in)

    return {
        "q_j": q_j,
        "normalization": norm_check,
        "supports": supports,
        "missing_weight": missing,
        "mode": mode,
    }


@njit(cache=True)
def geodesic_distance(R: int, R0: int, Nk: int, center_mode: int = 0) -> float:
    """
    Unsigned geodesic distance on a ring of length Nk from integer site R to a center:
      - site-centered:       c = R0
      - bond-centered:       c = R0 + 0.5   (midpoint between R0 and R0+1)
    Returns a float (can be n + 0.5 when bond-centered).

    Parameters
    ----------
    R : int          lattice site in [0, Nk-1]
    R0: int          reference site in [0, Nk-1]
    Nk: int          ring length
    center_mode: int 0 (site), 1 (bond)
    """
    c = R0 + 0.5 if center_mode == 1 else float(R0)
    d = abs(R - c)
    if d > Nk - d:
        d = Nk - d
    return d


@njit(cache=True, parallel=True)
def energy_functional(
    band_kernel: np.ndarray,  # M (Nk, Nk) complex
    k_physical: np.ndarray,  # (Nk,) physical k in radians
    theta_phases: np.ndarray,  # (Nk,) phases θ_k in radians, with θ[0]=0 for gauge
    offset: np.complex128,
) -> np.ndarray:
    """
    Explicit, readable computation of E_R(θ) for R=0..Nk-1 using the exact formula:
        E_R(θ) = (1/Nk) ∑_{k1,k2} v*(k1) M[k1,k2] v(k2) exp(i (k_phys[k1]-k_phys[k2]) R),
    where v(k) = exp(-i θ_k).

    Works for *any* k grid (uniform or not). Complexity O(Nk^3) (Nk^2 per R).
    """
    # Number of momentum sectors
    Nk = band_kernel.shape[0]
    # Define the vector with the fases v_k = exp(-i θ_k)
    v = np.empty(Nk, dtype=np.complex128)
    for kidx in range(Nk):
        v[kidx] = np.exp(-1j * theta_phases[kidx])
    # Pre-dressed kernel
    W = np.empty((Nk, Nk), dtype=np.complex128)
    for k1 in range(Nk):
        vi_conj = np.conjugate(v[k1])
        for k2 in range(Nk):
            W[k1, k2] = vi_conj * band_kernel[k1, k2] * v[k2]
    energy_vec = np.zeros(Nk, dtype=np.complex128)
    for R in prange(Nk):
        # per-R phases
        phase_plus = np.empty(Nk, dtype=np.complex128)
        phase_minus = np.empty(Nk, dtype=np.complex128)
        for k1 in range(Nk):
            exp_k1R = np.exp(1j * k_physical[k1] * R)
            phase_plus[k1] = exp_k1R
            phase_minus[k1] = np.conjugate(exp_k1R)

        for k1 in range(Nk):
            for k2 in range(Nk):
                energy_vec[R] += (W[k1, k2] * phase_plus[k1] * phase_minus[k2]) / Nk
        energy_vec[R] -= offset
    return np.abs(energy_vec)


@njit(cache=True)
def spread_functional(E_profile: np.ndarray, R_target: int, center_mode: int) -> float:
    """
    Anchored MLWF spread around the chosen site R_target:
        \sigma^2(R_target) = (Σ_R w[R] * d(R,R_target)^2) / (Σ_R w[R]),
    with w[R] = |E[R]|^p (p = weight_power, default 1).
    """
    Nk = E_profile.size
    wsum = 0.0
    num = 0.0
    for R in range(Nk):
        distance = geodesic_distance(R, R_target, Nk, center_mode)
        wsum += E_profile[R]
        num += E_profile[R] * (distance)
    return 0.0 if np.isclose(wsum, 0.0) else (num / wsum)


def localize_Wannier(
    sim_name,
    R_target=6,
    n_restarts=10,
    seed=0,
    maxiter=5000,
    center_mode=0,
    gs_energy=-4.580269235030599 - 1.251803175199139e-18j,
):
    # ================================================================
    def objective(x):
        """
        x: free variables = theta[1:], since we fix theta[0]=0 as gauge
        returns spread \sigma^2 around R_target for given phases
        """
        theta = np.empty(Nk, dtype=np.float64)
        # theta[0] = 0.0
        # theta[1:] = x
        theta = x
        # 1) energy profile E_R(θ)
        E = energy_functional(Mk1k2_matrix, k_physical, theta, offset)
        # 2) anchored spread around chosen site
        return spread_functional(E, R_target, center_mode)

    # ================================================================
    # We load the convolution results
    config_filename = f"scattering/{sim_name}"
    match = SimsQuery(group_glob=config_filename)
    ugrid, _ = uids_grid(match.uids, ["g", "m"])
    # (Nk, Nk) complex128 M_{k1,k2} = <k1|H_0|k2>
    Mk1k2_matrix = get_sim(ugrid[0][0]).res["k1k2matrix"]
    # (Nk,) integer indices of k T^2 vs TC
    k_indices = get_sim(ugrid[0][0]).res["k_indices"]
    # ------------------------------------------------
    Nk = len(k_indices)
    k_physical = 2 * np.pi * k_indices / Nk
    # Initialize the seed for random number generators
    rng = np.random.default_rng(seed)
    # Initialize the theta phases
    best_sigma = np.inf
    best_theta = None
    offset = np.complex128(gs_energy)
    # Optimize the theta phases
    logger.info("====================================================")
    logger.info("Localize Wannier function")
    logger.info("theta phases optimization")
    for indtry in range(n_restarts):
        logger.info(f"Restart {indtry+1}/{n_restarts}")
        # Random initialization of the theta
        x0 = rng.uniform(0.0, 2.0 * np.pi, size=Nk)  # random phases for k>=1
        res = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options=dict(maxiter=maxiter, xatol=1e-8, fatol=1e-10, disp=False),
        )
        if res.fun < best_sigma:
            best_sigma = float(res.fun)
            best_theta = np.empty(Nk, dtype=np.float64)
            # best_theta[0] = 0.0
            # best_theta[1:] = res.x
            best_theta = res.x
    # Get the optimal theta phases [0, 2π)
    best_theta = np.mod(best_theta, 2.0 * np.pi)
    # Obtain the energy profile
    energy_best = energy_functional(Mk1k2_matrix, k_physical, best_theta, offset)
    logger.info(f"Optimal \sigma^2 estimated: {best_sigma}")
    logger.info(f"Optimal theta phases (rad): {best_theta}")
    return energy_best, best_sigma, best_theta


def operator_to_mpo(
    operator: np.ndarray,
    local_dims: list[int],
    *,
    chi_max: int | None = None,
    svd_tol: float = 1e-12,
    normalize: bool = False,
):
    """
    Convert a square operator on a finite region S into an MPO living on that region.

    Parameters
    ----------
    operator : np.ndarray
        The full operator as a dense matrix of shape (D, D), where D = prod(local_dims).
    local_dims : list[int]
        The on-site Hilbert dimensions [d1, d2, ..., dn] for the sites in S,
        in the SAME site order you intend the MPO to have.
    chi_max : int or None, optional (default: None)
        Maximum allowed MPO bond dimension. If None, no explicit cap is applied.
    svd_tol : float, optional (default: 1e-12)
        Absolute SVD cutoff: singular values <= svd_tol are discarded.
    normalize : bool, optional (default: False)
        If True, normalizes the Frobenius norm of `operator` to 1 before
        factorization.

    Returns
    -------
    mpo : list[np.ndarray]
        A list of n tensors, one per site, each with shape
        (bond_left, d_in, d_out, bond_right).
        By construction:
          - mpo[0] has bond_left = 1
          - mpo[-1] has bond_right = 1
        The MPO is left-canonical (the left-orthonormal condition holds
        for all tensors except possibly the last).

    Notes
    -----
    • Conceptually, we reshape the operator into a 2n-leg tensor with legs:
        [d1_in, d1_out, d2_in, d2_out, ..., dn_in, dn_out]
      Then we sweep from left to right. At step i, we group everything we
      have on the left into a matrix (rows) and everything else into (cols),
      do an SVD, keep the leading components, and interpret U as the ith MPO tensor.

    • Truncation:
        - `svd_tol` removes tiny singular values (noise/roundoff).
        - `chi_max` enforces a hard cap on the bond dimension, which is crucial
          if you want a compact MPO.

    • Left-canonical form:
        - Each SVD produces U, S, Vh with U column-orthonormal. Interpreting
          U as the site tensor ensures left-orthonormality by construction.

    • Shapes are carefully annotated below to help you follow the algebra.
    """
    logger.info(f"----------------------------------------------------")
    logger.info(f"Project the operator to MPO with svdtol={svd_tol}")
    # ------------------------------------------------------
    # 0) Input checks & preparation
    # ------------------------------------------------------
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("operator_matrix must be a square 2D array.")
    if not isinstance(local_dims, list):
        logger.info(f"local_dims type: {type(local_dims)}")
        raise ValueError("Please pass a list/tuple like [d1, d2, ..., dn]")
    total_dim = operator.shape[0]
    n_sites = len(local_dims)
    for dim in local_dims:
        if dim <= 0:
            raise ValueError("All local dimensions must be positive integers.")
    prod_dims = np.prod(local_dims)
    if prod_dims != total_dim:
        raise ValueError(f"Prod {local_dims} = {prod_dims} != {total_dim}.")
    logger.info(f"loc dims {local_dims} {prod_dims}")
    # Optional normalization: useful if you want a unit-norm MPO of the map
    if normalize:
        frobenius_norm = np.linalg.norm(operator.ravel())
        if frobenius_norm > 0:
            operator = operator / frobenius_norm
    # ------------------------------------------------------
    # 1) Reshape to a 2n-leg tensor: [d1_in, d1_out, d2_in, d2_out, ..., dn_in, dn_out]
    # ------------------------------------------------------
    # This “tensorized” view makes it natural to peel off one site at a time.
    two_n_shape = [x for d in local_dims for x in (d, d)]
    T = operator.reshape(two_n_shape)
    # We'll iteratively factor T into left tensors and a remainder.
    mpo = []  # where we'll store W[1], W[2], ..., W[n]
    left_bond_dim = 1  # by convention, the first MPO tensor has left bond = 1
    # ------------------------------------------------------
    # 2) Left-to-right SVD sweep for sites 1 ... n-1
    #    At each step i, we:
    #      (a) reshape current tensor to a matrix:  (left_bond * d_i * d_i) x (rest)
    #      (b) SVD -> U S Vh
    #      (c) truncate using svd_tol and chi_max
    #      (d) reshape U -> site tensor W[i] with shape (left_bond, d_i, d_i, right_bond)
    #      (e) fold (S @ Vh) back into the remaining tensor for the next step
    # ------------------------------------------------------
    for i in range(n_sites - 1):
        d_in = local_dims[i]
        d_out = local_dims[i]
        # T currently has legs:
        # (left_bond, d_i_in, d_i_out, d_{i+1}_in, d_{i+1}_out, ..., d_n_in, d_n_out)
        # For i=0, "left_bond" is implicitly 1 (we add it by reshape below).
        # Step (a): merge left legs into rows; everything else into columns.
        T = T.reshape(left_bond_dim * d_in * d_out, -1)  # (rows) x (cols)
        # Step (b): SVD
        # U:  (left_bond*d_in*d_out) x r
        # S:  r
        # Vh: r x (rest)
        U, S, Vh = np.linalg.svd(T, full_matrices=False)
        # Step (c): truncation — first by tolerance, then by chi_max
        # Keep singular values strictly greater than svd_tol
        if svd_tol is not None and svd_tol > 0:
            keep = int(np.sum(S > svd_tol))
        else:
            keep = S.size
        if chi_max is not None:
            keep = min(keep, int(chi_max))
        if keep == 0:
            # Degenerate case: everything truncated. Preserve structure with a single zero.
            keep = 1
            U = U[:, :keep] * 0.0
            S = S[:keep] * 0.0
            Vh = Vh[:keep, :] * 0.0
        else:
            U = U[:, :keep]
            S = S[:keep]
            Vh = Vh[:keep, :]
        right_bond_dim = keep
        # Step (d): interpret U as the i-th MPO tensor (left-canonical)
        # U has shape (left_bond*d_in*d_out) x right_bond → reshape to (left_bond, d_in, d_out, right_bond)
        W_i = U.reshape(left_bond_dim, d_in, d_out, right_bond_dim)
        mpo.append(W_i)
        # Step (e): carry the remainder to the right:  (S @ Vh) is of shape (right_bond, rest)
        # We need to reshape it back to a tensor with the new left bond on the front,
        # followed by the remaining 2*(n-i-1) physical legs.
        remainder = np.diag(S) @ Vh
        # Remaining physical legs are d_{i+1}_in, d_{i+1}_out, ..., d_n_in, d_n_out
        remaining_shape = [x for d in local_dims[i + 1 :] for x in (d, d)]
        T = remainder.reshape(right_bond_dim, *remaining_shape)
        # Update bond
        left_bond_dim = right_bond_dim
    # ------------------------------------------------------
    # 3) Last site (i = n-1): whatever remains is the final tensor
    #    Shape it as (left_bond, d_n, d_n, right_bond=1)
    # ------------------------------------------------------
    d_last = local_dims[-1]
    W_last = T.reshape(left_bond_dim, d_last, d_last, 1)
    mpo.append(W_last)
    # Sanity checks on boundary bonds
    assert mpo[0].shape[0] == 1, "First MPO tensor must have left bond dimension 1."
    assert mpo[-1].shape[-1] == 1, "Last MPO tensor must have right bond dimension 1."

    return mpo


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
    logger.info("Vector to MPS")
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
        keep = truncate_singular_values_relative(
            S, rel_cut=svd_rel_tol, chi_max=chi_max
        )

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
    logger.info("SUM MPOs")
    if not mpo_list:
        raise ValueError("sum_mpos called with empty mpo_list")

    # Start from the first MPO
    mpo_sum = [W.copy() for W in mpo_list[0]]

    # Add others one by one
    for mpo in mpo_list[1:]:
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
      W[i]_{(aL,bL), σ_in, σ_out, (aR,bR)}
        = A[i]_{aL, σ_in, aR} * conj(B[i]_{bL, σ_out, bR})

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

    # Boundaries come from input MPS
    return mpo


def operator_to_mpo_via_mps(
    operator: np.ndarray,
    projector,  # sparse or dense (D_full x d_sec) projector P
    loc_dims: list[int],  # local dims on the support (e.g. [6,6,6,6,6,6])
    *,
    op_svd_rel_tol: float = 1e-4,
    max_rank: int | None = 10,
    mps_chi_max: int = 16,
    mps_svd_rel_tol: float = 1e-6,
) -> list[np.ndarray]:
    """
    Build an MPO approximation of the quasi-particle operator:

        O_full ≈ Σ_r |φ_r><ψ_r|

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

    Returns
    -------
    mpo:
        List of L MPO tensors representing the operator on the support.
    """
    logger.info("SVD of QP operator (sector basis)")
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("operator must be a square 2D array")

    d_sec = operator.shape[0]
    D_full = int(np.prod(loc_dims))

    # Sanity: projector should map sector -> full
    if projector.shape != (D_full, d_sec):
        raise ValueError(
            f"projector has shape {projector.shape}, expected ({D_full}, {d_sec})"
        )
    # ------------------------------------------------------------------
    # 1. SVD of operator in sector basis
    # ------------------------------------------------------------------
    U, S, Vh = np.linalg.svd(operator, full_matrices=False)
    # Choose rank R to control relative Frobenius error
    R = choose_rank_by_frobenius(S, rel_tol=op_svd_rel_tol, max_rank=max_rank)
    logger.info(
        f"QP operator SVD: keeping R={R} singular values out of {S.size} "
        f"(rel_tol={op_svd_rel_tol}, max_rank={max_rank})"
    )
    if R == 0:
        # Completely negligible operator ⇒ return zero MPO
        L = len(loc_dims)
        zero_mpo = [np.zeros((1, d, d, 1), dtype=np.complex128) for d in loc_dims]
        return zero_mpo
    U = U[:, :R]
    S = S[:R]
    Vh = Vh[:R, :]
    # ------------------------------------------------------------------
    # 2. For each singular triplet, lift to full space and build rank-1 MPO
    # ------------------------------------------------------------------
    mpo_terms: list[list[np.ndarray]] = []
    for r in range(R):
        sigma_r = S[r]
        u_r = U[:, r]  # left singular vector in sector basis
        v_r = Vh[r, :].conj()  # right singular vector in sector basis
        # Symmetric splitting of sqrt(S_r) into both sides:
        amp = np.sqrt(sigma_r)
        # Lift to full tensor-product basis on support:
        # φ_r = P (sqrt(S_r) u_r)
        # ψ_r = P (sqrt(S_r) v_r)
        phi_r_full = projector @ u_r  # shape (D_full,)
        psi_r_full = projector @ v_r  # shape (D_full,)
        # Convert lifted vectors to MPS on support with controlled bonds
        phi_mps = vector_to_mps(
            state_vector=phi_r_full,
            local_dims=loc_dims,
            chi_max=mps_chi_max,
            svd_rel_tol=mps_svd_rel_tol,
        )
        psi_mps = vector_to_mps(
            state_vector=psi_r_full,
            local_dims=loc_dims,
            chi_max=mps_chi_max,
            svd_rel_tol=mps_svd_rel_tol,
        )
        # Rank-1 MPO for |φ_r><ψ_r|
        mpo_r = outer_mps_to_mpo(phi_mps, psi_mps)
        mpo_terms.append(mpo_r)
    # ------------------------------------------------------------------
    # 3. Sum all rank-1 MPOs into a single MPO
    # ------------------------------------------------------------------
    mpo = sum_mpos(mpo_terms)
    logger.info("Final MPO tensors:")
    for i, W in enumerate(mpo):
        logger.info(f"  site {i}: shape {W.shape}")
    # Expect:
    #  - mpo[0].shape[0] == 1
    #  - mpo[-1].shape[3] == 1
    #  - internal bonds moderate (controlled by R and mps_chi_max)
    return mpo


def truncate_singular_values_relative(
    sing_vals: np.ndarray, rel_cut: float, chi_max: int | None = None
) -> int:
    """
    Choose how many singular values to keep based on relative Frobenius weight.
    Keep the smallest k such that:
        sum_{i>k} S[i]^2 / sum_i S[i]^2 <= rel_cut^2
    This directly controls the *relative* error in Frobenius norm.

    Args
    ----
    S : 1D array of singular values (non-negative, sorted descending).
    rel_cut : target relative tolerance (e.g. 1e-6).
    chi_max : optional hard cap on the number of singular values.

    Returns
    -------
    k : int
        Number of singular values to keep (>= 1 if any S != 0).
    """
    S2 = sing_vals * sing_vals
    total = S2.sum()
    if total == 0.0:
        return 1  # everything is zero: keep a dummy component

    cumsum = np.cumsum(S2)
    n_singvals = len(sing_vals)
    for r in range(1, n_singvals + 1):
        discarded = total - cumsum[r - 1]
        if discarded / total <= rel_cut * rel_cut:
            n_singvals = r
            break
    if chi_max is not None:
        n_singvals = min(n_singvals, chi_max)
    if n_singvals <= 0:
        n_singvals = 1
    return n_singvals


def choose_rank_by_frobenius(
    S: np.ndarray, rel_tol: float, max_rank: int | None = None
) -> int:
    """
    Global rank selection for an operator SVD.

    Given singular values S of O = U diag(S) V^†,
    choose the smallest R such that:
        sum_{r>R} S[r]^2 / sum_r S[r]^2 <= rel_tol^2

    This controls the *relative Frobenius error* of the truncated operator.

    Args
    ----
    S : 1D array of singular values (descending).
    rel_tol : target relative Frobenius error (e.g. 1e-4).
    max_rank : optional hard cap on R.

    Returns
    -------
    R : int
        Number of singular values to keep. At least 1 if there is any weight.
    """
    S2 = S * S
    total = S2.sum()
    if total == 0.0:
        return 0
    logger.info(f"total sum SVD {total}")
    cumsum = np.cumsum(S2)
    R = S.size
    for r in range(1, S.size + 1):
        discarded = total - cumsum[r - 1]
        if discarded / total <= rel_tol * rel_tol:
            R = r
            break

    if max_rank is not None:
        R = min(R, max_rank)
    return max(R, 0)
