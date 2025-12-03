import numpy as np
from simsio import *
from numba import njit, prange
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "get_data_from_sim",
    "get_Wannier_support",
    "geodesic_distance",
    "energy_functional",
    "spread_functional",
    "localize_Wannier",
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
        q_j = |E_j| / \sigma_R |E_R|
    as an effective localization profile for the Wannier.
    For each ε, the returned support S_ε is the smallest contiguous interval
    such that \sigma_{j∈S_ε} q_j ≥ 1 - ε.
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
        \sigma^2(R_target) = (\sigma_R w[R] * d(R,R_target)^2) / (\sigma_R w[R]),
    with w[R] = |E[R]|^p (p = weight_power, default 1).
    """
    Nk = E_profile.size
    wsum = 0.0
    num = 0.0
    for R in range(Nk):
        distance = geodesic_distance(R, R_target, Nk, center_mode)
        wsum += E_profile[R]
        num += E_profile[R] * (distance * distance)
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
    gs_energy = get_sim(ugrid[0][0]).res["gs_energy"]
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
    logger.info("Theta phases optimization")
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
    return energy_best, best_sigma, best_theta
