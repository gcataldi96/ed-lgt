# %%
import numpy as np
from simsio import *
from scipy.optimize import minimize
from numba import njit, prange
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


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


def get_Wannier_function(
    Mk1k2_matrix,
    k_indices,
    gs_energy,
    R_target=6,
    n_restarts=10,
    seed=0,
    maxiter=5000,
    center_mode=0,
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
    logger.info("=======================================")
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


# function to acquire the convolution matrix and k indices from the simulation
def get_simulation_data1(sim_name):
    # ================================================================
    # We load the convolution results
    config_filename = f"scattering/{sim_name}"
    match = SimsQuery(group_glob=config_filename)
    ugrid, _ = uids_grid(match.uids, ["g", "m"])
    # (Nk, Nk) complex128 M_{k1,k2} = <k1|H_0|k2>
    Mk1k2_matrix = get_sim(ugrid[0][0]).res["k1k2matrix"]
    # (Nk,) integer indices of k T^2 vs TC
    k_indices = get_sim(ugrid[0][0]).res["k_indices"]
    gs_energy = -4.580269235030599 - 1.251803175199139e-18j
    return Mk1k2_matrix, k_indices, gs_energy


def get_simulation_data2(sim_name):
    with np.load(sim_name, allow_pickle=False) as z:
        Mk1k2_matrix = z["matrix"]
        k_indices = z["kvals"]
        gs_energy = z["gs"]
    return Mk1k2_matrix, k_indices, gs_energy


# %%
Mk1k2_matrix, k_indices, gs_energy = get_simulation_data2("wannier_TC_MM.npz")
E_best, best_sigma, best_theta = get_Wannier_function(
    Mk1k2_matrix, k_indices, gs_energy, center_mode=1
)
Nk = E_best.shape[0]
# %%
Mk1k2_matrix, k_indices, gs_energy = get_simulation_data1("convolution1_N0")
E_best, best_sigma, best_theta = get_Wannier_function(
    Mk1k2_matrix, k_indices, gs_energy, center_mode=1
)
Nk = E_best.shape[0]
# %%
fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.plot(range(Nk), E_best, marker="o")
ax.set(xticks=np.arange(Nk))
ax.grid()
# ax.set(yscale="log")
# ax.set(ylim=[1e-4, max(E_best) + 1])

# %%
