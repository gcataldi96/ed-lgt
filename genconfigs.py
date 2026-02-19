# %%
from simsio import gen_configs
import numpy as np

# %%
# DFL
params = {"g": np.linspace(0, 15, 7), "m": np.linspace(0, 15, 7)}
gen_configs("template", params, f"DFL/HSF_dyn_P")
# %%
# SU2 PURE THEORY TOPOLOGY
params = {"g": np.logspace(-1, 1, 15)}
gen_configs("template", params, f"SU2/pure/topology")
# %%
# SU2 PURE THEORY FLUCTUATIONS
params = {"g": np.logspace(-1, 1, 15)}
gen_configs("template", params, f"SU2/pure/fluctuations")
# %%
# SU2 FULL THEORY TOPOLOGY pt1
params = {"g": np.logspace(-1, 1, 10), "m": np.array([1e-2, 1e-1, 1, 1e1, 1e2])}
gen_configs("template", params, f"SU2/full/topology1")
# %%
# SU2 FULL THEORY TOPOLOGY pt2
params = {"g": np.logspace(-1, 1, 5), "m": np.logspace(-2, 4, 7)}
gen_configs("template", params, f"SU2/full/topology2")
# %%
# SU2 FULL THEORY PHASE DIAGRAM
params = {"g": np.logspace(-2, 1, 25), "m": np.logspace(-2, 1, 25)}
gen_configs("template", params, f"LBO/su2_phase_diagram")
# %%
# SU2 FULL THEORY PHASE DIAGRAM
params = {"m": np.logspace(-2, 1, 25)}
gen_configs("template", params, f"LBO/su2_effectivebasis")
# %%
# SU2 FULL THEORY SUPERCONDUCTING ORDER PARAMETER
params = {"g": np.logspace(-1, 1, 10)}
gen_configs("template", params, f"SU2/full/SCOP")
# %%
# SU2 FULL THEORY ENERGY GAP
params = {
    "DeltaN": [0, 2],
    "m": np.logspace(-3, -1, 15),
    "k": np.logspace(-1, 1, 5),
}
gen_configs("template", params, f"SU2/full/energy_gaps")
# %%
# SU2 FULL THEORY CHARGE vs DENSITY
params = {"g": np.logspace(-1, 1, 15), "m": np.logspace(-1, 0, 10)}
gen_configs("template", params, f"SU2/full/charge_vs_density")

# %%
# QED U COMPARISON
params = {"spin": np.arange(1, 5, 1), "U": ["ladder", "spin"]}
gen_configs("template", params, f"QED/U_comparison")
# %%
# QED U CONVERGENCE
params = {
    "g": np.logspace(-2, -1, 4),
    "U": ["ladder", "spin"],
    "spin": np.arange(1, 11, 1),
}
gen_configs("template", params, f"QED/U_convergence")

# %%
# QED true convergence
beta = np.logspace(0, 1, 5)
params = {
    "g": 1 / beta,
    "U": ["ladder", "spin"],
    "spin": np.arange(1, 11, 1),
}
gen_configs("template", params, f"QED/convergence")

# %%
# QED DM MATRIX eigenvalue scaling
params = {"g": np.logspace(-2, -1, 10)}
gen_configs("template", params, f"QED/DM_scaling_PBC")
# %%
# QED ENTANGLEMENT vs spin REP
params = {"spin": np.arange(1, 6, 1), "g": np.logspace(-2, 0, 15)}
gen_configs("template", params, f"QED/entanglement")
# %%
params = {
    "spin": np.arange(1, 31, 1),
    "g": np.logspace(-2, 1, num=15),
}
gen_configs("template", params, f"QED/scaling_conv")
# %%
# Z2 FERMI HUBBARD MODEL
params = {"U": np.logspace(-1, 3, 33), "h": np.logspace(-3, 1, 33)}
gen_configs("template", params, f"Z2FermiHubbard/PBCxy/phase_diagram")
# %%
params = {"U": np.logspace(0, 3, 20), "h": np.arange(0, 0.21, 0.01)}
gen_configs("template", params, f"Z2FermiHubbard/energy_gap")
# %%
params = {"U": np.logspace(-1, 3, 100), "h": np.logspace(-3, 0, 10)}
gen_configs("template", params, f"Z2FermiHubbard/PBCxy/n_pair_scanU")
# %%
params = {"U": np.logspace(0, 3, 10), "h": np.logspace(1, 3, 100)}
gen_configs("template", params, f"Z2FermiHubbard/PBCxy/largeh")
# %%
params = {"U": np.logspace(-1, 3, 50), "h": np.logspace(-3, 3, 50)}
gen_configs("template", params, f"Z2FermiHubbard/OBC/4x2grid")
# %%
params = {"U": np.logspace(-1, 3, 50), "h": np.logspace(-3, 3, 50)}
gen_configs("template", params, f"Z2FermiHubbard/grid")
# %%
params = {"h": np.logspace(-3, -1, 50)}
gen_configs("template", params, f"Z2FermiHubbard/transition_c")
# %%
params = {"h": np.logspace(np.log10(0.05), np.log10(5), 16)}
gen_configs("template", params, f"Z2FermiHubbard/transition_a")
# %%
params = {"h": np.logspace(np.log10(40), np.log10(60), 8)}
gen_configs("template", params, f"Z2FermiHubbard/transition_b")
# %%
params = {"U": np.logspace(0.5, 1.5, 50)[26:]}
gen_configs("template", params, f"Z2FermiHubbard/finitesize")
# %%
params = {"J": np.logspace(-2, 0, 10)}
gen_configs("template", params, f"Z2FermiHubbard/PBCx/test")
# %%
params = {"m": np.logspace(-1, 1, 20), "g": np.logspace(-1, 1, 20)}
gen_configs("template", params, f"DFL/grid")
# %%
params = {"g": np.linspace(0, 15, 20)}
gen_configs("template", params, f"entropy_nobg")
# %%
params = {"g": np.linspace(0, 15, 30), "m": np.linspace(0, 15, 30)}
gen_configs("template", params, f"newDFL/phase_diagram_BG")
# %%
# ISING MODEL
params = {"lvals": [[6], [8], [10], [12]], "h": np.logspace(-2, 2, 20)}
gen_configs("template", params, f"Ising/Ising1D")
# %%
# SU2 scars
params = {"m": np.logspace(-1, 1, 5), "g": np.logspace(-1, 1, 5)}
gen_configs("template", params, f"scars/dynamics2Dgrid")
# %%
params = {"g": np.logspace(-1, 0.5, 30), "spin": np.arange(1, 10, 1)}
gen_configs("template", params, f"LBO/qed_scan")

# %%
params = {"g": np.logspace(-1, 0.5, 30)}
gen_configs("template", params, f"LBO/qed_error")
# %%
params = {"g": np.logspace(-1, 1, 10)}
gen_configs("template", params, f"LBO/qed_plaq_svd")
# %%
params = {"g": np.logspace(-1, 1, 10), "theta": np.linspace(0, 5, 20)}
gen_configs("template", params, f"su2_thetaterm/scan")
# %%
beta = np.array([0.75, 0.5, 0.3, 0.1, 0.05, 0.01, 0.005])
g = 1 / beta
params = {"g": g, "theta": np.linspace(-0.5, 0.5, 20)}
gen_configs("template", params, f"theta_term/qed_theta2")
# %%
params = {"g": np.logspace(-2, 2, 30), "m": np.logspace(-2, 2, 30)}
gen_configs("template", params, f"string_breaking/static/zd")
# %%
params = {
    "g": np.logspace(-1, 1, 7),
    "m": np.logspace(-1, 1, 7),
    "sector": [16, 18, 20],
    "momentum_k_vals": [0, 1, 2, 3, 4, 5, 6, 7, 8],
}
gen_configs("template", params, f"scattering/bands2")
# %%
params = {"g": np.logspace(-2, 2, 30), "m": np.logspace(-2, 2, 30)}
gen_configs("template", params, f"scattering/phasediagram_d1")
# %%
params = {"momentum_k_vals": np.arange(7, dtype=int)}
gen_configs("template", params, f"new_scattering/band_g1m3")
# %%
from scipy.stats import norm

# Parameters
n_points = 14  # Number of points per side
theta_center = 0.4  # Peak of the Gaussian
theta_width = 0.05  # Standard deviation
theta_range = (0.3, 0.6)  # Interval for each Gaussian

# Generate percentiles between 0 and 1 (excluding 0 and 1 to avoid infinities)
percentiles = np.linspace(0.001, 0.999, n_points)
# Inverse CDF (quantile function) for standard normal
z_vals = norm.ppf(percentiles)
# Rescale to desired width and shift to center
theta_plus = theta_center + theta_width * z_vals
# Keep only those in the interval [0.3, 0.5]
theta_plus = theta_plus[(theta_plus >= theta_range[0]) & (theta_plus <= theta_range[1])]
# Mirror to get negative side (centered at -0.4)
theta_minus = -theta_plus[::-1]
# Combine and sort
theta = np.linspace(-2, 2, 15)
theta_focused = np.sort(np.concatenate([theta_minus, theta_plus, theta]))
g = np.linspace(0.95, 2.5, 15)
params = {"g": g, "theta": theta_focused[20:]}
gen_configs("template", params, f"su2_thetaterm/scan2")

# %%
params = {"g": [2.328, 4, 10], "theta": np.linspace(0.4, 0.45, 20)}
gen_configs("template", params, f"qed_theta_term/kpipipi")
# %%
# SU2 FULL THEORY PHASE DIAGRAM
params = {"g": np.logspace(-3, 3, 30), "m": np.logspace(-3, 3, 30)}
gen_configs("template", params, f"string_breaking/phasediagram_bg")
gen_configs("template", params, f"string_breaking/phasediagram_nobg")


# %%
def build_theta_grid(
    theta_max: float = 3.0,
    theta_c: float = 0.52,
    coarse_step: float = 0.05,
    medium_window: tuple[float, float] = (0.35, 0.80),
    medium_step: float = 0.01,
    fine_halfwidth: float = 0.06,  # gives [0.46, 0.58] if theta_c=0.52
    fine_step: float = 0.002,
) -> np.ndarray:
    """
    Build a positive-theta grid with multi-resolution refinement near theta_c.

    Returns
    -------
    theta : np.ndarray, shape (N,)
        Sorted unique theta values in [0, theta_max], including theta=0.
    """
    # Coarse global grid
    theta_coarse = np.arange(0.0, theta_max + 0.5 * coarse_step, coarse_step)

    # Medium grid around the interesting region
    theta_medium = np.arange(
        medium_window[0],
        medium_window[1] + 0.5 * medium_step,
        medium_step,
    )

    # Fine grid around theta_c
    fine_min = max(0.0, theta_c - fine_halfwidth)
    fine_max = min(theta_max, theta_c + fine_halfwidth)
    theta_fine = np.arange(fine_min, fine_max + 0.5 * fine_step, fine_step)

    # Merge, unique (avoid float glitches by rounding), sort
    theta_all = np.concatenate([theta_coarse, theta_medium, theta_fine])
    theta_all = np.unique(np.round(theta_all, 12))
    theta_all.sort()
    return theta_all


theta_vals = build_theta_grid()
print(theta_vals[:10], "...", theta_vals[-10:])
print("N(theta) =", theta_vals.size)


def build_g_grid(g_min=0.8, g_max=10.0, n=10):
    """Geometrically spaced g grid (good default for terms ~g^2 and ~1/g^2)."""
    g_vals = np.geomspace(g_min, g_max, n)
    return np.round(g_vals, 6)


g_vals = build_g_grid()
params = {"g": g_vals, "theta": theta_vals}
gen_configs("template", params, f"su2_thetaterm/scan3")
# %%
