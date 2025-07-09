# %%
from simsio import gen_configs
import numpy as np

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
params = {"g": np.logspace(-1, 1, 30), "m": np.logspace(-3, 1, 30)}
gen_configs("template", params, f"SU2/full/phase_diagram")
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
params = {"g": np.logspace(-1, 2, 20)}
gen_configs("template", params, f"theta_term/qed_notheta")
# %%
beta = np.array([0.75, 0.5, 0.3, 0.1, 0.05, 0.01, 0.005])
g = 1 / beta
params = {"g": g, "theta": np.linspace(-0.5, 0.5, 20)}
gen_configs("template", params, f"theta_term/qed_theta2")
# %%
params = {"g": np.logspace(-2, 3, 40), "m": np.logspace(-3, 1, 40)}
gen_configs("template", params, f"string_breaking/phasediagram_nobg")
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
theta = np.linspace(-0.6, 0.6, 15)
theta_focused = np.sort(np.concatenate([theta_minus, theta_plus, theta]))
g = np.linspace(0.95, 2.5, 10)
params = {"g": g, "theta": theta_focused[20:]}
gen_configs("template", params, f"theta_term/qed_theta_eigvals")
# %%
# SU2 FULL THEORY PHASE DIAGRAM
params = {"g": np.logspace(-3, 3, 30), "m": np.logspace(-3, 3, 30)}
gen_configs("template", params, f"string_breaking/phasediagram_bg")
gen_configs("template", params, f"string_breaking/phasediagram_nobg")
# %%
