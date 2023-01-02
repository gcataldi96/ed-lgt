# %%
from simsio import gen_configs
import numpy as np

# %%
# GENERATE SIMULATION PARAMETERS FOR PURE THEORY
params = {"g": np.logspace(-1, 1, 15)}
for L in [2, 3, 4]:
    for bc in ["OBC", "PBC"]:
        gen_configs("template", params, f"pure_{L}x2_{bc}")
# %%
# PURE THEORY TOPOLOGY
params = {"g": np.logspace(-1, 1, 15)}
gen_configs("template", params, f"pure_topology")
# %%
# PARAMETERS FOR FULL THEORY
params = {"g": np.logspace(-1, 1, 20), "m": np.logspace(-1, 0, 10)}
for N in [0, 2]:
    for bc in ["OBC"]:
        gen_configs("template", params, f"full_Delta{N}_{bc}")
# %%
# SYMMETRY SECTORS
params = {"DeltaN": np.array([0, -4, -2, 2, 4]), "g": np.logspace(-1, 1, 20)}
gen_configs("template", params, f"full_sym_sectors")
# %%
# FULL THEORY TOPOLOGY
params = {"g": np.logspace(-1, 1, 15), "m": np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1])}
gen_configs("template", params, f"full_topology")

# %%
