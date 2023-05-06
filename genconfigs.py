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
# PURE THEORY TOPOLOGY
params = {"g": np.logspace(-1, 1, 15)}
gen_configs("template", params, f"pure_topology")
# %%
# FULL THEORY TOPOLOGY pt1
params = {"g": np.logspace(-1, 1, 10), "m": np.array([1e-2, 1e-1, 1, 1e1, 1e2])}
gen_configs("template", params, f"SU2/full/topology1")
# %%
# FULL THEORY TOPOLOGY pt2
params = {"g": np.logspace(-1, 1, 5), "m": np.logspace(-2, 4, 7)}
gen_configs("template", params, f"SU2/full/topology2")
