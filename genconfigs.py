# %%
from simsio import gen_configs
import numpy as np

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
# %%
# FULL THEORY PHASE DIAGRAM
params = {"g": np.logspace(-1, 1, 30), "m": np.logspace(-3, 1, 30)}
gen_configs("template", params, f"SU2/full/phase_diagram")
