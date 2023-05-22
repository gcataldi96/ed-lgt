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
# %%
# FULL THEORY SUPERCONDUCTING ORDER PARAMETER
params = {"g": np.logspace(-1, 1, 10)}
gen_configs("template", params, f"SU2/full/SCOP")

# %%
# QED U COMPARISON
params = {"spin": np.arange(1, 5, 1), "U": ["ladder", "spin"]}
gen_configs("template", params, f"QED/U_comparison")

# %%
# QED DM MATRIX eigenvalue scaling
params = {"g": np.logspace(-2, -1, 10)}
gen_configs("template", params, f"QED/DM_scaling_PBC")

# %%
# QED ENTANGLEMENT vs spin REP
params = {"spin": np.arange(1, 6, 1), "g": np.logspace(-2, 0, 15)}
gen_configs("template", params, f"QED/entanglement")

# %%
