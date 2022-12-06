# %%
from simsio import gen_configs
import numpy as np

# %%
# GENERATE SIMULATION PARAMETERS FOR PURE THEORY
params = {"g": np.logspace(-1, 1, 20)}
for L in [2, 3, 4]:
    for bc in ["OBC", "PBC"]:
        gen_configs("template", params, f"pure_{L}x2_{bc}")

# %%
# GENERATE SIMULATION PARAMETERS FOR FULL THEORY
params = {"g": np.logspace(-1, 1, 20), "m": np.logspace(-2, 0, 10)}
for N in [0, 2]:
    for bc in ["OBC", "PBC"]:
        gen_configs("template", params, f"full_Delta{N}_{bc}")

# %%
