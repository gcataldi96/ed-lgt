from simsio import gen_configs
import numpy as np

params = {
    "m": np.logspace(-4, 0, 10),
    "g": np.logspace(-1, 1, 20),
}
gen_configs("template", params, "deltaN2_grid")
