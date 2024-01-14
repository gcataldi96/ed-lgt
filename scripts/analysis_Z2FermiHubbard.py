import numpy as np
from simsio import *
from matplotlib import pyplot as plt
from ed_lgt.tools import load_dictionary, save_dictionary

config_filename = "Z2_FermiHubbard/V_potential"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["V"])
lvals = get_sim(ugrid[0]).par["lvals"]
res = {"V": vals["V"]}
# List of local observables
local_obs = [f"n_{s}{d}" for d in "xyz"[: len(lvals)] for s in "mp"]
local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
local_obs += ["X_Cross"]
# Acquire observables
for ii, V in enumerate(res["V"]):
    res["energy"].append(get_sim(ugrid[ii]).res["energy"])
    for obs in local_obs:
        res[obs].append(get_sim(ugrid[ii]).res[obs])
