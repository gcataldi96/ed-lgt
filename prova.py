# %%
from simsio import *
from math import prod
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, FuncFormatter
from ed_lgt.tools import save_dictionary, first_derivative
from scipy import stats


@plt.FuncFormatter
def fake_log(x, pos):
    "The two args are the value and tick position"
    return r"$10^{%d}$" % (x)


# Z2 HUBBARD WITH FINITE ELECTRIC FIELD
# List of local observables
local_obs = [f"n_{s}{d}" for d in "xy" for s in "mp"]
local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
local_obs += ["X_Cross", "S2_psi", "E"]
plaq_name = "plaq"

# %%
res = {}
# Acquire simulations of finite E field
config_filename = f"Z2FermiHubbard/transition_a"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["h"])
for obs in [
    "N_pair",
    "energy",
    "entropy",
    plaq_name,
    "X_Cross",
    "P_px",
    "S2_psi",
    "Efield",
]:
    res[obs] = np.zeros((len(vals["h"])), dtype=float)
for jj, h in enumerate(vals["h"]):
    res["P_px"][jj] = get_sim(ugrid[jj]).res["P_px"]
    res["energy"][jj] = get_sim(ugrid[jj]).res["energy"]
    for obs in ["N_pair", "entropy", plaq_name, "X_Cross", "S2_psi"]:
        res[obs][jj] = get_sim(ugrid[jj]).res[obs]
    res["Efield"][jj] = get_sim(ugrid[jj]).res["E_field"]

fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.grid()
axs.set(ylabel="E", xlabel="$h$", xscale="log")
axs.plot(
    vals["h"],
    res["entropy"],
    "o-",
    linewidth=1,
    markersize=3,
    markerfacecolor="black",
    label="ED",
)
# %%
res = {}
# Acquire simulations of finite E field
config_filename = f"Z2FermiHubbard/transition_b"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["h"])
for obs in ["N_pair", "energy", "entropy", plaq_name, "X_Cross", "S2_psi", "Efield"]:
    res[obs] = np.zeros((len(vals["h"]), 4), dtype=float)
for jj, h in enumerate(vals["h"]):
    res["energy"][jj, :] = get_sim(ugrid[jj]).res["energy"]
    for obs in ["N_pair", "entropy", plaq_name, "X_Cross", "S2_psi"]:
        res[obs][jj, :] = get_sim(ugrid[jj]).res[obs]
    res["Efield"][jj, :] = get_sim(ugrid[jj]).res["E_field"]

obs_name = "X_Cross"
fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.grid()
axs.set(ylabel=obs_name + " suscept", xlabel="$h$", xlim=[39, 61])
df = np.gradient(res[obs_name][:, 0], 0.046991532510082694)
axs.plot(
    vals["h"],
    res[obs_name][:, 0],
    "o-",
    linewidth=1,
    markersize=3,
    markerfacecolor="black",
)
# %%
