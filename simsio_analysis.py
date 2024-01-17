# %%
from simsio import *
from math import prod
from matplotlib import pyplot as plt

# %%
config_filename = "Z2_FermiHubbard/U_potential"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["has_obc", "U"])

res = {}
# List of local observables
lvals = get_sim(ugrid[0][0]).par["lvals"]
local_obs = [f"n_{s}{d}" for d in "xyz"[: len(lvals)] for s in "mp"]
local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
local_obs += ["X_Cross"]
for obs in local_obs:
    res[obs] = np.zeros((vals["has_obc"].shape[0], vals["U"].shape[0]))

res["energy"] = np.zeros((vals["has_obc"].shape[0], vals["U"].shape[0]))

for ii, has_obc in enumerate(vals["has_obc"]):
    for jj, U in enumerate(vals["U"]):
        res["energy"][ii, jj] = get_sim(ugrid[ii][jj]).res["energies"][0] / (
            prod(lvals)
        )
        for obs in local_obs:
            res[obs][ii, jj] = np.mean(get_sim(ugrid[ii][jj]).res[obs])

fig = plt.figure()
plt.ylabel(r"X_cross")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["X_Cross"][ii, :], "-o", label=BC_label)
plt.legend()

fig = plt.figure()
plt.ylabel(r"N")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["N_pair"][ii, :], "-o", label=f"pair ({BC_label})")
    plt.plot(vals["U"], res["N_single"][ii, :], "-o", label=f"single ({BC_label})")
plt.legend()

fig = plt.figure()
plt.ylabel(r"Energy Density")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["energy"][ii, :], "-o", label=BC_label)
plt.legend()

# %%
config_filename = "Ising/Ising1D"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["lvals", "h"])
res = {
    "th_gap": np.zeros((vals["lvals"].shape[0], vals["h"].shape[0])),
    "true_gap": np.zeros((vals["lvals"].shape[0], vals["h"].shape[0])),
}
for ii, lvals in enumerate(vals["lvals"]):
    for jj, h in enumerate(vals["h"]):
        for obs in res.keys():
            res[obs][ii, jj] = get_sim(ugrid[ii][jj]).res[obs]

res["abs_distance"] = np.zeros((vals["lvals"].shape[0], vals["h"].shape[0]))
res["rel_distance"] = np.zeros((vals["lvals"].shape[0], vals["h"].shape[0]))
for ii, lvals in enumerate(vals["lvals"]):
    for jj, h in enumerate(vals["h"]):
        res["abs_distance"][ii, jj] = np.abs(
            res["true_gap"][ii, jj] - res["th_gap"][ii, jj]
        )
        res["rel_distance"][ii, jj] = (
            res["abs_distance"][ii, jj] / res["true_gap"][ii, jj]
        )
for obs in ["Sz", "Sx"]:
    res[obs] = np.zeros((vals["lvals"].shape[0], vals["h"].shape[0]))
    for ii, lvals in enumerate(vals["lvals"]):
        for jj, h in enumerate(vals["h"]):
            res[obs][ii, jj] = np.mean(get_sim(ugrid[ii][jj]).res[obs])

fig = plt.figure()
plt.ylabel(r"abs distance")
plt.xlabel(r"h")
plt.xscale("log")
plt.grid()
for ii, lvals in enumerate(vals["lvals"]):
    plt.plot(vals["h"], res["abs_distance"][ii, :], "-o", label=f"L={lvals}")
plt.legend()

fig = plt.figure()
plt.ylabel(r"rel distance")
plt.xlabel(r"h")
plt.xscale("log")
plt.yscale("log")
plt.grid()
for ii, lvals in enumerate(vals["lvals"]):
    plt.plot(vals["h"][4:], res["rel_distance"][ii, 4:], "-o", label=f"L={lvals}")
plt.legend()

fig = plt.figure()
plt.ylabel(r"Sz")
plt.xlabel(r"h")
plt.xscale("log")
plt.grid()
for ii, lvals in enumerate(vals["lvals"]):
    plt.plot(vals["h"], res["Sz"][ii, :], "-o", label=f"L={lvals}")
plt.legend()
# %%

fig = plt.figure()
plt.ylabel(r"True Gap")
plt.xlabel(r"h")
plt.xscale("log")
plt.grid()
for ii, lvals in enumerate(vals["lvals"]):
    plt.plot(vals["h"][:], res["true_gap"][ii, :], "-o", label=f"L={lvals}")
plt.legend()
# %%
