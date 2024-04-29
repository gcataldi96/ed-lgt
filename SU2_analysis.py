# %%
from simsio import *
from matplotlib import pyplot as plt
from ed_lgt.tools import r_values

textwidth_pt = 510.0
textwidth_in = textwidth_pt / 72.27


@plt.FuncFormatter
def fake_log(x, pos):
    "The two args are the value and tick position"
    return r"$10^{%d}$" % (x)


# %%
# SPECTRUM
# ==========================================================================
config_filename = f"SU2"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
# N=10 8951 1790:7160
# N=8 1105 276:828
# N=6 139  35:104
N = 10
if N == 10:
    size = 3584
elif N == 8:
    size = 564
res = {
    # "entropy": np.zeros((vals["g"].shape[0], vals["m"].shape[0], size)),
    "energy": np.zeros((vals["g"].shape[0], vals["m"].shape[0], size)),
    "overlap_V": np.zeros((vals["g"].shape[0], vals["m"].shape[0], size)),
    # "overlap_PV": np.zeros((vals["g"].shape[0], vals["m"].shape[0], size)),
    # "r_values": np.zeros((vals["g"].shape[0], vals["m"].shape[0], size - 1)),
}

for kk, g in enumerate(vals["g"]):
    for ii, m in enumerate(vals["m"]):
        # res["entropy"][kk, ii, :] = get_sim(ugrid[kk][ii]).res["entropy"]
        res["energy"][kk, ii, :] = get_sim(ugrid[kk][ii]).res["energy"]
        res["overlap_V"][kk, ii, :] = get_sim(ugrid[kk][ii]).res["overlap_V"]
        # res["overlap_PV"][kk, ii, :] = get_sim(ugrid[kk][ii]).res["overlap_PV"]
        # res["r_values"][kk, ii, :], _ = r_values(res["energy"][kk, ii, :])
        print("===================================================")
        # print(g, m, np.mean(res["r_values"][kk, ii, 425:1275]))
# %%
# ==========================================================================
fig, ax = plt.subplots(
    1,
    1,
    sharex=True,
    sharey=False,
    figsize=(textwidth_in, 0.5 * textwidth_in),
    constrained_layout=True,
)

ax.plot(
    res["energy"][kk, ii, :],
    res["overlap_V"][kk, ii, :],
    "o",
    markersize=4,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.8,
)
"""
ax[1].plot(
    res["energy"][kk, ii, :],
    res["overlap_PV"][kk, ii, :],
    "o",
    markersize=4,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.8,
)

ax[2].plot(
    res["energy"][kk, ii, :],
    res["entropy"][kk, ii, :],
    "o",
    markersize=4,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.8,
)
"""
ax.set(ylabel="Ov Vacuum", yscale="log", ylim=[1e-9, 2])
# ax[1].set(ylabel="Ov Pol Vacuum", yscale="log", ylim=[1e-9, 2])
# ax[2].set(xlabel="Energy", ylabel="Ent. Entropy")
plt.savefig("momentum_m5g1_N10_PBC.pdf")
# ==========================================================================
# %%
# DYNAMICS
# ==========================================================================
config_filename = f"SU2_dynamics"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
res = {
    "entropy": get_sim(ugrid[0][0]).res["entropy"],
    "fidelity": get_sim(ugrid[0][0]).res["fidelity"],
}

fig, ax = plt.subplots(
    2,
    1,
    sharex=True,
    sharey=False,
    figsize=(textwidth_in, textwidth_in),
    constrained_layout=True,
)
start = 0
stop = 3
delta_n = 0.01
n_steps = int((stop - start) / delta_n)


ax[0].plot(np.arange(n_steps) * delta_n, res["fidelity"])
ax[0].grid()
ax[0].set(ylabel=r"Fidelity (vacuum)")

ax[1].plot(np.arange(n_steps) * delta_n, res["entropy"])
ax[1].set(xlabel="Time", ylabel="Entanglement entropy")
ax[1].grid()
plt.savefig("dynamics_spin12_m01_g01.pdf")
# ==========================================================================
# %%
import csv

with open("energy_overlap_PV.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Energy", "Overlap Pol Vacuum"])  # Writing header row
    for energy, overlap in zip(res["energy"][0, 0, :], res["overlap_PV"][0, 0, :]):
        writer.writerow([energy, overlap])  # Writing data rows

with open("energy_overlap_V.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Energy", "Overlap Vacuum"])  # Writing header row
    for energy, overlap in zip(res["energy"][0, 0, :], res["overlap_V"][0, 0, :]):
        writer.writerow([energy, overlap])  # Writing data rows
