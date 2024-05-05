# %%
from simsio import *
from matplotlib import pyplot as plt
from ed_lgt.tools import save_dictionary

textwidth_pt = 510.0
textwidth_in = textwidth_pt / 72.27


@plt.FuncFormatter
def fake_log(x, pos):
    "The two args are the value and tick position"
    return r"$10^{%d}$" % (x)


def custom_average(arr, staggered=None):
    # Determine indices to consider based on the consider_even parameter
    indices = np.arange(arr.shape[1])
    if staggered == "even":
        indices_to_consider = indices[indices % 2 == 0]  # Select even indices
    elif staggered == "odd":
        indices_to_consider = indices[indices % 2 != 0]  # Select odd indices
    else:
        indices_to_consider = indices
    # Calculate the mean across the selected indices
    mean_values = np.mean(arr[:, indices_to_consider], axis=1)
    return mean_values


# %%
# SPECTRUM
# ==========================================================================
config_filename = f"SU2/no_scars"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
# N=10 8951 1790:7160
# N=8 1105 276:828
# N=6 139  35:104
N = 10
if N == 10:
    size = 8951
elif N == 8:
    size = 564
res = {
    "entropy": np.zeros((len(vals["g"]), size)),
    "energy": np.zeros((len(vals["g"]), size)),
    "overlap_V": np.zeros((len(vals["g"]), size)),
    "overlap_PV": np.zeros((len(vals["g"]), size)),
    "g": vals["g"],
}

for kk, g in enumerate(vals["g"]):
    res["entropy"][kk, :] = get_sim(ugrid[kk]).res["entropy"]
    res["energy"][kk, :] = get_sim(ugrid[kk]).res["energy"]
    res["overlap_V"][kk, :] = get_sim(ugrid[kk]).res["overlap_V"]
    res["overlap_PV"][kk, :] = get_sim(ugrid[kk]).res["overlap_PV"]
save_dictionary(res, f"SU2_no_scars.pkl")
# ==========================================================================
fig, ax = plt.subplots(
    3,
    3,
    sharex="col",
    sharey="row",
    figsize=(textwidth_in, 1.0 * textwidth_in),
    constrained_layout=True,
)
for kk, g in enumerate(res["g"]):
    ax[0, kk].plot(
        res["energy"][kk, :],
        res["overlap_V"][kk, :],
        "o",
        markersize=2,
        markeredgecolor="darkblue",
        markerfacecolor="white",
        markeredgewidth=0.5,
    )

    ax[1, kk].plot(
        res["energy"][kk, :],
        res["overlap_PV"][kk, :],
        "o",
        markersize=2,
        markeredgecolor="darkblue",
        markerfacecolor="white",
        markeredgewidth=0.5,
    )

    ax[2, kk].plot(
        res["energy"][kk, :],
        res["entropy"][kk, :],
        "o",
        markersize=2,
        markeredgecolor="darkblue",
        markerfacecolor="white",
        markeredgewidth=0.5,
    )
    ax[2, kk].set(xlabel="Energy")
    ax[0, kk].annotate(rf"$m=g={g}$", xy=(0.5, 0.2))

ax[0, 0].set(ylabel="Ov Vacuum", yscale="log", ylim=[1e-9, 2])
ax[1, 0].set(ylabel="Ov Pol Vacuum", yscale="log", ylim=[1e-9, 2])
ax[2, 0].set(ylabel="Ent Entropy")
plt.savefig("phase_diagram.pdf")
# %%
# DYNAMICS
# ==========================================================================
start = 0
stop = 4
delta_n = 0.01
n_steps = int((stop - start) / delta_n)
time_steps = np.arange(n_steps) * delta_n
markersizes = [1, 1]
colors = ["darkblue", "darkred"]
col_densities = [
    ["green", "orchid", "orange"],
    ["darkgreen", "darkorchid", "darkorange"],
]
linestyles = ["-o", "--"]
linewidths = [1.5, 1.5]
# Acquire simulations
res = {}
for state_name in ["V", "PV"]:
    res[state_name] = {}
    for jirrep in ["12", "1"]:
        res[state_name][jirrep] = {}
        config_filename = f"SU2/dynamics/{state_name}/j{jirrep}"
        match = SimsQuery(group_glob=config_filename)
        ugrid, vals = uids_grid(match.uids, ["g"])
        for ii, g in enumerate(vals["g"]):
            res[state_name][jirrep][f"{ii}"] = {}
            for obs in ["entropy", f"overlap_{state_name}"]:
                res[state_name][jirrep][f"{ii}"][obs] = get_sim(ugrid[ii]).res[obs]

            res[state_name][jirrep][f"{ii}"]["N2"] = (
                custom_average(get_sim(ugrid[ii]).res["N_pair"], "even")
                + (1 - custom_average(get_sim(ugrid[ii]).res["N_pair"], "odd")) / 2
            )
            res[state_name][jirrep][f"{ii}"]["N1"] = custom_average(
                get_sim(ugrid[ii]).res["N_single"]
            )
            res[state_name][jirrep][f"{ii}"]["N0"] = (
                1
                - res[state_name][jirrep][f"{ii}"]["N1"]
                - res[state_name][jirrep][f"{ii}"]["N2"]
            )
    # -------------------------------------------------------------------------
    # Make a figure for each reference state
    fig, ax = plt.subplots(
        3,
        2,
        sharex=True,
        sharey="row",
        figsize=(textwidth_in, textwidth_in),
        constrained_layout=True,
    )

    for ii, g in enumerate(vals["g"]):
        for jj, jirrep in enumerate(["12", "1"]):
            ax[0, ii].plot(
                time_steps,
                res[state_name][jirrep][f"{ii}"][f"overlap_{state_name}"],
                linestyles[jj],
                linewidth=linewidths[jj],
                markersize=markersizes[jj],
                color=colors[jj],
                markeredgecolor=colors[jj],
                markerfacecolor=colors[jj],
                markeredgewidth=1,
            )
            for kk, obs in enumerate(["N1", "N2"]):
                ax[1, ii].plot(
                    time_steps,
                    res[state_name][jirrep][f"{ii}"][obs],
                    linestyles[jj],
                    linewidth=linewidths[jj],
                    markersize=markersizes[jj],
                    color=col_densities[jj][kk],
                    markeredgecolor=col_densities[jj][kk],
                    markerfacecolor=col_densities[jj][kk],
                    markeredgewidth=1,
                    label=rf"{obs} (j{jirrep})",
                )
            ax[2, ii].plot(
                time_steps,
                res[state_name][jirrep][f"{ii}"]["entropy"],
                linestyles[jj],
                linewidth=linewidths[jj],
                markersize=markersizes[jj],
                color=colors[jj],
                markeredgecolor=colors[jj],
                markerfacecolor=colors[jj],
                markeredgewidth=1,
            )
            ax[2, ii].set(xlabel="Time")

    ax[0, 0].set(ylabel=r"overlap_V")
    ax[1, 0].set(ylabel=r"density")
    ax[2, 0].set(ylabel=r"entropy")
    ax[1, 1].legend(loc="best", ncol=2)
    plt.savefig(f"dynamics_{state_name}.pdf")
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
