# %%
from simsio import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from edlgt.tools import (
    save_dictionary,
    load_dictionary,
    fake_log,
    get_tline,
    custom_average,
    time_integral,
    moving_time_integral,
    set_size,
)

textwidth_pt = 510.0
textwidth_in = textwidth_pt / 72.27
columnwidth_pt = 246.0
columnwidth_in = columnwidth_pt / 72.27


# %%
# ENTROPY BG
# ==========================================================================
res = {}
config_filename = f"entropy_bg"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
res["time_line"] = np.delete(get_sim(ugrid[0]).res["time_steps"], 11)
res["entropy"] = np.zeros((len(vals["g"]), len(res["time_line"])), dtype=float)
for ii, g in enumerate(vals["g"]):
    res["entropy"][ii] = np.delete(get_sim(ugrid[ii]).res["entropy"], 11)

sm = cm.ScalarMappable(cmap="magma")
palette = sm.to_rgba(vals["g"])

fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.grid()
for ii, g in enumerate(vals["g"]):
    ax.plot(
        res["time_line"][1:],
        res["entropy"][ii, 1:],
        "o-",
        label=f"g={g}",
        c=palette[ii],
        markersize=1,
        markeredgecolor=palette[ii],
        markerfacecolor="black",
        markeredgewidth=0.5,
    )
ax.set(xlabel="t", ylabel="entropy", xscale="log")
cb = fig.colorbar(
    sm, ax=ax, aspect=80, location="top", orientation="horizontal", pad=0.02
)
cb.set_label(label=r"$g^{2}$", labelpad=-22, x=-0.02, y=0)
plt.savefig(f"entropy.pdf")
save_dictionary(res, f"entropy_bg.pkl")
# %%
# ENTROPY NO BG
# ==========================================================================
res = {}
config_filename = f"entropy_nobg"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
res["time_line"] = get_sim(ugrid[0]).res["time_steps"]
res["entropy"] = np.zeros((len(vals["g"]), len(res["time_line"])), dtype=float)
for ii, g in enumerate(vals["g"]):
    res["entropy"][ii] = get_sim(ugrid[ii]).res["entropy"]

sm = cm.ScalarMappable(cmap="magma")
palette = sm.to_rgba(vals["g"])

fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.grid()
for ii, g in enumerate(vals["g"]):
    ax.plot(
        res["time_line"][1:],
        res["entropy"][ii, 1:],
        "o-",
        label=f"g={g}",
        c=palette[ii],
        markersize=1,
        markeredgecolor=palette[ii],
        markerfacecolor="black",
        markeredgewidth=0.5,
    )
ax.set(xlabel="t", ylabel="entropy", xscale="log")
cb = fig.colorbar(
    sm, ax=ax, aspect=80, location="top", orientation="horizontal", pad=0.02
)
cb.set_label(label=r"$g^{2}$", labelpad=-22, x=-0.02, y=0)
plt.savefig(f"entropy_nobg.pdf")
save_dictionary(res, f"entropy_nobg.pkl")
# %%
# FRAGMENTATION
# ==========================================================================
res = {}
idx = 0
config_filename = f"fragmentation_spectrum"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
res["overlap"] = get_sim(ugrid[idx]).res["overlap"]
res["energy"] = get_sim(ugrid[idx]).res["energy"]
res["entropy"] = get_sim(ugrid[idx]).res["entropy"]
res["r_array"] = get_sim(ugrid[idx]).res["r_array"]
fig, ax = plt.subplots(2, 1, constrained_layout=True)
ax[0].plot(
    res["energy"],
    res["overlap"],
    "o",
    markersize=2,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.5,
)
ax[0].set(ylabel="Overlap", yscale="log", xlabel="Energy", ylim=[1e-17, 1])
ax[1].plot(
    res["energy"],
    res["entropy"],
    "o",
    markersize=1,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.5,
)
ax[1].set(ylabel="Ent Entropy", xlabel="Energy")
plt.savefig(f"spectrum_fragmentation.pdf")
save_dictionary(res, f"frag_spectrum.pkl")
# %%
# DYNAMICS FRAGMENTATION
res = {}
idx = 0
config_filename = f"fragmentation_dynamics"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
for obs in [
    "overlap",
    "delta",
    "time_steps",
    "entropy",
    "N_single",
    "N_pair",
    "DE_N_single",
    "DE_N_pair",
    "DE_N_tot",
    "ME_N_single",
    "ME_N_pair",
    "ME_N_tot",
]:
    res[obs] = get_sim(ugrid[idx]).res[obs]


fig, ax = plt.subplots(4, 1, constrained_layout=True, sharex=True)
ax[0].plot(
    res["time_steps"],
    res["overlap"],
    "o-",
    markersize=0.5,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.2,
    linewidth=0.5,
)
ax[0].set(ylabel="Overlap")
ax[1].plot(
    res["time_steps"],
    res["entropy"],
    "o-",
    markersize=0.5,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.2,
    linewidth=0.5,
)
ax[1].set(ylabel="Ent Entropy")
ax[2].plot(
    res["time_steps"],
    res["delta"],
    "o-",
    markersize=0.5,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.2,
    linewidth=0.5,
)
ax[2].axhline(y=res["DE_N_tot"], linestyle="-", color="red")
ax[2].axhline(y=res["ME_N_tot"], linestyle="-.", color="darkgreen")
ax[2].set(ylabel="Imbalance")
ax[3].plot(
    res["time_steps"],
    res["N_single"],
    "o-",
    markersize=0.5,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.2,
    linewidth=0.5,
)
ax[3].set(ylabel="N_single", xlabel="time t")
ax[3].axhline(y=res["DE_N_single"], linestyle="-", color="red", label="DE")
ax[3].axhline(y=res["ME_N_single"], linestyle="-", color="darkgreen", label="ME")
fig.legend(
    bbox_to_anchor=(0.91, 0.25),
    ncol=2,
    frameon=True,
    labelspacing=0.1,
    bbox_transform=fig.transFigure,
)
save_dictionary(res, f"frag_dynamics.pkl")
plt.savefig(f"dynamics_fragmentation.pdf")


config_filename = f"fragment_dyn_bg"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
for obs in [
    "delta",
    "time_steps",
    "DE_N_tot",
    "ME_N_tot",
]:
    res[f"bg_{obs}"] = get_sim(ugrid[idx]).res[obs]
fig, ax = plt.subplots(1, 1, constrained_layout=True, sharex=True)
ax.plot(
    res["bg_time_steps"],
    moving_time_integral(res["bg_time_steps"], res["bg_delta"], 300),
    "o-",
    markersize=0.5,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.2,
    linewidth=0.5,
)
ax.plot(
    res["time_steps"],
    moving_time_integral(res["time_steps"], res["delta"], 300),
    "o-",
    markersize=0.5,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.2,
    linewidth=0.5,
)
save_dictionary(res, f"frag_imbalance.pkl")
# %%
# ==========================================================================
# 2D SPECTRUM
# ==========================================================================
index_list = ["g5m1", "g1m5"]
idx = 1
name = index_list[idx]
res = {}
config_filename = f"scars/PVspectrum2D"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
res["overlap"] = get_sim(ugrid[idx]).res["overlap"]
res["energy"] = get_sim(ugrid[idx]).res["energy"]
res["entropy"] = get_sim(ugrid[idx]).res["entropy"]

fig, ax = plt.subplots(2, 1, constrained_layout=True)
ax[0].plot(
    res["energy"],
    res["overlap"],
    "o",
    markersize=2,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.5,
)
ax[0].set(ylabel="Overlap", yscale="log", xlabel="Energy")
ax[1].plot(
    res["energy"],
    res["entropy"],
    "o",
    markersize=1,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.5,
)
ax[1].set(ylabel="Ent Entropy", xlabel="Energy")

# Define energy interval length
energy_interval = 17
start_energy = res["energy"][0]
end_energy = start_energy + energy_interval
lista = []
# Loop through energy intervals
while start_energy < max(res["energy"]):
    # Find indices within the current energy interval
    interval_indices = [
        i for i, e in enumerate(res["energy"]) if start_energy <= e < end_energy
    ]

    # Proceed if there are any points in this interval
    if interval_indices:
        # Find the index of the maximum overlap within the current energy interval
        max_overlap_idx = max(interval_indices, key=lambda i: res["overlap"][i])

        # Mark the maximum overlap point in the current interval with a red cross
        ax[0].scatter(
            res["energy"][max_overlap_idx],
            res["overlap"][max_overlap_idx],
            color="red",
            marker="x",
            s=50,  # size of the cross
        )
        ax[1].scatter(
            res["energy"][max_overlap_idx],
            res["entropy"][max_overlap_idx],
            color="red",
            marker="x",
            s=50,  # size of the cross
        )
        print(res["energy"][max_overlap_idx], res["overlap"][max_overlap_idx])
        lista.append(res["energy"][max_overlap_idx])
    # Move to the next energy interval
    start_energy = end_energy
    end_energy = start_energy + energy_interval

ax[0].set(ylim=[1e-9, 2])
plt.savefig(f"PV_{name}_spectrum.pdf")
save_dictionary(res, f"PV_2D_spectrum_{name}.pkl")
# for obs in ["energy", "overlap"]:
#    np.savetxt(f"{obs}.txt", res[obs], fmt="%f")
# %%
# ==========================================================================
# 2D DYNAMICS
# ==========================================================================
res = {}
config_filename = f"scars/PVdynamics2D"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
res["overlap"] = get_sim(ugrid[idx]).res["overlap"]
res["entropy"] = get_sim(ugrid[idx]).res["entropy"]
tline = (
    np.arange(res["overlap"].shape[0]) * get_sim(ugrid[0]).par["dynamics"]["delta_n"]
)
res["tline"] = tline
fig, ax = plt.subplots(2, 1, constrained_layout=True)
ax[0].plot(tline, res["overlap"])
ax[0].set(ylabel="Overlap", xlabel="T")
ax[1].plot(tline, res["entropy"])
ax[1].set(ylabel="Ent Entropy", xlabel="T")
# for obs in ["entropy", "overlap"]:
#    np.savetxt(f"{obs}_dyn.txt", res[obs], fmt="%f")
# np.savetxt(f"time_dyn.txt", tline, fmt="%f")
plt.savefig(f"PV2_{name}_dynamics.pdf")
save_dictionary(res, f"PV_2D_dynamics_{name}.pkl")
# ==========================================================================
# %%
# ==========================================================================
# DYNAMICS 1D
# ==========================================================================
state_list = ["PV", "V"]
j_list = ["12", "1", "32"]
idx = 1
res = {}
tline = np.arange(0, 10, 0.0025)

for ii, jmax in enumerate(j_list):
    config_filename = f"scars/1D/{state_list[idx]}/j{jmax}"
    match = SimsQuery(group_glob=config_filename)
    ugrid, vals = uids_grid(match.uids, ["g"])
    for kk, gval in enumerate(vals["g"]):
        res[f"overlap_j{jmax}_g{gval}"] = get_sim(ugrid[kk]).res["overlap"]
        res[f"entropy_j{jmax}_g{gval}"] = get_sim(ugrid[kk]).res["entropy"]

for kk, gval in enumerate(vals["g"]):
    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    for ii, jmax in enumerate(j_list):
        ax[0].plot(tline, res[f"overlap_j{jmax}_g{gval}"], label=f"j={jmax}")
        ax[0].set(ylabel="Overlap", xlabel="T")
        ax[1].plot(tline, res[f"entropy_j{jmax}_g{gval}"], label=f"j={jmax}")
        ax[1].set(ylabel="Ent Entropy", xlabel="T")
    plt.legend()
# for obs in ["entropy", "overlap"]:
#    np.savetxt(f"{obs}_dyn.txt", res[obs], fmt="%f")
# np.savetxt(f"time_dyn.txt", tline, fmt="%f")
# plt.savefig(f"BV_dynamics1D_m5g1_j32.pdf")
save_dictionary(res, f"{state_list[idx]}_1D_dynamics.pkl")

# %%
markersizes = [1.5, 1.5]
colors = ["darkblue", "darkgreen"]
col_densities = [
    ["green", "orchid", "orange"],
    ["darkgreen", "darkorchid", "darkorange"],
]
linestyles = ["-o", "-o"]
linewidths = [1, 1]
start = 0
stop = 10
delta_n = 0.1
n_steps = int((stop - start) / delta_n)
time_steps = np.arange(n_steps) * delta_n

res = {}
config_filename = f"SU2/PBC"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
res["g"] = vals["g"]
name = get_sim(ugrid[0]).par["dynamics"]["state"]
for obs in [
    "entropy",
    f"overlap_{name}",
    "canonical_avg",
    "microcan_avg",
    "diagonal_avg",
]:
    res[obs] = get_sim(ugrid[0]).res[obs]

res["N2"] = (
    custom_average(get_sim(ugrid[0]).res["N_pair"], "even")
    + (1 - custom_average(get_sim(ugrid[0]).res["N_pair"], "odd")) / 2
)
res["N1"] = custom_average(get_sim(ugrid[0]).res["N_single"])
res["N0"] = 1 - res["N1"] - res["N2"]


res1 = {}
config_filename = f"SU2/PBCk"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
res1["g"] = vals["g"]
name = get_sim(ugrid[0]).par["dynamics"]["state"]
res1[f"overlap_{name}"] = get_sim(ugrid[0]).res[f"overlap_{name}"]
# %%

fig, ax = plt.subplots(
    3,
    1,
    sharex=True,
    sharey="row",
    figsize=(textwidth_in, textwidth_in),
    constrained_layout=True,
)

jj = 0
kk = 0
ax[0].plot(
    time_steps,
    res[f"overlap_{name}"],
    linestyles[jj],
    linewidth=linewidths[jj],
    markersize=markersizes[jj],
    color=colors[jj],
    markeredgecolor=colors[jj],
    markerfacecolor=colors[jj],
    markeredgewidth=1,
)
ax[0].plot(
    time_steps,
    res1[f"overlap_{name}"] + 0.2,
    linestyles[jj],
    linewidth=linewidths[jj],
    markersize=markersizes[jj],
    color=colors[jj + 1],
    markeredgecolor=colors[jj + 1],
    markerfacecolor=colors[jj + 1],
    markeredgewidth=1,
)


ax[1].plot(
    time_steps,
    res["N1"],
    linestyles[jj],
    linewidth=linewidths[jj],
    markersize=markersizes[jj],
    color=colors[jj],
    markeredgecolor=colors[jj],
    markerfacecolor=colors[jj],
    markeredgewidth=1,
)
ax[1].axhline(
    y=np.mean(res["N1"]),
    color="orange",
    linestyle="-",
    linewidth=3,
    label=r"time average",
)
ax[1].axhline(
    y=res["canonical_avg"], color="red", linestyle="-", linewidth=2, label=r"canonical"
)

ax[1].axhline(
    y=res["microcan_avg"],
    color="black",
    linestyle="--",
    linewidth=2,
    label=r"microcanonical",
)
ax[1].axhline(
    y=res["diagonal_avg"],
    color="mediumseagreen",
    linestyle=":",
    linewidth=2.5,
    label=r"diagonal",
)

ax[2].plot(
    time_steps,
    res["entropy"],
    linestyles[jj],
    linewidth=linewidths[jj],
    markersize=markersizes[jj],
    color=colors[jj],
    markeredgecolor=colors[jj],
    markerfacecolor=colors[jj],
    markeredgewidth=1,
)

ax[2].set(xlabel="Time")

ax[0].set(ylabel=r"overlap $|\langle\psi(t)|\psi(0)\rangle|^{2}$", ylim=[0, 1.1])
ax[1].set(ylabel=r"density $\rho_{1}$", ylim=[-0.05, 0.5])
ax[2].set(ylabel=r"entropy $S$", ylim=[0, 5])

ax[1].legend(loc="best", ncol=1, fontsize=10)
plt.savefig(f"dynamics_{name}.pdf")

# %%
# %%
markersizes = [1.5, 1.5]
colors = ["darkblue", "darkgreen"]
col_densities = [
    ["green", "orchid", "orange"],
    ["darkgreen", "darkorchid", "darkorange"],
]
linestyles = ["-o", "-o"]
linewidths = [1, 1]
start = 0
stop = 10
delta_n = 0.1
n_steps = int((stop - start) / delta_n)
time_steps = np.arange(n_steps) * delta_n

res = {}
config_filename = f"SU2/no_scars"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
res["g"] = vals["g"]
name = get_sim(ugrid[0]).par["dynamics"]["state"]
for obs in [
    "entropy",
    f"overlap_{name}",
    "canonical_avg",
    "microcan_avg",
    "diagonal_avg",
]:
    res[obs] = get_sim(ugrid[0]).res[obs]

res["N2"] = (
    custom_average(get_sim(ugrid[0]).res["N_pair"], "even")
    + (1 - custom_average(get_sim(ugrid[0]).res["N_pair"], "odd")) / 2
)
res["N1"] = custom_average(get_sim(ugrid[0]).res["N_single"])
res["N0"] = 1 - res["N1"] - res["N2"]


res1 = {}
config_filename = f"SU2/PBCk"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
res1["g"] = vals["g"]
name = get_sim(ugrid[0]).par["dynamics"]["state"]
res1[f"overlap_{name}"] = get_sim(ugrid[0]).res[f"overlap_{name}"]
# %%

fig, ax = plt.subplots(
    3,
    1,
    sharex=True,
    sharey="row",
    figsize=(textwidth_in, textwidth_in),
    constrained_layout=True,
)

jj = 0
kk = 0
ax[0].plot(
    time_steps,
    res[f"overlap_{name}"],
    linestyles[jj],
    linewidth=linewidths[jj],
    markersize=markersizes[jj],
    color=colors[jj],
    markeredgecolor=colors[jj],
    markerfacecolor=colors[jj],
    markeredgewidth=1,
)
ax[0].plot(
    time_steps,
    res1[f"overlap_{name}"] + 0.2,
    linestyles[jj],
    linewidth=linewidths[jj],
    markersize=markersizes[jj],
    color=colors[jj + 1],
    markeredgecolor=colors[jj + 1],
    markerfacecolor=colors[jj + 1],
    markeredgewidth=1,
)


ax[1].plot(
    time_steps,
    res["N1"],
    linestyles[jj],
    linewidth=linewidths[jj],
    markersize=markersizes[jj],
    color=colors[jj],
    markeredgecolor=colors[jj],
    markerfacecolor=colors[jj],
    markeredgewidth=1,
)
ax[1].axhline(
    y=np.mean(res["N1"]),
    color="orange",
    linestyle="-",
    linewidth=3,
    label=r"time average",
)
ax[1].axhline(
    y=res["canonical_avg"], color="red", linestyle="-", linewidth=2, label=r"canonical"
)

ax[1].axhline(
    y=res["microcan_avg"],
    color="black",
    linestyle="--",
    linewidth=2,
    label=r"microcanonical",
)
ax[1].axhline(
    y=res["diagonal_avg"],
    color="mediumseagreen",
    linestyle=":",
    linewidth=2.5,
    label=r"diagonal",
)

ax[2].plot(
    time_steps,
    res["entropy"],
    linestyles[jj],
    linewidth=linewidths[jj],
    markersize=markersizes[jj],
    color=colors[jj],
    markeredgecolor=colors[jj],
    markerfacecolor=colors[jj],
    markeredgewidth=1,
)

ax[2].set(xlabel="Time")

ax[0].set(ylabel=r"overlap $|\langle\psi(t)|\psi(0)\rangle|^{2}$", ylim=[0, 1.1])
ax[1].set(ylabel=r"density $\rho_{1}$", ylim=[-0.05, 0.5])
ax[2].set(ylabel=r"entropy $S$", ylim=[0, 5])

ax[1].legend(loc="best", ncol=1, fontsize=10)
plt.savefig(f"dynamics_{name}.pdf")

# %%
res = {}
for state in ["PV", "micro"]:
    res[state] = {}
    for ii in range(1, 3, 1):
        config_filename = f"SU2/{state}{ii}"
        match = SimsQuery(group_glob=config_filename)
        ugrid, vals = uids_grid(match.uids, ["g"])
        res[state][ii] = {}
        res[state][ii]["g"] = vals["g"][0]
        res[state][ii]["m"] = get_sim(ugrid[0]).par["m"]
        name = get_sim(ugrid[0]).par["dynamics"]["state"]
        res[state][ii]["name"] = name
        ref_state = get_sim(ugrid[0]).par["ensemble"]["microcanonical"]["state"]
        res[state][ii]["ref_state"] = ref_state
        res[state][ii]["N1"] = custom_average(get_sim(ugrid[0]).res["N_single"])
        for obs in ["entropy", f"overlap_{name}"]:
            res[state][ii][obs] = get_sim(ugrid[0]).res[obs]

        config_filename = f"SU2/{state}{ii}_k"
        match = SimsQuery(group_glob=config_filename)
        ugrid, vals = uids_grid(match.uids, ["g"])
        for obs in ["canonical_avg", "microcan_avg", "diagonal_avg"]:
            res[state][ii][obs] = get_sim(ugrid[0]).res[obs]

save_dictionary(res, "dynamics_scars.pkl")

markersizes = [0.5, 1.5]
colors = ["darkblue", "darkgreen"]
col_densities = [
    ["green", "orchid", "orange"],
    ["darkgreen", "darkorchid", "darkorange"],
]
linestyles = ["-", "-o"]
linewidths = [1, 1]
start = 0
stop = 10
delta_n = 0.05
n_steps = int((stop - start) / delta_n)
time_steps = np.arange(n_steps) * delta_n
jj = 0
kk = 0

fig, ax = plt.subplots(
    2,
    4,
    sharex=True,
    sharey="row",
    figsize=(textwidth_in, textwidth_in),
    constrained_layout=True,
)

for s, state in enumerate(["PV", "micro"]):
    for ind in range(1, 3, 1):
        number = 2 * s + ind - 1
        ax[0, number].plot(
            time_steps,
            res[state][ind][f"overlap_{state}"],
            linestyles[jj],
            linewidth=linewidths[jj],
            markersize=markersizes[jj],
            color=colors[jj],
            markeredgecolor=colors[jj],
            markerfacecolor=colors[jj],
            markeredgewidth=1,
        )

        ax[1, number].plot(
            time_steps,
            res[state][ind]["N1"],
            linestyles[jj],
            linewidth=linewidths[jj],
            markersize=markersizes[jj],
            color=colors[jj],
            markeredgecolor=colors[jj],
            markerfacecolor=colors[jj],
            markeredgewidth=1,
            label=r"$\rho_{1}(t)$",
        )
        ax[1, number].axhline(
            y=res[state][ind]["canonical_avg"],
            color="red",
            linestyle="-",
            linewidth=2,
            label=r"canonical",
        )
        ax[1, number].axhline(
            y=np.mean(res[state][ind]["N1"]),
            color="orange",
            linestyle="-",
            linewidth=3,
            label=r"time average",
        )

        ax[1, number].axhline(
            y=res[state][ind]["microcan_avg"],
            color="black",
            linestyle="--",
            linewidth=2,
            label=r"microcanonical",
        )
        ax[1, number].axhline(
            y=res[state][ind]["diagonal_avg"],
            color="mediumseagreen",
            linestyle=":",
            linewidth=2.5,
            label=r"diagonal",
        )
        ax[0, number].annotate(
            r"$\psi_{"
            + f"{state}"
            + "}\,(m="
            + f"{res[state][ind]['m']}"
            + ",g^{2}="
            + f"{res[state][ind]['g']})$",
            xy=(0.9, 0.9),
            xycoords="axes fraction",
            fontsize=8,
            horizontalalignment="right",
            verticalalignment="bottom",
            bbox=dict(facecolor="white", edgecolor="black"),
        )

    ax[0, 0].set(
        ylabel=r"overlap $|\langle\psi(t)|\psi(0)\rangle|^{2}$", ylim=[-0.1, 1.1]
    )
    ax[1, 0].set(ylabel=r"density $\rho_{1}$", xlabel="Time")
    ax[1, 3].legend(loc="best", ncol=1, fontsize=8)
plt.savefig(f"dynamics_scars_FIGS2.pdf")
# %%
res = {}
for state in ["PV", "V"]:
    res[state] = {}
    config_filename = f"SU2/no_scars_{state}"
    match = SimsQuery(group_glob=config_filename)
    ugrid, vals = uids_grid(match.uids, ["g"])

    res[state]["g"] = vals["g"][0]
    res[state]["m"] = get_sim(ugrid[0]).par["m"]
    name = get_sim(ugrid[0]).par["dynamics"]["state"]
    res[state]["name"] = name

    ref_state = get_sim(ugrid[0]).par["ensemble"]["microcanonical"]["state"]
    res[state]["ref_state"] = ref_state
    for obs in ["entropy", f"overlap_{name}"]:
        res[state][obs] = get_sim(ugrid[0]).res[obs]

    res[state]["N2"] = (
        custom_average(get_sim(ugrid[0]).res["N_pair"], "even")
        + (1 - custom_average(get_sim(ugrid[0]).res["N_pair"], "odd")) / 2
    )
    res[state]["N1"] = custom_average(get_sim(ugrid[0]).res["N_single"])
    res[state]["N0"] = 1 - res[state]["N1"] - res[state]["N2"]

    config_filename = f"SU2/no_scars_{state}k"
    match = SimsQuery(group_glob=config_filename)
    ugrid, vals = uids_grid(match.uids, ["g"])
    for obs in ["canonical_avg", "microcan_avg", "diagonal_avg"]:
        res[state][obs] = get_sim(ugrid[0]).res[obs]

save_dictionary(res, "dynamics_no_scars.pkl")
# %%
res = load_dictionary("dynamics_no_scars.pkl")
markersizes = [0.5, 1.5]
colors = ["darkblue", "darkgreen"]
col_densities = [
    ["green", "orchid", "orange"],
    ["darkgreen", "darkorchid", "darkorange"],
]
linestyles = ["-o", "-o"]
linewidths = [1, 1]
start = 0
stop = 10
delta_n = 0.05
n_steps = int((stop - start) / delta_n)
time_steps = np.arange(n_steps) * delta_n
jj = 0
kk = 0

fig, ax = plt.subplots(
    3,
    2,
    sharex=True,
    sharey="row",
    figsize=(textwidth_in, textwidth_in),
    constrained_layout=True,
)

for s, state in enumerate(["PV", "V"]):
    ax[0, s].plot(
        time_steps,
        res[state][f"overlap_{state}"],
        linestyles[jj],
        linewidth=linewidths[jj],
        markersize=markersizes[jj],
        color=colors[jj],
        markeredgecolor=colors[jj],
        markerfacecolor=colors[jj],
        markeredgewidth=1,
    )

    ax[1, s].plot(
        time_steps,
        res[state]["N1"],
        linestyles[jj],
        linewidth=linewidths[jj],
        markersize=markersizes[jj],
        color=colors[jj],
        markeredgecolor=colors[jj],
        markerfacecolor=colors[jj],
        markeredgewidth=1,
        label=r"$\rho_{1}(t)$",
    )
    ax[1, s].axhline(
        y=res[state]["canonical_avg"],
        color="red",
        linestyle="-",
        linewidth=2,
        label=r"canonical",
    )
    ax[1, s].axhline(
        y=np.mean(res[state]["N1"]),
        color="orange",
        linestyle="-",
        linewidth=3,
        label=r"time average",
    )

    ax[1, s].axhline(
        y=res[state]["microcan_avg"],
        color="black",
        linestyle="--",
        linewidth=2,
        label=r"microcanonical",
    )
    ax[1, s].axhline(
        y=res[state]["diagonal_avg"],
        color="mediumseagreen",
        linestyle=":",
        linewidth=2.5,
        label=r"diagonal",
    )
    ax[0, s].annotate(
        r"$\psi({0})=\psi_{" + f"{res[state]['ref_state']}" + "}$",
        xy=(0.9, 0.9),
        xycoords="axes fraction",
        fontsize=15,
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox=dict(facecolor="white", edgecolor="black"),
    )

    ax[2, s].plot(
        time_steps,
        res[state]["entropy"],
        linestyles[jj],
        linewidth=linewidths[jj],
        markersize=markersizes[jj],
        color=colors[jj],
        markeredgecolor=colors[jj],
        markerfacecolor=colors[jj],
        markeredgewidth=1,
    )

    ax[0, 0].set(
        ylabel=r"overlap $|\langle\psi(t)|\psi(0)\rangle|^{2}$", ylim=[-0.1, 1.1]
    )
    ax[1, 0].set(ylabel=r"density $\rho_{1}$")
    ax[2, 0].set(ylabel=r"entropy $S$", xlabel="Time")
    ax[1, 1].legend(loc="best", ncol=1, fontsize=8.8)
plt.savefig(f"dynamics_no_scars_FIGS4.pdf")
# %%
# SPECTRUM

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
        res["g"] = vals["g"]
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
save_dictionary(res, "dynamics_ED.pkl")
# ==========================================================================

# %%
