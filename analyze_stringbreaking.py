# %%
from simsio import *
import matplotlib.pyplot as plt
from ed_lgt.tools import save_dictionary, fake_log, set_size

textwidth_pt = 510.0
textwidth_in = textwidth_pt / 72.27
columnwidth_pt = 246.0
columnwidth_in = columnwidth_pt / 72.27
# %%
# ===================================================================
# STRING BREAKING PHASE DIAGRAM
# ===================================================================
res = {}
for density in ["fd", "zd"]:
    config_filename = f"string_breaking/static/{density}"
    match = SimsQuery(group_glob=config_filename)
    ugrid, vals = uids_grid(match.uids, ["g", "m"])
    gvals = vals["g"]
    mvals = vals["m"]
    glen = len(gvals)
    mlen = len(mvals)
    res[density] = {
        "energy": np.zeros((glen, mlen)),
        "entropy": np.zeros((glen, mlen)),
        "E2": np.zeros((glen, mlen)),
        "N_single": np.zeros((glen, mlen)),
        "N_pair": np.zeros((glen, mlen)),
        "N_tot": np.zeros((glen, mlen)),
    }
    res["diff"] = np.zeros((glen, mlen))
    for ii, g in enumerate(gvals):
        for kk, m in enumerate(mvals):
            res[density]["energy"][ii, kk] = get_sim(ugrid[ii][kk]).res["energy"][0]
            res[density]["entropy"][ii, kk] = get_sim(ugrid[ii][kk]).res["entropy"][0]
            res[density]["E2"][ii, kk] = get_sim(ugrid[ii][kk]).res["E2"][0]
            res[density]["N_single"][ii, kk] = get_sim(ugrid[ii][kk]).res["N_single"][0]
            res[density]["N_pair"][ii, kk] = get_sim(ugrid[ii][kk]).res["N_pair"][0]
            res[density]["N_tot"][ii, kk] = get_sim(ugrid[ii][kk]).res["N_tot"][0]
for ii, g in enumerate(gvals):
    for kk, m in enumerate(mvals):
        res["diff"][ii, kk] = res["fd"]["energy"][ii, kk] - res["zd"]["energy"][ii, kk]
# diff (float)
diff = res["fd"]["energy"] - res["zd"]["energy"]  # shape (glen, mlen)
res["diff1"] = diff
# sign map: +1 if positive, -1 if negative, (0 if |diff|<=atol)
atol = 1e-13
sign = np.sign(diff)
sign[np.abs(diff) <= atol] = 0  # optional
res["diff_sign"] = sign.astype(np.int8)
from matplotlib.colors import ListedColormap, BoundaryNorm

X = res["diff_sign"]  # {-1,0,+1}
cmap = ListedColormap(["#2166ac", "#f7f7f7", "#b2182b"])  # blue, white, red
norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.set(
    xlabel=r"m",
    ylabel=r"$g^{2}$",
    xticks=[-2, -1, 0, 1, 2],
    yticks=[-2, -1, 0, 1, 2],
)
ax.xaxis.set_major_formatter(fake_log)
ax.yaxis.set_major_formatter(fake_log)
img = ax.imshow(X, origin="lower", extent=[-2, 2, -2, 2], cmap=cmap, norm=norm)
cb = fig.colorbar(
    img,
    ax=ax,
    ticks=[-1, 0, +1],
    aspect=20,
    location="right",
    orientation="vertical",
    pad=0.01,
    label="$E_{N=6}-E_{N=4}$",
)
cb.set_ticklabels([r"fd lower", r"tie", r"zd lower"])
plt.savefig("desity_SU2_nobg.pdf")
# %%
obs = "N_single"
fig, ax = plt.subplots(1, 1, constrained_layout=True)

X = res["diff"]
# X = res["fd"][obs]
img = plt.imshow(
    X, cmap="seismic", origin="lower", extent=[-2, 2, -2, 2]
)  # , vmin=-1, vmax=+1

ax.set(
    xlabel=r"m",
    ylabel=r"$g^{2}$",
    xticks=[-2, -1, 0, 1, 2],
    yticks=[-2, -1, 0, 1, 2],
)
ax.xaxis.set_major_formatter(fake_log)
ax.yaxis.set_major_formatter(fake_log)
cb = fig.colorbar(
    img,
    ax=ax,
    aspect=20,
    location="right",
    orientation="vertical",
    pad=0.01,
    label="$E_{N=6}-E_{N=4}$",
)
plt.savefig("density_SU2_phasediagram_nobg.pdf")
# %%
# ===================================================================
# STRING BREAKING PHASE DIAGRAM
# ===================================================================
res = {}
config_filename = f"string_breaking/phasediagram_nobg"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])

res = {
    "entropy": np.zeros((len(vals["g"]), len(vals["m"]))),
    "E_square": np.zeros((len(vals["g"]), len(vals["m"]))),
    "N_single": np.zeros((len(vals["g"]), len(vals["m"]))),
    "N_pair": np.zeros((len(vals["g"]), len(vals["m"]))),
    "N_zero": np.zeros((len(vals["g"]), len(vals["m"]))),
    "N_tot": np.zeros((len(vals["g"]), len(vals["m"]))),
}

for ii, g in enumerate(vals["g"]):
    for kk, m in enumerate(vals["m"]):
        res["entropy"][ii, kk] = get_sim(ugrid[ii][kk]).res["entropy"]
        res["E_square"][ii, kk] = get_sim(ugrid[ii][kk]).res["E_square"]
        res["N_single"][ii, kk] = get_sim(ugrid[ii][kk]).res["N_single"]
        res["N_pair"][ii, kk] = get_sim(ugrid[ii][kk]).res["N_pair"]
        res["N_zero"][ii, kk] = get_sim(ugrid[ii][kk]).res["N_zero"]
        res["N_tot"][ii, kk] = get_sim(ugrid[ii][kk]).res["N_tot"]
obs = "N_single"
fig, ax = plt.subplots(1, 1, constrained_layout=True)

X = np.transpose(res[obs])
img = plt.imshow(X, cmap="magma", origin="lower", extent=[-3, 3, -3, 3])
ax.set(
    ylabel=r"m",
    xlabel=r"g^{2}",
    xticks=[-3, -2, -1, 0, +1, +2, +3],
    yticks=[-3, -2, -1, 0, +1, +2, +3],
)
ax.xaxis.set_major_formatter(fake_log)
ax.yaxis.set_major_formatter(fake_log)
cb = fig.colorbar(
    img,
    ax=ax,
    aspect=20,
    location="right",
    orientation="vertical",
    pad=0.01,
    label=obs,
)
# %%
"""
0ae870a25cab11f096f4fa163ee812fb:
  <<<: common
  m: 37.5
"""
res = {}
config_filename = f"string_breaking/5x2/min_string"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["m"])

res = {"time_steps": get_sim(ugrid[0]).res["time_steps"]}
nsteps = len(res["time_steps"])

obs_list = [
    "entropy",
    "E_square",
    "N_single",
    "N_pair",
    "N_zero",
    "N_tot",
    "overlap0",
    "overlap1",
    "overlap2",
    "overlap3",
    "overlap4",
]

for obs in obs_list:
    res[f"{obs}"] = np.zeros((len(vals["m"]), nsteps))
    res[f"tot_overlap"] = np.zeros((len(vals["m"]), nsteps))
    for kk, m in enumerate(vals["m"]):
        res[obs][kk] = get_sim(ugrid[kk]).res[obs]
for kk, m in enumerate(vals["m"]):
    for ii in range(5):
        res[f"tot_overlap"][kk] += res[f"overlap{ii}"][kk]
# save_dictionary(res, f"minimal_string.pkl")

obs_color = ["darkblue", "darkred", "orange", "darkgreen"]
obs_names = [r"$E^{2}$", r"$N_{\rm{mes}}$", r"$N_{\rm{bar}}$", r"$N_{\rm{tot}}$"]
obs_size = [1, 1.5, 1.4, 1]
overlap_names = [
    r"$\rm{minS}_{1}$",
    r"$\rm{minS}_{2}$",
    r"$\rm{minS}_{3}$",
    r"$\rm{minS}_{4}$",
    r"$\rm{minS}_{5}$",
]
fig, ax = plt.subplots(3, 3, constrained_layout=True, sharex=True, sharey="row")
ax[0, 0].set(ylabel=r"$\rm{Ov}_{i}=|\langle \psi_{0}|\rm{minS}_{i}\rangle|^{2}$")
ax[1, 0].set(ylabel=r"observables")
ax[2, 0].set(ylabel=r"entropy $S$")
ax[2, 0].set(xlabel=r"time $t$ ($m=9.375$)")
ax[2, 1].set(xlabel=r"time $t$ ($m=18.75$)")
ax[2, 2].set(xlabel=r"time $t$ ($m=50$)")

for kk, m in enumerate(vals["m"]):
    for ii in range(5):
        ax[0, kk].plot(
            res["time_steps"],
            res[f"overlap{ii}"][kk],
            "o-",
            markersize=1.5,
            markeredgewidth=0.2,
            label=overlap_names[ii],
            linewidth=0.8,
        )
    ax[0, kk].plot(
        res["time_steps"],
        res["tot_overlap"][kk],
        "o-",
        markersize=2,
        markeredgewidth=0.2,
        linewidth=0.8,
        label=r"$\sum\rm{Ov}_{i}$",
    )

    for jj, obs in enumerate(["E_square", "N_single", "N_pair", "N_tot"]):
        ax[1, kk].plot(
            res["time_steps"],
            res[f"{obs}"][kk],
            "o-",
            c=obs_color[jj],
            markersize=obs_size[jj],
            markeredgewidth=0.2,
            label=f"{obs_names[jj]}",
            linewidth=0.8,
        )

    ax[2, kk].plot(
        res["time_steps"],
        res["entropy"][kk],
        "o-",
        markersize=2,
        markeredgewidth=0.2,
        linewidth=0.8,
    )

ax[0, 1].legend(
    bbox_to_anchor=(0.2, 0.45),
    ncol=2,
    handlelength=1,
    handletextpad=0.1,
    borderpad=0.1,
    framealpha=1.0,
    frameon=True,
    labelspacing=0.01,
)
ax[1, 0].legend(
    bbox_to_anchor=(0.98, 0.98),
    ncol=2,
    handlelength=1,
    handletextpad=0.1,
    borderpad=0.1,
    framealpha=1.0,
    frameon=True,
    labelspacing=0.01,
)
# plt.savefig(f"zerodensity.pdf")
# %%
res = {}
config_filename = f"string_breaking/3x2"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["m"])

res = {"time_steps": get_sim(ugrid[0]).res["time_steps"]}
nsteps = len(res["time_steps"])

obs_list = [
    "entropy",
    "E2",
    "N_single",
    "N_pair",
    "N_zero",
    "N_tot",
    "ov_max0",
]

for obs in obs_list:
    res[f"{obs}"] = np.zeros((len(vals["m"]), nsteps))
    res[f"tot_ov_max"] = np.zeros((len(vals["m"]), nsteps))
    for kk, m in enumerate(vals["m"]):
        res[obs][kk] = get_sim(ugrid[kk]).res[obs]
for kk, m in enumerate(vals["m"]):
    for ii in range(1):
        res[f"tot_ov_max"][kk] += res[f"ov_max{ii}"][kk]

obs_color = ["darkblue", "darkred", "orange", "darkgreen"]
obs_names = [r"$E^{2}$", r"$N_{\rm{mes}}$", r"$N_{\rm{bar}}$", r"$N_{\rm{tot}}$"]
obs_size = [1, 1.5, 1.4, 1]
overlap_names = [
    r"$\rm{minS}_{1}$",
    r"$\rm{minS}_{2}$",
    r"$\rm{minS}_{3}$",
    r"$\rm{minS}_{4}$",
    r"$\rm{minS}_{5}$",
]
fig, ax = plt.subplots(3, 1, constrained_layout=True, sharex=True, sharey="row")
ax[0].set(ylabel=r"$\mathcal{F}=|\langle \psi_{0}|\psi_{0}\rangle|^{2}$")
ax[1].set(ylabel=r"observables")
ax[2].set(ylabel=r"entropy $S$")
ax[2].set(xlabel=r"time $t$ ($m=31.5$ $g=50$)")

kk = 0
t = np.asarray(res["time_steps"], dtype=float)
e2 = np.asarray(res["E2"][kk], dtype=float)

# be robust to NaNs/Infs
mask = np.isfinite(t) & np.isfinite(e2)
idx_min = np.nanargmin(e2[mask])
t_min = t[mask][idx_min]

# pick a color (use the same as E2 if you like)
vline_color = "k"

for a in ax:
    a.axvline(
        t_min, linestyle="--", linewidth=1.0, color=vline_color, alpha=0.8, zorder=10
    )


for ii in range(1):
    ax[0].plot(
        res["time_steps"],
        res[f"ov_max{ii}"][kk],
        "o-",
        markersize=1.5,
        markeredgewidth=0.2,
        label=overlap_names[ii],
        linewidth=0.8,
    )

for jj, obs in enumerate(["E2", "N_single", "N_pair", "N_tot"]):
    ax[1].plot(
        res["time_steps"],
        res[f"{obs}"][kk],
        "o-",
        c=obs_color[jj],
        markersize=obs_size[jj],
        markeredgewidth=0.2,
        label=f"{obs_names[jj]}",
        linewidth=0.8,
    )

ax[2].plot(
    res["time_steps"],
    res["entropy"][kk],
    "o-",
    markersize=2,
    markeredgewidth=0.2,
    linewidth=0.8,
)

ax[1].legend(
    bbox_to_anchor=(0.7, 0.68),
    ncol=2,
    handlelength=1,
    handletextpad=0.1,
    borderpad=0.1,
    framealpha=1.0,
    frameon=True,
    labelspacing=0.01,
)
# save_dictionary(res, f"spin1.pkl")
# plt.savefig(f"spin1.pdf")
# %%
res = {}
config_filename = f"string_breaking/5x2/max_string"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["m"])

res = {"time_steps": get_sim(ugrid[0]).res["time_steps"]}
nsteps = len(res["time_steps"])

obs_list = [
    "entropy",
    "E_square",
    "N_single",
    "N_pair",
    "N_zero",
    "N_tot",
    "overlap_snake",
]

for obs in obs_list:
    res[f"{obs}"] = np.zeros((len(vals["m"]), nsteps))
    for kk, m in enumerate(vals["m"]):
        res[obs][kk] = get_sim(ugrid[kk]).res[obs]
save_dictionary(res, f"snake.pkl")
obs_color = ["darkblue", "darkred", "orange", "darkgreen"]
obs_names = [r"$E^{2}$", r"$N_{\rm{quarks}}$", r"$N_{\rm{baryon}}$", r"$N_{\rm{tot}}$"]
obs_size = [1, 1.5, 1.4, 1]

fig, ax = plt.subplots(3, 2, constrained_layout=True, sharex=True, sharey="row")
ax[0, 0].set(ylabel=r"fidelity $F=|\langle \psi_{0}|\psi(t)\rangle|^{2}$")
ax[1, 0].set(ylabel=r"observables")
ax[2, 0].set(ylabel=r"entropy $S$")
ax[2, 0].set(xlabel=r"time $t$ ($m=9.375$)")
ax[2, 1].set(xlabel=r"time $t$ ($m=18.75$)")

for kk, m in enumerate(vals["m"]):
    ax[0, kk].plot(
        res["time_steps"],
        res[f"overlap_snake"][kk],
        "o-",
        markersize=1.5,
        markeredgewidth=0.2,
        linewidth=0.8,
    )

    for jj, obs in enumerate(["E_square", "N_single", "N_pair", "N_tot"]):
        ax[1, kk].plot(
            res["time_steps"],
            res[f"{obs}"][kk],
            "o-",
            c=obs_color[jj],
            markersize=obs_size[jj],
            markeredgewidth=0.2,
            label=f"{obs_names[jj]}",
            linewidth=0.8,
        )

    ax[2, kk].plot(
        res["time_steps"],
        res["entropy"][kk],
        "o-",
        markersize=2,
        markeredgewidth=0.2,
        linewidth=0.8,
    )
ax[1, 0].legend(
    bbox_to_anchor=(0.6, 0.368),
    ncol=1,
    frameon=True,
    labelspacing=0.1,
)
# %%
res = {}
config_filename = f"string_breaking/6x2/sb_finite_density"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["sector"])

res = {"time_steps": get_sim(ugrid[0]).res["time_steps"]}
nsteps = len(res["time_steps"])

obs_list = [
    "E2",
    "N_single",
    "N_pair",
    "N_zero",
    "N_tot",
    "ov_min0",
    "ov_min1",
    "ov_min2",
    "ov_min3",
    "ov_min4",
    "ov_min5",
]

for obs in obs_list:
    res[f"{obs}"] = np.zeros((len(vals["sector"]), nsteps))
    res[f"tot_ov_min"] = np.zeros((len(vals["sector"]), nsteps))
    for kk, m in enumerate(vals["sector"]):
        res[obs][kk] = get_sim(ugrid[kk]).res[obs]
for kk, sec in enumerate(vals["sector"]):
    for ii in range(6):
        res[f"tot_ov_min"][kk] += res[f"ov_min{ii}"][kk]

obs_color = ["darkblue", "darkred", "orange", "darkgreen"]
obs_names = [r"$E^{2}$", r"$N_{\rm{mes}}$", r"$N_{\rm{bar}}$"]
obs_size = [1, 1.5, 1.4, 1]
overlap_names = [
    r"$\rm{minS}_{1}$",
    r"$\rm{minS}_{2}$",
    r"$\rm{minS}_{3}$",
    r"$\rm{minS}_{4}$",
    r"$\rm{minS}_{5}$",
    r"$\rm{minS}_{6}$",
]
fig, ax = plt.subplots(
    2,
    5,
    figsize=set_size(textwidth_pt, subplots=(3, 5)),
    constrained_layout=True,
    sharex=True,
    sharey="row",
)
ax[0, 0].set(ylabel=r"$\rm{Ov}_{i}=|\langle \psi_{0}|\rm{minS}_{i}\rangle|^{2}$")
ax[1, 0].set(ylabel=r"observables")
ax[1, 0].set(xlabel=r"time $t$")
ax[1, 1].set(xlabel=r"time $t$")
ax[1, 2].set(xlabel=r"time $t$")
ax[1, 3].set(xlabel=r"time $t$")
ax[1, 4].set(xlabel=r"time $t$")

for kk, m in enumerate(vals["sector"]):
    """t = np.asarray(res["time_steps"][:100], dtype=float)
    e2 = np.asarray(res["E2"][kk, :100], dtype=float)
    # be robust to NaNs/Infs
    mask = np.isfinite(t) & np.isfinite(e2)
    idx_min = np.nanargmin(e2[mask])
    t_min = t[mask][idx_min]
    for ii in range(3):
        ax[ii, kk].set(xlim=[-0.1, 5])
        ax[ii, kk].axvline(
            t_min, linestyle="--", linewidth=1.0, color="k", alpha=0.8, zorder=10
        )"""
    for ii in range(6):
        ax[0, kk].plot(
            res["time_steps"],
            res[f"ov_min{ii}"][kk],
            "o-",
            markersize=1,
            markeredgewidth=0.2,
            label=overlap_names[ii],
            linewidth=0.8,
        )
    ax[0, kk].plot(
        res["time_steps"],
        res["tot_ov_min"][kk],
        "o-",
        markersize=2,
        markeredgewidth=0.2,
        linewidth=0.8,
        label=r"sum $\rm{Ov}_{i}$",
    )

    for jj, obs in enumerate(["E2", "N_single", "N_pair"]):
        ax[1, kk].plot(
            res["time_steps"],
            res[f"{obs}"][kk],
            "o-",
            c=obs_color[jj],
            markersize=obs_size[jj],
            markeredgewidth=0.2,
            label=f"{obs_names[jj]}",
            linewidth=0.8,
        )
for ii in range(4):
    ax[0, ii].text(
        0.5,
        0.88,  # 5% in from left, 95% up from bottom
        rf"$N_b={ii}$",  # e.g. "(a)", "(b)", …
        transform=ax[0, ii].transAxes,  # interpret coords relative to the axes
        ha="left",
        va="top",  # align text box
        fontsize=9,  # tweak if you like
        bbox=dict(facecolor="white", alpha=0.2, edgecolor="black"),
    )
fig, ax = plt.subplots(
    1,
    2,
    figsize=set_size(columnwidth_pt, subplots=(2, 2)),
    constrained_layout=True,
)
ax[0].set(ylabel=r"Casimir $C(t)$")
ax[0].set(xlabel=r"time $t$")
ax[1].set(ylabel=r"break-time $t_{\rm{break}}$", xlabel=r"baryon number sector $N_{b}$")
res["tmin"] = np.zeros(5)
for kk, m in enumerate(vals["sector"]):
    t = np.asarray(res["time_steps"][:100], dtype=float)
    e2 = np.asarray(res["E2"][kk, :100], dtype=float)
    # be robust to NaNs/Infs
    mask = np.isfinite(t) & np.isfinite(e2)
    idx_min = np.nanargmin(e2[mask])
    t_min = t[mask][idx_min]
    res["tmin"][kk] = t_min
    ax[0].scatter(
        t_min,
        e2[mask][idx_min],
        s=7,
        marker="o",
        color="k",
    )
    ax[0].plot(
        res["time_steps"],
        res[f"E2"][kk],
        "o-",
        markersize=1,
        markeredgewidth=0.2,
        label=rf"$N_b={kk}$",
        linewidth=0.8,
    )
ax[1].plot(
    np.arange(5),
    res["tmin"],
    "o-",
    markersize=2,
    markeredgewidth=0.2,
    linewidth=1,
)
save_dictionary(res, f"6x2.pkl")
# %%
res = {}
config_filename = f"string_breaking/5x2/sb_finite_density"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["sector"])

res = {"time_steps": get_sim(ugrid[0]).res["time_steps"]}
nsteps = len(res["time_steps"])

obs_list = [
    "entropy",
    "E2",
    "N_single",
    "N_pair",
    "N_zero",
    "N_tot",
    "ov_min0",
    "ov_min1",
    "ov_min2",
    "ov_min3",
    "ov_min4",
]

for obs in obs_list:
    res[f"{obs}"] = np.zeros((len(vals["sector"]), nsteps))
    res[f"tot_ov_min"] = np.zeros((len(vals["sector"]), nsteps))
    for kk, m in enumerate(vals["sector"]):
        res[obs][kk] = get_sim(ugrid[kk]).res[obs]
for kk, sec in enumerate(vals["sector"]):
    for ii in range(5):
        res[f"tot_ov_min"][kk] += res[f"ov_min{ii}"][kk]

obs_color = ["darkblue", "darkred", "orange", "darkgreen"]
obs_names = [r"$E^{2}$", r"$N_{\rm{quarks}}$", r"$N_{\rm{baryon}}$", r"$N_{\rm{tot}}$"]
obs_size = [1, 1.5, 1.4, 1]
overlap_names = [
    r"$\rm{minS}_{1}$",
    r"$\rm{minS}_{2}$",
    r"$\rm{minS}_{3}$",
    r"$\rm{minS}_{4}$",
    r"$\rm{minS}_{5}$",
]
fig, ax = plt.subplots(
    3,
    4,
    figsize=set_size(textwidth_pt, subplots=(4, 3)),
    constrained_layout=True,
    sharex=True,
    sharey="row",
)
ax[0, 0].set(ylabel=r"$\rm{Ov}_{i}=|\langle \psi_{0}|\rm{minS}_{i}\rangle|^{2}$")
ax[1, 0].set(ylabel=r"observables")
ax[2, 0].set(ylabel=r"entropy $S$")
ax[2, 0].set(xlabel=r"time $t$")
ax[2, 1].set(xlabel=r"time $t$")
ax[2, 2].set(xlabel=r"time $t$")
ax[2, 3].set(xlabel=r"time $t$")

kk = 1
for kk, m in enumerate(vals["sector"]):
    t = np.asarray(res["time_steps"][:100], dtype=float)
    e2 = np.asarray(res["E2"][kk, :100], dtype=float)
    # be robust to NaNs/Infs
    mask = np.isfinite(t) & np.isfinite(e2)
    idx_min = np.nanargmin(e2[mask])
    t_min = t[mask][idx_min]
    for ii in range(3):
        ax[ii, kk].set(xlim=[-0.1, 5])
        ax[ii, kk].axvline(
            t_min, linestyle="--", linewidth=1.0, color="k", alpha=0.8, zorder=10
        )
    for ii in range(5):
        ax[0, kk].plot(
            res["time_steps"],
            res[f"ov_min{ii}"][kk],
            "o-",
            markersize=1,
            markeredgewidth=0.2,
            label=overlap_names[ii],
            linewidth=0.8,
        )
    ax[0, kk].plot(
        res["time_steps"],
        res["tot_ov_min"][kk],
        "o-",
        markersize=2,
        markeredgewidth=0.2,
        linewidth=0.8,
        label=r"sum $\rm{Ov}_{i}$",
    )

    for jj, obs in enumerate(["E2", "N_single", "N_pair"]):  # , "N_tot"]):
        ax[1, kk].plot(
            res["time_steps"],
            res[f"{obs}"][kk],
            "o-",
            c=obs_color[jj],
            markersize=obs_size[jj],
            markeredgewidth=0.2,
            label=f"{obs_names[jj]}",
            linewidth=0.8,
        )

    ax[2, kk].plot(
        res["time_steps"],
        res["entropy"][kk],
        "o-",
        markersize=1,
        markeredgewidth=0.2,
        linewidth=0.8,
    )
for ii in range(4):
    ax[0, ii].text(
        0.5,
        0.88,  # 5% in from left, 95% up from bottom
        rf"$N_b={ii}$",  # e.g. "(a)", "(b)", …
        transform=ax[0, ii].transAxes,  # interpret coords relative to the axes
        ha="left",
        va="top",  # align text box
        fontsize=9,  # tweak if you like
        bbox=dict(facecolor="white", alpha=0.2, edgecolor="black"),
    )
# save_dictionary(res, f"finite_density.pkl")
# plt.savefig(f"finite_density.pdf")
# %%
res = {}
config_filename = f"string_breaking/5x2/test_zerodensity"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["m"])

res = {"time_steps": get_sim(ugrid[0]).res["time_steps"]}
nsteps = len(res["time_steps"])

obs_list = [
    "entropy",
    "E2",
    "N_single",
    "N_pair",
    "N_zero",
    "N_tot",
    "ov_min0",
    "ov_min1",
    "ov_min2",
    "ov_min3",
    "ov_min4",
]

for obs in obs_list:
    res[f"{obs}"] = np.zeros((len(vals["m"]), nsteps))
    res[f"tot_ov_min"] = np.zeros((len(vals["m"]), nsteps))
    for kk, m in enumerate(vals["m"]):
        res[obs][kk] = get_sim(ugrid[kk]).res[obs]
for kk, m in enumerate(vals["m"]):
    for ii in range(5):
        res[f"tot_ov_min"][kk] += res[f"ov_min{ii}"][kk]
save_dictionary(res, f"minimal_string.pkl")

obs_color = ["darkblue", "darkred", "orange", "darkgreen"]
obs_names = [r"$E^{2}$", r"$N_{\rm{quarks}}$", r"$N_{\rm{baryon}}$", r"$N_{\rm{tot}}$"]
obs_size = [1, 1.5, 1.4, 1]
overlap_names = [
    r"$\rm{minS}_{1}$",
    r"$\rm{minS}_{2}$",
    r"$\rm{minS}_{3}$",
    r"$\rm{minS}_{4}$",
    r"$\rm{minS}_{5}$",
]
fig, ax = plt.subplots(3, 2, constrained_layout=True, sharex=True, sharey="row")
ax[0, 0].set(ylabel=r"$\rm{Ov}_{i}=|\langle \psi_{0}|\rm{minS}_{i}\rangle|^{2}$")
ax[1, 0].set(ylabel=r"observables")
ax[2, 0].set(ylabel=r"entropy $S$")
ax[2, 0].set(xlabel=r"time $t$ ($m=9.375,\, g=50$)")
ax[2, 1].set(xlabel=r"time $t$ ($m=18.75,\, g=50$)")

kk = 1
t = np.asarray(res["time_steps"], dtype=float)
e2 = np.asarray(res["E2"][kk], dtype=float)
# be robust to NaNs/Infs
mask = np.isfinite(t) & np.isfinite(e2)
idx_min = np.nanargmin(e2[mask])
t_min = t[mask][idx_min]
for ii in range(3):
    ax[ii, 1].axvline(
        t_min, linestyle="--", linewidth=1.0, color="k", alpha=0.8, zorder=10
    )

for kk, m in enumerate(vals["m"]):
    for ii in range(5):
        ax[0, kk].plot(
            res["time_steps"],
            res[f"ov_min{ii}"][kk],
            "o-",
            markersize=1.5,
            markeredgewidth=0.2,
            label=overlap_names[ii],
            linewidth=0.8,
        )
    ax[0, kk].plot(
        res["time_steps"],
        res["tot_ov_min"][kk],
        "o-",
        markersize=2,
        markeredgewidth=0.2,
        linewidth=0.8,
        label=r"$\rm{sum}\rm{Ov}_{i}$",
    )

    for jj, obs in enumerate(["E2", "N_single", "N_pair", "N_tot"]):
        ax[1, kk].plot(
            res["time_steps"],
            res[f"{obs}"][kk],
            "o-",
            c=obs_color[jj],
            markersize=obs_size[jj],
            markeredgewidth=0.2,
            label=f"{obs_names[jj]}",
            linewidth=0.8,
        )

    ax[2, kk].plot(
        res["time_steps"],
        res["entropy"][kk],
        "o-",
        markersize=2,
        markeredgewidth=0.2,
        linewidth=0.8,
    )

ax[0, 1].legend(
    bbox_to_anchor=(0.3, 0.35),
    ncol=2,
    handlelength=1,
    handletextpad=0.1,
    borderpad=0.1,
    framealpha=1.0,
    frameon=True,
    labelspacing=0.01,
)
ax[1, 0].legend(
    bbox_to_anchor=(0.9, 0.98),
    ncol=2,
    handlelength=1,
    handletextpad=0.1,
    borderpad=0.1,
    framealpha=1.0,
    frameon=True,
    labelspacing=0.01,
)
plt.savefig(f"zerodensity.pdf")
# =========================================================================

# %%
