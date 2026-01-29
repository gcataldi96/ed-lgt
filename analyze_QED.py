# %%
from simsio import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from ed_lgt.tools import save_dictionary, set_size
import matplotlib.lines as mlines

textwidth_pt = 510.0
columnwidth_pt = 246.0

# %%
config_filename = f"su2_thetaterm/scan2"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "theta"])

res = {
    "entropy": np.zeros((len(vals["g"]), len(vals["theta"]))),
    "E2": np.zeros((len(vals["g"]), len(vals["theta"]))),
    "plaq": np.zeros((len(vals["g"]), len(vals["theta"]))),
    "energy": np.zeros((len(vals["g"]), len(vals["theta"]))),
    "g": vals["g"],
    "theta": vals["theta"],
}
for ii, g in enumerate(vals["g"]):
    for kk, theta in enumerate(vals["theta"]):
        res["entropy"][ii, kk] = get_sim(ugrid[ii][kk]).res["entropy"][0]
        res["energy"][ii, kk] = get_sim(ugrid[ii][kk]).res["energy"][0]
        res["E2"][ii, kk] = get_sim(ugrid[ii][kk]).res["E2"][0]
        res["plaq"][ii, kk] += (
            get_sim(ugrid[ii][kk]).res["C_px,py_C_py,mx_C_my,px_C_mx,my"][0] / 3
        )
        res["plaq"][ii, kk] += (
            get_sim(ugrid[ii][kk]).res["C_px,pz_C_pz,mx_C_mz,px_C_mx,mz"][0] / 3
        )
        res["plaq"][ii, kk] += (
            get_sim(ugrid[ii][kk]).res["C_py,pz_C_pz,my_C_mz,py_C_my,mz"][0] / 3
        )
# save_dictionary(res, f"qed_thetaterm_ED1.pkl")
# %%
gindexmin = 0
sm_gvals = cm.ScalarMappable(
    cmap="magma",
)
palette_gvals = sm_gvals.to_rgba(res["g"][gindexmin:])
sm_theta = cm.ScalarMappable(cmap="seismic")
palette_theta = sm_theta.to_rgba(res["theta"])
fig, axs = plt.subplots(
    3,
    1,
    figsize=(set_size(columnwidth_pt, subplots=(3, 1))),
    sharex=True,
    constrained_layout=True,
)
axs[-1].set_xlabel(r"coupling $\tilde{\theta}$")
observables = ["E2", "plaq", "entropy"]
obs_names = [r"casimir $E^{2}$", r"plaq $B^{2}$", r"entropy $S$"]
for ax, obs, obs_name in zip(axs, observables, obs_names):
    for ii, g in enumerate(res["g"][gindexmin:]):
        ax.plot(
            res["theta"],
            res[obs][ii + gindexmin, :],
            "o-",
            color=palette_gvals[ii],
            markersize=2,
            markeredgecolor=palette_gvals[ii],
            markerfacecolor="black",
            markeredgewidth=0.5,
        )
    ax.set(ylabel=obs_name)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
cb = fig.colorbar(
    sm_gvals, ax=axs, aspect=50, location="top", orientation="horizontal", pad=0.005
)
cb.set_label(label=r"$g^{2}$", rotation=0, labelpad=-25, x=-0.04, y=+0.03)
plt.savefig(f"SU2thetaterm.pdf")
# %%
plaq1 = "C_px,py_C_py,mx_C_my,px_C_mx,my"
plaq2 = "C_px,pz_C_pz,mx_C_mz,px_C_mx,mz"
plaq3 = "C_py,pz_C_pz,my_C_mz,py_C_my,mz"
res = {}
for case in ["000", "pipipi"]:
    config_filename = f"su2_thetaterm/k{case}"
    match = SimsQuery(group_glob=config_filename)
    ugrid, vals = uids_grid(match.uids, ["g", "theta"])
    dim_tuple = (len(vals["g"]), len(vals["theta"]), 3)
    res[case] = {
        "E2": np.zeros(dim_tuple),
        "plaq": np.zeros(dim_tuple),
        "energy": np.zeros(dim_tuple),
        "g": vals["g"],
        "theta": vals["theta"],
    }
    for ii, g in enumerate(vals["g"]):
        for kk, theta in enumerate(vals["theta"]):
            for neig in range(3):
                res[case]["energy"][ii, kk, neig] = get_sim(ugrid[ii][kk]).res[
                    "energy"
                ][neig]
                res[case]["E2"][ii, kk, neig] = get_sim(ugrid[ii][kk]).res["E2"][neig]
                res[case]["plaq"][ii, kk, neig] += (
                    get_sim(ugrid[ii][kk]).res[plaq1][neig] / 3
                )
                res[case]["plaq"][ii, kk, neig] += (
                    get_sim(ugrid[ii][kk]).res[plaq2][neig] / 3
                )
                res[case]["plaq"][ii, kk, neig] += (
                    get_sim(ugrid[ii][kk]).res[plaq3][neig] / 3
                )
# %%
n_eigs = 3
sm_theta = cm.ScalarMappable(cmap="seismic")
palette_theta = sm_theta.to_rgba(res["000"]["theta"])
fig, axs = plt.subplots(
    3,
    1,
    figsize=(set_size(columnwidth_pt, subplots=(3, 1))),
    sharex=True,
    constrained_layout=True,
)
axs[-1].set_xlabel(r"coupling $\tilde{\theta}$")
color_k = ["black", "red"]
k_labels = [r"$k\!=\!(0,0,0)$", r"$k\!=\!(\pi,\pi,\pi)$"]
# choose the x-window you want (can be different per axis if needed)
xlo, xhi = 0.5, 0.7
ylim_list = [(-9.7, -8.8), (-0.63, -0.49), (-0.65, 0.62)]
for gidx, g in enumerate(res["000"]["g"]):
    axs[gidx].set(
        ylabel=rf"energy $\mathcal{{E}}\;(g^{{2}}={g})$",
        xlim=(xlo, xhi),
        yscale="log",
        # ylim=ylim_list[gidx],
    )
    axs[gidx].grid(True, which="both", linestyle="--", linewidth=0.5)
    for caseidx, case in enumerate(["000", "pipipi"]):
        for ii in range(n_eigs):
            axs[gidx].plot(
                res[case]["theta"],
                res[case]["energy"][gidx, :, ii] - res["000"]["energy"][gidx, :, 0],
                "o-",
                color=color_k[caseidx],
                markersize=2,
                markeredgecolor="black",
                markerfacecolor=color_k[caseidx],
                markeredgewidth=0.3,
            )

# --- one legend for the whole figure: only two cases ---
handles = [
    mlines.Line2D(
        [],
        [],
        color=color_k[0],
        marker="o",
        linestyle="-",
        markersize=4,
        label=k_labels[0],
    ),
    mlines.Line2D(
        [],
        [],
        color=color_k[1],
        marker="o",
        linestyle="-",
        markersize=4,
        label=k_labels[1],
    ),
]
leg = fig.legend(
    handles=handles,
    loc="upper center",
    ncol=1,
    bbox_to_anchor=(0.56, 0.87),
    frameon=True,  # <- box on
    fancybox=True,  # rounded corners (set False for sharp)
    framealpha=1.0,  # opaque box (e.g. 0.6 for semi-transparent)
    edgecolor="black",
    facecolor="white",
    borderpad=0.3,
)

# optional: tweak line width of the frame
leg.get_frame().set_linewidth(0.8)
plt.savefig(f"SU2_momentum.pdf")
# %%
plaq1 = "C_px,py_C_py,mx_C_my,px_C_mx,my"
plaq2 = "C_px,pz_C_pz,mx_C_mz,px_C_mx,mz"
plaq3 = "C_py,pz_C_pz,my_C_mz,py_C_my,mz"
res = {}
for case in ["000", "pipipi"]:
    config_filename = f"qed_theta_term/k{case}"
    match = SimsQuery(group_glob=config_filename)
    ugrid, vals = uids_grid(match.uids, ["g", "theta"])
    dim_tuple = (len(vals["g"]), len(vals["theta"]), 3)
    res[case] = {
        "E2": np.zeros(dim_tuple),
        "plaq": np.zeros(dim_tuple),
        "energy": np.zeros(dim_tuple),
        "g": vals["g"],
        "theta": vals["theta"],
    }
    for ii, g in enumerate(vals["g"]):
        for kk, theta in enumerate(vals["theta"]):
            for neig in range(3):
                res[case]["energy"][ii, kk, neig] = get_sim(ugrid[ii][kk]).res[
                    "energy"
                ][neig]
                res[case]["E2"][ii, kk, neig] = get_sim(ugrid[ii][kk]).res["E2"][neig]
                res[case]["plaq"][ii, kk, neig] += (
                    get_sim(ugrid[ii][kk]).res[plaq1][neig] / 3
                )
                res[case]["plaq"][ii, kk, neig] += (
                    get_sim(ugrid[ii][kk]).res[plaq2][neig] / 3
                )
                res[case]["plaq"][ii, kk, neig] += (
                    get_sim(ugrid[ii][kk]).res[plaq3][neig] / 3
                )
# %%
n_eigs = 3
sm_theta = cm.ScalarMappable(cmap="seismic")
palette_theta = sm_theta.to_rgba(res["000"]["theta"])
fig, axs = plt.subplots(
    4,
    1,
    figsize=(set_size(columnwidth_pt, subplots=(4, 1))),
    sharex=True,
    constrained_layout=True,
)
axs[-1].set_xlabel(r"coupling $\tilde{\theta}$")
color_k = ["black", "red"]
k_labels = [r"$k\!=\!(0,0,0)$", r"$k\!=\!(\pi,\pi,\pi)$"]
# choose the x-window you want (can be different per axis if needed)
xlo, xhi = 0.4, 0.7
indices = [0, 1, 2, 4]
# ylim_list = [(-9.7, -8.8), (-0.63, -0.49), (-0.65, 0.62)]
gvals = res["000"]["g"][indices]
for kk in range(4):
    g = gvals[kk]
    axs[kk].set(
        ylabel=rf"energy $\mathcal{{E}}\;(g^{{2}}={g})$",
        # xlim=(xlo, xhi),
        # ylim=ylim_list[gidx],
    )
    axs[kk].grid(True, which="both", linestyle="--", linewidth=0.5)
    for caseidx, case in enumerate(["000", "pipipi"]):
        for ii in range(n_eigs):
            axs[kk].plot(
                res[case]["theta"],
                res[case]["energy"][kk, :, ii],  # - res["000"]["energy"][gidx, :, 0],
                "o-",
                color=color_k[caseidx],
                markersize=2,
                markeredgecolor="black",
                markerfacecolor=color_k[caseidx],
                markeredgewidth=0.3,
            )

# --- one legend for the whole figure: only two cases ---
handles = [
    mlines.Line2D(
        [],
        [],
        color=color_k[0],
        marker="o",
        linestyle="-",
        markersize=4,
        label=k_labels[0],
    ),
    mlines.Line2D(
        [],
        [],
        color=color_k[1],
        marker="o",
        linestyle="-",
        markersize=4,
        label=k_labels[1],
    ),
]
leg = fig.legend(
    handles=handles,
    loc="upper center",
    ncol=1,
    bbox_to_anchor=(0.56, 0.93),
    frameon=True,  # <- box on
    fancybox=True,  # rounded corners (set False for sharp)
    framealpha=1.0,  # opaque box (e.g. 0.6 for semi-transparent)
    edgecolor="black",
    facecolor="white",
    borderpad=0.3,
)

# optional: tweak line width of the frame
leg.get_frame().set_linewidth(0.8)
plt.savefig(f"QED_momentum.pdf")
# %%
fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=True)
for ii, theta in enumerate(res["theta"][:10]):
    axs.plot(
        res["g"],
        res["entropy"][:, ii],
        "o-",
        color=palette_theta[ii],
        markersize=2,
        markeredgecolor=palette_theta[ii],
        markerfacecolor="black",
        markeredgewidth=0.5,
    )
axs.set(
    xlabel=r"coupling $g^{2}$",
    ylabel=r"entropy $S$",
    xscale="log",
)
cb = fig.colorbar(
    sm_theta, ax=axs, aspect=50, location="right", orientation="vertical", pad=0.005
)
cb.set_label(label=r"$\tilde{\theta}$", rotation=0, labelpad=-30, x=-0.02, y=-0.03)
# %%
config_filename = f"qed_theta_term/qed_theta"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "theta"])

res = {
    "entropy": np.zeros((len(vals["g"]), len(vals["theta"]))),
    "E_square": np.zeros((len(vals["g"]), len(vals["theta"]))),
    "plaq": np.zeros((len(vals["g"]), len(vals["theta"]))),
    "energy": np.zeros((len(vals["g"]), len(vals["theta"]))),
    "g": vals["g"],
    "theta": vals["theta"],
}

# %%
for ii, g in enumerate(vals["g"]):
    for kk, theta in enumerate(vals["theta"]):
        res["entropy"][ii, kk] = get_sim(ugrid[ii][kk]).res["entropy"]
        res["energy"][ii, kk] = get_sim(ugrid[ii][kk]).res["energy"]
        res["E_square"][ii, kk] = get_sim(ugrid[ii][kk]).res["E_square"]
        res["plaq"][ii, kk] += (
            get_sim(ugrid[ii][kk]).res["C_px,py_C_py,mx_C_my,px_C_mx,my"] / 3
        )
        res["plaq"][ii, kk] += (
            get_sim(ugrid[ii][kk]).res["C_px,pz_C_pz,mx_C_mz,px_C_mx,mz"] / 3
        )
        res["plaq"][ii, kk] += (
            get_sim(ugrid[ii][kk]).res["C_py,pz_C_pz,my_C_mz,py_C_my,mz"] / 3
        )
save_dictionary(res, f"qed_thetaterm_ED1.pkl")
# %%
gindexmin = 0
sm_gvals = cm.ScalarMappable(cmap="magma", norm=LogNorm())
palette_gvals = sm_gvals.to_rgba(res["g"][gindexmin:])
sm_theta = cm.ScalarMappable(cmap="seismic")
palette_theta = sm_theta.to_rgba(res["theta"])

fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=True)
for ii, g in enumerate(res["g"][gindexmin:]):
    print(g)
    axs.plot(
        res["theta"],
        res["E_square"][ii + gindexmin, :],
        "o-",
        color=palette_gvals[ii],
        markersize=2,
        markeredgecolor=palette_gvals[ii],
        markerfacecolor="black",
        markeredgewidth=0.5,
    )
axs.set(
    xlabel=r"coupling $\tilde{\theta}$",
    ylabel=r"casimir $E^{2}$",
)
cb = fig.colorbar(
    sm_gvals, ax=axs, aspect=50, location="right", orientation="vertical", pad=0.005
)
cb.set_label(label=r"$g^{2}$", rotation=0, labelpad=-30, x=-0.02, y=-0.03)

# %%
fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=True)
for ii, theta in enumerate(res["theta"][:10]):
    axs.plot(
        res["g"],
        res["entropy"][:, ii],
        "o-",
        color=palette_theta[ii],
        markersize=2,
        markeredgecolor=palette_theta[ii],
        markerfacecolor="black",
        markeredgewidth=0.5,
    )
axs.set(
    xlabel=r"coupling $g^{2}$",
    ylabel=r"entropy $S$",
    xscale="log",
)
cb = fig.colorbar(
    sm_theta, ax=axs, aspect=50, location="right", orientation="vertical", pad=0.005
)
cb.set_label(label=r"$\tilde{\theta}$", rotation=0, labelpad=-30, x=-0.02, y=-0.03)
# %%
config_filename = f"qed_theta_term/qed_theta2"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "theta"])

res = {
    "entropy": np.zeros((len(vals["g"]), len(vals["theta"]))),
    "E_square": np.zeros((len(vals["g"]), len(vals["theta"]))),
    "plaq": np.zeros((len(vals["g"]), len(vals["theta"]))),
    "energy": np.zeros((len(vals["g"]), len(vals["theta"]))),
    "g": vals["g"],
    "theta": vals["theta"],
}

for ii, g in enumerate(vals["g"]):
    for kk, theta in enumerate(vals["theta"]):
        res["entropy"][ii, kk] = get_sim(ugrid[ii][kk]).res["entropy"]
        res["energy"][ii, kk] = get_sim(ugrid[ii][kk]).res["energy"]
        res["E_square"][ii, kk] = get_sim(ugrid[ii][kk]).res["E_square"]
        res["plaq"][ii, kk] += (
            get_sim(ugrid[ii][kk]).res["C_px,py_C_py,mx_C_my,px_C_mx,my"] / 3
        )
        res["plaq"][ii, kk] += (
            get_sim(ugrid[ii][kk]).res["C_px,pz_C_pz,mx_C_mz,px_C_mx,mz"] / 3
        )
        res["plaq"][ii, kk] += (
            get_sim(ugrid[ii][kk]).res["C_py,pz_C_pz,my_C_mz,py_C_my,mz"] / 3
        )

save_dictionary(res, f"qed_thetaterm_ED2.pkl")

sm_gvals = cm.ScalarMappable(cmap="magma")
palette_gvals = sm_gvals.to_rgba(res["g"])
sm_theta = cm.ScalarMappable(cmap="seismic")
palette_theta = sm_theta.to_rgba(res["theta"])

fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=True)
for ii, g in enumerate(res["g"]):
    axs.plot(
        res["theta"],
        res["entropy"][ii, :],
        "o-",
        color=palette_gvals[ii],
        markersize=2,
        markeredgecolor=palette_gvals[ii],
        markerfacecolor="black",
        markeredgewidth=0.5,
    )
axs.set(
    xlabel=r"coupling $\tilde{\theta}$",
    # ylabel=r"casimir $E^{2}$",
    ylabel=r"entropy",
)
cb = fig.colorbar(
    sm_gvals, ax=axs, aspect=50, location="right", orientation="vertical", pad=0.005
)
cb.set_label(label=r"$g^{2}$", rotation=0, labelpad=-30, x=-0.02, y=-0.03)
# %%
config_filename = f"qed_theta_term/eigvals"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "theta"])
n_eigs = 5
res = {
    "E_square": np.zeros((len(vals["g"]), len(vals["theta"]), n_eigs)),
    "plaq": np.zeros((len(vals["g"]), len(vals["theta"]), n_eigs)),
    "energy": np.zeros((len(vals["g"]), len(vals["theta"]), n_eigs)),
    "g": vals["g"],
    "theta": vals["theta"],
}

plaq1 = "C_px,py_C_py,mx_C_my,px_C_mx,my"
plaq2 = "C_px,pz_C_pz,mx_C_mz,px_C_mx,mz"
plaq3 = "C_py,pz_C_pz,my_C_mz,py_C_my,mz"

for ii, g in enumerate(vals["g"]):
    for kk, theta in enumerate(vals["theta"]):
        for neig in range(5):
            res["energy"][ii, kk, neig] = get_sim(ugrid[ii][kk]).res["energy"][neig]
            res["E_square"][ii, kk, neig] = get_sim(ugrid[ii][kk]).res["E_square"][neig]
            res["plaq"][ii, kk, neig] += get_sim(ugrid[ii][kk]).res[plaq1][neig] / 3
            res["plaq"][ii, kk, neig] += get_sim(ugrid[ii][kk]).res[plaq2][neig] / 3
            res["plaq"][ii, kk] += get_sim(ugrid[ii][kk]).res[plaq3][neig] / 3

sm_gvals = cm.ScalarMappable(cmap="magma")
palette_gvals = sm_gvals.to_rgba(res["g"])
sm_theta = cm.ScalarMappable(cmap="seismic")
palette_theta = sm_theta.to_rgba(res["theta"])

g_index = 0
for g_index, g in enumerate(res["g"]):
    fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=True)
    axs.grid()
    for ii in range(1, n_eigs, 1):
        axs.plot(
            res["theta"],
            res["energy"][g_index, :, ii] - res["energy"][g_index, :, 0],
            "o-",
            markersize=2,
            markerfacecolor="black",
            markeredgewidth=0.5,
            label=f"E {ii}",
        )
    axs.set(
        xlabel=r"coupling $\theta$",
        # ylabel=r"casimir $E^{2}$",
        ylabel=rf"energy levels g^{2}={round(g,3)}",
        xlim=[0.34, 0.45],
        yscale="log",
        # ylim=[0, 0.],
    )
    fig.legend()
save_dictionary(res, f"qed_thetaterm_ED3.pkl")
# %%
config_filename = f"qed_theta_term/k0"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["theta"])
n_eigs = 4
res1 = {
    "E_square": np.zeros((len(vals["theta"]), n_eigs)),
    "plaq": np.zeros((len(vals["theta"]), n_eigs)),
    "energy": np.zeros((len(vals["theta"]), n_eigs)),
    "theta": vals["theta"],
}

plaq1 = "C_px,py_C_py,mx_C_my,px_C_mx,my"
plaq2 = "C_px,pz_C_pz,mx_C_mz,px_C_mx,mz"
plaq3 = "C_py,pz_C_pz,my_C_mz,py_C_my,mz"


for kk, theta in enumerate(vals["theta"]):
    for neig in range(4):
        res1["energy"][kk, neig] = get_sim(ugrid[kk]).res["energy"][neig]
        res1["E_square"][kk, neig] = get_sim(ugrid[kk]).res["E_square"][neig]
        res1["plaq"][kk, neig] += get_sim(ugrid[kk]).res[plaq1][neig] / 3
        res1["plaq"][kk, neig] += get_sim(ugrid[kk]).res[plaq2][neig] / 3
        res1["plaq"][kk] += get_sim(ugrid[kk]).res[plaq3][neig] / 3


config_filename = f"qed_theta_term/kpi"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["theta"])
n_eigs = 4
res2 = {
    "E_square": np.zeros((len(vals["theta"]), n_eigs)),
    "plaq": np.zeros((len(vals["theta"]), n_eigs)),
    "energy": np.zeros((len(vals["theta"]), n_eigs)),
    "theta": vals["theta"],
}

plaq1 = "C_px,py_C_py,mx_C_my,px_C_mx,my"
plaq2 = "C_px,pz_C_pz,mx_C_mz,px_C_mx,mz"
plaq3 = "C_py,pz_C_pz,my_C_mz,py_C_my,mz"

gs = np.zeros(len(vals["theta"]))

for kk, theta in enumerate(vals["theta"]):
    for neig in range(4):
        res2["energy"][kk, neig] = get_sim(ugrid[kk]).res["energy"][neig]
        res2["E_square"][kk, neig] = get_sim(ugrid[kk]).res["E_square"][neig]
        res2["plaq"][kk, neig] += get_sim(ugrid[kk]).res[plaq1][neig] / 3
        res2["plaq"][kk, neig] += get_sim(ugrid[kk]).res[plaq2][neig] / 3
        res2["plaq"][kk] += get_sim(ugrid[kk]).res[plaq3][neig] / 3
    gs[kk] = min(res2["energy"][kk, 0], res1["energy"][kk, 0])

sm_theta = cm.ScalarMappable(cmap="seismic")
palette_theta = sm_theta.to_rgba(res1["theta"])


fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=True)
axs.grid()
for ii in range(n_eigs):
    axs.plot(
        res1["theta"],
        res1["energy"][:, ii],
        "o-",
        c="black",
        markersize=2,
        markerfacecolor="black",
        markeredgewidth=0.5,
        label=rf"E {ii} ($0, 0, 0$)",
    )
    axs.plot(
        res2["theta"],
        res2["energy"][:, ii],
        "o-",
        c="red",
        markersize=2,
        markerfacecolor="black",
        markeredgewidth=0.5,
        label=rf"E {ii} ($\pi, \pi, \pi$)",
    )
axs.set(
    xlabel=r"coupling $\theta$",
    # ylabel=r"casimir $E^{2}$",
    ylabel=rf"energy levels g^{2}=2.327",
    # ylim=[0, 0.],
)
fig.legend()
# save_dictionary(res, f"qed_thetaterm_ED3.pkl")

# %%
