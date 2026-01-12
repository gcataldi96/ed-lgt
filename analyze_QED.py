# %%
from simsio import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from ed_lgt.tools import save_dictionary

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
