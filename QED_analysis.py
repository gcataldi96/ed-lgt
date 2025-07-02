# %%
from simsio import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from ed_lgt.tools import save_dictionary, load_dictionary


def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    fig_height_pt = fig_width_pt * golden_ratio * (subplots[0] / subplots[1])
    print(fig_width_pt, fig_height_pt)
    return (fig_width_in, fig_height_in)


textwidth_pt = 510.0
textwidth_in = textwidth_pt / 72.27
columnwidth_pt = 246.0
columnwidth_in = columnwidth_pt / 72.27
"""
To extract simulations use
    op1) energy[ii][jj] = extract_dict(ugrid[ii][jj], key="res", glob="energy")
    op2) energy[ii][jj] = get_sim(ugrid[ii][jj]).res["energy"])
To acquire the psi file
    sim= get_sim(ugrid[ii][jj])
    sim.link("psi")
    psi= sim.load("psi", cache=True)
"""


@plt.FuncFormatter
def fake_log(x, pos):
    "The two args are the value and tick position"
    return r"$10^{%d}$" % (x)


# %%
config_filename = f"QED/scaling_conv"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "spin"])

res = {
    "entropy": np.zeros((len(vals["g"]), len(vals["spin"]))),
    "E_square": np.zeros((len(vals["g"]), len(vals["spin"]))),
    "plaq": np.zeros((len(vals["g"]), len(vals["spin"]))),
}

for ii, g in enumerate(vals["g"]):
    for kk, spin in enumerate(vals["spin"]):
        res["entropy"][ii, kk] = get_sim(ugrid[ii][kk]).res["entropy"]
        res["E_square"][ii, kk] = np.mean(get_sim(ugrid[ii][kk]).res["E_square"])
        res["plaq"][ii, kk] = get_sim(ugrid[ii][kk]).res[
            "C_px,py_C_py,mx_C_my,px_C_mx,my"
        ]
abs_convergence = 1e-6
rel_convergence = 1e-7
res["convergence"] = np.zeros((len(vals["g"])))
for ii, g in enumerate(vals["g"]):
    for kk, spin in enumerate(vals["spin"]):
        if kk > 0:
            abs_delta = np.abs(res["plaq"][ii, kk] - res["plaq"][ii, kk - 1])
            rel_delta = abs_delta / np.abs(res["plaq"][ii, kk])
            if abs_delta < abs_convergence and rel_delta < rel_convergence:
                print(g, spin)
                res["convergence"][ii] = spin
                break
print("===========================")
for ii, g in enumerate(vals["g"]):
    if res["convergence"][ii] == 0:
        res["convergence"][ii] = 30

prefactor = 2.9
fig, ax = plt.subplots()
ax.plot(
    1 / vals["g"],
    res["convergence"],
    "-o",
    label=r"$j_{\min} (\Delta_{re\ell} =10^{-7}, \Delta_{abs} =10^{-6})$",
)
ax.set(
    xscale="log",
    yscale="log",
    ylim=[3, 50],
    xlim=[3e-1, 10],
    xlabel="1/g",
    ylabel=r"$j_{\min}$",
)
ax.plot(1 / vals["g"], prefactor / vals["g"], label=r"$3*g^{-1}$")
ax.legend()
plt.savefig("QED_conv.pdf")
# %%
dim_list = [19, 85, 231, 489, 891, 1469, 2255, 3281, 4579]
config_filename = f"LBO/qed_scan"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "spin"])

res1 = {
    "entropy": np.zeros((len(vals["g"]), len(vals["spin"]))),
    "E_square": np.zeros((len(vals["g"]), len(vals["spin"]))),
    "plaq": np.zeros((len(vals["g"]), len(vals["spin"]))),
    "energy": np.zeros((len(vals["g"]), len(vals["spin"]))),
}

for ii, g in enumerate(vals["g"]):
    for kk, spin in enumerate(vals["spin"]):
        res1["entropy"][ii, kk] = get_sim(ugrid[ii][kk]).res["entropy"]
        res1["E_square"][ii, kk] = np.mean(get_sim(ugrid[ii][kk]).res["E_square"])
        res1["plaq"][ii, kk] = get_sim(ugrid[ii][kk]).res[
            "C_px,py_C_py,mx_C_my,px_C_mx,my"
        ]
        res1["energy"][ii, kk] = get_sim(ugrid[ii][kk]).res["energy"]

sm = cm.ScalarMappable(cmap="magma")
palette = sm.to_rgba(vals["spin"])
marker_size = 2
fig, axs = plt.subplots(3, 1, sharex=True, constrained_layout=True)
for kk, spin in enumerate(vals["spin"]):
    axs[0].plot(
        vals["g"],
        0.5 * res1["E_square"][:, kk],
        "o-",
        c=palette[kk],
        markersize=marker_size,
        markeredgecolor=palette[kk],
        markerfacecolor="black",
        markeredgewidth=0.5,
    )
    axs[0].set(ylabel="Casimir $E^{2}$", xscale="log")
    axs[1].plot(
        vals["g"],
        1 - res1["plaq"][:, kk],
        "o-",
        c=palette[kk],
        markersize=marker_size,
        markeredgecolor=palette[kk],
        markerfacecolor="black",
        markeredgewidth=0.5,
    )
    axs[1].set(ylabel="plaquette $B^{2}$", xscale="log")

    axs[2].plot(
        vals["g"],
        res1["entropy"][:, kk],
        "o-",
        c=palette[kk],
        markersize=marker_size,
        markeredgecolor=palette[kk],
        markerfacecolor="black",
        markeredgewidth=0.5,
    )
    axs[2].set(ylabel="entropy $\mathcal{S}$", xscale="log")

axs[-1].set(xlabel=r"coupling $g$")
cb = fig.colorbar(
    sm, ax=axs, aspect=50, location="right", orientation="vertical", pad=0.005
)
cb.set_label(label=r"spin", rotation=0, labelpad=-15, x=-0.02, y=-0.03)
plt.savefig(f"QED_scan.pdf")


fig, axs = plt.subplots(3, 1, sharex=True, constrained_layout=True)
obs_names = ["E_square", "plaq", "entropy"]
colors = ["darkblue", "darkgreen", "darkred"]

obs_list = ["energy", "E_square", "plaq", "entropy"]
obs = obs_list[0]
for ii, gindex in enumerate([0, 12, 20]):
    axs[ii].plot(
        dim_list[:-1],
        np.abs(res1[obs][gindex, :-1] - res1[obs][gindex, -1])
        / np.abs(res1[obs][gindex, -1]),
        "o-",
        c=colors[ii],
        markersize=marker_size + 2,
        markeredgecolor=colors[ii],
        markerfacecolor="white",
        markeredgewidth=0.5,
        label=f"g={round(vals['g'][gindex], 2)}",
    )
    axs[ii].set(ylabel=f"{obs}", yscale="log", xscale="log")
axs[-1].set(xlabel=r"local dimension")
fig.legend(
    loc="upper center",
    bbox_to_anchor=(0.85, 0.35),
    ncol=1,
    frameon=False,
    fontsize=10,
)
# save_dictionary(res1, "QED_scan.pkl")
# %%
oldres = load_dictionary("QED_error.pkl")

res = {}
for obs in ["entropy", "E_square", "plaq", "energy"]:
    res[obs] = np.zeros(30, dtype=float)
    res[f"eff_{obs}"] = np.zeros((30, 9), dtype=float)
    for ii in range(20):
        res[obs][ii] = oldres[obs][ii]
        res[f"eff_{obs}"][ii, :] = oldres[f"eff_{obs}"][ii, :]

res["eff_dims"] = np.array(
    [
        [173, 173, 2037, 3747, 4349, 4527, 4575, 4575, 4579],
        [189, 189, 2021, 3719, 4341, 4527, 4575, 4575, 4579],
        [209, 209, 1979, 3655, 4341, 4519, 4559, 4575, 4579],
        [225, 225, 1955, 3575, 4325, 4515, 4559, 4575, 4579],
        [257, 257, 1851, 3427, 4261, 4505, 4559, 4575, 4579],
        [289, 289, 1691, 3255, 4193, 4465, 4551, 4575, 4575],
        [305, 305, 1579, 2949, 3983, 4369, 4515, 4559, 4575],
        [305, 305, 1373, 2565, 3695, 4217, 4449, 4527, 4559],
        [305, 305, 1073, 2131, 3205, 3967, 4301, 4449, 4523],
        [305, 305, 877, 1659, 2549, 3413, 3999, 4261, 4441],
        [257, 257, 667, 1297, 1955, 2709, 3405, 3879, 4147],
        [15, 225, 551, 953, 1429, 2019, 2691, 3221, 3689],
        [19, 165, 411, 731, 1057, 1445, 1931, 2395, 2965],
        [27, 145, 305, 543, 773, 1057, 1389, 1747, 2017],
        [27, 121, 233, 387, 587, 787, 1025, 1265, 1541],
        [27, 87, 165, 305, 427, 571, 739, 925, 1121],
        [19, 63, 145, 225, 305, 411, 535, 651, 813],
        [19, 59, 103, 161, 225, 305, 387, 491, 595],
        [15, 43, 63, 121, 165, 221, 289, 339, 451],
        [13, 27, 51, 71, 121, 161, 197, 249, 313],
        [9, 15, 27, 59, 71, 103, 137, 165, 237],
        [9, 13, 19, 35, 51, 67, 103, 129, 149],
        [9, 9, 13, 27, 35, 51, 59, 87, 103],
        [9, 9, 9, 15, 27, 35, 51, 59, 83],
        [9, 9, 9, 9, 15, 27, 35, 51, 59],
        [9, 9, 9, 9, 13, 15, 27, 35, 35],
        [9, 9, 9, 9, 9, 13, 19, 27, 35],
        [9, 9, 9, 9, 9, 9, 13, 19, 35],
        [9, 9, 9, 9, 9, 9, 9, 15, 27],
        [9, 9, 9, 9, 9, 9, 9, 9, 15],
    ]
)


res["gvals"] = np.logspace(-1, 0.5, 30, dtype=float)
res["precision"] = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
sm = cm.ScalarMappable(cmap="magma")
palette = sm.to_rgba(res["gvals"])
fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=True)
for ii, gindex in enumerate(res["gvals"]):
    axs.plot(
        res["precision"],
        res["eff_dims"][ii, :],
        "o-",
        c=palette[ii],
        markersize=marker_size + 2,
        markeredgecolor=palette[ii],
        markerfacecolor="white",
        markeredgewidth=0.5,
        label=f"eff g={round(res['gvals'][ii], 2)}",
    )
axs.set(xscale="log", yscale="log", xlabel="precision", ylabel="local dimension")
cb = fig.colorbar(
    sm, ax=axs, aspect=50, location="right", orientation="vertical", pad=0.005
)
cb.set_label(label=r"g", rotation=0, labelpad=-15, x=-0.02, y=-0.03)
# %%


config_filename = f"LBO/qed_error"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])


for ii, g in enumerate(vals["g"]):
    for obs in ["entropy", "E_square", "energy"]:
        res[obs][20 + ii] = get_sim(ugrid[ii]).res[obs]
        res[f"eff_{obs}"][20 + ii] = get_sim(ugrid[ii]).res[f"eff_{obs}"].reshape(9)
    res["plaq"][20 + ii] = get_sim(ugrid[ii]).res["C_px,py_C_py,mx_C_my,px_C_mx,my"]
    res["eff_plaq"][20 + ii] = (
        get_sim(ugrid[ii]).res["eff_C_px,py_C_py,mx_C_my,px_C_mx,my"].reshape(9)
    )


# %%
for gindex in range(len(res["gvals"])):
    fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=True)
    colors = ["darkblue", "darkgreen", "darkred"]

    obs_list = ["energy", "E_square", "plaq", "entropy"]
    obs_name = obs_list[0]

    axs.plot(
        res["eff_dims"][gindex, :],
        np.abs(res[f"eff_{obs_name}"][gindex, :] - res[obs_name][gindex])
        / np.abs(res[obs_name][gindex]),
        "o-",
        c=colors[0],
        markersize=marker_size + 2,
        markeredgecolor=colors[0],
        markerfacecolor="white",
        markeredgewidth=0.5,
        label=f"eff g={round(res['gvals'][gindex], 2)}",
    )

    axs.plot(
        dim_list[:-1],
        np.abs(res1[obs_name][gindex, :-1] - res1[obs_name][gindex, -1])
        / np.abs(res1[obs_name][gindex, -1]),
        "o-",
        c=colors[1],
        markersize=marker_size + 2,
        markeredgecolor=colors[1],
        markerfacecolor="white",
        markeredgewidth=0.5,
        label=f"irrep g={round(res['gvals'][gindex], 2)}",
    )

    axs.set(ylabel=f"delta {obs_name}", yscale="log", xscale="log")
    axs.set(xlabel=r"local dimension")
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.85, 0.95),
        ncol=1,
        frameon=False,
        fontsize=10,
    )
# %%
config_filename = f"LBO/qed_plaq_svd"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
sm = cm.ScalarMappable(cmap="magma", norm=LogNorm())
palette = sm.to_rgba(vals["g"])


res = {"eigvals": np.zeros((len(vals["g"]), 5299), dtype=float)}
for ii, g in enumerate(vals["g"]):
    res["eigvals"][ii, :] = get_sim(ugrid[ii]).res["eigvals"]
fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=True)
for ii, g in enumerate(vals["g"]):
    axs.plot(
        np.arange(1, 5300, 1),
        res["eigvals"][ii, :],
        "o-",
        color=palette[ii],
        markersize=2,
        markeredgecolor=palette[ii],
        markerfacecolor="black",
        markeredgewidth=0.5,
    )
axs.set(
    yscale="log",
    xscale="log",
    xlabel=r"eigenvalue index $k$",
    ylabel=r"singular values $\lambda_{k}$",
)
cb = fig.colorbar(
    sm, ax=axs, aspect=50, location="right", orientation="vertical", pad=0.005
)
cb.set_label(label=r"$g^{2}$", rotation=0, labelpad=-30, x=-0.02, y=-0.03)
# save_dictionary(res, "QED_svd.pkl")
# plt.savefig("QED_plaq_svd.pdf")
# %%
config_filename = f"theta_term/qed_theta"
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
config_filename = f"theta_term/qed_theta2"
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
config_filename = f"theta_term/qed_theta_eigvals"
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
    for ii in range(n_eigs):
        axs.plot(
            res["theta"],
            res["energy"][g_index, :, ii],
            "o-",
            markersize=2,
            markerfacecolor="black",
            markeredgewidth=0.5,
            label=f"E {ii}",
        )
    axs.set(
        xlabel=r"coupling $\theta$",
        # ylabel=r"casimir $E^{2}$",
        ylabel=rf"energy levels g={round(g,3)}",
        xlim=[0.34, 0.45],
    )
    fig.legend()
save_dictionary(res, f"qed_thetaterm_ED3.pkl")
# %%
