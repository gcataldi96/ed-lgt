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
config_filename = f"LBO/qed_scan"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "spin"])

res = {
    "entropy": np.zeros((len(vals["g"]), len(vals["spin"]))),
    "E_square": np.zeros((len(vals["g"]), len(vals["spin"]))),
    "plaq": np.zeros((len(vals["g"]), len(vals["spin"]))),
    "energy": np.zeros((len(vals["g"]), len(vals["spin"]))),
}

for ii, g in enumerate(vals["g"]):
    for kk, spin in enumerate(vals["spin"]):
        res["entropy"][ii, kk] = get_sim(ugrid[ii][kk]).res["entropy"]
        res["E_square"][ii, kk] = np.mean(get_sim(ugrid[ii][kk]).res["E_square"])
        res["plaq"][ii, kk] = get_sim(ugrid[ii][kk]).res[
            "C_px,py_C_py,mx_C_my,px_C_mx,my"
        ]
        res["energy"][ii, kk] = get_sim(ugrid[ii][kk]).res["energy"]

sm = cm.ScalarMappable(cmap="magma")
palette = sm.to_rgba(vals["spin"])
marker_size = 2
fig, axs = plt.subplots(3, 1, sharex=True, constrained_layout=True)
for kk, spin in enumerate(vals["spin"]):
    axs[0].plot(
        vals["g"],
        0.5 * res["E_square"][:, kk],
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
        -res["plaq"][:, kk],
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
        res["entropy"][:, kk],
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
for ii, gindex in enumerate([0, 12, 20]):
    axs[ii].plot(
        vals["spin"],
        (res["energy"][gindex, :] - res["energy"][gindex, -1])
        / np.abs(res["energy"][gindex, -1]),
        "o-",
        c=colors[ii],
        markersize=marker_size + 2,
        markeredgecolor=colors[ii],
        markerfacecolor="white",
        markeredgewidth=0.5,
        label=vals["g"][gindex],
    )
    axs[ii].set(ylabel="energy diff $\Delta E$", yscale="log")
axs[-1].set(xlabel=r"spin $s$")
fig.legend(
    loc="upper center",
    bbox_to_anchor=(0.55, 1.05),
    ncol=3,
    frameon=False,
    fontsize=10,
)
# %%
