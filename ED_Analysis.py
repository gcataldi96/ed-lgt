# %%
import numpy as np
from simsio import *
import pickle
from copy import deepcopy
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backend_bases import register_backend


def save_dictionary(dict, filename):
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(dict, outp, pickle.HIGHEST_PROTOCOL)
    outp.close


@plt.FuncFormatter
def fake_log(x, pos):
    "The two args are the value and tick position"
    return r"$10^{%d}$" % (x)


register_backend("pdf", "matplotlib.backends.backend_pgf")

path_simulations = "Results/SU2_Full_PBC"

latex_preamble = [
    r"\usepackage{amsmath}",
    r"\usepackage{amsfonts}",
    r"\usepackage{amssymb}",
    r"\usepackage{amsthm}",
    r"\usepackage{mathtools}",
    r"\usepackage{physics}",
    r"\newcommand{\avg}[1]{\left\langle#1\right\rangle}",
]

textwidth = 455.24411


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

    return (fig_width_in, fig_height_in)


plot_params = {
    "font": {"family": "serif", "serif": ["Helvetica"], "size": 10},
    "text": {"usetex": True, "latex.preamble": "\n".join(latex_preamble)},
    "pgf": {
        "texsystem": "xelatex",  # "pdflatex"
        "rcfonts": False,
        "preamble": "\n".join(latex_preamble),
    },
    "xtick": {"labelsize": 15},
    "ytick": {"labelsize": 15},
    "legend": {"fontsize": 12, "title_fontsize": 10},
    "axes": {"labelsize": 15, "titlesize": 10},
    "figure": {"titlesize": 10},
    "savefig": {"format": "pdf", "directory": path_simulations},
}

for key in plot_params.keys():
    plt.rc(key, **plot_params[key])

default_params = {
    "save_plot": {"bbox_inches": "tight", "transparent": True},
}

# %%
# ACQUIRE SIMULATION RESULTS
match = SimsQuery(group_glob="M_G_new")

ugrid, vals = uids_grid(match.uids, ["mass", "gSU2"])

res = {}
res["energy"] = np.vectorize(extract_dict)(ugrid, key="res", glob="energy")
res["gamma"] = np.vectorize(extract_dict)(ugrid, key="res", glob="gamma")
res["plaq"] = np.vectorize(extract_dict)(ugrid, key="res", glob="plaq")
res["n_single_EVEN"] = np.vectorize(extract_dict)(
    ugrid, key="res", glob="n_single_EVEN"
)
res["n_single_ODD"] = np.vectorize(extract_dict)(ugrid, key="res", glob="n_single_ODD")
res["n_pair_EVEN"] = np.vectorize(extract_dict)(ugrid, key="res", glob="n_pair_EVEN")
res["n_pair_ODD"] = np.vectorize(extract_dict)(ugrid, key="res", glob="n_pair_ODD")

res["n_tot_EVEN"] = np.vectorize(extract_dict)(ugrid, key="res", glob="n_tot_EVEN")
res["n_tot_ODD"] = np.vectorize(extract_dict)(ugrid, key="res", glob="n_tot_ODD")

res["params"] = vals

save_dictionary(res, "ED_2x2_M_G_grid.pkl")

# %%
# LOADING SIMULATION RESULTS
with open("ED_2x2_M_G_grid.pkl", "rb") as dict:
    ED_data = pickle.load(dict)

obs_labels = [
    r"$E_{GS}$",
    r"$\avg{\Gamma}$",
    r"$\avg{C}$",
    r"$\avg{n_{\text{single}}}_{+}$",
    r"$\avg{n_{\text{single}}}_{-}$",
    r"$\avg{n_{\text{pair}}}_{+}$",
    r"$\avg{n_{\text{pair}}}_{-}$",
    r"$\avg{n_{\text{tot}}}_{+}$",
    r"$\avg{n_{\text{tot}}}_{-}$",
]

obs_names = [
    "energy",
    "gamma",
    "plaq",
    "n_single_EVEN",
    "n_single_ODD",
    "n_pair_EVEN",
    "n_pair_ODD",
    "n_tot_EVEN",
    "n_tot_ODD",
]
# SCALING TOTAL MASS
y = ED_data["n_tot_EVEN"].T

gSU2 = np.square(np.logspace(-2, 1, 30))
sm = cm.ScalarMappable(cmap="RdYlBu_r", norm=LogNorm())
palette = sm.to_rgba(gSU2)

fig, ax = plt.subplots(1, 1, figsize=set_size(textwidth, subplots=(1, 1)))
for ii, g in enumerate(gSU2):
    ax.plot(
        ED_data["params"]["mass"][30:],
        y[ii][30:],
        "o-",
        c=palette[ii],
        linewidth=1.2,
        markersize=5,
        markerfacecolor="white",
    )
ax.set(xscale="log", xlabel=r"$m$", ylabel=r"$\avg{n_{\text{tot}}}_{+}$")
ax.set_xticks(ticks=[1e-5, 1e-3, 1e-1])
ax.grid()
plt.colorbar(sm, label=r"$g_{SU(2)}$")
plt.savefig("SU2_ED_n_tot_PT.pdf", **default_params["save_plot"])
# %%
# SCALING OBSERVABLE VARYING THE MASS
y = ED_data["n_tot_EVEN"].T


gSU2 = np.square(np.logspace(-2, 1, 30))
sm = cm.ScalarMappable(cmap="RdYlBu_r", norm=LogNorm())
palette = sm.to_rgba(gSU2[10:])

fig, axs = plt.subplots(
    1,
    2,
    figsize=set_size(textwidth, subplots=(1, 2)),
    sharex=False,
    sharey=True,
    constrained_layout=True,
)

for ii in range(20):
    axs[0].plot(
        ED_data["params"]["mass"][30:],
        y[10 + ii][:30],
        "o-",
        c=palette[ii],
        linewidth=1,
        markersize=2,
        markerfacecolor="white",
    )

axs[0].set(xscale="log", xlabel=r"$m$", ylabel=r"$\avg{n_{\text{tot}}}_{+}$")
axs[0].set_xticks(
    ticks=[1e-5, 1e-3, 1e-1], labels=[r"$-10^{-1}$", r"$-10^{-3}$", r"$-10^{-5}$"]
)
axs[0].grid()


for ii in range(20):
    axs[1].plot(
        ED_data["params"]["mass"][30:],
        y[10 + ii][30:],
        "o-",
        c=palette[ii],
        linewidth=1,
        markersize=2,
        markerfacecolor="white",
    )
axs[1].set(xscale="log", xlabel=r"$m$")
axs[1].set_xticks(ticks=[1e-5, 1e-3, 1e-1])
axs[1].grid()
plt.colorbar(sm, label=r"$g_{SU(2)}$")
plt.savefig("SU2_ED_n_tot_+-mass.pdf", **default_params["save_plot"])

# %%
# # UNIVERSAL SCALING
x_plus = ED_data["params"]["mass"][30:]
x_minus = ED_data["params"]["mass"][:30]

fig, axs = plt.subplots(
    1,
    2,
    figsize=set_size(textwidth, subplots=(1, 2)),
    sharex=False,
    sharey=True,
    constrained_layout=True,
)

n_offset_list = []
for ii in range(20):
    y_minus = (ED_data["n_tot_EVEN"].T)[10 + ii][:30]
    x_fit = np.log10(-x_minus[np.nonzero((y_minus > 1.2) & (y_minus < 1.8))])
    y_minus_fit = y_minus[np.nonzero((y_minus > 1.2) & (y_minus < 1.8))]
    m, q = np.polyfit(x_fit, y_minus_fit, 1)
    y_fit = np.poly1d([m, q])
    offset = (1.5 - q) / m
    n_offset_list.append(offset)
    axs[0].plot(
        x_minus / 10 ** (offset),
        y_minus,
        "o-",
        c=palette[ii],
        linewidth=1,
        markersize=2,
        markerfacecolor="white",
    )
axs[0].set_xscale("symlog", linthresh=0.0001)

axs[0].set_xlabel(r"$\frac{m}{m^{*}(g_{SU(2)})}$")
axs[0].set_ylabel(r"$\avg{n_{\text{tot}}}_{+}$")
axs[0].grid()


p_offset_list = []
for ii in range(20):
    y_plus = (ED_data["n_tot_EVEN"].T)[10 + ii][30:]
    x_fit = np.log10(x_plus[np.nonzero((y_plus > 0.35) & (y_plus < 0.65))])
    y_plus_fit = y_plus[np.nonzero((y_plus > 0.35) & (y_plus < 0.65))]
    m, q = np.polyfit(x_fit, y_plus_fit, 1)
    y_fit = np.poly1d([m, q])
    offset = (0.5 - q) / m
    p_offset_list.append(offset)

    axs[1].plot(
        x_plus / 10 ** (offset),
        y_plus,
        "o-",
        c=palette[ii],
        linewidth=1,
        markersize=2,
        markerfacecolor="white",
    )
axs[1].set(xscale="log", xlabel=r"$\frac{m}{m^{*}(g_{SU(2)})}$")
axs[1].grid()
plt.colorbar(sm, label=r"$g_{SU(2)}$")
plt.sca(axs[0])
plt.xticks([-1e3, -1e0, -1e-3])
plt.yticks([0, 1, 2])
plt.sca(axs[1])
plt.xticks([1e-3, 1e-0, 1e3])
plt.savefig("SU2_ED_universal_n_tot.pdf", **default_params["save_plot"])

# %%
# POWER LAW SCALING OF UNIVERSAL MASS
m0 = 10 ** np.array(n_offset_list)
fig, ax = plt.subplots(1, 1, figsize=set_size(textwidth, subplots=(1, 1)))
ax.plot(
    gSU2[10:],
    m0,
    "ro-",
    linewidth=1.3,
    markersize=5,
    markerfacecolor="white",
)

ax.set(xscale="log", yscale="log", ylabel=r"$m^{*}(g_{SU(2)})$", xlabel=r"$g_{SU(2)}$")
m, q = np.polyfit(np.log(gSU2[10:20]), np.log(m0[:10]), 1)
pow_law_fit = np.poly1d([m, q])
ax.text(0.25, 0.005, r"$m^{*}\sim 10^{-3}\cdot g^{1.09}$", fontsize=20)
ax.plot(gSU2[10:20], np.exp(pow_law_fit(np.log(gSU2[10:20]))), "b-", linewidth=1.2)
plt.grid()
plt.savefig("SU2_ED_universal_mass_powerlaw.pdf", **default_params["save_plot"])

# %%
# MASS AND COUPLING GRID
fig, axs = plt.subplots(
    4,
    2,
    figsize=set_size(textwidth, subplots=(3, 2)),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

for ii, ax in enumerate(axs.flat):
    # IMSHOW
    img = ax.imshow(
        ED_data[obs_names[ii + 1]][30:],
        origin="lower",
        cmap="magma",
        extent=[-4, 2, -6, 0],
    )
    # SET AXES LABELS
    if ii in [0, 2, 4, 6]:
        ax.set_ylabel(r"$m$")
    if ii in [6, 7]:
        ax.set_xlabel(r"$g_{SU(2)}^{2}$")
    ax.set(yticks=[-5, -3, -1], xticks=[-3, -1, 1])
    ax.xaxis.set_major_formatter(fake_log)
    ax.yaxis.set_major_formatter(fake_log)

    # BUILD THE COLORMAP
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="15%", pad=0.15)
    fig.colorbar(img, cax=cax, orientation="vertical", label=obs_labels[ii + 1])
plt.savefig("SU2_ED_grid_positive_masses.pdf", **default_params["save_plot"])

# %%
# ACQUIRE SIMULATION DATA FROM FILES
def get_simulation_data(data_file_name, whitespaces=False, first_line_labels=True):
    # CHECK ON TYPES
    if not isinstance(data_file_name, str):
        raise TypeError(f"data_file_name be a STR, not a {type(data_file_name)}")
    if not isinstance(whitespaces, bool):
        raise TypeError(f"whitespaces must be a BOOL, not a {type(whitespaces)}")
    if not isinstance(first_line_labels, bool):
        raise TypeError(
            f"first_line_labels must be a BOOL, not a {type(first_line_labels)}"
        )
    # Open the file and acquire all the lines
    f = open(data_file_name, "r+")
    line = f.readlines()
    f.close()

    # CREATE A DICTIONARY TO HOST THE LISTS OBTAINED FROM EACH COLUMN OF data_file_name
    data = {}
    # Generate a list for the values of the first column of the file
    data["x"] = list()
    # Get the first line of the File as a list of entries.
    if not whitespaces:
        n = line[0].strip().split(",")
    else:
        n = re.sub("\s+", ",", line[0].strip())
        n = n.split(",")

    if first_line_labels:
        # Get the x_name from the first line
        data["x_label"] = str(n[0])
        # IGNORE THE FIRST LINE OF line (ALREAY USED FOR THE LABELS)
        del line[0]

    for ii in range(1, len(n)):
        # Generate a list for each column of data_file
        data[f"y_{str(ii)}"] = list()
        # Generate a label for each list acquiring the ii+1 entry of the first line n
        data[f"y_label_{str(ii)}"] = str(n[ii])

    # Fill the lists with the entries of Columns
    for ii in range(len(line)):
        if not whitespaces:
            a = line[ii].strip().split(",")
        else:
            a = re.sub("\s+", ",", line[ii].strip())
            a = a.split(",")
        # x AXIS
        data["x"].append(float(a[0]))
        # y AXIS
        for jj in range(1, len(n)):
            data[f"y_{str(jj)}"].append(float(a[jj]))

    # CONVERT THE LISTS INTO ARRAYS
    for ii in range(1, len(n)):
        data[f"y_{str(ii)}"] = np.asarray(data[f"y_{str(ii)}"])

    data["x"] = np.asarray(data["x"])

    return data


# %%
# PHASE TRANSITION VARYING THE MASS
mass_labels = [
    r"$m=-10^{-5}$",
    r"$m=-10^{-4}$",
    r"$m=-10^{-3}$",
    r"$m=-10^{-2}$",
    r"$m=-10^{-1}$",
    r"$m=2\cdot10^{-1}$",
    r"$m=4\cdot10^{-1}$",
    r"$m=6\cdot10^{-1}$",
    r"$m=8\cdot10^{-1}$",
    r"$m=10^{0}$",
]

params = {
    "mass": {
        "values": [
            "%.5f" % elem
            for elem in [
                -0.00001,
                -0.0001,
                -0.001,
                -0.01,
                -0.1,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ]
        ],
        "label": r"$m$",
    },
    "g_SU2": {
        "values": np.logspace(start=-1, stop=1, num=30),
        "label": r"$g_{SU(2)}^{2}$",
    },
}

results = {
    "energy": [],
    "gamma": [],
    "plaq": [],
    "n_single_EVEN": [],
    "n_single_ODD": [],
    "n_pair_EVEN": [],
    "n_pair_ODD": [],
    "n_tot_EVEN": [],
    "n_tot_ODD": [],
}

for ii, m in enumerate(params["mass"]["values"][:5]):
    # ----------------------------------------------------------------------
    # ACQUIRE DATA FROM SIMULATION FILE
    simulation = get_simulation_data(f"{path_simulations}/Simulation_m_{m}.txt")
    # ----------------------------------------------------------------------
    # GET Y DATA
    for jj, obs in enumerate(obs_names):
        results[obs].append(simulation[f"y_{jj+1}"])

fig, axs = plt.subplots(
    3,
    2,
    figsize=set_size(textwidth, subplots=(3, 2)),
    sharex=True,
)
x_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
for jj, ax in enumerate(axs.flat):
    # SET LABELS
    ax.set(ylabel=obs_labels[jj + 3])
    if jj > 3:
        ax.set(xlabel=params["g_SU2"]["label"])
    # SET SCALE
    ax.set(xscale="log")
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    # PLOT GRID
    ax.grid()
    # PLOT CURVES
    for kk, m in enumerate(params["mass"]["values"][:5]):
        ax.plot(
            np.square(params["g_SU2"]["values"]),
            results[obs_names[jj + 3]][kk],
            "-o",
            label=mass_labels[kk],
            linewidth=1,
            markersize=3,
            c=palette[kk],
            markerfacecolor="white",
        )

fig.legend(
    mass_labels[:5],
    loc="lower right",
    bbox_to_anchor=(1.15, 0.74),
    ncol=1,
    bbox_transform=fig.transFigure,
)
plt.tight_layout()
plt.savefig("SU2_ED_Negative_Masses.pdf", **default_params["save_plot"])

# %%
# SU2 PURE HAMILTONIAN
params = {
    "BC_list": ["OBC", "PBC"],
    "sizes": [[2, 2], [3, 2], [4, 2]],
    "g_SU2": {
        "values": np.logspace(start=-1, stop=1, num=30),
        "label": r"$g_{SU(2)}^{2}$",
    },
}
results = {
    "OBC": {"energy": [], "gamma": [], "plaq": []},
    "PBC": {"energy": [], "gamma": [], "plaq": []},
}


for BC in params["BC_list"]:
    data_path = f"Results/SU2_Pure_{BC}"
    for ii, (x, y) in enumerate(params["sizes"]):
        # ----------------------------------------------------------------------
        # ACQUIRE DATA FROM SIMULATION FILE
        simulation = get_simulation_data(f"{data_path}/Simulation_{x}x{y}.txt")
        # ----------------------------------------------------------------------
        # GET Y DATA
        for jj, obs in enumerate(results[BC].keys()):
            results[BC][obs].append(simulation[f"y_{jj+1}"])

# SU2 PURE HAMILTONIAN PLOTS
fig, axs = plt.subplots(3, 2, figsize=set_size(textwidth, subplots=(3, 2)), sharex=True)

for ii, BC in enumerate(params["BC_list"]):
    for jj, ax in enumerate(axs[:, ii]):
        # SET LABELS
        if ii == 0:
            ax.set(ylabel=obs_labels[jj])
        if jj == 0:
            ax.set_title(BC, backgroundcolor="black", color="white", fontsize=12)
        elif jj == 2:
            ax.set(xlabel=params["g_SU2"]["label"])
        # SET SCALE
        ax.set(xscale="log")
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        # PLOT GRID
        ax.grid()
        # PLOT CURVES
        for kk, (x, y) in enumerate(params["sizes"]):
            ax.plot(
                np.square(params["g_SU2"]["values"]),
                results[BC][list(results[BC].keys())[jj]][kk],
                "o-",
                linewidth=1.3,
                markersize=3,
                markerfacecolor="white",
            )
fig.legend(
    [r"$" + str(x) + "\\times" + str(y) + "$" for (x, y) in params["sizes"]],
    bbox_to_anchor=(0.45, 0.9),
    ncol=1,
    fontsize=15,
    bbox_transform=fig.transFigure,
)
# SET COMMON RANGES IN Y AXIS
for ax1, ax2 in axs[:]:
    ax1.set_ylim(ax2.get_ylim())
plt.tight_layout()
plt.savefig("SU2_ED_Pure_gauge_fields.pdf", **default_params["save_plot"])

# %%
# SU2 FULL HAMILTONIAN
x_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
params = {
    "BC_list": ["OBC", "PBC"],
    "mass": {
        "values": [0.2, 0.4, 0.6, 0.8, 1.0],
        "label": [
            r"$m=2\cdot10^{-1}$",
            r"$m=4\cdot10^{-1}$",
            r"$m=6\cdot10^{-1}$",
            r"$m=8\cdot10^{-1}$",
            r"$m=10^{0}$",
        ],
    },
    "g_SU2": {
        "values": np.logspace(start=-1, stop=1, num=30),
        "label": r"$g_{SU(2)}^{2}$",
    },
}

results = {
    "OBC": {
        "energy": [],
        "gamma": [],
        "plaq": [],
        "n_single_EVEN": [],
        "n_single_ODD": [],
        "n_pair_EVEN": [],
        "n_pair_ODD": [],
        "n_tot_EVEN": [],
        "n_tot_ODD": [],
    },
    "PBC": {
        "energy": [],
        "gamma": [],
        "plaq": [],
        "n_single_EVEN": [],
        "n_single_ODD": [],
        "n_pair_EVEN": [],
        "n_pair_ODD": [],
        "n_tot_EVEN": [],
        "n_tot_ODD": [],
    },
}


for BC in params["BC_list"]:
    data_path = f"Results/SU2_Full_{BC}"
    for ii, m in enumerate(params["mass"]["values"]):
        # ----------------------------------------------------------------------
        # ACQUIRE DATA FROM SIMULATION FILE
        simulation = get_simulation_data(
            f"{data_path}/Simulation_m_{format(m,'.2f')}.txt"
        )
        # ----------------------------------------------------------------------
        # GET Y DATA
        for jj, obs in enumerate(obs_names):
            results[BC][obs].append(simulation[f"y_{jj+1}"])

# SU2 PURE HAMILTONIAN PLOTS
fig, axs = plt.subplots(1, 2, figsize=set_size(textwidth, subplots=(1, 2)), sharex=True)


for jj, ax in enumerate(axs.flat):
    # SET LABELS
    ax.set(ylabel=obs_labels[jj + 1])
    if jj > 1:
        ax.set(xlabel=params["g_SU2"]["label"])
    # SET SCALE
    ax.set(xscale="log")
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    # PLOT GRID
    ax.grid()
    # PLOT CURVES
    for kk, m in enumerate(params["mass"]["values"]):
        ax.plot(
            np.square(params["g_SU2"]["values"]),
            results["PBC"][obs_names[jj + 1]][kk],
            "o-",
            linewidth=1,
            markersize=3,
            markerfacecolor="white",
        )
fig.legend(
    params["mass"]["label"],
    bbox_to_anchor=(1.1, 0.9),
    ncol=1,
    fontsize=10,
    bbox_transform=fig.transFigure,
)
plt.tight_layout()
plt.savefig("SU2_ED_Full_gauge_fields.pdf", **default_params["save_plot"])

# %%
fig, ax = plt.subplots(1, 1, figsize=set_size(textwidth, subplots=(1, 1)))

# SET LABELS
ax.set(ylabel=obs_labels[0])
ax.set(xlabel=params["g_SU2"]["label"])
# SET SCALE
ax.set(xscale="log")
ax.xaxis.set_minor_locator(x_minor)
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
# PLOT GRID
# ax.grid()
# PLOT CURVES

left, bottom, width, height = [0.4, 0.25, 0.45, 0.45]
ax2 = fig.add_axes([left, bottom, width, height])

for kk, m in enumerate(params["mass"]["values"]):
    ax.plot(
        np.square(params["g_SU2"]["values"]),
        results["PBC"][obs_names[0]][kk],
        "o-",
        linewidth=1,
        markersize=3,
        markerfacecolor="white",
    )
    ax2.plot(
        np.square(params["g_SU2"]["values"]),
        results["PBC"][obs_names[0]][kk],
        "o-",
        linewidth=1,
        markersize=3,
        markerfacecolor="white",
    )

ax2.set(yscale="symlog", xscale="log")
plt.savefig("SU2_ED_Full_energy.pdf", **default_params["save_plot"])
# %%
fig, axs = plt.subplots(3, 2, figsize=set_size(textwidth, subplots=(3, 2)), sharex=True)


for jj, ax in enumerate(axs.flat):
    # SET LABELS
    ax.set(ylabel=obs_labels[jj + 3])
    if jj > 3:
        ax.set(xlabel=params["g_SU2"]["label"])
    # SET SCALE
    ax.set(xscale="log")
    ax.xaxis.set_minor_locator(x_minor)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    # PLOT GRID
    ax.grid()
    # PLOT CURVES
    for kk, m in enumerate(params["mass"]["values"]):
        ax.plot(
            np.square(params["g_SU2"]["values"]),
            results["PBC"][obs_names[jj + 3]][kk],
            "o-",
            linewidth=1.3,
            markersize=3,
            markerfacecolor="white",
        )
fig.legend(
    params["mass"]["label"],
    bbox_to_anchor=(1.17, 0.3),
    ncol=1,
    fontsize=13,
    bbox_transform=fig.transFigure,
)
# SET COMMON RANGES IN Y AXIS
plt.tight_layout()

plt.savefig("SU2_ED_Full_numbers.pdf", **default_params["save_plot"])
