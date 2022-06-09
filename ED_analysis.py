# %%
import numpy as np
import pickle
from copy import deepcopy
import re
from Hamitonian_Functions.QMB_Operations.Mappings_1D_2D import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.backend_bases import register_backend

register_backend("pdf", "matplotlib.backends.backend_pgf")

path_simulations = "/Users/giovannicataldi/Dropbox/PhD/Models/"

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
    "legend": {"fontsize": 10, "title_fontsize": 10},
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
# EXACT DIAGONALIZATION
x_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
data = {
    "params": {
        "theory": ["Full", "Pure"],
        "BC": ["OBC", "PBC"],
        "size": [(2, 2), (3, 2), (4, 2)],
        "mass": ["%.2f" % elem for elem in np.arange(0.2, 1.1, 0.2)],
        "g_SU2": {
            "label": r"$g_{SU(2)}^{2}$",
        },
    },
    "Full": {
        "results": {
            "energy": {
                "label": r"$E_{GS}$",
                "OBC_vals": [],
                "PBC_vals": [],
            },
            "gamma": {
                "label": r"$\avg{\Gamma}$",
                "OBC_vals": [],
                "PBC_vals": [],
            },
            "plaq": {
                "label": r"$\avg{C}$",
                "OBC_vals": [],
                "PBC_vals": [],
            },
            "n_single_EVEN": {
                "label": r"$\avg{n_{\text{single}}}_{+}$",
                "OBC_vals": [],
                "PBC_vals": [],
            },
            "n_single_ODD": {
                "label": r"$\avg{n_{\text{single}}}_{-}$",
                "OBC_vals": [],
                "PBC_vals": [],
            },
            "n_pair_EVEN": {
                "label": r"$\avg{n_{\text{pair}}}_{+}$",
                "OBC_vals": [],
                "PBC_vals": [],
            },
            "n_pair_ODD": {
                "label": r"$\avg{n_{\text{pair}}}_{-}$",
                "OBC_vals": [],
                "PBC_vals": [],
            },
        },
        "syms_sectors": {
            "label": r"$\Delta E=\abs{E_{\Delta N}-E_{0}}$",
            "list_sectors": [-4, -2, 0, 2, 4],
            "vals": [],
            "labels_plots": [
                r"$\Delta N=-4$",
                r"$\Delta N=-2$",
                r"$\Delta N=+2$",
                r"$ \Delta N=+4$",
            ],
        },
    },
    "Pure": {
        "results": {
            "energy": {
                "label": r"$E_{GS}$",
                "OBC_vals": [],
                "PBC_vals": [],
            },
            "gamma": {
                "label": r"$\avg{\Gamma}$",
                "OBC_vals": [],
                "PBC_vals": [],
            },
            "plaq": {
                "label": r"$\avg{C}$",
                "OBC_vals": [],
                "PBC_vals": [],
            },
        },
    },
}

ED_folder = (
    "/Users/giovannicataldi/Dropbox/PhD/Models/Results/SU2/Exact_Diagonalization"
)


# %%
# FULL HAMILTONIAN
theory = data["params"]["theory"][0]

for BC in data["params"]["BC"]:
    data_path = f"{ED_folder}/{theory}/{BC}"
    for ii, mass in enumerate(data["params"]["mass"]):
        # ----------------------------------------------------------------------
        # ACQUIRE DATA FROM SIMULATION FILE
        simulation = get_simulation_data(f"{data_path}/Simulation_m_{mass}.txt")
        data["params"]["g_SU2"]["vals"] = simulation["x"]
        # ----------------------------------------------------------------------
        # GET Y DATA
        for jj, obs in enumerate(data[theory]["results"].keys()):
            data[theory]["results"][obs][f"{BC}_vals"].append(simulation[f"y_{jj+1}"])

# %%
# SU2 FULL HAMILTONIAN PLOTS
fig, axs = plt.subplots(
    3, 2, figsize=set_size(textwidth, subplots=(3, 2)), sharex=False
)

for ii, BC in enumerate(data["params"]["BC"]):
    for jj, ax in enumerate(axs[:, ii]):
        # GET OBSERVABLES NAMES
        obs = list(data[theory]["results"].keys())[jj]
        # SET LABELS
        if ii == 0:
            ax.set(ylabel=data[theory]["results"][obs]["label"])
        if jj == 0:
            ax.set_title(BC, backgroundcolor="black", color="white")
        elif jj == 2:
            ax.set(xlabel=data["params"]["g_SU2"]["label"])
        # SET SCALE
        ax.set(xscale="log")
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        # PLOT GRID
        ax.grid()
        # PLOT CURVES
        for kk, mass in enumerate(data["params"]["mass"]):
            ax.plot(
                np.square(data["params"]["g_SU2"]["vals"]),
                data[theory]["results"][obs][f"{BC}_vals"][kk],
                label=r"$m=" + format(float(mass), ".1f") + "$",
                linewidth=1,
            )
        # PLOT LEGEND
        ax.legend()
# SET COMMON RANGES IN Y AXIS
for ax1, ax2 in axs[:]:
    ax1.set_ylim(ax2.get_ylim())
plt.tight_layout()
plt.savefig("SU2_ED_Full_gauge_fields.pdf", **default_params["save_plot"])


# %%
fig, axs = plt.subplots(
    4, 2, figsize=set_size(textwidth, subplots=(4, 2)), sharex=False
)

for ii, BC in enumerate(data["params"]["BC"]):
    for jj, ax in enumerate(axs[:, ii]):
        # GET OBSERVABLES NAMES
        obs = list(data[theory]["results"].keys())[jj + 3]
        # SET LABELS
        if ii == 0:
            ax.set(ylabel=data[theory]["results"][obs]["label"])
        if jj == 0:
            ax.set_title(BC, backgroundcolor="black", color="white")
        elif jj == 3:
            ax.set(xlabel=data["params"]["g_SU2"]["label"])
        # SET SCALE
        ax.set(xscale="log")
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        # PLOT GRID
        ax.grid()
        # PLOT CURVES
        for kk, mass in enumerate(data["params"]["mass"]):
            ax.plot(
                np.square(data["params"]["g_SU2"]["vals"]),
                data[theory]["results"][obs][f"{BC}_vals"][kk],
                label=r"$m=" + format(float(mass), ".1f") + "$",
                linewidth=1.3,
            )
        # PLOT LEGEND
        ax.legend()
# SET COMMON RANGES IN Y AXIS
for ax1, ax2 in axs[:]:
    ax1.set_ylim(ax2.get_ylim())
plt.tight_layout()
plt.savefig("SU2_ED_Full_Number_Operators.pdf", **default_params["save_plot"])

# %%
# SYMMETRY SECTORS
BC = "OBC"
# SYMMETRY SECTORS of the HAMILTONIAN
data_path = f"{ED_folder}/{theory}/{BC}/Sym_sectors_m_0.1.txt"
sym_sectors = get_simulation_data(data_path, whitespaces=True)
data["params"]["g_SU2"]["vals"] = sym_sectors["x"]


# for ii, sec in enumerate(data[theory]["syms_sectors"]["list_sectors"]):
#    data[theory]["syms_sectors"]["vals"].append(sym_sectors[f"y_{ii+1}"])

ground_state = data[theory]["syms_sectors"]["vals"][2]
excited_sectors = [0, 1, 3, 4]

fig = plt.figure(figsize=set_size(textwidth))
shape = ["-", "-", "--", "--"]

for ii, sec in enumerate(excited_sectors):
    plt.plot(
        np.square(data["params"]["g_SU2"]["vals"]),
        np.abs(np.subtract(data[theory]["syms_sectors"]["vals"][sec], ground_state)),
        shape[ii],
        linewidth=2,
        label=data[theory]["syms_sectors"]["labels_plots"][ii],
    )
plt.xscale("log")
plt.grid()
plt.xlabel(data["params"]["g_SU2"]["label"])
plt.ylabel(data[theory]["syms_sectors"]["label"])
plt.legend(fontsize=15)
plt.savefig("SU2_ED_Full_Syms_Sec.pdf", **default_params["save_plot"])

# %%
# SU2 PURE HAMILTONIAN
theory = "Pure"
for BC in data["params"]["BC"]:
    data_path = f"{ED_folder}/{theory}/{BC}"
    for ii, (x, y) in enumerate(data["params"]["size"]):
        # ----------------------------------------------------------------------
        # ACQUIRE DATA FROM SIMULATION FILE
        simulation = get_simulation_data(f"{data_path}/Simulation_{x}x{y}.txt")
        data["params"]["g_SU2"]["vals"] = simulation["x"]
        # ----------------------------------------------------------------------
        # GET Y DATA
        for jj, obs in enumerate(data[theory]["results"].keys()):
            data[theory]["results"][obs][f"{BC}_vals"].append(simulation[f"y_{jj+1}"])

# %%
# SU2 PURE HAMILTONIAN PLOTS
fig, axs = plt.subplots(
    3, 2, figsize=set_size(textwidth, subplots=(3, 2)), sharex=False
)

for ii, BC in enumerate(data["params"]["BC"]):
    for jj, ax in enumerate(axs[:, ii]):
        # GET OBSERVABLES NAMES
        obs = list(data[theory]["results"].keys())[jj]
        # SET LABELS
        if ii == 0:
            ax.set(ylabel=data[theory]["results"][obs]["label"])
        if jj == 0:
            ax.set_title(BC, backgroundcolor="black", color="white")
        elif jj == 2:
            ax.set(xlabel=data["params"]["g_SU2"]["label"])
        # SET SCALE
        ax.set(xscale="log")
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        # PLOT GRID
        ax.grid()
        # PLOT CURVES
        for kk, (x, y) in enumerate(data["params"]["size"]):
            ax.plot(
                np.square(data["params"]["g_SU2"]["vals"]),
                data[theory]["results"][obs][f"{BC}_vals"][kk],
                label=r"$" + str(x) + "\\times" + str(y) + "$",
                linewidth=1.3,
            )
        # PLOT LEGEND
        ax.legend()
# SET COMMON RANGES IN Y AXIS
for ax1, ax2 in axs[:]:
    ax1.set_ylim(ax2.get_ylim())
plt.tight_layout()
plt.savefig("SU2_ED_Pure_gauge_fields.pdf", **default_params["save_plot"])
