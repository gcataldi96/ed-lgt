# %%
from simsio import *
from math import prod
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, FuncFormatter
from ed_lgt.tools import save_dictionary, first_derivative


@plt.FuncFormatter
def fake_log(x, pos):
    "The two args are the value and tick position"
    return r"$10^{%d}$" % (x)


# Z2 HUBBARD WITH FINITE ELECTRIC FIELD
# List of local observables
local_obs = [f"n_{s}{d}" for d in "xy" for s in "mp"]
local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
local_obs += ["X_Cross", "S2_psi", "E"]
plaq_name = "plaq"
# %%
res = {"finiteE": {}, "zeroE": {}}
# Acquire simulations of finite E field
config_filename = f"Z2FermiHubbard/PBCxy/new_test"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["U", "h"])
for obs in local_obs + ["energy", plaq_name]:
    res["finiteE"][obs] = np.zeros((len(vals["U"]), len(vals["h"])), dtype=float)
for ii, U in enumerate(vals["U"]):
    for jj, h in enumerate(vals["h"]):
        print(ii, jj, U, h)
        res["finiteE"]["energy"][ii, jj] = get_sim(ugrid[ii, jj]).res["energy"] / 4
        for obs in local_obs + [plaq_name]:
            res["finiteE"][obs][ii, jj] = get_sim(ugrid[ii, jj]).res[obs]
# %%
# Acquire simulations at zero E field
config_filename = f"Z2FermiHubbard/PBCxy/test"
match = SimsQuery(group_glob=config_filename)
ugrid1, vals1 = uids_grid(match.uids, ["U"])
for obs in local_obs + ["energy", plaq_name]:
    res["zeroE"][obs] = np.zeros(len(vals1["U"]), dtype=float)
for ii, U in enumerate(vals["U"]):
    res["zeroE"]["energy"][ii] = get_sim(ugrid1[ii]).res["energy"] / 4
    for obs in local_obs + [plaq_name]:
        res["zeroE"][obs][ii] = get_sim(ugrid1[ii]).res[obs]
save_dictionary(res, f"res_Z2Hubbard.pkl")
# ============================================================================
# Imshow
obs_name = "X_Cross"
fig, axs = plt.subplots(
    1,
    1,
    sharex=True,
    constrained_layout=True,
)
img = axs.imshow(
    np.transpose(res["finiteE"][obs_name]),
    origin="lower",
    cmap="magma",
    extent=[-1, 2, -1, 1],
)
axs.set(xticks=[-1, 0, 1, 2], yticks=[-1, 0, 1], xlabel="U", ylabel="h")
axs.xaxis.set_major_formatter(fake_log)
axs.yaxis.set_major_formatter(fake_log)

cb = fig.colorbar(
    img,
    ax=axs,
    aspect=20,
    location="right",
    orientation="vertical",
    pad=0.01,
    label=obs_name,
)
# Single Curves
sm = cm.ScalarMappable(cmap="plasma", norm=LogNorm())
palette = sm.to_rgba(vals["h"])
fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.set(ylabel=obs_name, xscale="log", xlabel="$U$")
for jj, h in enumerate(vals["h"]):
    axs.plot(
        vals["U"],
        res["finiteE"][obs_name][:, jj],
        "o-",
        linewidth=1,
        markersize=3,
        c=palette[jj],
        markerfacecolor="black",
    )
# Zero E field
axs.plot(
    vals["U"],
    res["zeroE"][obs_name],
    "x-",
    linewidth=1.5,
    markersize=7,
    c="black",
    markerfacecolor="black",
    label=r"$E=0$",
)
axs.legend(loc=(0.15, 0.11))
cb = fig.colorbar(
    sm, ax=axs, aspect=80, location="top", orientation="horizontal", pad=0.02
)
cb.set_label(label=r"$h$", labelpad=-22, x=-0.02, y=0)
# %%
# Susceptibility
obs_name = "plaq"
sm = cm.ScalarMappable(cmap="plasma", norm=LogNorm())
palette = sm.to_rgba(vals["U"])
fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.grid()
axs.set(ylabel=obs_name + " suscept", xlabel="$h$")
for ii, U in enumerate(vals["U"]):
    df = np.gradient(res["finiteE"][obs_name][ii, :], 0.01)
    axs.plot(
        vals["h"],
        df,
        "o-",
        linewidth=1,
        markersize=3,
        c=palette[ii],
        markerfacecolor="black",
    )
cb = fig.colorbar(
    sm, ax=axs, aspect=80, location="top", orientation="horizontal", pad=0.02
)
cb.set_label(label=r"$U$", labelpad=-22, x=-0.02, y=0)
plt.savefig(f"suscept.pdf")
# %%
from scipy import stats

hmax = np.zeros(len(vals["U"]), dtype=float)
for ii, U in enumerate(vals["U"]):
    df = np.gradient(res["finiteE"][obs_name][ii, :], 0.01)
    hmax[ii] = vals["h"][np.argmax(df)]
fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.plot(
    vals["U"],
    hmax,
    "o-",
    linewidth=1,
    markersize=3,
    c=palette[0],
    markerfacecolor="black",
    label="Data",
)
axs.set(xscale="log", xlabel="U", yscale="log", ylabel=r"$h_{\max}$")
# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    np.log(vals["U"]), np.log(hmax)
)

# Convert slope and intercept back to the original scale
b = slope
a = np.exp(intercept)

axs.plot(vals["U"], a * vals["U"] ** b, label=r"$h_{\max} = a\cdot U^b$", color="red")
# Adjust annotation to fit the data range
# Use values from the plot to pick suitable positions within the visible range
x_annot = np.mean(
    vals["U"]
)  # Use the mean of 'vals["U"]' for a better annotation position
y_annot_a = np.mean(hmax)  # Use the mean of 'hmax' for a reasonable y position for 'a'
y_annot_b = y_annot_a / 2  # Adjust y position for 'b' annotation a bit lower

# Annotate with the values of 'a' and 'b'
axs.annotate(
    f"a = {a:.3f}",
    xy=(x_annot, y_annot_a),
    xytext=(x_annot, y_annot_a * 1.5),
    fontsize=12,
    color="black",
)
axs.annotate(
    f"b = {b:.3f}",
    xy=(x_annot, y_annot_b),
    xytext=(x_annot, y_annot_b * 1.5),
    fontsize=12,
    color="black",
)

# Show legend
axs.legend()
plt.savefig(f"hmax_fit.pdf")
# %%
x = np.arange(0, 10, 0.1)
sin = np.sin(x)
cos = np.gradient(sin, 0.1)
fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.grid()
axs.plot(
    x,
    sin,
    "o-",
    linewidth=1,
    markersize=3,
    c=palette[0],
    markerfacecolor="black",
)
axs.plot(
    x,
    cos,
    "o-",
    linewidth=1,
    markersize=3,
    c=palette[0],
    markerfacecolor="red",
)


# Set x-axis to pi/2 fractions
def format_func(value, tick_number):
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\frac{\pi}{2}$"
    elif N == 2:
        return r"$\pi$"
    elif N == 3:
        return r"$\frac{3\pi}{2}$"
    elif N == 4:
        return r"$2\pi$"
    elif N == -1:
        return r"-$\frac{\pi}{2}$"
    elif N == -2:
        return r"-$\pi$"
    elif N == -3:
        return r"-$\frac{3\pi}{2}$"
    else:
        return r"${0}\pi$".format(N / 2)


axs.xaxis.set_major_locator(
    MultipleLocator(base=np.pi / 2)
)  # Set major ticks at multiples of pi
axs.xaxis.set_major_formatter(
    FuncFormatter(format_func)
)  # Format the ticks as multiples of pi

# Add labels and legend
axs.set_xlabel("x")
axs.set_ylabel("y")
axs.legend()

# %%
# List of local observables
local_obs = [f"n_{s}{d}" for d in "xy" for s in "mp"]
local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
local_obs += ["X_Cross", "S2"]

BC_list = ["PBCxy"]
lattize_size_list = ["2x2", "3x2"]
for ii, BC in enumerate(BC_list):
    res = {}
    # define the observables arrays
    res["energy"] = np.zeros((len(lattize_size_list), 25), dtype=float)
    for obs in local_obs:
        res[obs] = np.zeros((len(lattize_size_list), 25), dtype=float)
    for jj, size in enumerate(lattize_size_list):
        # look at the simulation
        config_filename = f"Z2FermiHubbard/{BC}/{size}"
        match = SimsQuery(group_glob=config_filename)
        ugrid, vals = uids_grid(match.uids, ["U"])
        lvals = get_sim(ugrid[0]).par["model"]["lvals"]
        for kk, U in enumerate(vals["U"]):
            for obs in local_obs:
                res[obs][jj, kk] = np.mean(get_sim(ugrid[kk]).res[obs])
                res["energy"][jj, kk] = get_sim(ugrid[kk]).res["energies"][0] / (
                    prod(lvals)
                )
    save_dictionary(res, f"{BC}.pkl")
# %%
for obs in ["X_Cross", "N_pair", "N_single", "energy", "S2"]:
    print(obs)
    fig = plt.figure()
    plt.ylabel(rf"{obs}")
    plt.xlabel(r"U")
    plt.xscale("log")
    plt.grid()
    for ii, label in enumerate(lattize_size_list):
        plt.plot(vals["U"], res[obs][ii, :], "-o", label=label)
    plt.legend(loc=(0.05, 0.11))
    # plt.savefig(f"{obs}.pdf")
# %%


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set(xscale="log", yscale="log")
config_filename = f"Z2FermiHubbard/entropy"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["U"])
lvals = get_sim(ugrid[0]).par["model"]["lvals"]
entropies = np.zeros((len(vals["U"]), 3))
for kk, U in enumerate(vals["U"]):
    C = get_sim(ugrid[kk]).res["Sz_Sz"][:]
    entropies[kk, :] = get_sim(ugrid[kk]).res["entropy"]
    ax.plot(C[:, 0], C[:, 1], "-o", label=f"U={U}")
ax.set(xlabel="r=|i-j|", ylabel="<SzSz>")
plt.legend()
# %%
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

sm = cm.ScalarMappable(cmap="plasma", norm=LogNorm())
palette = sm.to_rgba(vals["U"])
fig1, ax1 = plt.subplots()
for kk, U in enumerate(vals["U"]):
    ax1.plot(np.arange(3), entropies[kk, :], "-o", color=palette[kk])
ax1.set(xlabel="A", ylabel="EE")
plt.legend()
# %%
config_filename = "Z2_FermiHubbard/entropy"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["has_obc", "U"])

res = {}
# List of local observables
lvals = get_sim(ugrid[0][0]).par["lvals"]
local_obs = [f"n_{s}{d}" for d in "xyz"[: len(lvals)] for s in "mp"]
local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
local_obs += ["X_Cross"]
for obs in local_obs:
    res[obs] = np.zeros((vals["has_obc"].shape[0], vals["U"].shape[0]))

res["energy"] = np.zeros((vals["has_obc"].shape[0], vals["U"].shape[0]))

for ii, has_obc in enumerate(vals["has_obc"]):
    for jj, U in enumerate(vals["U"]):
        res["energy"][ii, jj] = get_sim(ugrid[ii][jj]).res["energies"][0] / (
            prod(lvals)
        )
        for obs in local_obs:
            res[obs][ii, jj] = np.mean(get_sim(ugrid[ii][jj]).res[obs])

fig = plt.figure()
plt.ylabel(r"X_cross")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["X_Cross"][ii, :], "-o", label=BC_label)
plt.legend()

fig = plt.figure()
plt.ylabel(r"N")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["N_pair"][ii, :], "-o", label=f"pair ({BC_label})")
    plt.plot(vals["U"], res["N_single"][ii, :], "-o", label=f"single ({BC_label})")
plt.legend()

fig = plt.figure()
plt.ylabel(r"Energy Density")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["energy"][ii, :], "-o", label=BC_label)
plt.legend(loc="lower right")

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set(xscale="log", yscale="log")
config_filename = f"Z2FermiHubbard/prova1"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["U"])
lvals = get_sim(ugrid[0]).par["model"]["lvals"]
entropies = []
for kk, U in enumerate(vals["U"]):
    C = get_sim(ugrid[kk]).res["C"]
    entropies.append(get_sim(ugrid[kk]).res["entropy"])
    ax.plot(C[:, 0], C[:, 1], "-o", label=f"U={U}")
ax.set(xlabel="r=|i-j|", ylabel="<SzSz>")
plt.legend()

fig1, ax1 = plt.subplots()
for kk, U in enumerate(vals["U"]):
    ax1.plot(np.arange(4), entropies[kk][:4], "-o", label=f"U={U}")
ax1.set(xlabel="A", ylabel="EE")
plt.legend()
# %%
config_filename = "Z2_FermiHubbard/U_potential"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["has_obc", "U"])

res = {}
# List of local observables
lvals = get_sim(ugrid[0][0]).par["lvals"]
local_obs = [f"n_{s}{d}" for d in "xyz"[: len(lvals)] for s in "mp"]
local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
local_obs += ["X_Cross"]
for obs in local_obs:
    res[obs] = np.zeros((vals["has_obc"].shape[0], vals["U"].shape[0]))

res["energy"] = np.zeros((vals["has_obc"].shape[0], vals["U"].shape[0]))

for ii, has_obc in enumerate(vals["has_obc"]):
    for jj, U in enumerate(vals["U"]):
        res["energy"][ii, jj] = get_sim(ugrid[ii][jj]).res["energies"][0] / (
            prod(lvals)
        )
        for obs in local_obs:
            res[obs][ii, jj] = np.mean(get_sim(ugrid[ii][jj]).res[obs])

fig = plt.figure()
plt.ylabel(r"X_cross")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["X_Cross"][ii, :], "-o", label=BC_label)
plt.legend()

fig = plt.figure()
plt.ylabel(r"N")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["N_pair"][ii, :], "-o", label=f"pair ({BC_label})")
    plt.plot(vals["U"], res["N_single"][ii, :], "-o", label=f"single ({BC_label})")
plt.legend()

fig = plt.figure()
plt.ylabel(r"Energy Density")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["energy"][ii, :], "-o", label=BC_label)
plt.legend(loc="lower right")
