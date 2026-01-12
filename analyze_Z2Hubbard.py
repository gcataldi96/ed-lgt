# %%
from simsio import *
from math import prod
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, FuncFormatter
from ed_lgt.tools import save_dictionary, first_derivative
from scipy import stats


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
# string
res = {}
# Acquire simulations of finite E field
config_filename = f"Z2FermiHubbard/PBCx/test"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["J"])
for obs in ["energy", "plaq", "string"]:
    res[obs] = np.zeros((10, 6), dtype=float)
    for ii, J in enumerate(vals["J"]):
        res[obs][ii, :] = get_sim(ugrid[ii]).res[obs]
# %%
fig, ax = plt.subplots(1, 1, constrained_layout=True)
for ii in range(6):
    ax.plot(
        vals["J"],
        res["energy"][:, ii],
        "-o",
        markersize=4,
        markeredgecolor="darkblue",
        markerfacecolor="white",
        markeredgewidth=0.5,
        label=f"E_{ii}",
    )
ax.set(ylabel="String", xscale="log", xlim=[0.01, 0.1], xlabel="J")
plt.legend()
# %%
res = {}
# Acquire simulations of finite E field
config_filename = f"Z2FermiHubbard/energy_gap"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["U", "h"])
res["gap"] = np.zeros((len(vals["U"]), len(vals["h"])), dtype=float)
for ii, U in enumerate(vals["U"]):
    for jj, h in enumerate(vals["h"]):
        res["gap"][ii, jj] = (
            get_sim(ugrid[ii, jj]).res["energy"][1]
            - get_sim(ugrid[ii, jj]).res["energy"][0]
        )
# %%
obs_name = "gap"
sm = cm.ScalarMappable(cmap="plasma")
palette = sm.to_rgba(vals["h"])
fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.set(xscale="log", xlabel="$U$", yscale="log")
for jj, h in enumerate(vals["h"][:1]):
    axs.plot(
        vals["U"],
        res[obs_name][:, jj],
        "o-",
        linewidth=1,
        markersize=3,
        c=palette[jj],
        markerfacecolor="black",
    )
# %%
obs_name = "gap"
sm = cm.ScalarMappable(cmap="plasma")
palette = sm.to_rgba(vals["h"])
fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.set(ylabel=rf"$\Delta=E_{1}-E_{0}$", xscale="log", xlabel="$U$", yscale="log")
for jj, h in enumerate(vals["h"]):
    axs.plot(
        vals["U"],
        res[obs_name][:, jj],
        "o-",
        linewidth=1,
        markersize=3,
        c=palette[jj],
        markerfacecolor="black",
    )
cb = fig.colorbar(
    sm, ax=axs, aspect=80, location="top", orientation="horizontal", pad=0.02
)
cb.set_label(label=r"$h$", labelpad=-22, x=-0.02, y=0)


# Perform linear regression
slope, intercept, _, _, _ = stats.linregress(
    np.log(vals["U"][-8:]), np.log(res[obs_name][-8:, -1])
)
# Convert slope and intercept back to the original scale
b = slope
a = np.exp(intercept)
print(f"a0={a}, b0={b}")
axs.plot(vals["U"], a * vals["U"] ** b, label=r"$\Delta = a\cdot U^b$", color="black")
# Perform linear regression
slope, intercept, _, _, _ = stats.linregress(
    np.log(vals["U"][-8:]), np.log(res[obs_name][-8:, 0])
)
# Convert slope and intercept back to the original scale
b = slope
a = np.exp(intercept)
print(f"a1={a}, b1={b}")
axs.plot(
    vals["U"], a * vals["U"] ** b, "--", label=r"$\Delta = a\cdot U^b$", color="black"
)
axs.axvline(x=vals["U"][-8], ymin=10 ** (-6), ymax=10, linestyle="--", color="black")
plt.savefig(f"energy_gap.pdf")
# %%
# =======================================================================================
res = {}
# Acquire simulations of finite E field
config_filename = f"Z2FermiHubbard/PBCxy/largeh"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["U", "h"])
for obs in local_obs + ["energy", plaq_name]:
    res[obs] = np.zeros((len(vals["U"]), len(vals["h"])), dtype=float)
for ii, U in enumerate(vals["U"]):
    for jj, h in enumerate(vals["h"]):
        res["energy"][ii, jj] = get_sim(ugrid[ii, jj]).res["energy"] / 4
        for obs in local_obs + [plaq_name]:
            res[obs][ii, jj] = get_sim(ugrid[ii, jj]).res[obs]
# save_dictionary(res, f"phase_diagram.pkl")
# %%
# Susceptibility
obs_name = "plaq"
sm = cm.ScalarMappable(cmap="plasma", norm=LogNorm())
palette = sm.to_rgba(vals["U"])
fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.grid()
axs.set(ylabel=obs_name + " suscept", xlabel="$h$", xscale="log")
hmax = np.zeros(len(vals["U"]), dtype=float)
for ii, U in enumerate(vals["U"]):
    df = np.gradient(res[obs_name][ii, :], 10)
    hmax[ii] = vals["h"][np.argmax(df)]
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
plt.savefig(f"plaq_suscept_large_h.pdf")
fig1, axs = plt.subplots(1, 1, constrained_layout=True)
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
    np.log(vals["U"][4:]), np.log(hmax)[4:]
)

# Convert slope and intercept back to the original scale
b = slope
a = np.exp(intercept)

axs.plot(
    vals["U"],
    a * vals["U"] ** b,
    label=r"$h_{\max} = a\cdot U^b$",
    linestyle="--",
    color="red",
)
# Adjust annotation to fit the data range
# Use values from the plot to pick suitable positions within the visible range
x_annot = np.mean(vals["U"])
# Use the mean of 'vals["U"]' for a better annotation position
y_annot_a = (
    np.mean(hmax) - 50
)  # Use the mean of 'hmax' for a reasonable y position for 'a'
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
plt.savefig(f"hmax_fit_large_h.pdf")
# %%
# Susceptibility Npair
obs_name = "N_pair"
sm = cm.ScalarMappable(cmap="plasma", norm=LogNorm())
palette = sm.to_rgba(vals["h"])
fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.grid()
axs.set(ylabel=obs_name + " suscept", xlabel="$U$", xscale="log")
hmin = np.zeros(len(vals["h"]), dtype=float)
for jj, h in enumerate(vals["h"]):
    df = np.gradient(res[obs_name][:, jj], 0.09303374113107266)
    hmin[jj] = vals["U"][np.argmin(df)]
    axs.plot(
        vals["U"],
        df,
        "o-",
        linewidth=1,
        markersize=3,
        c=palette[jj],
        markerfacecolor="black",
    )
cb = fig.colorbar(
    sm, ax=axs, aspect=80, location="top", orientation="horizontal", pad=0.02
)
cb.set_label(label=r"$h$", labelpad=-22, x=-0.02, y=0)
plt.savefig(f"Umin_Npair.pdf")
# %%
fig1, axs = plt.subplots(1, 1, constrained_layout=True)
axs.plot(
    vals["h"][:-2],
    hmin[:-2],
    "o-",
    linewidth=1,
    markersize=3,
    c=palette[0],
    markerfacecolor="black",
    label="Data",
)
axs.set(xscale="log", xlabel="h", yscale="log", ylabel=r"$U_{\min}$")
plt.savefig(f"Umin_constant.pdf")
# %%
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
# plt.savefig(f"hmax_fit.pdf")
# %%
res = {}
# Acquire simulations of finite E field
config_filename = f"Z2FermiHubbard/grid"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["U", "h"])
for obs in ["N_pair", "energy", "entropy", plaq_name, "X_Cross", "S2_psi", "Efield"]:
    res[obs] = np.zeros((len(vals["U"]), len(vals["h"]), 4), dtype=float)
for ii, U in enumerate(vals["U"]):
    for jj, h in enumerate(vals["h"]):
        res["energy"][ii, jj, :] = get_sim(ugrid[ii, jj]).res["energy"]
        for obs in ["N_pair", "entropy", plaq_name, "X_Cross", "S2_psi"]:
            res[obs][ii, jj, :] = get_sim(ugrid[ii, jj]).res[obs]
        if ii < 36 or (ii == 36 and jj < 36):
            res["Efield"][ii, jj, :] = get_sim(ugrid[ii, jj]).res["E_field"]
        else:
            res["Efield"][ii, jj, :] = get_sim(ugrid[ii, jj]).res["Efield"]
save_dictionary(res, "phase_diagram.pkl")
# %%
# VERTICAL TRANSITION
Emin = 0
Emax = 18
obs_name = "N_pair"
Umax = np.zeros(len(vals["h"][Emin:Emax]), dtype=float)

sm = cm.ScalarMappable(cmap="plasma", norm=LogNorm())
palette = sm.to_rgba(vals["h"][Emin:Emax])
fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.grid()
axs.set(ylabel=obs_name + " suscept", xlabel="$U$", xscale="log")
axs.axvline(x=5.179474679231209, color="black", linestyle="--", label=r"$h_{c}=5.179$")
for jj, h in enumerate(vals["h"][Emin:Emax]):
    df = np.gradient(res[obs_name][:, jj + Emin, 0], 0.1879661300403302)
    Umax[jj] = vals["U"][np.argmin(df)]
    axs.plot(
        vals["U"],
        df,
        "o-",
        linewidth=1,
        markersize=3,
        c=palette[jj],
        markerfacecolor="black",
    )
cb = fig.colorbar(
    sm, ax=axs, aspect=80, location="top", orientation="horizontal", pad=0.02
)
cb.set_label(label=r"$h$", labelpad=-22, x=-0.02, y=0)
axs.legend()
plt.savefig(f"vertical_transition_{obs_name}.pdf")
fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.plot(
    vals["h"][Emin:Emax],
    Umax,
    "o-",
    linewidth=1,
    markersize=3,
    c=palette[0],
    markerfacecolor="black",
    label="Data",
)
axs.set(xscale="log", xlabel="U")

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    np.log(vals["h"][Emin:Emax]), np.log(Umax)
)
# Convert slope and intercept back to the original scale
b = slope
a = np.exp(intercept)

axs.plot(
    vals["h"][Emin:Emax],
    a * vals["h"][Emin:Emax] ** b,
    label=r"$h_{\max} = a\cdot U^b$",
    linestyle="--",
    color="red",
)
# Show legend
axs.legend()
print("---", a, b)
hfit = a * vals["U"][Emin:Emax] ** b  # Only using the slice [21:] for hmax
# %%
# HORIZONTAL TRANSITION
Umin = 0
Umax = 14
obs_name = "plaq"
hmax = np.zeros(len(vals["U"][Umin:Umax]), dtype=float)

sm = cm.ScalarMappable(cmap="plasma", norm=LogNorm())
palette = sm.to_rgba(vals["U"][Umin:Umax])
fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.grid()
axs.set(ylabel=obs_name + " suscept", xlabel="$h$", xscale="log")
axs.axvline(
    x=0.49417133613238334, color="black", linestyle="--", label=r"$h_{c}=0.494$"
)
for jj, U in enumerate(vals["U"][Umin:Umax]):
    print(U)
    df = np.gradient(res[obs_name][jj + Umin, :, 0], 0.28194919506049576)
    if obs_name == "Efield":
        hmax[jj] = vals["h"][np.argmin(df)]
    elif obs_name == "plaq":
        hmax[jj] = vals["h"][np.argmax(df)]
    axs.plot(
        vals["h"],
        df,
        "o-",
        linewidth=1,
        markersize=3,
        c=palette[jj],
        markerfacecolor="black",
    )
cb = fig.colorbar(
    sm, ax=axs, aspect=80, location="top", orientation="horizontal", pad=0.02
)
cb.set_label(label=r"$U$", labelpad=-22, x=-0.02, y=0)
axs.legend()
plt.savefig(f"horizontal_transition_{obs_name}.pdf")

fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.plot(
    vals["U"][Umin:Umax],
    hmax,
    "o-",
    linewidth=1,
    markersize=3,
    c=palette[0],
    markerfacecolor="black",
    label="Data",
)
axs.set(xscale="log", yscale="log", xlabel="U")

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    np.log(vals["U"][Umin:Umax]), np.log(hmax)
)

# Convert slope and intercept back to the original scale
b = slope
a = np.exp(intercept)

axs.plot(
    vals["U"][Umin:Umax],
    a * vals["U"][Umin:Umax] ** b,
    label=r"$h_{\max} = a\cdot U^b$",
    linestyle="--",
    color="red",
)
# Show legend
axs.legend()
print("---", a, b)

hfit = a * vals["U"][Umin:Umax] ** b  # Only using the slice [21:] for hmax
# %%
Umin = 23
Umax = 31
hhmax = 16
obs_name = "plaq"
hmax = np.zeros(len(vals["U"][Umin:Umax]), dtype=float)

sm = cm.ScalarMappable(cmap="plasma", norm=LogNorm())
palette = sm.to_rgba(vals["U"][Umin:Umax])
fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.grid()
axs.set(ylabel=obs_name + " suscept", xlabel="$h$", xscale="log")
for jj, U in enumerate(vals["U"][Umin:Umax]):
    print(U)
    df = np.gradient(res[obs_name][jj + Umin, :hhmax, 0], 0.28194919506049576)
    hmax[jj] = vals["h"][:hhmax][np.argmax(df)]
    axs.plot(
        vals["h"][:hhmax],
        df,
        "o-",
        linewidth=1,
        markersize=3,
        c=palette[jj],
        markerfacecolor="black",
    )
cb = fig.colorbar(
    sm, ax=axs, aspect=80, location="top", orientation="horizontal", pad=0.02
)
cb.set_label(label=r"$U$", labelpad=-22, x=-0.02, y=0)

fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.plot(
    vals["U"][Umin:Umax],
    hmax,
    "o-",
    linewidth=1,
    markersize=3,
    c=palette[0],
    markerfacecolor="black",
    label="Data",
)
axs.set(xscale="log", yscale="log", xlabel="U")

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    np.log(vals["U"][Umin:Umax]), np.log(hmax)
)

# Convert slope and intercept back to the original scale
b = slope
a = np.exp(intercept)

axs.plot(
    vals["U"][Umin:Umax],
    a * vals["U"][Umin:Umax] ** b,
    label=r"$h_{\max} = a\cdot U^b$",
    linestyle="--",
    color="red",
)
# Show legend
axs.legend()
print("---", a, b)

hfit = a * vals["U"][Umin:Umax] ** b  # Only using the slice [21:] for hmax
# %%
# LARGE h Transition
Umin = 31
Umax = 49
hhmin = 35
obs_name = "N_pair"
hmax = np.zeros(len(vals["U"][Umin:Umax]), dtype=float)
sm = cm.ScalarMappable(cmap="plasma", norm=LogNorm())
palette = sm.to_rgba(vals["U"][Umin:Umax])
fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.grid()
axs.set(ylabel=obs_name + " suscept", xlabel="$h$", xscale="log")
for jj, U in enumerate(vals["U"][Umin:Umax]):
    df = np.gradient(res[obs_name][jj + Umin, hhmin:, 0], 0.28194919506049576)
    hmax[jj] = vals["h"][hhmin:][np.argmax(df)]
    axs.plot(
        vals["h"][hhmin:],
        df,
        "o-",
        linewidth=1,
        markersize=3,
        c=palette[jj],
        markerfacecolor="black",
    )
cb = fig.colorbar(
    sm, ax=axs, aspect=80, location="top", orientation="horizontal", pad=0.02
)
cb.set_label(label=r"$U$", labelpad=-22, x=-0.02, y=0)

fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.plot(
    vals["U"][Umin:Umax],
    hmax,
    "o-",
    linewidth=1,
    markersize=3,
    c=palette[0],
    markerfacecolor="black",
    label="Data",
)
axs.set(xscale="log", yscale="log", xlabel="U")

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    np.log(vals["U"][Umin:Umax]), np.log(hmax)
)

# Convert slope and intercept back to the original scale
b = slope
a = np.exp(intercept)

axs.plot(
    vals["U"][Umin:Umax],
    a * vals["U"][Umin:Umax] ** b,
    label=r"$h_{\max} = a\cdot U^b$",
    linestyle="--",
    color="red",
)
# Show legend
axs.legend()
print(a, b)

hfit = a * vals["U"][Umin:Umax] ** b  # Only using the slice [21:] for hmax

# %%
# Set up the plot
fig, axs = plt.subplots(
    1,
    1,
    sharex=True,
    constrained_layout=True,
)

"""# Plot hmax as a dashed line
axs.plot(
    np.log10(vals["U"][Umin:Umax]),
    np.log10(hfit),
    color="white",
    linestyle="--",
    linewidth=2,
)"""
obs_name = "Efield"
# Logarithmic scale formatting (assuming fake_log is a log formatter)
axs.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"$10^{{{int(x)}}}$"))
axs.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"$10^{{{int(y)}}}$"))

# Plot imshow with extent reflecting the log10 scale
extent = [-1, 3, -3, 3]
img = axs.imshow(
    np.transpose(res[obs_name][:, :, 0]),
    origin="lower",
    cmap="magma",
    extent=extent,
)
# Add colorbar
cb = fig.colorbar(
    img,
    ax=axs,
    aspect=20,
    location="right",
    orientation="vertical",
    pad=0.01,
    label=obs_name,
)
# Set xticks and yticks using log scale values
axs.set(
    xticks=np.log10([0.1, 1, 10, 100, 1000]),
    yticks=np.log10([0.001, 0.01, 0.1, 1, 10, 100, 1000]),
    xlabel="U",
    ylabel="h",
)
plt.savefig(f"phase_diagram_{obs_name}.pdf")

# %%
# FINITE SIZE
res = {}
# Acquire simulations of finite E field
config_filename = f"Z2FermiHubbard/finitesize"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["U"])
obs_list = ["energy", "N_pair", "entropy", "X_Cross", "S2_psi", plaq_name]
for obs in obs_list:
    res[obs] = np.zeros(len(vals["U"]), dtype=float)
for ii, U in enumerate(vals["U"]):
    res["energy"][ii] = get_sim(ugrid[ii]).res["energy"] / 8
    for obs in obs_list[1:]:
        res[obs][ii] = get_sim(ugrid[ii]).res[obs]
# %%
delta = 0
lim = 50
obs_name = "N_pair"
data = np.loadtxt(f"{obs_name}_Lx_Ly_8_2.txt")
data1 = np.loadtxt(f"{obs_name}_Lx_Ly_16_2.txt")
column_data = data[:]
fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.grid()
axs.set(ylabel=obs_name, xlabel="$U$", xscale="log")
axs.plot(
    vals["U"],
    res[obs_name] * (4**delta),
    "o-",
    linewidth=1,
    markersize=3,
    markerfacecolor="black",
    label=r"$4\times2$",
)
axs.plot(
    vals["U"][:lim],
    data * (8**delta),
    "o-",
    linewidth=1,
    markersize=3,
    markerfacecolor="black",
    label=r"$8\times2$",
)
axs.plot(
    vals["U"][:lim],
    data1 * (16**delta),
    "o-",
    linewidth=1,
    markersize=3,
    markerfacecolor="black",
    label=r"$16\times2$",
)
plt.legend()
plt.savefig(f"{obs_name}.pdf")

dU = np.log(vals["U"][1] / vals["U"][0])
fig, axs = plt.subplots(1, 1, constrained_layout=True)
axs.grid()
axs.set(ylabel=f"d{obs_name}/dU", xlabel="$U$", xscale="log")
axs.plot(
    vals["U"],
    np.gradient(res[obs_name], dU),
    "o-",
    linewidth=1,
    markersize=3,
    markerfacecolor="black",
    label=r"$4\times2$",
)
axs.plot(
    vals["U"][:lim],
    np.gradient(data, dU),
    "o-",
    linewidth=1,
    markersize=3,
    markerfacecolor="black",
    label=r"$8\times2$",
)
axs.plot(
    vals["U"][:lim],
    np.gradient(data1, dU),
    "o-",
    linewidth=1,
    markersize=3,
    markerfacecolor="black",
    label=r"$16\times2$",
)
plt.legend()
plt.savefig(f"der_{obs_name}.pdf")
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

# %%
# # %%
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
obs_name = "N_pair"
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
