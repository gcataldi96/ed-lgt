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


def time_integral(time, M):
    """
    Computes the running time integral/average of an observable M over a given time line.

    Parameters:
    time (numpy.ndarray): Array of time points.
    M (numpy.ndarray): Array of observable values corresponding to each time point.

    Returns:
    numpy.ndarray: Array of the running average of M at each time point.
    """
    Mavg = np.zeros_like(M)
    Mavg[0] = M[0]

    for cnt in range(1, len(time)):
        for tnc in range(1, cnt + 1):
            Mavg[cnt] += (
                0.5 * (M[tnc] + M[tnc - 1]) * (time[tnc] - time[tnc - 1]) / time[cnt]
            )
    return Mavg


def get_tline(par: dict):
    start = par["start"]
    stop = par["stop"]
    delta_n = par["delta_n"]
    n_steps = int((stop - start) / delta_n)
    return np.arange(n_steps) * delta_n


def custom_average(arr, staggered=None, norm=None):
    # Determine indices to consider based on the staggered parameter
    indices = np.arange(arr.shape[1])
    if staggered == "even":
        indices_to_consider = indices[indices % 2 == 0]  # Select even indices
    elif staggered == "odd":
        indices_to_consider = indices[indices % 2 != 0]  # Select odd indices
    else:
        indices_to_consider = indices

    if norm is not None:
        # Ensure norm is a 1D array with the same length as the number of columns in arr
        if norm.shape[0] != arr.shape[1]:
            raise ValueError(
                f"norm vector length {norm.shape[0]} must match the number of columns in arr {arr.shape[1]}"
            )
        # Calculate the scalar product of each row and the norm vector
        # then divide by the number of columns
        mean_values = np.dot(arr, norm) / arr.shape[1]
    else:
        # Calculate the mean across the selected indices
        mean_values = np.mean(arr[:, indices_to_consider], axis=1)

    return mean_values


# List of local observables
local_obs = [f"T2_{s}{d}" for d in "x" for s in "mp"]
local_obs += ["E_square"]
local_obs += [f"N_{label}" for label in ["r", "g", "tot", "single", "pair"]]


# %%
def calculate_preallocation_memory(sparsity, sector_dim, N):
    """
    Calculate the memory used in GB for preallocating arrays based on sparsity, sector_dim, and N.

    Args:
        sparsity (float): Estimated sparsity of the operator.
        sector_dim (int): Dimension of the symmetry sector (number of configurations).
        N (int): Number of lattice sites.

    Returns:
        float: Memory used in GB for preallocation.
    """
    # Estimate the number of non-zero elements based on sparsity
    estimated_nonzero_elements = int(sparsity * sector_dim**2)

    # Memory requirements for row_list, col_list, and value_list
    # row_list and col_list use np.int32 (4 bytes per entry)
    # value_list uses np.float64 (8 bytes per entry)
    memory_bytes = 16 * estimated_nonzero_elements
    # Convert to GB
    memory_gb = memory_bytes / (1024**3)
    return memory_gb


from scipy import stats

N = np.array([4, 6, 8, 10, 12])
X = 1 - 0.01 * np.array([86.285, 97.530, 99.600, 99.940, 99.9912])
dim = np.array([38, 282, 2214, 17906, 147578])
fig, ax = plt.subplots(2, 1, constrained_layout=True)
ax[0].plot(N, X, "o-")
ax[0].set(xlabel="N", ylabel="X", yscale="log")
ax[1].plot(N, dim, "o-")
ax[1].set(xlabel="N", ylabel="hilbert space", yscale="log")
# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(N, np.log(X))
# Convert slope and intercept back to the original scale
b = slope
a = np.exp(intercept)
ax[0].plot(N, a * np.exp(b * N), "--")
print(a, b, r_value**2)
# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(N, np.log(dim))
# Convert slope and intercept back to the original scale
b = slope
a = np.exp(intercept)
print(a, b, r_value**2)
for ii in range(N.shape[0]):
    print(N[ii], calculate_preallocation_memory(X[ii], dim[ii], N[ii]))
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
# ==========================================================================
# 1D DFL TIME INTEGRAL
# ==========================================================================
res = {}
config_filename = f"DFL/DW2sitesmicro"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = ["delta", "entropy", "overlap"]
# Get the time line
tline = get_tline(get_sim(ugrid[0]).par["dynamics"])
res["micro"] = get_sim(ugrid[0]).res["microcan_avg"]
for obs in obs_list:
    res[obs] = np.zeros((6, len(tline)), dtype=float)
    for ii, g in enumerate(vals["g"]):
        res[obs][ii, :] = get_sim(ugrid[ii]).res[obs]

res1 = {}
config_filename = f"DFL/DW2sitesBGmicro"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = ["delta"]  # , "entropy", "overlap"]
# Get the time line
tline1 = get_tline(get_sim(ugrid[0]).par["dynamics"])
res1["micro"] = get_sim(ugrid[0]).res["microcan_avg"]
for obs in obs_list:
    res1[obs] = np.zeros((6, len(tline1)), dtype=float)
    for ii, g in enumerate(vals["g"]):
        res1[obs][ii, :] = get_sim(ugrid[ii]).res[obs]

obs = "delta"
for ii, g in enumerate(vals["g"]):
    m = get_sim(ugrid[ii]).par["m"]
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    ax.plot(
        tline,
        time_integral(tline, res[obs][ii, :]),
        "--",
        label=f"t_DW m={m},g={g}",
    )
    ax.plot(
        tline,
        res[obs][ii, :],
        "--",
        label=f"DW m={m},g={g}",
    )
    ax.axhline(y=res["micro"], linestyle="-")
    ax.axhline(y=res1["micro"], linestyle="-.")
    ax.plot(tline1, time_integral(tline1, res1[obs][ii, :]), "-", label="t_DW-BG")
    ax.plot(tline1, res1[obs][ii, :], "-", label="DW-BG")
    ax.set(xlabel=r"$t$", ylabel=f"time integral({obs})")
    ax.grid()
    plt.legend()
# %%
# ==========================================================================
# DFL SPECTRUM
# ==========================================================================
res = {}
config_filename = f"DFL/spectra"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = ["energy", "entropy", "overlap"]

for obs in obs_list:
    res[obs] = np.zeros((6, 2214), dtype=float)
    for ii, g in enumerate(vals["g"]):
        print(g, get_sim(ugrid[ii]).par["m"])
        res[obs][ii, :] = get_sim(ugrid[ii]).res[obs]

for ii, g in enumerate(vals["g"]):
    m = get_sim(ugrid[ii]).par["m"]
    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    ax[0].plot(
        res["energy"][ii, :],
        res["overlap"][ii, :],
        "o",
        markersize=2,
        markeredgecolor="darkblue",
        markerfacecolor="white",
        markeredgewidth=0.5,
    )
    ax[0].set(yscale="log", ylim=[1e-12, 1])
    ax[1].grid()
    ax[1].plot(
        res["energy"][ii, :],
        res["entropy"][ii, :],
        "o",
        markersize=2,
        markeredgecolor="darkblue",
        markerfacecolor="white",
        markeredgewidth=1,
        label=f"DW m={m},g={g}",
    )
    ax[1].set(ylabel="Ent Entropy", xlabel="Energy")
    ax[1].legend()
    # plt.savefig(f"spectra_m{m}_g{g}.pdf")
# %%
# ==========================================================================
# 1D DFL TIME INTEGRAL & SPECTRA
# ==========================================================================
res = {}
config_filename = f"DFL/DW2sites"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = ["delta", "entropy", "overlap"][:1]
# Get the time line
tline = get_tline(get_sim(ugrid[0]).par["dynamics"])

for obs in obs_list:
    res[obs] = np.zeros((6, len(tline)), dtype=float)
    for ii, g in enumerate(vals["g"]):
        res[obs][ii, :] = get_sim(ugrid[ii]).res[obs]

res1 = {}
config_filename = f"DFL/DW2sitesBG"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = ["delta", "entropy", "overlap"][:1]
# Get the time line
tline1 = get_tline(get_sim(ugrid[0]).par["dynamics"])

for obs in obs_list:
    res1[obs] = np.zeros((6, len(tline1)), dtype=float)
    for ii, g in enumerate(vals["g"]):
        res1[obs][ii, :] = get_sim(ugrid[ii]).res[obs]

res_spectra = {}
config_filename = f"DFL/spectra"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = ["energy", "entropy", "overlap"]

for obs in obs_list:
    res_spectra[obs] = np.zeros((6, 2214), dtype=float)
    for ii, g in enumerate(vals["g"]):
        res_spectra[obs][ii, :] = get_sim(ugrid[ii]).res[obs]

obs = "delta"
for ii, g in enumerate(vals["g"]):
    m = get_sim(ugrid[ii]).par["m"]
    delta = time_integral(tline, res[obs][ii, :])
    deltaDLF = time_integral(tline, res1[obs][ii, :])
    # TIME INTEGRAL DYNAMICS
    fig, ax = plt.subplots(
        2,
        1,
        figsize=set_size(textwidth_pt),
        constrained_layout=True,
    )
    ax[0].grid()
    ax[0].plot(tline, delta, "--", label=f"t_DW m={m},g={g}")
    ax[0].plot(tline1, deltaDLF, "-", label="t_DW-BG")
    ax[0].set(xlabel=r"$t$", ylabel=f"T.I.({obs})")
    ax[0].legend()
    # SPECTRA
    ax[1].grid()
    ax[1].plot(
        res_spectra["energy"][ii, :],
        res_spectra["entropy"][ii, :],
        "o",
        markersize=2,
        markeredgecolor="darkblue",
        markerfacecolor="white",
        markeredgewidth=1,
        label=f"DW m={m},g={g}",
    )
    ax[1].set(ylabel="Entropy", xlabel="Energy")
    plt.savefig(f"DFL_m{m}_g{g}.pdf")
# %%
# ===================================================================
# DFL PHASE DIAGRAM
# ===================================================================
res = {}
config_filename = f"DFL/gridtest_DFL"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
res["Deff"] = np.zeros((len(vals["g"]), len(vals["m"]), 128), dtype=float)
res["delta"] = np.zeros((len(vals["g"]), len(vals["m"]), 201), dtype=float)
res["imshow_delta"] = np.zeros((len(vals["g"]), len(vals["m"])), dtype=float)
res["imshow_deff"] = np.zeros((len(vals["g"]), len(vals["m"])), dtype=float)
res["time_line"] = get_sim(ugrid[0, 0]).res["time_steps"]
for ii, g in enumerate(vals["g"]):
    for jj, m in enumerate(vals["m"]):
        res["delta"][ii, jj, :] = get_sim(ugrid[ii, jj]).res["delta"]
        res["imshow_delta"][ii, jj] = np.mean(get_sim(ugrid[ii, jj]).res["delta"])
        res["imshow_deff"][ii, jj] = np.mean(get_sim(ugrid[ii, jj]).res["Deff"])

res1 = {}
config_filename = f"DFL/gridtest_BV"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
res1["Deff"] = np.zeros((len(vals["g"]), len(vals["m"])), dtype=float)
res1["delta"] = np.zeros((len(vals["g"]), len(vals["m"]), 201), dtype=float)
res1["imshow_delta"] = np.zeros((len(vals["g"]), len(vals["m"])), dtype=float)
res1["imshow_deff"] = np.zeros((len(vals["g"]), len(vals["m"])), dtype=float)
res1["time_line"] = get_sim(ugrid[0, 0]).res["time_steps"]
for ii, g in enumerate(vals["g"]):
    for jj, m in enumerate(vals["m"]):
        res1["delta"][ii, jj, :] = get_sim(ugrid[ii, jj]).res["delta"]
        res1["imshow_delta"][ii, jj] = np.mean(get_sim(ugrid[ii, jj]).res["delta"])
        res1["imshow_deff"][ii, jj] = get_sim(ugrid[ii, jj]).res["Deff"]
# %%
# ===================================================================
# Plot phase diagram
# ===================================================================
fig, ax = plt.subplots(1, 1, constrained_layout=True)
# IMSHOW
measure = "deff"
X = np.transpose(res[f"imshow_{measure}"])
centers = [0, 4, 0, 4]
(dx,) = np.diff(centers[:2]) / (X.shape[1] - 1)
(dy,) = np.diff(centers[2:]) / (X.shape[0] - 1)
extent = [
    centers[0] - dx / 2,
    centers[1] + dx / 2,
    centers[2] - dy / 2,
    centers[3] + dy / 2,
]
img = plt.imshow(X, cmap="magma", origin="lower", extent=extent)

xticks = np.arange(centers[0], centers[1] + dx, dx)
yticks = np.arange(centers[2], centers[3] + dy, dy)
tick_labels = np.array([0.1, 0.8, 5, 10, 15])
ax.set(ylabel=r"m", xlabel=r"g^{2}")

# Custom ticks
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels(tick_labels)
ax.set_yticklabels(tick_labels)

cb = fig.colorbar(
    img,
    ax=ax,
    aspect=20,
    location="right",
    orientation="vertical",
    pad=0.01,
    label=rf"<{measure}>",
)
plt.savefig(f"small_diagram{measure}.pdf")
# %%
ii = 2
jj = 4
m = vals["m"]
g = vals["g"]
fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.plot(
    res["time_line"],
    res["delta"][ii, jj, :],
    "-",
    label=f"BG m={m[jj]},g={g[ii]}",
)
ax.plot(
    res1["time_line"],
    res1["delta"][ii, jj, :],
    "-",
    label=f"DW m={m[jj]},g={g[ii]}",
)
ax.set(xlabel=r"$t$", ylabel=f"delta(t)")
ax.grid()
plt.legend()
# %%
# ===================================================================
# ENTROPY
sm = cm.ScalarMappable(cmap="magma", norm=LogNorm())
palette = sm.to_rgba(vals["g"])
jj = 10
for ii, obs in enumerate(obs_list):
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    for kk, g in enumerate(vals["g"]):
        ax.plot(
            tline,
            res[obs][kk, jj],
            "o-",
            linewidth=0.7,
            markersize=2,
            c=palette[kk],
            markerfacecolor="black",
            markeredgewidth=0.6,
        )
        ax.set(xlabel=r"$t$", ylabel=obs)
        if ii == 0:
            ax.set(ylim=[0, 1.1])
        if ii == 1:
            ax.set(xscale="log")
        ax.grid(visible=True)

    cb = fig.colorbar(
        sm, ax=ax, aspect=80, location="top", orientation="horizontal", pad=0.02
    )
    cb.set_label(label=r"$g^{2}$", labelpad=-22, x=-0.02, y=0)
    plt.savefig(
        f"m{vals['m'][jj]}_{obs}.pdf",
    )

# %%
res = {}
config_filename = f"DFL/gscale"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = ["delta", "N_single", "N_pair", "N_zero", "Deff", "Hspace_size"]
time_line = get_sim(ugrid[0]).res["time_steps"]

for obs in obs_list[:4]:
    res[obs] = np.zeros((len(vals["g"]), len(time_line)), dtype=float)
    res[f"micro_{obs}"] = np.zeros(len(vals["g"]), dtype=float)
    for ii, g in enumerate(vals["g"]):
        if obs in ["N_pair", "N_zero"]:
            res[obs][ii, :] = get_sim(ugrid[ii]).res[obs] / (128**2)
        else:
            res[obs][ii, :] = get_sim(ugrid[ii]).res[obs]
        res[f"micro_{obs}"][ii] = get_sim(ugrid[ii]).res[f"micro_{obs}"]

res["D"] = np.zeros((len(vals["g"]), 128), dtype=float)
for obs in obs_list[4:]:
    res[obs] = np.zeros((len(vals["g"]), 128), dtype=float)
    for ii, g in enumerate(vals["g"]):
        res[obs][ii, :] = get_sim(ugrid[ii]).res[obs]
for ii, g in enumerate(vals["g"]):
    for sec in range(128):
        res["D"][ii, sec] = -np.log(res["Deff"][ii, sec]) / np.log(
            res["Hspace_size"][ii, sec]
        )
# %%
res1 = {}
config_filename = f"newDFL/gscale_NOBG"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = ["N_single", "N_pair", "N_zero"]
time_line1 = get_sim(ugrid[0]).res["time_steps"]
res1["entropy"] = np.zeros((len(vals["g"]), len(time_line1)), dtype=float)
for obs in obs_list:
    res1[obs] = np.zeros((len(vals["g"]), len(time_line1)), dtype=float)
    res1[f"ME_{obs}"] = np.zeros(len(vals["g"]), dtype=float)
    res1[f"DE_{obs}"] = np.zeros(len(vals["g"]), dtype=float)
    for ii, g in enumerate(vals["g"]):
        res1[obs][ii, :] = get_sim(ugrid[ii]).res[obs]
        res1[f"ME_{obs}"][ii] = get_sim(ugrid[ii]).res[f"ME_{obs}"]
        res1[f"DE_{obs}"][ii] = get_sim(ugrid[ii]).res[f"DE_{obs}"]
for ii, g in enumerate(vals["g"]):
    res1["entropy"][ii, :] = get_sim(ugrid[ii]).res["entropy"]
# %%
obs_names = [
    r"$\hat{\rho}_{\rm{1}}$",
    r"$\hat{\rho}_{\rm{2}}$",
    r"$\hat{\rho}_{\rm{0}}$",
]
sizes = [5, 8, 5]
styles = ["--", "-", ":"]
colors = ["darkgreen", "darkblue", "darkred"]
m = get_sim(ugrid[ii]).par["m"]
fig, ax = plt.subplots(1, 1, constrained_layout=True, sharex=True, sharey=True)
for ii, obs in enumerate(["N_single", "N_pair", "N_zero"]):
    ax.plot(
        vals["g"],
        res1[f"ME_{obs}"],
        styles[ii],
        c=colors[ii],
        label=f"{obs_names[ii]} ME",
    )
    ax.plot(
        vals["g"],
        res1[f"DE_{obs}"],
        f"{styles[ii]}o",
        c=colors[ii],
        linewidth=2,
        markeredgecolor=colors[ii],
        markerfacecolor="white",
        markeredgewidth=1,
        markersize=sizes[ii],
        label=f"{obs_names[ii]} DE",
    )
    ax.set(xlabel=r"$g$", ylabel=r"Pariticle Densities $\hat{\rho}(m=1)$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.05, 0.45))
plt.savefig(f"gscale_NOBG.pdf")
sm = cm.ScalarMappable(cmap="copper")
palette = sm.to_rgba(vals["g"])
fig, ax = plt.subplots(1, 1, constrained_layout=True, sharex=True, sharey=True)
for ii, m in enumerate(vals["g"]):
    if ii % 2 == 0:
        ax.plot(
            time_line1,
            res1[f"entropy"][ii, :],
            "-",
            label=f"g={g}",
            c=palette[ii],
        )
cb = fig.colorbar(
    sm, ax=ax, aspect=80, location="top", orientation="horizontal", pad=0.02
)
cb.set_label(label=r"$g$", labelpad=-22, x=-0.02, y=0)
ax.set(xlabel=r"$t$", ylabel=r"Entanglement Entropy", xscale="log")
plt.savefig(f"gscale_NOBG_entropy_log.pdf")
# %%
sm = cm.ScalarMappable(cmap="copper")
palette = sm.to_rgba(np.arange(128))
fig, ax = plt.subplots(1, 1, constrained_layout=True, sharex=True, sharey=True)
m = get_sim(ugrid[ii]).par["m"]

for sec in range(128):
    ax.plot(
        vals["g"],
        res["D"][:, sec],
        "-",
        c=palette[sec],
        linewidth=0.7,
    )
ax.plot(
    vals["g"],
    np.mean(res["D"], axis=1),
    "-o",
    c="black",
    linewidth=2,
    markeredgecolor="black",
    markerfacecolor="white",
    markeredgewidth=1,
    label=r"$\langle D \rangle$ (BG)",
)
ax.plot(
    vals["g"],
    res1["D"],
    "-o",
    c="blue",
    linewidth=2,
    markeredgecolor="blue",
    markerfacecolor="yellow",
    markeredgewidth=1,
    label=r"$D$ (NO BG)",
)
ax.set(
    xlabel=r"$g^{2}$", ylabel=r"Fractal Dimension $D=\log(\rm{IPR})/\log(\mathcal{N})$"
)
ax.legend(loc="upper left", bbox_to_anchor=(0.5, 0.9))
plt.savefig(f"fractal_dimension.pdf")
# %%
fig, ax = plt.subplots(1, 1, constrained_layout=True, sharex=True, sharey=True)
m = get_sim(ugrid[ii]).par["m"]
ax.plot(vals["g"], np.mean(res["delta"], axis=1), "-", label="BG")
ax.plot(vals["g"], res1["DE_delta"], "-", label="no BG")
ax.plot(vals["g"], res["micro_delta"], "-", label="ME (BG)")
# ax.plot(vals["g"], res1["micro_delta"], "--", label="ME (NO BG)")
ax.set(xlabel=r"$g^{2}$", ylabel=r"Imbalance $\Delta(t\gg1, g^{2},m=1)$")
ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.8))
plt.savefig(f"imbalance.pdf")
# %%
res = {}
config_filename = f"DFL/mscale"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["m"])
obs_list = ["delta", "N_single", "N_pair", "N_zero", "Deff", "Hspace_size"]
time_line = get_sim(ugrid[0]).res["time_steps"]

for obs in obs_list[:4]:
    res[obs] = np.zeros((len(vals["m"]), len(time_line)), dtype=float)
    res[f"micro_{obs}"] = np.zeros(len(vals["m"]), dtype=float)
    for ii, g in enumerate(vals["m"]):
        if obs in ["N_pair", "N_zero"]:
            res[obs][ii, :] = get_sim(ugrid[ii]).res[obs] / (128**2)
        else:
            res[obs][ii, :] = get_sim(ugrid[ii]).res[obs]
        res[f"micro_{obs}"][ii] = get_sim(ugrid[ii]).res[f"micro_{obs}"]

res["D"] = np.zeros((len(vals["m"]), 128), dtype=float)
for obs in obs_list[4:]:
    res[obs] = np.zeros((len(vals["m"]), 128), dtype=float)
    for ii, g in enumerate(vals["m"]):
        res[obs][ii, :] = get_sim(ugrid[ii]).res[obs]
for ii, g in enumerate(vals["m"]):
    for sec in range(128):
        res["D"][ii, sec] = -np.log(res["Deff"][ii, sec]) / np.log(
            res["Hspace_size"][ii, sec]
        )


res1 = {}
config_filename = f"DFL/mscale_NOBG"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["m"])
obs_list = ["delta", "N_single", "N_pair", "N_zero", "Deff", "Hspace_size"]
time_line1 = get_sim(ugrid[0]).res["time_steps"]
for obs in obs_list[:4]:
    res1[obs] = np.zeros((len(vals["m"]), len(time_line1)), dtype=float)
    res1[f"micro_{obs}"] = np.zeros(len(vals["m"]), dtype=float)
    res1[f"DE_{obs}"] = np.zeros(len(vals["m"]), dtype=float)
    for ii, m in enumerate(vals["m"]):
        res1[obs][ii, :] = get_sim(ugrid[ii]).res[obs]
        res1[f"micro_{obs}"][ii] = get_sim(ugrid[ii]).res[f"micro_{obs}"]
        res1[f"DE_{obs}"][ii] = get_sim(ugrid[ii]).res[f"DE_{obs}"]

res1["D"] = np.zeros(len(vals["m"]), dtype=float)
for obs in obs_list[4:]:
    res1[obs] = np.zeros(len(vals["m"]), dtype=float)
    for ii, g in enumerate(vals["m"]):
        res1[obs][ii] = get_sim(ugrid[ii]).res[obs]
for ii, g in enumerate(vals["m"]):
    res1["D"][ii] = -np.log(res1["Deff"][ii]) / np.log(res1["Hspace_size"][ii])


# %%
fig, ax = plt.subplots(1, 1, constrained_layout=True, sharex=True, sharey=True)
g = get_sim(ugrid[ii]).par["g"]
ax.plot(vals["m"], np.mean(res["delta"], axis=1), "-", label="BG")
ax.plot(vals["m"], res1["DE_delta"], "-", label="no BG")
ax.plot(vals["m"], res["micro_delta"], "-", label="ME (BG)")
# ax.plot(vals["g"], res1["micro_delta"], "--", label="ME (NO BG)")
ax.set(xlabel=r"$m$", ylabel=r"Imbalance $\Delta(t\gg1,m ,g^{2}=1)$")
ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.8))
plt.savefig(f"imbalance_m.pdf")
# %%
obs_names = [
    r"$\hat{\rho}_{\rm{mes}}$",
    r"$\hat{\rho}_{\rm{bar}}$",
    r"$\hat{\rho}_{\rm{vac}}$",
]
colors = ["darkred", "darkblue", "darkgreen"]
g = get_sim(ugrid[ii]).par["g"]
fig, ax = plt.subplots(1, 1, constrained_layout=True, sharex=True, sharey=True)
for ii, obs in enumerate(["N_single", "N_pair", "N_zero"]):
    """
    ax.plot(
        vals["g"],
        np.mean(res[obs], axis=1),
        "-o",
        c=colors[ii],
        linewidth=2,
        markeredgecolor=colors[ii],
        markerfacecolor="white",
        markeredgewidth=1,
        label=f"{obs_names[ii]} DE",
    )
    ax.plot(
        vals["g"],
        res[f"micro_{obs}"],
        "--",
        c=colors[ii],
        label=f"{obs_names[ii]} ME",
    )"""
    ax.plot(
        vals["m"],
        res1[f"DE_{obs}"],
        "-o",
        c=colors[ii],
        linewidth=2,
        markeredgecolor=colors[ii],
        markerfacecolor="white",
        markeredgewidth=1,
        label=f"{obs_names[ii]} DE NO BG",
    )
    ax.plot(
        vals["m"],
        res1[f"micro_{obs}"],
        "--",
        c=colors[ii],
        label=f"{obs_names[ii]} ME NO BG",
    )
    """
    ax.plot(
        vals["m"],
        np.mean(res["N_single"], axis=1)
        + np.mean(res["N_pair"], axis=1)
        + np.mean(res["N_zero"], axis=1),
        "-o",
        label=f"{obs} DE",
    )"""
    ax.set(xlabel="m", ylabel="obs")
    ax.legend()
# %%
sm = cm.ScalarMappable(cmap="copper")
palette = sm.to_rgba(np.arange(128))
fig, ax = plt.subplots(1, 1, constrained_layout=True, sharex=True, sharey=True)
g = get_sim(ugrid[ii]).par["g"]

for sec in range(128):
    ax.plot(
        vals["m"],
        res["D"][:, sec],
        "-",
        c=palette[sec],
        linewidth=0.7,
    )
ax.plot(
    vals["m"],
    np.mean(res["D"], axis=1),
    "-o",
    c="black",
    linewidth=2,
    markeredgecolor="black",
    markerfacecolor="white",
    markeredgewidth=1,
    label=r"$\langle D \rangle$ (BG)",
)
ax.plot(
    vals["m"],
    res1["D"],
    "-o",
    c="blue",
    linewidth=2,
    markeredgecolor="blue",
    markerfacecolor="yellow",
    markeredgewidth=1,
    label=r"$D$ (NO BG)",
)
ax.set(
    xlabel=r"$g^{2}$", ylabel=r"Fractal Dimension $D=\log(\rm{IPR})/\log(\mathcal{N})$"
)
ax.legend(loc="upper left", bbox_to_anchor=(0.5, 1))
plt.savefig(f"fractal_dimension_m.pdf")
# %%
# ==========================================================================
# 1D DFL TIME INTEGRAL
# ==========================================================================
res = {}
config_filename = f"DFL/test1"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = ["delta", "N_single", "S2_matter"]
# Get the time line
time_line = get_sim(ugrid[0]).res["time_steps"]

for obs in obs_list:
    res[obs] = np.zeros((4, len(time_line)), dtype=float)
    for ii, g in enumerate(vals["g"]):
        res[obs][ii, :] = get_sim(ugrid[ii]).res[obs]
for micro in ["microcan_avg", "micro_Nsingle", "micro_S2_matter"]:
    res[micro] = np.zeros((4), dtype=float)
    for ii, g in enumerate(vals["g"]):
        res[micro][ii] = get_sim(ugrid[ii]).res[micro]

res1 = {}
config_filename = f"DFL/testnobg"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = ["delta", "N_single", "N_pair"]
# Get the time line
time_line1 = get_sim(ugrid[0]).res["time_steps"]


for obs in obs_list:
    res1[obs] = np.zeros((4, len(time_line1)), dtype=float)
    for ii, g in enumerate(vals["g"]):
        res1[obs][ii, :] = get_sim(ugrid[ii]).res[obs]
for micro in ["micro_delta", "micro_N_single", "micro_N_pair", "Deff", "Hspace_size"]:
    res1[micro] = np.zeros(4, dtype=float)
    for ii, g in enumerate(vals["g"]):
        res1[micro][ii] = get_sim(ugrid[ii]).res[micro]

obs = "N_single"
fig, ax = plt.subplots(2, 2, constrained_layout=True, sharex=True, sharey=True)
for ii, g in enumerate(vals["g"]):
    ir = ii // 2
    ic = ii % 2
    m = get_sim(ugrid[ii]).par["m"]
    ax[ir, ic].grid()
    ax[ir, ic].axhline(
        y=res1[f"micro_{obs}"][ii],
        linestyle="-",
        c="black",
        linewidth=0.9,
        label="micro NOBG",
    )
    ax[ir, ic].axhline(
        y=res["micro_Nsingle"][ii],
        linestyle="-",
        c="darkgreen",
        linewidth=0.9,
        label="micro BG",
    )
    """
    ax[ir, ic].plot(
        time_line,
        time_integral(time_line, res[obs][ii, :]),
        "--",
        label=f"BGs m={m},g={g}",
    )
    ax[ir, ic].plot(
        time_line1,
        time_integral(time_line1, res1[obs][ii, :]),
        "--",
        label=f"NOBG  D= {round(-np.log(res1['Deff'][ii])/np.log(res1['Hspace_size'][ii]),4)}",
        c="darkred",
    )
    """
    ax[ir, ic].plot(
        time_line,
        res[obs][ii, :],
        "--",
        label=f"BG m={m},g={g}",
    )

    ax[ir, ic].plot(
        time_line1,
        res1[obs][ii, :],
        "--",
        label=f"nobg",
    )
    if ic == 0:
        ax[ir, ic].set(ylabel=f"{obs}")
    if ir == 1:
        ax[ir, ic].set(xlabel=f"time")
    ax[ir, ic].legend()
plt.savefig(
    f"snapshots_{obs}.pdf",
)
# %%
# DYNAMICS
data = np.loadtxt(f"delta_Q0_N8g5m1.txt")
dataDFL = np.loadtxt(f"delta4.txt")
tline3 = 10 * np.arange(len(dataDFL)) / len(dataDFL)
tline4 = 60 * np.arange(len(data)) / len(data)

res1 = {}

config_filename = f"DFL/testgiuseppe"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = ["delta"]
for obs in obs_list:
    res1[obs] = get_sim(ugrid[0]).res[obs]
tline1 = get_sim(ugrid[0]).res["time_steps"]

res2 = {}
config_filename = f"DFL/tested2"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
for obs in obs_list:
    res2[obs] = get_sim(ugrid[0]).res[obs]
tline2 = get_sim(ugrid[0]).res["time_steps"]

obs = "delta"
fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.plot(tline4, data, "-", label="Giuseppe")
ax.plot(tline1, res1[obs], "x", label="ED")
ax.plot(tline2, res2[obs], "--", label="ED2")
ax.set(xlabel=r"$t$", ylabel=f"overlap")
ax.grid()
plt.legend()
# %%
fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.plot(tline4, np.abs(data - res1[obs]), "-", label="diff")
# ax.plot(tline, res1[obs], "+", label="ED2")
ax.set(xlabel=r"$t$", ylabel=f"overlap", yscale="log")
ax.grid()
plt.legend()
# %%
res = {}
config_filename = f"test/dfl"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = ["delta"]
for obs in obs_list:
    res[obs] = get_sim(ugrid[0]).res[obs]
tline = get_tline(get_sim(ugrid[0]).par["dynamics"])
obs = "delta"
fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.plot(tline, res[obs], "--", label="ED")
ax.plot(tline1, dataDFL, "-", label="Giuseppe")
ax.set(xlabel=r"$t$", ylabel=f"overlap")
ax.grid()
plt.legend()
# %%
res = {}
config_filename = f"DFL/testgiuseppe"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = ["delta"]
for obs in obs_list:
    res[obs] = get_sim(ugrid[0]).res[obs]
tline = get_tline(get_sim(ugrid[0]).par["dynamics"])
obs = "delta"
fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.plot(tline, res[obs], "--", label="ED")
ax.set(xlabel=r"$t$", ylabel=f"overlap")
ax.grid()
plt.legend()
# %%
# DYNAMICS
res1 = {}
config_filename = f"DFL/test"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = ["delta", "entropy", "overlap"][:2]
for obs in obs_list:
    res1[obs] = get_sim(ugrid[0]).res[obs]
tline1 = (
    np.arange(res1["delta"].shape[0]) * get_sim(ugrid[0]).par["dynamics"]["delta_n"]
)
obs = "delta"
fig, ax = plt.subplots(1, 1, constrained_layout=True)
for ii, obs in enumerate(obs_list[:1]):
    ax.plot(tline[1:], time_integral(tline, res[obs])[1:], "--", label="DW")
    ax.plot(tline1[1:], time_integral(tline1, res1[obs])[1:], "-", label="DW-BG")
    ax.set(xlabel=r"$t$", ylabel=f"time integral({obs})")
    # ax.set(ylim=[0, 1.1])
    # if ii == 1:
    # ax[ii].set(xscale="log")
    ax.grid()
plt.legend()
# np.savetxt(f"overlap6.txt", res["overlap"], fmt="%f")
# np.savetxt(f"delta6.txt", res["delta"], fmt="%f")
plt.savefig(f"time_integral_N8m1g5_2siteDW.pdf")
# %%
# DYNAMICS
res = {}
config_filename = f"scars/testDFL"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
for obs in ["delta", "E_square", "overlap"]:
    res[obs] = get_sim(ugrid[0]).res[obs]
fig, ax = plt.subplots(1, 1, constrained_layout=True)
tline = (
    np.arange(res["overlap"].shape[0]) * get_sim(ugrid[0]).par["dynamics"]["delta_n"]
)
ax.plot(tline, res["overlap"], label="Vacuum")
# ax.plot(tline, res["delta"], label="DFL")
ax.set(xlabel=r"$t$", ylabel="overlap")
ax.grid()
plt.legend()
# %%
tline = np.arange(res["delta"].shape[0]) * get_sim(ugrid[0]).par["dynamics"]["delta_n"]
fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.plot(tline, res["N_tot"], label="Vacuum")
# ax.plot(tline, res["delta"], label="DFL")
ax.set(xlabel=r"$t$", ylabel="Delta")
ax.grid()
plt.legend()
# %%
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
delta_n = 0.1
n_steps = int((stop - start) / delta_n)
time_steps = np.arange(n_steps) * delta_n

res = {}
for state in ["PV", "V"]:
    res[state] = {}
    for ii in range(1, 3, 1):
        config_filename = f"SU2/{state}_no_scars{ii}"
        match = SimsQuery(group_glob=config_filename)
        ugrid, vals = uids_grid(match.uids, ["g"])
        res[state][ii] = {}
        res[state][ii]["g"] = vals["g"][0]
        name = get_sim(ugrid[0]).par["dynamics"]["state"]
        res[state][ii]["name"] = name
        m = get_sim(ugrid[0]).par["m"]
        res[state][ii]["m"] = m
        ref_state = get_sim(ugrid[0]).par["ensemble"]["microcanonical"]["state"]
        res[state][ii]["ref_state"] = ref_state
        for obs in [
            f"overlap_{name}",
            "canonical_avg",
            "microcan_avg",
            "diagonal_avg",
        ]:
            res[state][ii][obs] = get_sim(ugrid[0]).res[obs]

        res[state][ii]["N2"] = (
            custom_average(get_sim(ugrid[0]).res["N_pair"], "even")
            + (1 - custom_average(get_sim(ugrid[0]).res["N_pair"], "odd")) / 2
        )
        res[state][ii]["N1"] = custom_average(get_sim(ugrid[0]).res["N_single"])
        res[state][ii]["N0"] = 1 - res[state][ii]["N1"] - res[state][ii]["N2"]

save_dictionary(res, "dynamics_thermal.pkl")
# %%
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

for s, state in enumerate(["PV", "V"]):
    for ind in range(1, 3, 1):
        number = 2 * s + ind - 1
        ax[0, number].plot(
            time_steps,
            res[state][ind][f"overlap_{name}"],
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
        )
        ax[1, number].axhline(
            y=res[state][ind]["canonical_avg"],
            color="red",
            linestyle="-",
            linewidth=2,
            label=r"canonical",
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
            + f"{res[state][ind]['ref_state']}"
            + "}^{TH} (m="
            + f"{res[state][ind]['m']}"
            + ",g^{2}="
            + f"{res[state][ind]['g']})$",
            xy=(0.935, 0.9),
            xycoords="axes fraction",
            fontsize=10,
            horizontalalignment="right",
            verticalalignment="bottom",
            bbox=dict(facecolor="white", edgecolor="black"),
        )

    ax[0, 0].set(
        ylabel=r"overlap $|\langle\psi(t)|\psi(0)\rangle|^{2}$", ylim=[-0.1, 1.1]
    )
    ax[1, 0].set(ylabel=r"density $\rho_{1}$", xlabel="Time")
    ax[1, 1].legend(loc="best", ncol=1, fontsize=8.8)
plt.savefig(f"dynamics_thermal.pdf")


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
# %%
# ========================================================================
# SU(2) SIMULATIONS PURE FLUCTUATIONS
# ========================================================================
config_filename = "SU2/pure/fluctuations"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = LGT_obs_list(model="SU2", pure=True, has_obc=True)
res = {"g": vals["g"]}
for obs in obs_list:
    res[obs] = []
    for ii in range(len(res["g"])):
        res[obs].append(get_sim(ugrid[ii]).res[obs])
    res[obs] = np.asarray(res[obs])

fig, ax = plt.subplots()
ax.plot(res["g"], res["E_square"], "-o", label=f"E2")
ax.plot(res["g"], res["delta_E_square"], "-o", label=f"Delta")
ax.set(xscale="log")
ax2 = ax.twinx()
ax2.plot(res["g"], -res["plaq"] + max(res["plaq"]), "-^", label=f"B2")
ax2.plot(res["g"], res["delta_plaq"], "-*", label=f"DeltaB")
ax.legend()
ax2.legend()
ax.grid()
save_dictionary(res, "saved_dicts/SU2_pure_fluctuations.pkl")
# %%
# ========================================================================
# SU(2) SIMULATIONS PURE TOPOLOGY
# ========================================================================
config_filename = "SU2/pure/topology"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = LGT_obs_list(model="SU2", pure=True, has_obc=False)
res = {"g": vals["g"]}

for obs in ["energy", "py_sector", "px_sector"]:
    res[obs] = np.zeros((vals["g"].shape[0], 5))
    for ii in range(len(res["g"])):
        for n in range(5):
            res[obs][ii][n] = get_sim(ugrid[ii]).res[obs][n]
fig = plt.figure()
for n in range(1, 5):
    plt.plot(
        vals["g"],
        res["energy"][:, n] - res["energy"][:, 0],
        "-o",
        label=f"{format(res['px_sector'][0, n],'.5f')}",
    )
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.ylabel("energy")
save_dictionary(res, "saved_dicts/SU2_pure_topology.pkl")
# %%
# ========================================================================
# SU(2) FULL TOPOLOGY 1
# ========================================================================
config_filename = "SU2/full/topology1"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=True)
res = {"g": vals["g"], "m": vals["m"]}
for obs in ["energy", "py_sector", "px_sector"]:
    res[obs] = np.zeros((res["g"].shape[0], res["m"].shape[0]))
    for ii, g in enumerate(res["g"]):
        for jj, m in enumerate(res["m"]):
            res[obs][ii][jj] = get_sim(ugrid[ii][jj]).res[obs]
fig = plt.figure()
for jj, m in enumerate(res["m"]):
    plt.plot(vals["g"], 1 - res["py_sector"][:, jj], "-o", label=f"m={format(m,'.3f')}")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.ylabel("1-py_sector")
save_dictionary(res, "saved_dicts/SU2_full_topology1.pkl")
# %%
# ========================================================================
# SU(2) FULL TOPOLOGY 2
# ========================================================================
config_filename = "SU2/full/topology2"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=True)
res = {"g": vals["g"], "m": vals["m"]}
for obs in ["energy", "py_sector", "px_sector"]:
    res[obs] = np.zeros((res["g"].shape[0], res["m"].shape[0]))
    for ii, g in enumerate(res["g"]):
        for jj, m in enumerate(res["m"]):
            res[obs][ii][jj] = get_sim(ugrid[ii][jj]).res[obs]
fig = plt.figure()
for ii, g in enumerate(res["g"]):
    plt.plot(vals["m"], 1 - res["py_sector"][ii, :], "-o", label=f"g={format(g,'.3f')}")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.ylabel("1-py_sector")
save_dictionary(res, "saved_dicts/SU2_full_topology2.pkl")
# %%
# ========================================================================
# SU(2) PHASE DIAGRAM
# ========================================================================
config_filename = "SU2/full/phase_diagram"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=False)
res = {"g": vals["g"], "m": vals["m"]}
for obs in obs_list:
    res[obs] = np.zeros((res["g"].shape[0], res["m"].shape[0]))
    for ii, g in enumerate(res["g"]):
        for jj, m in enumerate(res["m"]):
            res[obs][ii][jj] = get_sim(ugrid[ii][jj]).res[obs]

fig, axs = plt.subplots(
    3,
    1,
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
obs = ["E_square", "rho", "spin"]
for ii, ax in enumerate(axs.flat):
    # IMSHOW
    img = ax.imshow(
        np.transpose(res[obs[ii]]),
        origin="lower",
        cmap="magma",
        extent=[-2, 2, -3, 1],
    )
    ax.set_ylabel(r"m")
    axs[2].set_xlabel(r"g2")
    ax.set(xticks=[-2, -1, 0, 1, 2], yticks=[-3, -2, -1, 0, 1])
    ax.xaxis.set_major_formatter(fake_log)
    ax.yaxis.set_major_formatter(fake_log)

    cb = fig.colorbar(
        img,
        ax=ax,
        aspect=20,
        location="right",
        orientation="vertical",
        pad=0.01,
        label=obs[ii],
    )
save_dictionary(res, "saved_dicts/SU2_full_phase_diagram.pkl")
# %%
# ========================================================================
# SU(2) FULL THEORY CHARGE vs DENSITY
# ========================================================================
config_filename = "SU2/full/charge_vs_density"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=True)
res = {"g": vals["g"], "m": vals["m"]}
for obs in ["n_tot_even", "n_tot_odd"]:
    res[obs] = np.zeros((res["g"].shape[0], res["m"].shape[0]))
    for ii, g in enumerate(res["g"]):
        for jj, m in enumerate(res["m"]):
            res[obs][ii][jj] = get_sim(ugrid[ii][jj]).res[obs]
fig = plt.figure()
for ii, m in enumerate(res["m"]):
    plt.plot(
        vals["g"],
        2 + res["n_tot_even"][:, ii] - res["n_tot_odd"][:, ii],
        "-o",
        label=f"g={format(m,'.3f')}",
    )
plt.xscale("log")
plt.legend()
plt.ylabel("rho")
save_dictionary(res, "saved_dicts/charge_vs_density.pkl")
# %%
# ========================================================================
# SU(2) FULL THEORY TTN COMPARISON
# ========================================================================
config_filename = "SU2/full/TTN_comparison"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=True)
res = {"g": vals["g"]}
for obs in ["energy", "n_tot_even", "n_tot_odd", "E_square"]:
    res[obs] = np.zeros(res["g"].shape[0])
    for ii, g in enumerate(res["g"]):
        res[obs][ii] = get_sim(ugrid[ii]).res[obs]
fig = plt.figure()
plt.plot(vals["g"], 2 + res["n_tot_even"][:] - res["n_tot_odd"][:], "-o")
plt.xscale("log")
plt.legend()
plt.ylabel("rho")
save_dictionary(res, "saved_dicts/TTN_comparison.pkl")
# %%
# ========================================================================
# SU(2) FULL THEORY ENERGY GAPS
# ========================================================================
config_filename = "SU2/full/energy_gaps"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["DeltaN", "g", "k"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=True)
res = {"DeltaN": vals["DeltaN"], "g": vals["g"], "k": vals["k"]}
res_shape = (res["DeltaN"].shape[0], res["g"].shape[0], res["k"].shape[0])
res["energy"] = np.zeros(res_shape)
res["m"] = np.zeros(res_shape)
for ii, DeltaN in enumerate(res["DeltaN"]):
    for jj, g in enumerate(res["g"]):
        for kk, k in enumerate(res["k"]):
            res["m"][ii, jj, kk] = get_sim(ugrid[ii, jj, kk]).res["m"]
            res["energy"][ii, jj, kk] = get_sim(ugrid[ii, jj, kk]).res["energy"][0]

fig = plt.figure()
for kk, k in enumerate(res["k"]):
    plt.plot(
        vals["g"] ** 2,
        res["energy"][1, :, kk] - res["energy"][0, :, kk],
        "--o",
        label=f"k={k}, TOT",
    )


plt.xscale("log")
plt.yscale("log")
plt.xlabel("g2")
plt.legend()
plt.ylabel("DEltaE")

fig = plt.figure()
for kk, k in enumerate(res["k"]):
    plt.plot(
        vals["g"] ** 2,
        res["energy"][1, :, kk] - res["energy"][0, :, kk] - 0.5 * res["m"][1, :, kk],
        "-^",
        label=f"k={k} RES",
    )
plt.xscale("log")
plt.yscale("log")
plt.xlabel("g2")
plt.legend()
plt.ylabel("DEltaE_res")
save_dictionary(res, "saved_dicts/SU2_energy_gap.pkl")

# %%
# ========================================================================
# SU(2) FULL THEORY ENERGY GAPS
# ========================================================================
config_filename = "SU2/full/energy_gaps"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["DeltaN", "m", "k"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=False)
res = {"DeltaN": vals["DeltaN"], "m": vals["m"], "k": vals["k"]}
res_shape = (res["DeltaN"].shape[0], res["m"].shape[0], res["k"].shape[0])
res["energy"] = np.zeros(res_shape)
res["g"] = np.zeros(res_shape)
for ii, DeltaN in enumerate(res["DeltaN"]):
    for jj, m in enumerate(res["m"]):
        for kk, k in enumerate(res["k"]):
            res["g"][ii, jj, kk] = get_sim(ugrid[ii, jj, kk]).res["g"]
            res["energy"][ii, jj, kk] = get_sim(ugrid[ii, jj, kk]).res["energy"][0]

fig = plt.figure()
for kk, k in enumerate(res["k"]):
    plt.plot(
        res["m"],
        res["energy"][1, :, kk] - res["energy"][0, :, kk],
        "--o",
        label=f"k={k}",
    )
plt.xscale("log")
plt.yscale("log")
plt.xlabel("m")
plt.legend()
plt.ylabel("DEltaE")

fig = plt.figure()
for kk, k in enumerate(res["k"]):
    plt.plot(
        vals["m"],
        res["energy"][1, :, kk] - res["energy"][0, :, kk] - 0.5 * res["m"],
        "-^",
        label=f"k={k} RES",
    )
plt.xscale("log")
plt.yscale("log")
plt.xlabel("g2")
plt.legend()
plt.ylabel("DEltaE_res")
save_dictionary(res, "SU2_energy_gap_new.pkl")
