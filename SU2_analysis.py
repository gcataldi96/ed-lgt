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


def gaussian_time_integral(time, M, sigma=None):
    """
    Computes a locally averaged version of the observable M using a Gaussian window.

    For each time point t, the function computes a weighted average of M over all times,
    where the weights are given by a Gaussian function centered at t. This helps to
    suppress the influence of the initial condition and improves convergence.

    Parameters:
        time (numpy.ndarray): 1D array of time points (can be non-uniform).
        M (numpy.ndarray): 1D array of observable values corresponding to each time point.
        sigma (float, optional): Width of the Gaussian window (in the same units as time).
            If None, sigma defaults to one-tenth of the total time range.

    Returns:
        numpy.ndarray: Array of the locally averaged observable.
    """
    # Choose a default sigma if none is provided.
    if sigma is None:
        sigma = (time[-1] - time[0]) / 10.0

    M_smoothed = np.zeros_like(M)

    # For each time point, compute the Gaussian-weighted average.
    for i, t in enumerate(time):
        # Compute Gaussian weights centered at t.
        weights = np.exp(-0.5 * ((time - t) / sigma) ** 2)
        # Use numerical integration (trapezoidal rule) to perform the weighted average.
        weighted_sum = np.trapz(weights * M, time)
        weight_norm = np.trapz(weights, time)
        M_smoothed[i] = weighted_sum / weight_norm

    return M_smoothed


def moving_time_integral(time, M, max_points=100):
    """
    Computes a running time average of an observable M over a moving window of at most `max_points`
    time steps. In the beginning, when there are fewer than `max_points` steps, the average is taken
    over all available time points. This way, after some time the average "forgets" the early transient.

    Parameters:
        time (numpy.ndarray): 1D array of time points (can be non-uniformly spaced).
        M (numpy.ndarray): 1D array of observable values corresponding to each time point.
        max_points (int): Maximum number of points in the moving window for averaging.

    Returns:
        numpy.ndarray: Array of the running averaged observable.
    """
    M_avg = np.zeros_like(M)

    for i in range(len(time)):
        # Determine the starting index of the moving window.
        start = max(0, i - max_points + 1)
        t_segment = time[start : i + 1]
        M_segment = M[start : i + 1]

        # Compute the integral over the selected time window using the trapezoidal rule.
        # Then normalize by the width of the time window to get an average.
        dt = t_segment[-1] - t_segment[0]
        if dt != 0:
            integrated_value = np.trapz(M_segment, t_segment)
            M_avg[i] = integrated_value / dt
        else:
            M_avg[i] = M_segment[0]

    return M_avg


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
# ===================================================================
# STRING BREAKING PHASE DIAGRAM
# ===================================================================
res = {}
config_filename = f"LBO/su2_phase_diagram"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])

res = {
    "entropy": np.zeros((len(vals["g"]), len(vals["m"]))),
    "E_square": np.zeros((len(vals["g"]), len(vals["m"]))),
    "N_single": np.zeros((len(vals["g"]), len(vals["m"]))),
    "N_pair": np.zeros((len(vals["g"]), len(vals["m"]))),
    "N_tot": np.zeros((len(vals["g"]), len(vals["m"]))),
}

for ii, g in enumerate(vals["g"]):
    for kk, m in enumerate(vals["m"]):
        res["entropy"][ii, kk] = get_sim(ugrid[ii][kk]).res["entropy"]
        res["E_square"][ii, kk] = get_sim(ugrid[ii][kk]).res["E_square"]
        res["N_single"][ii, kk] = get_sim(ugrid[ii][kk]).res["N_single"]
        res["N_pair"][ii, kk] = get_sim(ugrid[ii][kk]).res["N_pair"]
        res["N_tot"][ii, kk] = get_sim(ugrid[ii][kk]).res["N_tot"]

# %%
obs = "N_tot"
fig, ax = plt.subplots(1, 1, constrained_layout=True)

X = np.transpose(res[obs])
img = plt.imshow(X, cmap="magma", origin="lower", extent=[-2, 1, -2, 1])
ax.set(
    ylabel=r"m",
    xlabel=r"g^{2}",
    xticks=[-2, -1, 0, +1],
    yticks=[-2, -1, 0, +1],
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
eff_res = {}
config_filename = f"LBO/su2_effectivebasis"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["m"])
truncation_values = [
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
    1e-8,
    1e-9,
    1e-10,
    1e-11,
    1e-12,
    1e-13,
    1e-14,
]
n_trunc = len(truncation_values)
eff_res = {
    "energy": np.zeros((len(vals["m"]), n_trunc)),
    "E_square": np.zeros((len(vals["m"]), n_trunc)),
    "N_single": np.zeros((len(vals["m"]), n_trunc)),
    "N_pair": np.zeros((len(vals["m"]), n_trunc)),
    "N_tot": np.zeros((len(vals["m"]), n_trunc)),
    "eff_basis": np.zeros((len(vals["m"]), n_trunc)),
}
true_res = {
    "energy": np.zeros(len(vals["m"])),
    "E_square": np.zeros(len(vals["m"])),
    "N_single": np.zeros(len(vals["m"])),
    "N_pair": np.zeros(len(vals["m"])),
    "N_tot": np.zeros(len(vals["m"])),
}

for ii, m in enumerate(vals["m"]):
    eff_res["eff_basis"][ii] = get_sim(ugrid[ii]).res["eff_basis"][:]
    eff_res["energy"][ii] = get_sim(ugrid[ii]).res["eff_energy"][:, 0]
    eff_res["E_square"][ii] = get_sim(ugrid[ii]).res["eff_E_square"][:, 0]
    eff_res["N_single"][ii] = get_sim(ugrid[ii]).res["eff_N_single"][:, 0]
    eff_res["N_pair"][ii] = get_sim(ugrid[ii]).res["eff_N_pair"][:, 0]
    eff_res["N_tot"][ii] = get_sim(ugrid[ii]).res["eff_N_tot"][:, 0]
    true_res["energy"][ii] = get_sim(ugrid[ii]).res["energy"][0]
    true_res["E_square"][ii] = get_sim(ugrid[ii]).res["E_square"][0]
    true_res["N_single"][ii] = get_sim(ugrid[ii]).res["N_single"][0]
    true_res["N_pair"][ii] = get_sim(ugrid[ii]).res["N_pair"][0]
    true_res["N_tot"][ii] = get_sim(ugrid[ii]).res["N_tot"][0]

# %%
from ed_lgt.operators import spin_space

irrep_res = {}
config_filename = f"LBO/su2_irrepbasis"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["m"])
spin_list = np.arange(0, spin_space(4.5), 1)[1:] / 2
n_spins = len(spin_list)
irrep_basis = np.array([6, 10, 14, 18, 22, 26, 30, 34, 38])
irrep_res = {
    "energy": np.zeros((len(vals["m"]), n_spins)),
    "E_square": np.zeros((len(vals["m"]), n_spins)),
    "N_single": np.zeros((len(vals["m"]), n_spins)),
    "N_pair": np.zeros((len(vals["m"]), n_spins)),
    "N_tot": np.zeros((len(vals["m"]), n_spins)),
}

for ii, m in enumerate(vals["m"]):
    irrep_res["energy"][ii] = get_sim(ugrid[ii]).res["energy"][:, 0]
    irrep_res["E_square"][ii] = get_sim(ugrid[ii]).res["E_square"][:, 0]
    irrep_res["N_single"][ii] = get_sim(ugrid[ii]).res["N_single"][:, 0]
    irrep_res["N_pair"][ii] = get_sim(ugrid[ii]).res["N_pair"][:, 0]
    irrep_res["N_tot"][ii] = get_sim(ugrid[ii]).res["N_tot"][:, 0]

# %%
for ii, m in enumerate(vals["m"]):
    print(f"-{ii}------{m}----------------------")
    print(true_res["energy"][ii])
    print(np.abs(irrep_res["energy"][ii] - true_res["energy"][ii]))
    print("-----------------------------")

# %%
fig, ax = plt.subplots(1, 1, constrained_layout=True, sharex=True, sharey="row")
sm = cm.ScalarMappable(
    cmap="magma", norm=LogNorm(vmin=vals["m"].min(), vmax=vals["m"].max())
)
palette = sm.to_rgba(vals["m"])

ax.grid()
ax.set(
    yscale="log",
    ylim=[1e-17, 1e-1],
    ylabel=r"$\Delta E = |E_{j} - E_{5}|$",
    xlabel=r"local dim $d(j)$",
)
for ii, m in enumerate(vals["m"]):
    ax.plot(
        irrep_basis,
        np.abs(irrep_res["energy"][ii] - true_res["energy"][ii]),
        "o-",
        label=f"m={m}",
        c=palette[ii],
        markersize=2,
        markeredgecolor=palette[ii],
        markerfacecolor="black",
        markeredgewidth=1,
    )
cb = fig.colorbar(
    sm, ax=ax, aspect=80, location="top", orientation="horizontal", pad=0.02
)
cb.set_label(label=r"$m$", labelpad=-22, x=-0.02, y=0)


# %%
def trim_from_first_nonzero(a, tol=0.0):
    """
    Return (a_trimmed, start_idx) where a_trimmed = a[start_idx:],
    and start_idx is the index of the first entry with |a[i]| > tol.
    If no such entry exists, returns (empty_array, len(a)).
    """
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    mask = np.abs(a) > tol
    nz = np.flatnonzero(mask)
    start = int(nz[0]) if nz.size else len(a)
    return a[start:], start


for ii, m in enumerate(vals["m"]):
    eff_energy, index = trim_from_first_nonzero(eff_res["energy"][ii], tol=1e-10)
    print(f"-------- {ii}-------")
    fig, ax = plt.subplots(1, 1, constrained_layout=True, sharex=True, sharey="row")
    ax.grid()
    ax.set(yscale="log")
    ax.plot(
        irrep_basis,
        np.abs(irrep_res["energy"][ii] - true_res["energy"][ii]),
        "o-",
        linewidth=1,
        label=f"irrep",
        markersize=6,
        markerfacecolor="black",
        markeredgewidth=2,
    )
    ax.plot(
        eff_res["eff_basis"][ii, index:],
        np.abs(eff_energy - true_res["energy"][ii]),
        "o-",
        label=f"eff",
        linewidth=1,
        markersize=3,
        markerfacecolor="white",
        markeredgewidth=0.5,
    )
    ax.text(
        0.5,
        0.88,  # 5% in from left, 95% up from bottom
        rf"$m={round(m,4)}$",  # e.g. "(a)", "(b)", …
        transform=ax.transAxes,  # interpret coords relative to the axes
        ha="left",
        va="top",  # align text box
        fontsize=12,  # tweak if you like
        bbox=dict(facecolor="white", alpha=0.2, edgecolor="black"),
    )
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
save_dictionary(res, f"minimal_string.pkl")

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
plt.savefig(f"zerodensity.pdf")
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
save_dictionary(res, f"spin1.pkl")
plt.savefig(f"spin1.pdf")
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
config_filename = f"string_breaking/test"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["m"])

res = {"time_steps": get_sim(ugrid[0]).res["time_steps"]}
nsteps = len(res["time_steps"])

obs_list = ["entropy", "E2", "N_single", "N_pair", "N_zero", "N_tot", "tot_ov_max"]
for ii in range(24):
    obs_list.append(f"ov_max{ii}")

res["self_cross"] = np.zeros((len(vals["m"]), nsteps))

for obs in obs_list:
    res[f"{obs}"] = np.zeros((len(vals["m"]), nsteps))
    for kk, m in enumerate(vals["m"]):
        res[obs][kk] = get_sim(ugrid[kk]).res[obs]

res["self_cross"] = np.zeros((len(vals["m"]), nsteps))
for ii in [5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
    res["self_cross"][0] += res[f"ov_max{ii}"][0]

res["self_cross1"] = np.zeros((len(vals["m"]), nsteps))
for ii in [5, 10, 12, 14, 16, 18, 20]:
    res["self_cross1"][0] += res[f"ov_max{ii}"][0]

res["self_cross2"] = np.zeros((len(vals["m"]), nsteps))
for ii in [6, 11, 13, 15, 17, 19, 21, 22]:
    res["self_cross2"][0] += res[f"ov_max{ii}"][0]

obs_color = ["darkblue", "darkred", "orange", "darkgreen"]
obs_names = [r"$E^{2}$", r"$N_{\rm{quarks}}$", r"$N_{\rm{baryon}}$", r"$N_{\rm{tot}}$"]
obs_size = [1, 1.5, 1.4, 1]
save_dictionary(res, f"4x3.pkl")
fig, ax = plt.subplots(2, 1, constrained_layout=True, sharex=True, sharey="row")
ax[0].set(ylabel=r"fidelity $F=|\langle \psi_{0}|\psi(t)\rangle|^{2}$")
ax[1].set(ylabel=r"observables")
ax[1].set(xlabel=r"time $t$ ($m=25$)")


ax[0].plot(
    res["time_steps"],
    res[f"ov_max0"][0],
    "o-",
    markersize=1.5,
    markeredgewidth=0.2,
    label="snake",
    linewidth=0.8,
)
ax[0].plot(
    res["time_steps"],
    res[f"tot_ov_max"][0],
    "o-",
    markersize=1.5,
    markeredgewidth=0.2,
    label="tot",
    linewidth=0.8,
)
ax[0].plot(
    res["time_steps"],
    res[f"ov_max23"][0],
    "o-",
    markersize=1.5,
    markeredgewidth=0.2,
    label="11",
    linewidth=0.8,
)
ax[0].plot(
    res["time_steps"],
    res[f"self_cross"][0],
    "o-",
    markersize=1.5,
    markeredgewidth=0.2,
    label="self-crossing",
    linewidth=0.8,
)
ax[0].plot(
    res["time_steps"],
    res[f"self_cross1"][0],
    "o-",
    markersize=1.5,
    markeredgewidth=0.2,
    label="self-cross1",
    linewidth=0.8,
)
ax[0].plot(
    res["time_steps"],
    res[f"self_cross2"][0],
    "o-",
    markersize=1.5,
    markeredgewidth=0.2,
    label="self-cross2",
    linewidth=0.8,
)
ax[0].plot(
    res["time_steps"],
    res[f"ov_max1"][0] + res[f"ov_max4"][0],
    "o-",
    markersize=1.5,
    markeredgewidth=0.2,
    label="loops2",
    linewidth=0.8,
)
ax[0].plot(
    res["time_steps"],
    res[f"ov_max2"][0] + res[f"ov_max3"][0],
    "o-",
    markersize=1.5,
    markeredgewidth=0.2,
    label="loops1",
    linewidth=0.8,
)

ax[0].legend(
    bbox_to_anchor=(0.6, 0.368),
    ncol=1,
    frameon=True,
    labelspacing=0.1,
)
for jj, obs in enumerate(["E2", "N_single", "N_pair", "N_tot"]):
    ax[1].plot(
        res["time_steps"],
        res[f"{obs}"][0],
        "o-",
        c=obs_color[jj],
        markersize=obs_size[jj],
        markeredgewidth=0.2,
        label=f"{obs_names[jj]}",
        linewidth=0.8,
    )
ax[1].legend(
    bbox_to_anchor=(0.6, 0.368),
    ncol=1,
    frameon=True,
    labelspacing=0.1,
)
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
save_dictionary(res, f"finite_density.pkl")
plt.savefig(f"finite_density.pdf")
# %%
"""ax[0, 1].legend(
    bbox_to_anchor=(0.3, 0.45),
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
)"""

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
# %%
res = {}
config_filename = f"LBO/qed"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
res["eigvals"] = np.zeros((len(vals["g"]), 231), dtype=float)
for obs in ["E_px", "E_mx", "E_my", "E_py", "E_square"]:
    res[f"exp_{obs}"] = np.zeros((len(vals["g"]), 231), dtype=float)
    for ii, g in enumerate(vals["g"]):
        res[f"exp_{obs}"][ii] = get_sim(ugrid[ii]).res[f"exp_{obs}"]
for ii in range(len(vals["g"])):
    res["eigvals"][ii] = get_sim(ugrid[ii]).res["eigvals"]

sm = cm.ScalarMappable(cmap="copper")
palette = sm.to_rgba(vals["g"])
fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.grid()
for ii in range(len(vals["g"])):
    ax.plot(
        np.arange(231),
        res["eigvals"][ii],
        "o-",
        c=palette[ii],
        markersize=2,
        markerfacecolor=palette[ii],
        markeredgewidth=0.2,
    )
ax.set(yscale="log", ylim=[1e-10, 1])


custom_yticks = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
custom_yticklabels = [
    r"$0$",
    r"$2$",
    r"$4$",
    r"$6$",
    r"$8$",
    r"$10$",
    r"$12$",
    r"$14$",
    r"$16$",
    r"$18$",
]
fig, ax = plt.subplots(3, 2, constrained_layout=True, sharex=True, sharey=True)
for ii, axs in enumerate(ax.flatten()):
    axs.plot(
        np.arange(231),
        res[f"exp_E_square"][ii],
        "o",
        c=palette[ii],
        markersize=2,
        markerfacecolor=palette[ii],
        markeredgecolor="black",
        markeredgewidth=0.2,
    )
for ii in range(3):
    ax[ii, 0].set_yticks(custom_yticks)
    ax[ii, 0].set_yticklabels(custom_yticklabels)

fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.grid()
for obs in ["E_px", "E_mx", "E_my", "E_py"]:
    ax.plot(
        np.arange(231),
        res[f"exp_{obs}"][0],
        "o",
        markersize=2,
        markeredgewidth=0.5,
    )
ax.set(xlim=[0, 60])
# %%
custom_yticks = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
custom_yticklabels = [
    r"$0$",
    r"$2$",
    r"$4$",
    r"$6$",
    r"$8$",
    r"$10$",
    r"$12$",
    r"$14$",
    r"$16$",
    r"$18$",
]
fig, ax = plt.subplots(5, 1, constrained_layout=True)
ax.grid()
ax.plot(
    np.arange(231),
    res[f"exp_E_square"],
    "o",
    markersize=2,
    markeredgewidth=0.5,
)
ax.set_yticks(custom_yticks)
ax.set_yticklabels(custom_yticklabels)
"""ax.plot(
    np.arange(len(res["expvals1"])),
    res["expvals1"],
    "x",
    markersize=8,
    markerfacecolor="red",
    markeredgewidth=0.5,
)"""
# %%
# ENTROPY BG
# ==========================================================================
res = {}
config_filename = f"entropy_bg"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
res["time_line"] = np.delete(get_sim(ugrid[0]).res["time_steps"], 11)
res["entropy"] = np.zeros((len(vals["g"]), len(res["time_line"])), dtype=float)
for ii, g in enumerate(vals["g"]):
    res["entropy"][ii] = np.delete(get_sim(ugrid[ii]).res["entropy"], 11)

sm = cm.ScalarMappable(cmap="magma")
palette = sm.to_rgba(vals["g"])

fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.grid()
for ii, g in enumerate(vals["g"]):
    ax.plot(
        res["time_line"][1:],
        res["entropy"][ii, 1:],
        "o-",
        label=f"g={g}",
        c=palette[ii],
        markersize=1,
        markeredgecolor=palette[ii],
        markerfacecolor="black",
        markeredgewidth=0.5,
    )
ax.set(xlabel="t", ylabel="entropy", xscale="log")
cb = fig.colorbar(
    sm, ax=ax, aspect=80, location="top", orientation="horizontal", pad=0.02
)
cb.set_label(label=r"$g^{2}$", labelpad=-22, x=-0.02, y=0)
plt.savefig(f"entropy.pdf")
save_dictionary(res, f"entropy_bg.pkl")
# %%
# ENTROPY NO BG
# ==========================================================================
res = {}
config_filename = f"entropy_nobg"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
res["time_line"] = get_sim(ugrid[0]).res["time_steps"]
res["entropy"] = np.zeros((len(vals["g"]), len(res["time_line"])), dtype=float)
for ii, g in enumerate(vals["g"]):
    res["entropy"][ii] = get_sim(ugrid[ii]).res["entropy"]

sm = cm.ScalarMappable(cmap="magma")
palette = sm.to_rgba(vals["g"])

fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.grid()
for ii, g in enumerate(vals["g"]):
    ax.plot(
        res["time_line"][1:],
        res["entropy"][ii, 1:],
        "o-",
        label=f"g={g}",
        c=palette[ii],
        markersize=1,
        markeredgecolor=palette[ii],
        markerfacecolor="black",
        markeredgewidth=0.5,
    )
ax.set(xlabel="t", ylabel="entropy", xscale="log")
cb = fig.colorbar(
    sm, ax=ax, aspect=80, location="top", orientation="horizontal", pad=0.02
)
cb.set_label(label=r"$g^{2}$", labelpad=-22, x=-0.02, y=0)
plt.savefig(f"entropy_nobg.pdf")
save_dictionary(res, f"entropy_nobg.pkl")
# %%
# FRAGMENTATION
# ==========================================================================
res = {}
idx = 0
config_filename = f"fragmentation_spectrum"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
res["overlap"] = get_sim(ugrid[idx]).res["overlap"]
res["energy"] = get_sim(ugrid[idx]).res["energy"]
res["entropy"] = get_sim(ugrid[idx]).res["entropy"]
res["r_array"] = get_sim(ugrid[idx]).res["r_array"]
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
ax[0].set(ylabel="Overlap", yscale="log", xlabel="Energy", ylim=[1e-17, 1])
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
plt.savefig(f"spectrum_fragmentation.pdf")
save_dictionary(res, f"frag_spectrum.pkl")
# %%
# DYNAMICS FRAGMENTATION
res = {}
idx = 0
config_filename = f"fragmentation_dynamics"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
for obs in [
    "overlap",
    "delta",
    "time_steps",
    "entropy",
    "N_single",
    "N_pair",
    "DE_N_single",
    "DE_N_pair",
    "DE_N_tot",
    "ME_N_single",
    "ME_N_pair",
    "ME_N_tot",
]:
    res[obs] = get_sim(ugrid[idx]).res[obs]


fig, ax = plt.subplots(4, 1, constrained_layout=True, sharex=True)
ax[0].plot(
    res["time_steps"],
    res["overlap"],
    "o-",
    markersize=0.5,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.2,
    linewidth=0.5,
)
ax[0].set(ylabel="Overlap")
ax[1].plot(
    res["time_steps"],
    res["entropy"],
    "o-",
    markersize=0.5,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.2,
    linewidth=0.5,
)
ax[1].set(ylabel="Ent Entropy")
ax[2].plot(
    res["time_steps"],
    res["delta"],
    "o-",
    markersize=0.5,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.2,
    linewidth=0.5,
)
ax[2].axhline(y=res["DE_N_tot"], linestyle="-", color="red")
ax[2].axhline(y=res["ME_N_tot"], linestyle="-.", color="darkgreen")
ax[2].set(ylabel="Imbalance")
ax[3].plot(
    res["time_steps"],
    res["N_single"],
    "o-",
    markersize=0.5,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.2,
    linewidth=0.5,
)
ax[3].set(ylabel="N_single", xlabel="time t")
ax[3].axhline(y=res["DE_N_single"], linestyle="-", color="red", label="DE")
ax[3].axhline(y=res["ME_N_single"], linestyle="-", color="darkgreen", label="ME")
fig.legend(
    bbox_to_anchor=(0.91, 0.25),
    ncol=2,
    frameon=True,
    labelspacing=0.1,
    bbox_transform=fig.transFigure,
)
save_dictionary(res, f"frag_dynamics.pkl")
plt.savefig(f"dynamics_fragmentation.pdf")


def moving_time_integral(time, M, max_points=100):
    """
    Computes a running time average of an observable M over a moving window of at most `max_points`
    time steps. In the beginning, when there are fewer than `max_points` steps, the average is taken
    over all available time points. This way, after some time the average "forgets" the early transient.

    Parameters:
        time (numpy.ndarray): 1D array of time points (can be non-uniformly spaced).
        M (numpy.ndarray): 1D array of observable values corresponding to each time point.
        max_points (int): Maximum number of points in the moving window for averaging.

    Returns:
        numpy.ndarray: Array of the running averaged observable.
    """
    M_avg = np.zeros_like(M)

    for i in range(len(time)):
        # Determine the starting index of the moving window.
        start = max(0, i - max_points + 1)
        t_segment = time[start : i + 1]
        M_segment = M[start : i + 1]

        # Compute the integral over the selected time window using the trapezoidal rule.
        # Then normalize by the width of the time window to get an average.
        dt = t_segment[-1] - t_segment[0]
        if dt != 0:
            integrated_value = np.trapz(M_segment, t_segment)
            M_avg[i] = integrated_value / dt
        else:
            M_avg[i] = M_segment[0]

    return M_avg


config_filename = f"fragment_dyn_bg"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
for obs in [
    "delta",
    "time_steps",
    "DE_N_tot",
    "ME_N_tot",
]:
    res[f"bg_{obs}"] = get_sim(ugrid[idx]).res[obs]
fig, ax = plt.subplots(1, 1, constrained_layout=True, sharex=True)
ax.plot(
    res["bg_time_steps"],
    moving_time_integral(res["bg_time_steps"], res["bg_delta"], 300),
    "o-",
    markersize=0.5,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.2,
    linewidth=0.5,
)
ax.plot(
    res["time_steps"],
    moving_time_integral(res["time_steps"], res["delta"], 300),
    "o-",
    markersize=0.5,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.2,
    linewidth=0.5,
)
save_dictionary(res, f"frag_imbalance.pkl")
# %%
# ===================================================================
# DFL PHASE DIAGRAM
# ===================================================================
res = {}
color = ["red", "blue", "green", "orange"]
config_filename = f"lbo"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
res["llambdas"] = np.zeros((len(vals["g"]), 58), dtype=float)
for ii, g in enumerate(vals["g"]):
    res["llambdas"][ii, :] = get_sim(ugrid[ii]).res["lambdas"][0, :][::-1]
fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.grid()
for ii, g in enumerate(vals["g"]):
    ax.plot(
        np.arange(58),
        res["llambdas"][ii, :],
        "o-",
        label=f"g={g}",
        c=color[ii],
        markersize=4,
        markeredgecolor=color[ii],
        markerfacecolor="white",
        markeredgewidth=0.5,
    )
ax.set(xlabel="lambda index", ylabel="lambda", yscale="log")
ax.set(ylim=[1e-11, 1])
ax.legend()


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
res = {}
config_filename = f"newDFL/phase_diagram_BG"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
for obs in ["N_tot", "N_single", "N_pair"]:
    res[f"ME_{obs}"] = np.zeros((len(vals["g"]), len(vals["m"])), dtype=float)
    res[f"DE_{obs}"] = np.zeros((len(vals["g"]), len(vals["m"])), dtype=float)
    res[f"diff_{obs}"] = np.zeros((len(vals["g"]), len(vals["m"])), dtype=float)
    for ii, g in enumerate(vals["g"]):
        for jj, m in enumerate(vals["m"]):
            res[f"DE_{obs}"][ii, jj] = get_sim(ugrid[ii, jj]).res[f"DE_{obs}"]
            res[f"ME_{obs}"][ii, jj] = get_sim(ugrid[ii, jj]).res[f"ME_{obs}"]
            res[f"diff_{obs}"][ii, jj] = np.abs(
                res[f"ME_{obs}"][ii, jj] - res[f"DE_{obs}"][ii, jj]
            )
save_dictionary(res, "phase_dictionary_BG.pkl")
# %%
# %%
# ==========================================================================
# 1D DFL TIME INTEGRAL
# ==========================================================================
res = {}
config_filename = f"newDFL/dynamics_BG"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
tline = get_sim(ugrid[0]).res["time_steps"]
res["delta"] = np.zeros((2, len(vals["g"]), len(tline)), dtype=float)
res["tline"] = tline
for ii, g in enumerate(vals["g"]):
    res["delta"][0, ii, :] = get_sim(ugrid[ii]).res["delta"]
for obs in ["N_tot", "N_single", "N_pair"]:
    res[obs] = np.zeros((2, len(vals["g"]), len(tline)), dtype=float)
    for en in ["DE", "ME"]:
        res[f"{en}_{obs}"] = np.zeros((2, len(vals["g"])), dtype=float)

for ii, g in enumerate(vals["g"]):
    for obs in ["N_tot", "N_single", "N_pair"]:
        res[obs][0, ii, :] = get_sim(ugrid[ii]).res[obs]
        for en in ["DE", "ME"]:
            res[f"{en}_{obs}"][0, ii] = get_sim(ugrid[ii]).res[f"{en}_{obs}"]
config_filename = f"newDFL/dynamics_NOBG"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
for ii, g in enumerate(vals["g"]):
    res["delta"][1, ii, :] = get_sim(ugrid[ii]).res["delta"]
    for obs in ["N_tot", "N_single", "N_pair"]:
        res[obs][1, ii, :] = get_sim(ugrid[ii]).res[obs]
        for en in ["DE", "ME"]:
            res[f"{en}_{obs}"][1, ii] = get_sim(ugrid[ii]).res[f"{en}_{obs}"]
save_dictionary(res, "dynamics_DFL.pkl")
# %%
fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.plot(
    tline,
    moving_time_integral(res["tline"], res["delta"][0, 1, :], 80),
    "--",
)
ax.plot(
    tline,
    moving_time_integral(res["tline"], res["delta"][1, 1, :], 80),
    "--",
)
ax.axhline(y=res["ME_N_tot"][0, 1], linestyle="-")
ax.axhline(y=res["ME_N_tot"][1, 1], linestyle="-.")
ax.axhline(y=res["DE_N_tot"][0, 1], linestyle="-")
ax.axhline(y=res["DE_N_tot"][1, 1], linestyle="-.")
# %%
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
fig, ax = plt.subplots(1, 1, constrained_layout=True)
X = np.transpose(res[f"DE_{obs}"])[:22, :22]
# X = np.transpose(res[f"diff_{obs}"])[:22,:22]
img = plt.imshow(X, cmap="magma", origin="lower", extent=[0, 10, 0, 10])

ax.set(ylabel=r"m", xlabel=r"g^{2}")

cb = fig.colorbar(
    img,
    ax=ax,
    aspect=20,
    location="right",
    orientation="vertical",
    pad=0.01,
    label=r"$\Delta$",
)
plt.savefig(f"phase_diagram_BG_delta.pdf")
# %%
res1 = {}
config_filename = f"newDFL/gscale_BG_obs"
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
save_dictionary(res1, "gscale_BG.pkl")
# for ii, g in enumerate(vals["g"]):
#    res1["entropy"][ii, :] = get_sim(ugrid[ii]).res["entropy"]

obs_names = [
    r"$\hat{\rho}_{\rm{1}}$",
    r"$\hat{\rho}_{\rm{2}}$",
    r"$\hat{\rho}_{\rm{0}}$",
]
sizes = [5, 8, 5]
styles = ["--", "-", ":"]
colors = ["darkgreen", "purple", "darkred"]
m = get_sim(ugrid[ii]).par["m"]
fig, ax = plt.subplots(1, 1, constrained_layout=True, sharex=True, sharey=True)
for ii, obs in enumerate(["N_single", "N_pair"]):
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
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.75))
plt.savefig(f"gscale_BG.pdf")
# %%
res1 = {}
config_filename = f"newDFL/gscale_NOBG"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = ["N_single", "N_pair", "N_zero"]
time_line1 = get_sim(ugrid[0]).res["time_steps"]
for obs in obs_list:
    res1[obs] = np.zeros((len(vals["g"]), len(time_line1)), dtype=float)
    res1[f"ME_{obs}"] = np.zeros(len(vals["g"]), dtype=float)
    res1[f"DE_{obs}"] = np.zeros(len(vals["g"]), dtype=float)
    for ii, g in enumerate(vals["g"]):
        res1[obs][ii, :] = get_sim(ugrid[ii]).res[obs]
        res1[f"ME_{obs}"][ii] = get_sim(ugrid[ii]).res[f"ME_{obs}"]
        res1[f"DE_{obs}"][ii] = get_sim(ugrid[ii]).res[f"DE_{obs}"]
save_dictionary(res1, "gscale_NOBG.pkl")
# %%
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
config_filename = f"newDFL/mscale"
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
