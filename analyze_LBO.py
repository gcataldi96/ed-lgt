# %%
from simsio import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from ed_lgt.tools import save_dictionary, fake_log, load_dictionary

textwidth_pt = 510.0
textwidth_in = textwidth_pt / 72.27
columnwidth_pt = 246.0
columnwidth_in = columnwidth_pt / 72.27

# %%
su2_res = {}
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
su2_res["phase_diagram"] = res

obs = "E_square"
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
su2_res["effective_basis"] = eff_res
su2_res["true"] = true_res
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
su2_res["irrep_basis"] = irrep_res


# %%
for ii, m in enumerate(vals["m"]):
    print(f"-{ii}------{m}----------------------")
    print(true_res["energy"][ii])
    print(np.abs(irrep_res["energy"][ii] - true_res["energy"][ii]))
    print("-----------------------------")


def first_index_below(a, threshold, *, inclusive=False, ignore_nan=True):
    """
    Return the first index i where a[i] < threshold (or <= if inclusive=True).
    If no such element exists, return None.

    Parameters
    ----------
    a : array-like (1D)
    threshold : float
    inclusive : bool, optional
        If True, use <= threshold instead of < threshold.
    ignore_nan : bool, optional
        If True, NaNs are ignored (not considered matches).

    Returns
    -------
    int or None
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 1:
        raise ValueError("Input must be a 1D array.")

    cond = (a <= threshold) if inclusive else (a < threshold)
    if ignore_nan:
        cond &= np.isfinite(a)

    idxs = np.flatnonzero(cond)
    return int(idxs[0]) if idxs.size else None


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


ratio = np.zeros((len(vals["m"]), 3), dtype=float)
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
    difference = np.abs(eff_energy - true_res["energy"][ii])
    print("==============================")
    print(f"{difference}")
    for kk, threshold in enumerate([1e-4, 1e-6, 1e-8]):
        idx = first_index_below(difference, threshold)
        if idx is None:
            single_ratio = 1
        else:
            single_ratio = eff_res["eff_basis"][ii, index + idx] / 42
        ratio[ii, kk] = single_ratio
        print(f"{ii}_{kk}__{idx}_________")
        print(single_ratio)
su2_res["mvals"] = vals["m"]
su2_res["ratio"] = ratio
save_dictionary(su2_res, "LBO_su2.pkl")
# %%
qed_res = {}
irrep_res = {}
config_filename = f"LBO/qed_irrepbasis"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
spin_list = np.arange(1, 10, 1)
n_spins = len(spin_list)
irrep_basis = np.array([19, 85, 231, 489, 891, 1469, 2255, 3281])
irrep_res = {
    "energy": np.zeros((len(vals["g"]), n_spins)),
    "E_square": np.zeros((len(vals["g"]), n_spins)),
}

for ii, g in enumerate(vals["g"]):
    irrep_res["energy"][ii] = get_sim(ugrid[ii]).res["energy"][:, 0]
    irrep_res["E_square"][ii] = get_sim(ugrid[ii]).res["E_square"][:, 0]

qed_res["irrep_basis"] = irrep_res

fig, ax = plt.subplots(1, 1, constrained_layout=True, sharex=True, sharey="row")
sm = cm.ScalarMappable(
    cmap="magma", norm=LogNorm(vmin=vals["g"].min(), vmax=vals["g"].max())
)
palette = sm.to_rgba(vals["g"])

ax.grid()
ax.set(
    yscale="log",
    xscale="log",
    ylabel=r"$\Delta E = |E_{j} - E_{9}|$",
    xlabel=r"local dim $d(j)$",
)
for ii, g in enumerate(vals["g"]):
    ax.plot(
        irrep_basis,
        np.abs(irrep_res["energy"][ii, :-1] - irrep_res["energy"][ii, -1]),
        "o-",
        label=f"g={g}",
        c=palette[ii],
        markersize=2,
        markeredgecolor=palette[ii],
        markerfacecolor="black",
        markeredgewidth=1,
    )
cb = fig.colorbar(
    sm, ax=ax, aspect=80, location="top", orientation="horizontal", pad=0.02
)
cb.set_label(label=r"$g$", labelpad=-22, x=-0.02, y=0)
# %%
eff_res = {}
config_filename = f"LBO/qed_effectivebasis"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
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
    "energy": np.zeros((len(vals["g"]), n_trunc)),
    "E_square": np.zeros((len(vals["g"]), n_trunc)),
    "eff_basis": np.zeros((len(vals["g"]), n_trunc)),
}
true_res = {
    "energy": np.zeros(len(vals["g"])),
    "E_square": np.zeros(len(vals["g"])),
}

for ii, g in enumerate(vals["g"]):
    eff_res["eff_basis"][ii] = get_sim(ugrid[ii]).res["eff_basis"][:]
    eff_res["energy"][ii] = get_sim(ugrid[ii]).res["eff_energy"][:, 0]
    eff_res["E_square"][ii] = get_sim(ugrid[ii]).res["eff_E_square"][:, 0]
    true_res["energy"][ii] = get_sim(ugrid[ii]).res["energy"][0]
    true_res["E_square"][ii] = get_sim(ugrid[ii]).res["E_square"][0]
qed_res["effective_basis"] = eff_res
qed_res["true"] = true_res


def qed_trim_from_first_nonzero(a, tol=0.0):
    """
    Return (a_trimmed, start_idx) where a_trimmed = a[start_idx:],
    and start_idx is the index of the first entry with |a[i]| > tol.
    If no such entry exists, returns (empty_array, len(a)).
    """
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    mask = np.abs(a) < tol
    nz = np.flatnonzero(mask)
    end = int(nz[0]) if nz.size else len(a)
    return a[:end], end


def first_index_below(a, threshold, *, inclusive=False, ignore_nan=True):
    """
    Return the first index i where a[i] < threshold (or <= if inclusive=True).
    If no such element exists, return None.

    Parameters
    ----------
    a : array-like (1D)
    threshold : float
    inclusive : bool, optional
        If True, use <= threshold instead of < threshold.
    ignore_nan : bool, optional
        If True, NaNs are ignored (not considered matches).

    Returns
    -------
    int or None
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 1:
        raise ValueError("Input must be a 1D array.")

    cond = (a <= threshold) if inclusive else (a < threshold)
    if ignore_nan:
        cond &= np.isfinite(a)

    idxs = np.flatnonzero(cond)
    return int(idxs[0]) if idxs.size else None


ratio = np.zeros((len(vals["g"]), 3), dtype=float)
for ii, g in enumerate(vals["g"]):
    eff_energy, index = qed_trim_from_first_nonzero(eff_res["energy"][ii], tol=1e-10)
    fig, ax = plt.subplots(1, 1, constrained_layout=True, sharex=True, sharey="row")
    ax.grid()
    ax.set(yscale="log", xscale="log")
    ax.plot(
        irrep_basis,
        np.abs(irrep_res["energy"][ii, :-1] - true_res["energy"][ii]),
        "o-",
        linewidth=1,
        label=f"irrep",
        markersize=6,
        markerfacecolor="black",
        markeredgewidth=2,
    )
    difference = np.abs(eff_energy - true_res["energy"][ii])
    print("==============================")
    print(f"{difference}")
    for kk, threshold in enumerate([1e-4, 1e-6, 1e-8]):
        idx = first_index_below(difference, threshold)
        if idx is None:
            single_ratio = 1
        else:
            single_ratio = eff_res["eff_basis"][ii, idx] / 4579
        ratio[ii, kk] = single_ratio
        print(f"{ii}_{kk}__{idx}_________")
        print(single_ratio)
    ax.plot(
        eff_res["eff_basis"][ii, :index],
        difference,
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
        rf"$g={round(g,4)}$",  # e.g. "(a)", "(b)", …
        transform=ax.transAxes,  # interpret coords relative to the axes
        ha="left",
        va="top",  # align text box
        fontsize=12,  # tweak if you like
        bbox=dict(facecolor="white", alpha=0.2, edgecolor="black"),
    )

fig, ax = plt.subplots(1, 1, constrained_layout=True, sharex=True, sharey="row")
ax.grid()
ax.set(yscale="log", xscale="log")
for kk, threshold in enumerate([1e-4, 1e-6, 1e-8]):
    ax.plot(
        vals["g"],
        ratio[:, kk],
        "o-",
        label=f"< {threshold:.0e}",
        linewidth=1,
        markersize=3,
        markerfacecolor="white",
        markeredgewidth=0.5,
    )
fig.legend()
qed_res["gvals"] = vals["g"]
qed_res["ratio"] = ratio
save_dictionary(qed_res, "LBO_qed.pkl")
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

# %%

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
