# %%
from simsio import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from ed_lgt.tools import save_dictionary, fake_log

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
