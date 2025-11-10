# %%
import numpy as np
from simsio import *
import math
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys
import logging

logger = logging.getLogger(__name__)
default_params = {
    "save_plot": {"bbox_inches": "tight", "transparent": False},
}


def bz_axis(Nk: int, *, numeric_labels: bool = False):
    """
    Build a symmetric Brillouin-zone axis for Nk momenta.
    Returns:
      k_path      : np.ndarray, length Nk+1, from -π to +π (inclusive)
      order       : list of indices (length Nk+1) that reorders data to [-π..π] with 0 in the middle and closes the loop
      tick_pos    : same as k_path (you can subsample if you want fewer ticks)
      tick_labels : LaTeX labels for ticks (either π-fractions or simple integers)
    Notes:
      - Assumes Nk is even.
      - `order` matches the common pattern [Nk//2 .. Nk-1, 0 .. Nk//2, Nk//2] used to “wrap” the curve.
    """
    assert Nk % 2 == 0, "Use an even Nk for a symmetric grid with ±π endpoints."

    # 1) x-positions: include both endpoints so you can 'close' the band plot.
    #    This is equivalent to m * (2π/Nk) for m in [-Nk/2, ..., +Nk/2]
    k_path = np.linspace(-np.pi, np.pi, Nk + 1)

    # 2) Reordering indices so momentum 0 sits in the middle and the curve is closed
    #    Example (Nk=16): [8,9,10,11,12,13,14,15, 0,1,2,3,4,5,6,7, 8]
    order = list(range(Nk // 2, Nk)) + list(range(0, Nk // 2)) + [Nk // 2]

    # 3) Tick labels
    #    We label at every point by default; you can slice these arrays if you want fewer ticks.
    mvals = np.arange(-Nk // 2, Nk // 2 + 1)  # matches k_path
    if numeric_labels:
        # Simple integer labels from -Nk//2 .. 0 .. +Nk//2
        tick_labels = [rf"${m}$" if m != 0 else r"$0$" for m in mvals]
    else:
        # Pretty π-fraction labels: k = (m / (Nk/2)) * π
        den_base = Nk // 2
        tick_labels = []
        for m in mvals:
            if m == 0:
                tick_labels.append(r"$0$")
                continue
            if abs(m) == den_base:
                tick_labels.append(r"$-\pi$" if m < 0 else r"$+\pi$")
                continue
            # reduce fraction |m| / (Nk/2)
            num, den = abs(m), den_base
            g = math.gcd(num, den)
            num //= g
            den //= g
            sign = "-" if m < 0 else "+"
            if den == 1:
                label = rf"${sign}\pi$" if num == 1 else rf"${sign}{num}\pi$"
                tick_labels.append(label)
            else:
                label = (
                    rf"${sign}\frac{{\pi}}{{{den}}}$"
                    if num == 1
                    else rf"${sign}\frac{{{num}\pi}}{{{den}}}$"
                )
                tick_labels.append(label)
    return k_path, order, tick_labels


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def set_size(width_pt, fraction=1, subplots=(1, 1), height_factor=1.0):
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
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in * height_factor)


def plot_energy_bands(sim_name_list, textwidth_pt=510.0, columnwidth_pt=246.0):
    res = {}
    for sim_name in sim_name_list:
        res[sim_name] = {}
        config_filename = f"scattering/{sim_name}"
        match = SimsQuery(group_glob=config_filename)
        ugrid, vals = uids_grid(match.uids, ["momentum_k_vals"])
        # Universal parameters across the simulations
        g = get_sim(ugrid[0]).par["g"]
        m = get_sim(ugrid[0]).par["m"]
        n_eigs = get_sim(ugrid[0]).par["hamiltonian"]["n_eigs"]
        # Momentum parameters
        k_indices = vals["momentum_k_vals"]
        Nk = len(k_indices)
        # Sort k values centering 0 in the BZ
        k_values, order, tick_labels = bz_axis(Nk)
        # Specific parameters of the simulation
        res[sim_name] = {
            "energy": np.zeros((Nk, n_eigs)),
            "E2": np.zeros((Nk, n_eigs)),
            "N_single": np.zeros((Nk, n_eigs)),
            "N_pair": np.zeros((Nk, n_eigs)),
        }
        for ll in k_indices:
            sim_res = get_sim(ugrid[ll]).res
            res[sim_name]["energy"][ll] = sim_res["energy"]
            res[sim_name]["E2"][ll] = sim_res["E2"]
            res[sim_name]["N_single"][ll] = sim_res["N_single"]
            res[sim_name]["N_pair"][ll] = sim_res["N_pair"]
        # Sort the energy bands centering k=0 in the BZ
        res[sim_name]["bands"] = res[sim_name]["energy"][..., order, :]
    # BUILD THE FIGURE
    fig, ax = plt.subplots(
        1,
        1,
        figsize=set_size(2 * columnwidth_pt, subplots=(1, 1), height_factor=3),
        constrained_layout=True,
        sharex=True,
    )

    ax.set_xticks(k_values)
    ax.set_xticklabels(tick_labels)
    ax.set(ylabel=r"energy E")
    ax.set(xlabel=r"momentum $k$")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.annotate(
        rf"$g^{2}\!=\!{g}, m\!=\!{m}$",
        xy=(0.935, 0.1),
        xycoords="axes fraction",
        fontsize=10,
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox=dict(facecolor="white", edgecolor="black"),
    )
    colors = ["green", "blue", "red"]
    markersize_list = [7, 6, 5]
    # PLOT THE DATA
    for sim_idx, sim_name in enumerate(sim_name_list):
        for ss in range(n_eigs):
            ax.plot(
                k_values,
                res[sim_name]["bands"][:, ss],
                "o",
                color=colors[sim_idx],  # line & marker edge color
                markeredgecolor=colors[sim_idx],
                markerfacecolor=lighten_color(colors[sim_idx], 0.6),
                markeredgewidth=1.3,
                markersize=markersize_list[sim_idx],
            )
    # MAKE THE LEGEND
    sector_labels = [r"$N_{\rm bar}=0$", r"$N_{\rm bar}=+1$", r"$N_{\rm bar}=+2$"]
    handles = []
    for c, ms, lab in zip(["green", "blue", "red"], [7, 6, 5], sector_labels):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                color=c,  # legend line color (not shown since linestyle=None)
                markeredgecolor=c,
                markerfacecolor=lighten_color(c, 0.6),
                markeredgewidth=1.3,
                markersize=ms,
                label=lab,
            )
        )
    # Put the legend where you like:
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.3, 0.1),  # tweak or remove to place inside
        frameon=True,
        ncol=1,
        handlelength=1.0,
        handletextpad=0.3,
        borderpad=0.3,
        labelspacing=0.15,
        fontsize=10,
        title=r"Baryon sectors",  # ← title here
        title_fontsize=10,
    )
    plt.savefig(f"bands_g{g}_m{m}.pdf", **default_params["save_plot"])


# %%
plot_energy_bands(["band1_N0", "band1_N1"])

# %%
plot_energy_bands(["band2_N0", "band2_N1"])
# %%
plot_energy_bands(["band3_N0", "band3_N1"])

# %%
plot_energy_bands(["band4_N0", "band4_N1"])

# %%
