import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "set_size",
    "fake_log",
    "get_tline",
    "time_integral",
    "custom_average",
    "moving_time_integral",
    "gaussian_time_integral",
]

default_params = {
    "save_plot": {"bbox_inches": "tight", "transparent": False},
}


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
    if Nk % 2 != 0:
        raise ValueError("Nk must be even for a symmetric Brillouin zone within ±π.")
    # 1) x-positions: include both endpoints so you can 'close' the band plot.
    # This is equivalent to m * (2π/Nk) for m in [-Nk/2, ..., +Nk/2]
    k_path = np.linspace(-np.pi, np.pi, Nk + 1)
    # 2) Reordering indices so momentum 0 sits in the middle and the curve is closed
    # Example (Nk=16): [8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,8]
    order = list(range(Nk // 2, Nk)) + list(range(0, Nk // 2)) + [Nk // 2]
    # 3) Tick labels at every point by default (same length as k_path);
    # you can slice these arrays if you want fewer ticks.
    tick_vals = np.arange(-Nk // 2, Nk // 2 + 1)
    if numeric_labels:
        # Simple integer labels from -Nk//2 .. 0 .. +Nk//2
        tick_labels = [rf"${m}$" if m != 0 else r"$0$" for m in tick_vals]
    else:
        # Pretty π-fraction labels: k = (m / (Nk/2)) * π
        den_base = Nk // 2
        tick_labels = []
        for m in tick_vals:
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
