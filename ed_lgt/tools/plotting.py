from simsio import *
import numpy as np
import matplotlib.pyplot as plt
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
