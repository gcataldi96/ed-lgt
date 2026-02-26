"""Post-processing helpers for spectral and correlation-based observables.

This module collects lightweight analysis utilities used in scripts and example
workflows, including:

- level-spacing statistics (`r_values`),
- distance-resolved correlator summaries (`analyze_correlator`),
- structure-factor evaluation on 2D lattices (`structure_factor`),
- simple charge/density combinations from two occupancies.

Only a subset of functions is exported as public API via ``__all__``.
"""

import numpy as np
from math import prod
from itertools import product
from scipy.linalg import eigh
from .lattice_mappings import zig_zag

__all__ = [
    "structure_factor",
    "get_charge",
    "get_density",
    "analyze_correlator",
    "r_values",
]


def r_values(energy):
    """Compute adjacent-gap ratios from a spectrum.

    Parameters
    ----------
    energy : array-like
        One-dimensional array of energy eigenvalues. The input is sorted
        internally before computing level spacings.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(r_array, delta_E)`` where:

        - ``delta_E[i] = E[i+1] - E[i]`` for the sorted spectrum,
        - ``r_array[i]`` is the ratio between adjacent spacings.

    Notes
    -----
    The first entry of ``r_array`` is set to ``1`` by convention because it has
    no left neighbor spacing.
    """
    energy = np.sort(energy)
    delta_E = np.zeros(energy.shape[0] - 1)
    r_array = np.zeros(energy.shape[0] - 1)
    for ii in range(energy.shape[0] - 1):
        delta_E[ii] = energy[ii + 1] - energy[ii]
    for ii in range(delta_E.shape[0]):
        if ii == 0:
            r_array[ii] = 1
        else:
            r_array[ii] = min(delta_E[ii], delta_E[ii - 1]) / max(
                delta_E[ii], delta_E[ii - 1]
            )
    return r_array, delta_E


def analyze_correlator(corr):
    """Average a 2D correlator by Euclidean distance.

    Parameters
    ----------
    corr : numpy.ndarray
        Correlator array indexed as ``corr[x1, y1, x2, y2]``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_distances, 3)`` with columns:

        1. distance ``r``,
        2. mean absolute correlator value at that distance,
        3. standard error of the mean (computed from the sample variance).

    Notes
    -----
    Distances are rounded to 3 decimal places before grouping to avoid small
    floating-point differences splitting equivalent distances.
    """
    # Get the dimensions of the lattice
    dim1, dim2 = corr.shape[0], corr.shape[1]
    # Initialize dictionaries to store correlator values for each distance
    corr_values = {}
    # Create all possible combinations of lattice points
    points = list(product(range(dim1), range(dim2)))
    # Iterate over all pairs of points in the lattice using itertools.product
    for (x1, y1), (x2, y2) in product(points, repeat=2):
        if x1 != x2 or y1 != y2:
            # Calculate the distance between the two points
            r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            # Round the distance to handle numerical precision issues
            r_rounded = round(r, 3)  # Adjust the rounding as necessary
            # Collect the correlator values for this distance
            if r_rounded not in corr_values:
                corr_values[r_rounded] = []
            corr_values[r_rounded].append(corr[x1, y1, x2, y2])
    # Prepare the output array
    num_distances = len(corr_values)
    # Initialize the 2D results array
    results = np.zeros((num_distances, 3))
    # Sort the distances to maintain an ascending order
    sorted_distances = sorted(corr_values.keys())
    # Calculate the average correlator and its standard error for each distance
    for i, r in enumerate(sorted_distances):
        values = np.abs(corr_values[r])
        avg = np.mean(values)
        # Calculate the variance and standard error
        var = np.var(values, ddof=1)  # ddof=1 for sample variance
        std_error = np.sqrt(var) / np.sqrt(len(values))
        # Store the results in the results array
        results[i, :] = [r, avg, std_error]
    return results


def structure_factor(corr, lvals):
    """Compute the 2D static structure factor on a rectangular lattice.

    Parameters
    ----------
    corr : numpy.ndarray
        Correlator array indexed as ``corr[x1, y1, x2, y2]``.
    lvals : sequence[int]
        Lattice dimensions ``[Lx, Ly]``.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(Lx, Ly)`` containing the structure factor on
        the discrete Brillouin-zone grid.

    Notes
    -----
    This routine calls :func:`single_structure_factor` for each momentum point.
    """
    # DEFINE THE BRILLUOIN ZONE
    Lx = lvals[0]
    Ly = lvals[1]
    str_factor = np.zeros((Lx, Ly), dtype=complex)
    for nx in range(Lx):
        for ny in range(Ly):
            kx = 2 * np.pi * nx / Lx
            ky = 2 * np.pi * ny / Ly
            str_factor[nx, ny] = single_structure_factor(lvals, kx, ky, corr)
    return str_factor


def single_structure_factor(lvals, kx, ky, corr):
    # DEFINE THE BRILLUOIN ZONE
    Lx = lvals[0]
    Ly = lvals[1]
    n_sites = Lx * Ly
    counter = 0
    sf_sum = 0.0
    for i in range(n_sites):
        for j in range(n_sites):
            if i != j:
                counter += 1
                # get the coordinates of the lattice points
                ix, iy = zig_zag([Lx, Ly], i)
                jx, jy = zig_zag([Lx, Ly], j)
                exp_factor = kx * (ix - jx) + ky * (iy - jy)
                sf_sum += np.exp(complex(0.0, 1.0) * exp_factor) * corr[ix, iy, jx, jy]
    return sf_sum / counter


def get_density(N_plus, N_minus):
    """Return the density-like combination used by the project conventions.

    Parameters
    ----------
    N_plus, N_minus : float or numpy.ndarray
        Inputs combined element-wise if arrays are provided.

    Returns
    -------
    float or numpy.ndarray
        ``N_plus - N_minus + 2``.
    """
    return N_plus - N_minus + 2


def get_charge(N_plus, N_minus):
    """Return the charge-like combination used by the project conventions.

    Parameters
    ----------
    N_plus, N_minus : float or numpy.ndarray
        Inputs combined element-wise if arrays are provided.

    Returns
    -------
    float or numpy.ndarray
        ``N_plus + N_minus - 2``.
    """
    return N_plus + N_minus - 2


def get_energy_gap(self, parity_sector):
    if parity_sector:
        Den = get_Q_operator(self.lvals, self.res)
        Num = get_P_operator(self.lvals, self.has_obc[0], self.res, self.coeffs)
    else:
        Den = get_N_operator(self.lvals, self.res)
        Num = get_M_operator(self.lvals, self.has_obc[0], self.res, self.coeffs)
    self.res["th_gap"] = eigh(a=Num, b=Den, eigvals_only=True)[0]


# ====================================================================================
# EXCITATIONS that breaks the parity sectors
def get_M_operator(lvals, has_obc, obs, coeffs):
    n_sites = prod(lvals)
    M = np.zeros((n_sites, n_sites), dtype=complex)
    for ii, jj in product(range(n_sites), repeat=2):
        nn_condition = [
            all([ii > 0, jj == ii - 1]),
            all([ii < n_sites - 1, jj == ii + 1]),
            all([not has_obc, ii == 0, jj == n_sites - 1]),
            all([not has_obc, ii == n_sites - 1, jj == 0]),
        ]
        if any(nn_condition):
            M[ii, jj] += coeffs["J"] * obs["Sz_Sz"][ii, jj]
        elif jj == ii:
            M[ii, jj] += 2 * coeffs["h"] * obs["Sz"][ii]
            if 0 < ii < n_sites - 1 or all(
                [(ii == 0 or ii == n_sites - 1), not has_obc]
            ):
                M[ii, jj] += complex(0, 0.5 * coeffs["J"]) * (
                    obs["Sm_Sx"][ii, (ii + 1) % n_sites]
                    - obs["Sp_Sx"][ii, (ii + 1) % n_sites]
                    + obs["Sx_Sm"][(ii - 1) % n_sites, ii]
                    - obs["Sx_Sp"][(ii - 1) % n_sites, ii]
                )
    return M


def get_N_operator(lvals, obs):
    n_sites = prod(lvals)
    N = np.zeros((n_sites, n_sites), dtype=float)
    for ii in range(n_sites):
        N[ii, ii] += obs["Sz"][ii]
    return N


# ====================================================================================
# Parity preserving excitations


def get_P_operator(lvals, has_obc, obs, coeffs):
    n_sites = prod(lvals)
    B = np.zeros((n_sites, n_sites), dtype=complex)

    for ii, jj in product(range(n_sites), repeat=2):
        if ii == jj - 1 and any(
            [
                0 < jj < n_sites - 1,
                jj == 0 and not has_obc,
                ii == n_sites - 1 and not has_obc,
            ]
        ):
            B[ii, jj] += complex(2 * coeffs["J"], 0) * (
                obs["Sz_Sp_Sm"][ii % n_sites, jj, (jj + 1) % n_sites]
                - obs["Sm_Sp_Sz"][ii % n_sites, jj, (jj + 1) % n_sites]
            )
            B[jj, ii] = -complex(0, 1) * B[ii, jj]
        if ii == jj - 2 and any(
            [
                1 < jj < n_sites - 1,
                jj <= 1 and not has_obc,
                ii == n_sites - 1 and not has_obc,
            ]
        ):
            B[ii, jj] += (
                -coeffs["J"]
                * obs["Sm_Sz_Sz_Sm"][
                    ii % n_sites, (ii + 1) % n_sites, jj, (jj + 1) % n_sites
                ]
            )
            B[jj, ii] = -complex(0, 1) * B[ii, jj]
        elif ii == jj:
            if any([ii < n_sites - 1, ii == n_sites - 1 and not has_obc]):
                B[ii, jj] += 4 * (
                    -coeffs["J"] * obs["Sm_Sp"][ii, (ii + 1) % n_sites]
                    + coeffs["h"] * obs["Sz_Sz"][ii, (ii + 1) % n_sites]
                )
            if any(
                [
                    0 < ii < n_sites - 1,
                    ii == 0 and not has_obc,
                    ii == n_sites - 1 and not has_obc,
                ]
            ):
                B[ii, jj] += complex(0, 2 * coeffs["J"]) * (
                    obs["Sx_Sm_Sz"][(ii - 1) % n_sites, ii, (ii + 1) % n_sites]
                )
            if any([ii < n_sites - 2, ii >= n_sites - 2 and not has_obc]):
                B[ii, jj] += complex(0, 2 * coeffs["J"]) * (
                    obs["Sz_Sp_Sx"][ii, (ii + 1) % n_sites, (ii + 2) % n_sites]
                )

    Q = 0.5 * (B + np.conjugate(B.T))
    return Q


def get_Q_operator(lvals, obs):
    n_sites = prod(lvals)
    P = np.zeros((n_sites, n_sites), dtype=float)
    for ii in range(n_sites):
        P[ii, ii] += (
            obs["SpSm_Sz"][ii, (ii + 1) % n_sites]
            - obs["Sz_SpSm"][ii, (ii + 1) % n_sites]
        )
    return P
