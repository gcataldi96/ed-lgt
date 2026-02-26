"""Finite-difference derivative helpers on uniformly spaced 1D grids.

This module provides simple central-difference routines for first and second
derivatives. Both functions return the derivative evaluated on the interior
points only (the first and last grid points are dropped).
"""

import numpy as np

__all__ = ["first_derivative", "second_derivative"]


def first_derivative(x, f, dx):
    """Compute the first derivative using a central-difference stencil.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional grid values. Only interior points are returned.
    f : numpy.ndarray
        Function values sampled on ``x``.
    dx : float
        Uniform grid spacing.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(x_interior, df_dx)`` where both arrays have length ``len(x) - 2``.

    Notes
    -----
    This routine assumes a uniformly spaced grid and uses the standard
    second-order central-difference approximation on interior points.
    """
    # COMPUTE THE 1st OR THE 2nd DERIVATIVE
    # f_der OF A FUNCTION f WRT A VARIABLE x
    f_der = np.zeros(x.shape[0] - 2)
    # COMPUTE THE 1ST ORDER CENTRAL DERIVATIVE
    for ii in range(f_der.shape[0]):
        jj = ii + 1
        f_der[ii] = (f[jj + 1] - f[jj - 1]) / (2 * dx)
    # USE AN UPDATE VERSION OF X WHERE THE FIRST
    # AND THE LAST ENTRY ARE ELIMINATED IN ORDER
    # TO GET AN ARRAY OF THE SAME DIMENSION OF
    # THE ONE WITH THE DERIVATIVE OF F
    x_copy = np.zeros(f_der.shape[0])
    for ii in range(f_der.shape[0]):
        x_copy[ii] = x[ii + 1]
    return x_copy, f_der


def second_derivative(x, f, dx):
    """Compute the second derivative using a central-difference stencil.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional grid values. Only interior points are returned.
    f : numpy.ndarray
        Function values sampled on ``x``.
    dx : float
        Uniform grid spacing.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(x_interior, d2f_dx2)`` where both arrays have length ``len(x) - 2``.

    Notes
    -----
    This routine assumes a uniformly spaced grid and uses the standard
    second-order central-difference approximation on interior points.
    """
    # COMPUTE THE 1st OR THE 2nd DERIVATIVE
    # f_der OF A FUNCTION f WRT A VARIABLE x
    f_der = np.zeros(x.shape[0] - 2)
    # COMPUTE THE 2ND ORDER CENTRAL DERIVATIVE
    for ii in range(f_der.shape[0]):
        jj = ii + 1
        f_der[ii] = (f[jj + 1] - 2 * f[jj] + f[jj - 1]) / (dx**2)
    # USE AN UPDATE VERSION OF X WHERE THE FIRST
    # AND THE LAST ENTRY ARE ELIMINATED IN ORDER
    # TO GET AN ARRAY OF THE SAME DIMENSION OF
    # THE ONE WITH THE DERIVATIVE OF F
    x_copy = np.zeros(f_der.shape[0])
    for ii in range(f_der.shape[0]):
        x_copy[ii] = x[ii + 1]
    return x_copy, f_der
