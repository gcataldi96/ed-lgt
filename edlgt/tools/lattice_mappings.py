"""Mappings between a linear site index and lattice coordinates.

This module contains helpers that map a 1D index ``d`` to lattice coordinates
and vice versa using simple scan orders (zig-zag / snake) and a Hilbert curve
for 2D square lattices.

Public API
----------
The exported functions in ``__all__`` are:

- :func:`zig_zag` and :func:`inverse_zig_zag` for generic multi-dimensional
  coordinates (documented use: square / hypercubic lattices with 0-based
  indexing),
- :func:`coords`, a small formatting helper for human-readable 2D labels.

Notes
-----
- All mapping functions in this module use 0-based indexing for coordinates and
  linear indices unless explicitly stated otherwise.
"""

import numpy as np
from math import prod

__all__ = [
    "zig_zag",
    "inverse_zig_zag",
    "coords",
]


def zig_zag(lvals, d):
    """Map a linear index to lattice coordinates using a zig-zag scan order.

    Parameters
    ----------
    lvals : list[int]
        Lattice sizes along each axis, e.g. ``[Lx, Ly]`` or ``[Lx, Ly, Lz]``.
        The documented and recommended use of this generic implementation is for
        square / hypercubic lattices (equal linear size along all axes).
    d : int
        0-based linear index. It must satisfy ``0 <= d < prod(lvals)``.

    Returns
    -------
    tuple[int, ...]
        Coordinate tuple ``(x, y, z, ...)`` corresponding to ``d``.

    Raises
    ------
    TypeError
        If ``lvals`` or ``d`` has an invalid type.
    ValueError
        If ``d`` is outside the valid range.

    Notes
    -----
    This is a simple row-major style mapping (not a snake curve): the fastest
    direction is the first coordinate in the inverse mapping convention used by
    :func:`inverse_zig_zag`.
    """
    if not isinstance(lvals, list):
        raise TypeError(f"lvals should be a list, not a {type(lvals)}")
    elif not all(np.isscalar(dim) and isinstance(dim, int) for dim in lvals):
        raise TypeError("All items of lvals must be scalar integers.")

    if not np.isscalar(d) or not isinstance(d, int):
        raise TypeError(f"d must be a scalar integer, not {type(d)}")

    tot_size = prod(lvals)
    if d > tot_size - 1:
        raise ValueError(
            f"d must be a smaller than the total number of lattice sites {tot_size}, not {d}"
        )
    lattice_dim = len(lvals)
    coords = [0] * lattice_dim
    for ii, p in zip(range(lattice_dim), reversed(range(lattice_dim))):
        coords[p] = d // lvals[ii] ** p
        d -= coords[p] * lvals[ii] ** p
    return tuple(coords)


def inverse_zig_zag(lvals, coords):
    """Map lattice coordinates back to the 0-based zig-zag linear index.

    Parameters
    ----------
    lvals : list[int]
        Lattice sizes along each axis. The documented use is square /
        hypercubic lattices for consistency with :func:`zig_zag`.
    coords : sequence[int]
        Coordinate tuple/list ``(x, y, z, ...)`` using 0-based indexing.

    Returns
    -------
    int
        0-based linear index corresponding to ``coords``.

    Raises
    ------
    TypeError
        If the input types are invalid.
    ValueError
        If any coordinate lies outside the corresponding lattice range.

    Notes
    -----
    ``inverse_zig_zag(lvals, zig_zag(lvals, d)) == d`` for supported inputs.
    """
    if not isinstance(lvals, list):
        raise TypeError(f"lvals should be a list, not a {type(lvals)}")
    elif not all(np.isscalar(dim) and isinstance(dim, int) for dim in lvals):
        raise TypeError("All items of lvals must be scalar integers.")
    if not all(np.isscalar(c) and isinstance(c, int) for c in coords):
        raise TypeError("All coords must be scalar integers.")
    lattice_dim = len(lvals)
    d = 0
    coords = tuple(coords)
    for ii, c in zip(range(lattice_dim), "xyz"[:lattice_dim]):
        if coords[ii] > (lvals[ii] - 1):
            raise ValueError(
                f"The {c} coord should be smaller than {lvals[ii]}, not {coords[ii]}"
            )
        d += coords[ii] * (lvals[ii - 1] ** ii)
    return d


def zig_zag1(nx, ny, d):
    """
    Given the 1d point at position d of the zigzag curve in a (nx,ny) discrete lattice,
    it provides the corresponding 2d coordinates (x,y) of the point.

    NOTE: The zigzag curve is built by always counting from 0 (not 1), hence the points
    of the 1d curve start from 0 to (nx * ny)-1 and the coordinates (x,y) are supposed to go from (0,0) to (nx-1,ny-1).

    Args:
        nx (int): x number of lattice sites
        ny (int): y number of lattice sites
        d (int): point of a 1D curve covering the 2D lattice

    Returns:
        (int, int): 2D coordinates of the 1D point of the ZigZag curve in the lattice
    """
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f"nx must be SCALAR & INTEGER, not {type(nx)}")
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f"ny must be SCALAR & INTEGER, not {type(ny)}")
    if not np.isscalar(d) and not isinstance(d, int):
        raise TypeError(f"d must be SCALAR & INTEGER, not {type(d)}")
    if d == 0:
        x = 0
        y = 0
    elif d < nx:
        y = 0
        x = d
    else:
        # COMPUTE THE REST OF THE DIVISION
        x = d % nx
        # COMPUTE THE INTEGER PART OF THE DIVISION
        y = d // nx
    return x, y


def inverse_zig_zag1(nx, ny, x, y):
    """
    Inverse zigzag curve mapping (from 2D coords to the 1D points).

    NOTE: Given the sizes (nx,ny) of a lattice, the coords (x,y)
    has to start from (0,0) and arrive to (nx-1,ny-1).
    Correspondingly, the points of the zigzag curve range from 0 to (nx * ny -1).

    Args:
        nx (int): x number of lattice sites
        ny (int): y number of lattice sites
        x (int): x coordinate of the lattice
        y (int): y coordinate of the lattice

    Returns:
        int: 1D point of the zigzag curve
    """
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f"nx must be SCALAR & INTEGER, not {type(nx)}")
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f"ny must be SCALAR & INTEGER, not {type(ny)}")
    if not np.isscalar(x) and not isinstance(x, int):
        raise TypeError(f"x must be SCALAR & INTEGER, not {type(x)}")
    if not np.isscalar(y) and not isinstance(y, int):
        raise TypeError(f"y must be SCALAR & INTEGER, not {type(y)}")
    d = (y * nx) + x
    return d


def snake(nx, ny, d):
    """
    Given the 1d point of the snake curve in a (nx,ny) discrete lattice,
    it provides the corresponding 2d coordinates (x,y) of the point.

    NOTE: The snake curve is built by always counting from 0 (not 1)
    hence the points of the 1d curve start from 0 to (nx*ny)-1
    and the coords x and y are supposed to go from 0 to n-1.

    Args:
        nx (int): x number of lattice sites
        ny (int): y number of lattice sites
        d (int): point of a 1D curve covering the 2D lattice.

    Returns:
        (int, int): 2D coordinates of the 1D point of the ZigZag curve in the lattice
    """
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f"nx must be SCALAR & INTEGER, not {type(nx)}")
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f"ny must be SCALAR & INTEGER, not {type(ny)}")
    if not np.isscalar(d) and not isinstance(d, int):
        raise TypeError(f"d must be SCALAR & INTEGER, not {type(d)}")
    if d == 0:
        x = 0
        y = 0
    elif d < nx:
        y = 0
        x = d
    else:
        # COMPUTE THE INTEGER PART OF THE DIVISION
        y = d // nx
        if (y % 2) == 0:
            x = d % nx
        else:
            x = nx - 1 - d % nx
    return x, y


def inverse_snake(nx, ny, x, y):
    """
    Inverse snake curve mapping (from coords to the 1d points)

    NOTE: Given the sizes (nx,ny) of a lattice, the coords (x,y)
    has to start from (0,0) and arrive to (nx-1,ny-1).
    Correspondingly, the points of the zigzag curve range from 0 to (nx * ny -1).

    Args:
        nx (int): x number of lattice sites
        ny (int): y number of lattice sites
        x (int): x coordinate of the lattice
        y (int): y coordinate of the lattice

    Returns:
        int: 1D point of the zigzag curve
    """
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f"nx must be SCALAR & INTEGER, not {type(nx)}")
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f"ny must be SCALAR & INTEGER, not {type(ny)}")
    if not np.isscalar(x) and not isinstance(x, int):
        raise TypeError(f"x must be SCALAR & INTEGER, not {type(x)}")
    if not np.isscalar(y) and not isinstance(y, int):
        raise TypeError(f"y must be SCALAR & INTEGER, not {type(y)}")
    d = 0
    # Notice that the first (and hence odd) column is the 0^th column
    if (y % 2) == 0:
        # EVEN COLUMNS (0,2,4,...n-2)
        d = (y * nx) + x
    else:
        # ODD COLUMNS (1,3,5...n-1)
        d = (y * nx) + nx - 1 - x
    return d


def regions(num, x, y, s):
    if num == 0:
        # BOTTOM LEFT: CLOCKWISE ROTATE THE COORDS (x,y) OF 90 DEG
        # THE ROTATION MAKES (x,y) INVERT (y,x)
        t = x
        x = y
        y = t
    elif num == 1:
        # TOP LEFT: TRANSLATE UPWARDS (x,y) OF THE PREVIOUS LEVE
        x = x
        y = y + s
    elif num == 2:
        # TOP RIGHT: TRANSLATE UPWARDS AND RIGHTFORWARD (x,y)
        x = x + s
        y = y + s
    elif num == 3:
        # BOTTOM RIGHT: COUNTER CLOCKWISE ROTATE OF 90 DEG THE (x,y)
        t = x
        x = (s - 1) - y + s
        y = (s - 1) - t
    return x, y


def bitconv(num):
    # GIVEN THE POSITION OF THE HILBERT CURVE IN A 2x2 SQUARE,
    # IT RETURNS THE CORRESPONDING PAIR OF COORDINATES (rx,ry)
    if num == 0:
        # BOTTOM LEFT
        rx = 0
        ry = 0
    elif num == 1:
        # TOP LEFT
        rx = 0
        ry = 1
    elif num == 2:
        # TOP RIGHT
        rx = 1
        ry = 1
    elif num == 3:
        # BOTTOM RIGHT
        rx = 1
        ry = 0
    return rx, ry


def hilbert(n, d):
    # MAPPING THE POSITION d OF THE HILBERT CURVE
    # LIVING IN A nxn SQUARE LATTIVE INTO THE
    # CORRESPONDING 2D (x,y) COORDINATES OF A S
    s = 1  # FIX THE INITIAL LEVEL OF DESCRIPTION
    n1 = d & 3  # FIX THE 2 BITS CORRESPONDING TO THE LEVEL
    x = 0
    y = 0
    # CONVERT THE POSITION OF THE POINT IN THE CURVE AT LEVEL 0 INTO
    # THE CORRESPONDING (x,y) COORDINATES
    x, y = bitconv(n1)
    s *= 2  # UPDATE THE LEVEL OF DESCRIPTION
    tmp = d  # COPY THE POINT d OF THE HILBERT CURVE
    while s < n:
        tmp = tmp >> 2  # MOVE TO THE RIGHT THE 2 BITS OF THE POINT dÅ
        n2 = tmp & 3  # FIX THE 2 BITS CORRESPONDING TO THE LEVEL
        x, y = regions(n2, x, y, s)  # UPDATE THE COORDINATES OF THAT LEVEL
        s *= 2  # UPDATE THE LEVEL OF DESCRIPTION
        s = int(s)
    return x, y


def inverse_regions(num, x, y, s):
    if num == 0:
        # BOTTOM LEFT
        t = x
        x = y
        y = t
    elif num == 1:
        # TOP LEFT
        x = x
        y = y - s
    elif num == 2:
        # TOP RIGHT
        x = x - s
        y = y - s
    elif num == 3:
        # BOTTOM RIGHT
        tt = x
        x = (s - 1) - y
        y = (s - 1) - tt + s
    return x, y


def inverse_bitconv(rx, ry):
    # GIVEN A PAIR OF COORDINATES (x,y) IN A 2x2 LATTICE, IT
    # RETURNS THE POINT num OF THE CORRESPONDING HILBERT CURVE
    if rx == 0:
        if ry == 0:
            # BOTTOM LEFT
            num = 0
        elif ry == 1:
            # TOP LEFT
            num = 1
    elif rx == 1:
        if ry == 0:
            # BOTTOM RIGHT
            num = 3
        elif ry == 1:
            # TOP RIGHT
            num = 2
    return num


def inverse_hilbert(n, x, y):
    # MAPPING THE 2D (x,y) OF A nxn SQUARE INTO THE POSITION d
    # OF THE HILBERT CURVE. REMEMBER THAT THE FINAL POINT
    # HAS TO BE SHIFTED BY 1
    d = 0
    n0 = 0
    s = int(n / 2)
    while s > 1:
        rx = int(x / s)
        ry = int(y / s)
        n0 = inverse_bitconv(rx, ry)
        x, y = inverse_regions(n0, x, y, s)
        d += n0
        d = d << 2
        s /= 2
        s = int(s)
    n0 = inverse_bitconv(x, y)
    d += n0
    return d


def coords(x, y):
    """Format 2D coordinates as a 1-based string label.

    Parameters
    ----------
    x, y : int
        0-based coordinates.

    Returns
    -------
    str
        Coordinate label formatted as ``"(x+1,y+1)"``.

    Notes
    -----
    This is a display helper only; it does not perform validation.
    """
    return "(" + str(x + 1) + "," + str(y + 1) + ")"
