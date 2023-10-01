import numpy as np

__all__ = [
    "zig_zag",
    "inverse_zig_zag",
    "snake",
    "inverse_snake",
    "hilbert",
    "inverse_hilbert",
    "coords",
]


def zig_zag(nx, ny, d):
    """
    Given the 1d point at position d of the zigzag curve in a (nx,ny) discrete lattice,
    it provides the corresponding 2d coordinates (x,y) of the point
    NOTE: the zigzag curve is built by always counting from 0 (not 1),
        hence the points of the 1d curve start from 0 to (nx * ny)-1
        and the coordinates (x,y) are supposed to go from (0,0) to (nx-1,ny-1.

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
        # COMPUTE THE REST OF THE DIVISION
        x = d % nx
        # COMPUTE THE INTEGER PART OF THE DIVISION
        y = d // nx
    return x, y


def inverse_zig_zag(nx, ny, x, y):
    """
    Inverse zigzag curve mapping (from 2D coords to the 1D points)
    NOTE: given the sizes (nx,ny) of a lattice, the coords (x,y)
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
    it provides the corresponding 2d coordinates (x,y) of the point
    NOTE: the snake curve is built by always counting from 0 (not 1)
        hence the points of the 1d curve start from 0 to (nx*ny)-1
        and the coord.s x and y are supposed to go from 0 to n-1

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
    NOTE: given the size l of a square lattice, the coords x,y has to
        start from 0 and arrive to l-1. at the end, the points of the
        snake curve start from 0. add 1 if you want to start from 1

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
    return "(" + str(x + 1) + "," + str(y + 1) + ")"
