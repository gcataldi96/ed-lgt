import numpy as np
from math import prod
from ed_lgt.tools import validate_parameters

__all__ = [
    "zig_zag",
    "inverse_zig_zag",
]


def zig_zag(lvals, d):
    """
    Given the 1d point at position d of the zigzag curve in a discrete lattice with arbitrary dimensions,
    it provides the corresponding multidimensional coordinates of the point.

    NOTE: d has to be smaller than the total number of lattice sites

    Args:
        lvals (list or tuple of int): The dimensions of the lattice in each direction (Lx, Ly, Lz, ...)

        d (int): Point of a 1D curve covering the multi-dimensional lattice.

    Returns:
        tuple of int: Multi-dimensional coordinates of the 1D point of the ZigZag curve in the lattice (x, y, z, ...).
    """
    # Validate type of parameters
    validate_parameters(lvals=lvals)
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
    """
    Inverse zigzag curve mapping (from d coords to the 1D points).

    NOTE: Given the sizes of a multidimensional lattice, the d-dimensional coords
    are supposed to start from 0 and have to be smaller than each lattice dimension
    Correspondingly, the points of the zigzag curve start from 0.

    Args:
        lvals (list or tuple of int): The dimensions of the lattice in each direction (Lx, Ly, Lz, ...)

        coords (list or tuple of int): Multi-dimensional coordinates of the 1D point of the ZigZag curve in the lattice (x, y, z, ...).

    Returns:
        int: 1D point of the zigzag curve
    """
    # Validate type of parameters
    validate_parameters(lvals=lvals, coords=coords)
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
