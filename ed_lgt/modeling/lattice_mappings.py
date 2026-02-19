import numpy as np
from math import prod
from ed_lgt.tools import validate_parameters

__all__ = [
    "zig_zag",
    "inverse_zig_zag",
]


def _lattice_strides(lvals):
    """Return row-major (C-order) strides for a mixed-radix lattice index mapping."""
    lattice_dim = len(lvals)
    strides = [1] * lattice_dim
    for axis_index in range(1, lattice_dim):
        strides[axis_index] = strides[axis_index - 1] * int(lvals[axis_index - 1])
    return strides


def zig_zag(lvals, index_1d):
    """
    Convert a linear site index into lattice coordinates.
    The mapping is a mixed-radix (stride-based) conversion, consistent with row-major
    ordering where the first axis ("x") is the fastest-changing coordinate.

    Parameters
    ----------
    lvals : sequence of int
        Lattice sizes per axis, e.g. "[Lx, Ly, Lz]".
    index_1d : int
        Linear index in "[0, prod(lvals) - 1]".

    Returns
    -------
    tuple of int
        Lattice coordinates "(x, y, z, ...)" with "0 <= coord[i] < lvals[i]".

    Notes
    -----
    This is not a geometric "zig-zag curve" in the sense of path reflections; it is a
    standard lattice linearization / delinearization based on strides.
    """
    validate_parameters(lvals=lvals)
    if not isinstance(index_1d, (int, np.integer)):
        raise TypeError(f"index_1d must be an integer, not {type(index_1d)}")
    n_sites = prod(lvals)
    if index_1d < 0 or index_1d >= n_sites:
        raise ValueError(f"index_1d must be in [0, {n_sites - 1}], not {index_1d}")
    strides = _lattice_strides(lvals)
    coords = []
    for axis_size, axis_stride in zip(lvals, strides):
        coords.append((int(index_1d) // axis_stride) % int(axis_size))
    return tuple(coords)


def inverse_zig_zag(lvals, coords):
    """
    Convert lattice coordinates into a linear site index.

    The mapping is the inverse of :func:`zig_zag` and uses the same row-major convention.

    Parameters
    ----------
    lvals : sequence of int
        Lattice sizes per axis, e.g. "[Lx, Ly, Lz]".
    coords : sequence of int
        Lattice coordinates "(x, y, z, ...)" with "0 <= coord[i] < lvals[i]".

    Returns
    -------
    int
        Linear index in "[0, prod(lvals) - 1]".

    Raises
    ------
    ValueError
        If any coordinate is outside its axis bounds, or if the dimensionality mismatches.
    """
    validate_parameters(lvals=lvals, coords=coords)
    coords = tuple(coords)
    lattice_dim = len(lvals)
    if len(coords) != lattice_dim:
        raise ValueError(f"coords must have length {lattice_dim}, got {len(coords)}")
    axis_names = "xyz"
    for axis_index, (coord_value, axis_size) in enumerate(zip(coords, lvals)):
        if coord_value < 0 or coord_value >= axis_size:
            axis = axis_names[axis_index] if axis_index < 3 else f"axis{axis_index}"
            msg = f"{axis}-coord must be in [0,{axis_size - 1}]: got {coord_value}"
            raise ValueError(msg)
    strides = _lattice_strides(lvals)
    index_1d = 0
    for coord_value, axis_stride in zip(coords, strides):
        index_1d += int(coord_value) * int(axis_stride)
    return int(index_1d)
