# %%
import numpy as np
from math import prod
from copy import copy, deepcopy
from itertools import product
from numba import typed
from .lattice_mappings import inverse_zig_zag, zig_zag
from edlgt.tools import validate_parameters

__all__ = [
    "get_site_label",
    "lattice_base_configs",
    "get_neighbor_sites",
    "get_plaquette_neighbors",
    "get_origin_surfaces",
    "get_lattice_borders_labels",
    "LGT_border_configs",
    "get_lattice_link_site_pairs",
]


def get_site_label(coords, lvals, has_obc, staggered_basis=False, all_sites_equal=True):
    """
    This function associate a label to each lattice site according
    to the presence of a staggered basis, the choice of the boundary
    conditions and the position of the site in the lattice.

    Args:
        coords (tuple of ints): d-dimensional coordinates of a point in the lattice

        lvals (list of ints): lattice dimensions

        has_obc (list of bool): true for OBC, false for PBC along each direction

        staggered_basis (bool, optional): if True, a staggered basis is required. Defaults to False.

        all_sites_equal (bool, optional): if False, a different basis can be used for sites
            on borders and corners of the lattice

    Returns:
        (np.array((lvals)),np.array((lvals))): the d-dimensional array with the labels
            of the site and the corresponding site-basis dimensions
    """
    # Validate type of parameters
    validate_parameters(
        coords=coords,
        lvals=lvals,
        has_obc=has_obc,
        staggered_basis=staggered_basis,
        all_sites_equal=all_sites_equal,
    )
    # Define the list of lattice axes
    dimension = "xyz"[: len(lvals)]
    # STAGGERED LABEL
    stag_label = (
        "even"
        if staggered_basis and (-1) ** sum(coords) > 0
        else "odd" if staggered_basis else "site"
    )
    # SITE LABEL
    site_label = ""
    if not all_sites_equal:
        for ii, c in enumerate(coords):
            if has_obc[ii]:
                if c == 0:
                    site_label += f"_m{dimension[ii]}"
                elif c == lvals[ii] - 1:
                    site_label += f"_p{dimension[ii]}"
                elif c < 0 or c > lvals[ii] - 1:
                    raise ValueError(
                        f"coords[{ii}] must be in betweem 0 and {lvals[ii]-1}: got {c}"
                    )
    label = f"{stag_label}{site_label}"
    return label


def lattice_base_configs(gauge_basis, lvals, has_obc, staggered_basis=False):
    """
    This function associates the basis to each lattice site and the corresponding dimension.

    Args:
        gauge_basis (dict of sparse matrices): dict with the proper hilbert basis
            of a given LGT for each lattice site

        lvals (list of ints): lattice dimensions

        has_obc (list of bool): true for OBC, false for PBC along each direction

        staggered_basis (bool, optional): if True, a staggered  basis is required. Default to False.

    Returns:
        (np.array((lvals)),np.array((lvals))): the d-dimensional array with the labels of
            the site and the d-dimensional array with the corresponding site-basis dimensions
    """
    # Validate type of parameters
    validate_parameters(lvals=lvals, has_obc=has_obc, staggered_basis=staggered_basis)
    # Construct the lattice base
    lattice_base = np.zeros(tuple(lvals), dtype=object)
    loc_dims = np.zeros(tuple(lvals), dtype=int)
    for coords in product(*[range(l) for l in lvals]):
        # PROVIDE A LABEL TO THE LATTICE SITE
        label = get_site_label(
            coords, lvals, has_obc, staggered_basis, all_sites_equal=False
        )
        lattice_base[tuple(coords)] = label
        loc_dims[tuple(coords)] = gauge_basis[label].shape[1]
    return lattice_base, loc_dims


def get_neighbor_sites(coords, lvals, axis, has_obc):
    """
    Calculates the neighboring sites along a specified axis for a given lattice site.

    This function is used to determine the neighboring sites in a lattice system,
    taking into account the lattice dimensions, specified axis, and whether periodic
    boundary conditions (PBC) along that axis are applied.
    It is applicable for finding close sites and two-body term sites in a lattice.

    Args:
        coords (tuple/list of ints): The coordinates of the initial site in the lattice.

        lvals (list of ints ): The dimensions of the lattice. Represents the number of
            sites along each axis (e.g., (Lx, Ly, Lz) for a 3D lattice).

        axis (str): The axis along which the neighboring sites are to be found.
            Should be a character 'x', 'y', or 'z', corresponding to the axis.

        has_obc (list of bools): Indicates whether open boundary conditions (OBC) are
            applied along each axis. List of booleans corresponding to each axis in 'lvals'.
            True for OBC, False for PBC.

    Returns:
        coords_list: A list of tuples representing the coordinates of the initial
            site and its neighbor. Returns None if no neighbor is found.

        sites_list: A list of integers representing the 1D lattice indices of the
            initial site and its neighbor, as converted by the inverse_zig_zag
            function. Returns None if no neighbor is found.

    Example:
        >>> get_neighbor_sites(coords=[0, 0], lvals=[3, 3], axis='x', has_obc=[False, False])
            ([(0, 0), (1, 0)], [0, 1])
    """
    # Validate type of parameters
    validate_parameters(coords=coords, lvals=lvals, axes=[axis], has_obc=has_obc)
    dimensions = "xyz"[: len(lvals)]
    coords1 = list(coords)
    i1 = inverse_zig_zag(lvals, coords1)
    coords2 = deepcopy(coords1)
    # Check if the site admits a neighbor along the direction axis
    # Look at the specific index of the axis
    indx = dimensions.index(axis)
    # Handles both normal and PBC cases
    if coords1[indx] < lvals[indx] - 1 or (
        coords1[indx] == lvals[indx] - 1 and not has_obc[indx]
    ):
        coords2[indx] = (coords2[indx] + 1) % lvals[indx]
        i2 = inverse_zig_zag(lvals, coords2)
        sites_list = [i1, i2]
        coords_list = [tuple(coords1), tuple(coords2)]
    else:
        sites_list, coords_list = None, None
    return coords_list, sites_list


def get_plaquette_neighbors(coords, lvals, axes, has_obc):
    """
    Given a “lower-left” corner coords in an len(lvals)-D lattice,
    and a 2-element list of axes (e.g. ['x','y'] or ['x','z']), return the
    four sites of the elementary plaquette in that plane, or None if it
    doesn’t fit (because you’re at the boundary in one direction under OBC).

    Args:
      coords   - tuple of length D
      lvals    - list of D integers
      axes     - ['x','y'], ['x','z'], etc.
      has_obc  - list of D bools (True=open BC, False=PBC)

    Returns:
      (coords_list, sites_list) or (None, None)
    """
    D = len(lvals)
    dims = "xyz"[:D]
    # figure out which integer indices correspond to your two axes
    ia = dims.index(axes[0])
    ib = dims.index(axes[1])

    # check you can step +1 along each axis
    ca, cb = coords[ia], coords[ib]
    if not (ca < lvals[ia] - 1 or (ca == lvals[ia] - 1 and not has_obc[ia])):
        return None, None
    if not (cb < lvals[ib] - 1 or (cb == lvals[ib] - 1 and not has_obc[ib])):
        return None, None

    coords_list = []
    sites_list = []

    # 0) the “origin” corner
    base = list(coords)
    coords_list.append(tuple(base))
    sites_list.append(inverse_zig_zag(lvals, base))

    # 1) step along axis a
    c1 = copy(base)
    c1[ia] = (c1[ia] + 1) % lvals[ia]
    coords_list.append(tuple(c1))
    sites_list.append(inverse_zig_zag(lvals, c1))

    # 2) step along axis b
    c2 = copy(base)
    c2[ib] = (c2[ib] + 1) % lvals[ib]
    coords_list.append(tuple(c2))
    sites_list.append(inverse_zig_zag(lvals, c2))

    # 3) diagonal: both +1
    c3 = copy(base)
    c3[ia] = (c3[ia] + 1) % lvals[ia]
    c3[ib] = (c3[ib] + 1) % lvals[ib]
    coords_list.append(tuple(c3))
    sites_list.append(inverse_zig_zag(lvals, c3))

    return coords_list, sites_list


def get_origin_surfaces(lvals):
    """
    For a 2D or 3D lattice of size lvals, return the “origin” links (in 2D)
    or faces (in 3D), both as coordinate tuples and as 1D indices along the
    zig-zag curve.

    - If len(lvals)==2 (2D), returns two full edges through (0,0):
        'x': all sites with y=0 and x=0..Lx-1
        'y': all sites with x=0 and y=0..Ly-1

    - If len(lvals)==3 (3D), returns the three coordinate-planes through (0,0,0):
        'yz' at x=0, 'xz' at y=0, 'xy' at z=0

    Args:
        lvals (list of int): [Lx, Ly] or [Lx, Ly, Lz]

    Returns:
        dict: mapping keys → (coords, sites), where
              coords is a list of tuples and
              sites  is a list of ints via inverse_zig_zag.
    """
    if not all(isinstance(d, int) and d > 0 for d in lvals):
        raise ValueError("All dimensions in lvals must be positive ints")
    # 2D: return the two full “edges” through the origin
    if len(lvals) == 2:
        Lx, Ly = lvals
        if Lx < 2 or Ly < 2:
            raise ValueError("lvals must be at least [2,2]")

        surfaces = {}
        # edge along x at y=0
        coords_x = [(x, 0) for x in range(Lx)]
        sites_x = [inverse_zig_zag(lvals, c) for c in coords_x]
        surfaces["x"] = (coords_x, sites_x)

        # edge along y at x=0
        coords_y = [(0, y) for y in range(Ly)]
        sites_y = [inverse_zig_zag(lvals, c) for c in coords_y]
        surfaces["y"] = (coords_y, sites_y)

        return surfaces

    # 3D: return the three full surfaces through the origin
    elif len(lvals) == 3:
        Lx, Ly, Lz = lvals
        surfaces = {}

        coords_yz = [(0, y, z) for y in range(Ly) for z in range(Lz)]
        sites_yz = [inverse_zig_zag(lvals, c) for c in coords_yz]
        surfaces["yz"] = (coords_yz, sites_yz)

        coords_xz = [(x, 0, z) for x in range(Lx) for z in range(Lz)]
        sites_xz = [inverse_zig_zag(lvals, c) for c in coords_xz]
        surfaces["xz"] = (coords_xz, sites_xz)

        coords_xy = [(x, y, 0) for x in range(Lx) for y in range(Ly)]
        sites_xy = [inverse_zig_zag(lvals, c) for c in coords_xy]
        surfaces["xy"] = (coords_xy, sites_xy)

        return surfaces

    else:
        raise ValueError("get_origin_surfaces only supports 2D or 3D lattices")


def get_lattice_link_site_pairs(lvals, has_obc):
    """
    Acquire all the pairs of sites sharing a lattice link.
    Pairs form an array for each lattice dimension
    """
    site_pairs = typed.List()
    for d in "xyz"[: len(lvals)]:
        dir_list = []
        for ii in range(prod(lvals)):
            # Compute the corresponding coords
            coords = zig_zag(lvals, ii)
            # Check if it admits a twobody term according to the lattice geometry
            _, sites_list = get_neighbor_sites(coords, lvals, d, has_obc)
            if sites_list is not None:
                dir_list.append(sites_list)
        site_pairs.append(np.array(dir_list, dtype=np.uint8))
    return site_pairs


def get_lattice_borders_labels(lattice_dim):
    if lattice_dim == 1:
        return ["mx", "px"]
    elif lattice_dim == 2:
        return ["mx", "px", "my", "py", "mx_my", "mx_py", "px_my", "px_py"]
    elif lattice_dim == 3:
        return [
            "mx",
            "px",
            "my",
            "py",
            "mz",
            "pz",
            "mx_my",
            "mx_py",
            "mx_mz",
            "mx_pz",
            "px_my",
            "px_py",
            "px_mz",
            "px_pz",
            "my_mz",
            "my_pz",
            "py_mz",
            "py_pz",
            "mx_my_mz",
            "mx_my_pz",
            "mx_py_mz",
            "mx_py_pz",
            "px_my_mz",
            "px_my_pz",
            "px_py_mz",
            "px_py_pz",
        ]


def LGT_border_configs(config, offset, pure_theory, get_only_bulk=False):
    """
    This function fixes the value of the electric field on
    lattices with open boundary conditions (has_obc=True).

    For the moment, it works only with integer spin representation
    where the offset of E is naturally the central value assumed by the rishon number.

    Args:
        config (list of ints): configuration of internal rishons in the single dressed site basis

        offset (scalar, int): offset corresponding to a trivial description of the gauge field.
            In QED, where the U(1) Gauge field is truncated with a spin-s representation (integer for the moment)
            and lives in a gauge Hilbert space of dimension (2s +1), the offset is exactly s.
            In SU2, the offset is the size of the Hilbert space of the J=0 spin rep, which is 1.

        pure_theory (bool): True if the theory does not include matter

    Returns:
        list of strings: list of lattice borders/corners displaying that trivial description the gauge field
    """
    if not isinstance(config, list) and not isinstance(config, tuple):
        raise TypeError(f"config should be a LIST, not a {type(config)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    # Look at the configuration
    label = []
    if get_only_bulk:
        return label
    else:
        if not np.isscalar(offset) or not isinstance(offset, int):
            raise TypeError(f"offset must be SCALAR & INTEGER, not {type(offset)}")
        if not pure_theory:
            config = config[1:]
        lattice_dim = len(config) // 2
        if lattice_dim == 1:
            if config[0] == offset:
                label.append("mx")
            if config[1] == offset:
                label.append("px")
        elif lattice_dim == 2:
            if config[0] == offset:
                label.append("mx")
            if config[1] == offset:
                label.append("my")
            if config[2] == offset:
                label.append("px")
            if config[3] == offset:
                label.append("py")
            if (config[0] == offset) and (config[1] == offset):
                label.append("mx_my")
            if (config[0] == offset) and (config[3] == offset):
                label.append("mx_py")
            if (config[1] == offset) and (config[2] == offset):
                label.append("px_my")
            if (config[2] == offset) and (config[3] == offset):
                label.append("px_py")
        elif lattice_dim == 3:
            if config[0] == offset:
                label.append("mx")
            if config[1] == offset:
                label.append("my")
            if config[2] == offset:
                label.append("mz")
            if config[3] == offset:
                label.append("px")
            if config[4] == offset:
                label.append("py")
            if config[5] == offset:
                label.append("pz")
            if (config[0] == offset) and (config[1] == offset):
                label.append("mx_my")
            if (config[0] == offset) and (config[2] == offset):
                label.append("mx_mz")
            if (config[0] == offset) and (config[4] == offset):
                label.append("mx_py")
            if (config[0] == offset) and (config[5] == offset):
                label.append("mx_pz")
            if (config[3] == offset) and (config[1] == offset):
                label.append("px_my")
            if (config[3] == offset) and (config[2] == offset):
                label.append("px_mz")
            if (config[3] == offset) and (config[4] == offset):
                label.append("px_py")
            if (config[3] == offset) and (config[5] == offset):
                label.append("px_pz")
            if (config[1] == offset) and (config[2] == offset):
                label.append("my_mz")
            if (config[1] == offset) and (config[5] == offset):
                label.append("my_pz")
            if (config[4] == offset) and (config[2] == offset):
                label.append("py_mz")
            if (config[4] == offset) and (config[5] == offset):
                label.append("py_pz")
            if (
                (config[0] == offset)
                and (config[1] == offset)
                and (config[2] == offset)
            ):
                label.append("mx_my_mz")
            if (
                (config[0] == offset)
                and (config[1] == offset)
                and (config[5] == offset)
            ):
                label.append("mx_my_pz")
            if (
                (config[0] == offset)
                and (config[4] == offset)
                and (config[2] == offset)
            ):
                label.append("mx_py_mz")
            if (
                (config[0] == offset)
                and (config[4] == offset)
                and (config[5] == offset)
            ):
                label.append("mx_py_pz")
            if (
                (config[3] == offset)
                and (config[1] == offset)
                and (config[2] == offset)
            ):
                label.append("px_my_mz")
            if (
                (config[3] == offset)
                and (config[1] == offset)
                and (config[5] == offset)
            ):
                label.append("px_my_pz")
            if (
                (config[3] == offset)
                and (config[4] == offset)
                and (config[2] == offset)
            ):
                label.append("px_py_mz")
            if (
                (config[3] == offset)
                and (config[4] == offset)
                and (config[5] == offset)
            ):
                label.append("px_py_pz")
    return label


# %%
