# %%
import numpy as np
from math import prod
from copy import deepcopy
from itertools import product
from numba import typed
from .lattice_mappings import inverse_zig_zag, zig_zag
from ed_lgt.tools import validate_parameters

__all__ = [
    "get_site_label",
    "lattice_base_configs",
    "get_neighbor_sites",
    "get_plaquette_neighbors",
    "get_lattice_borders_labels",
    "LGT_border_configs",
    "get_lattice_link_site_pairs",
]


def get_site_label(coords, lvals, has_obc, staggered=False, all_sites_equal=True):
    """
    This function associate a label to each lattice site according
    to the presence of a staggered basis, the choice of the boundary
    conditions and the position of the site in the lattice.

    Args:
        coords (tuple of ints): d-dimensional coordinates of a point in the lattice

        lvals (list of ints): lattice dimensions

        has_obc (list of bool): true for OBC, false for PBC along each direction

        staggered (bool, optional): if True, a staggered basis is required. Defaults to False.

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
        staggered=staggered,
        all_sites_equal=all_sites_equal,
    )
    # Define the list of lattice axes
    dimension = "xyz"[: len(lvals)]
    # STAGGERED LABEL
    stag_label = (
        "even"
        if staggered and (-1) ** sum(coords) > 0
        else "odd" if staggered else "site"
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


def lattice_base_configs(base, lvals, has_obc, staggered=False):
    """
    This function associates the basis to each lattice site and the corresponding dimension.

    Args:
        base (dict of sparse matrices): dict with the proper hilbert basis
            of a given LGT for each lattice site

        lvals (list of ints): lattice dimensions

        has_obc (list of bool): true for OBC, false for PBC along each direction

        staggered (bool, optional): if True, a staggered basis is required. Default to False.

    Returns:
        (np.array((lvals)),np.array((lvals))): the d-dimensional array with the labels of
            the site and the d-dimensional array with the corresponding site-basis dimensions
    """
    # Validate type of parameters
    validate_parameters(lvals=lvals, has_obc=has_obc, staggered=staggered)
    # Construct the lattice base
    lattice_base = np.zeros(tuple(lvals), dtype=object)
    loc_dims = np.zeros(tuple(lvals), dtype=int)
    for coords in product(*[range(l) for l in lvals]):
        # PROVIDE A LABEL TO THE LATTICE SITE
        label = get_site_label(coords, lvals, has_obc, staggered, all_sites_equal=False)
        lattice_base[tuple(coords)] = label
        loc_dims[tuple(coords)] = base[label].shape[1]
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
    Calculates the neighboring sites forming a plaquette for a given lattice site.

    This function determines the neighboring sites in a lattice system that can
    form a plaquette with the given site. It considers the lattice dimensions,
    specified axes for the plaquette, and whether periodic boundary conditions
    (PBC) are applied.

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
            site and its plaquette neighbors. Returns None if neighbors
            are not found or plaquette cannot be formed.

        sites_list: A list of integers representing the 1D lattice indices of the
            initial site and its plaquette neighbors, as converted by the
            inverse_zig_zag function. Returns None if neighbors are not found
            or plaquette cannot be formed.

    Example:
        >>> get_plaquette_neighbors(coords=[0, 0], lvals=[3, 3], axes=['x', 'y'], has_obc=[False, False])
            ([(0, 0), (1, 0), (0, 1), (1, 1)], [0, 1, 3, 4])
    """
    # Validate type of parameters
    validate_parameters(coords=coords, lvals=lvals, axes=axes, has_obc=has_obc)
    dimensions = "xyz"[: len(lvals)]
    coords1 = list(coords)
    i1 = inverse_zig_zag(lvals, coords1)
    coords_list = [tuple(coords1)]
    sites_list = [i1]
    # Build the neighboring sites along the 2 axes.
    # At the end of the cycle, we have 3 Plaquette sites out of 4.
    for ax in axes:
        coords2 = deepcopy(coords1)
        # Get the indices of the axis
        indx = dimensions.index(ax)
        # Look at the neighbor site along each axis
        if coords1[indx] < lvals[indx] - 1 or (
            coords1[indx] == lvals[indx] - 1 and not has_obc[indx]
        ):
            coords2[indx] = (coords2[indx] + 1) % lvals[indx]
            coords_list.append(tuple(coords2))
            sites_list.append(inverse_zig_zag(lvals, coords2))
        else:
            # Plaquette cannot be formed in this case
            return None, None
    # Add the last diagonal neighbor, in case both axes adimts neighbors
    if len(coords_list) == 3:
        coords3 = [coords_list[1][0], coords_list[2][1]]
        if len(lvals) > 2:
            # For 3D lattices, maintain the third coordinate
            coords3.append(coords1[2])
        coords_list.append(tuple(coords3))
        sites_list.append(inverse_zig_zag(lvals, coords3))
    return coords_list, sites_list


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


def LGT_border_configs(config, offset, pure_theory):
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
    if not np.isscalar(offset) or not isinstance(offset, int):
        raise TypeError(f"offset must be SCALAR & INTEGER, not {type(offset)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    # Look at the configuration
    label = []
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
        if (config[0] == offset) and (config[1] == offset) and (config[2] == offset):
            label.append("mx_my_mz")
        if (config[0] == offset) and (config[1] == offset) and (config[5] == offset):
            label.append("mx_my_pz")
        if (config[0] == offset) and (config[4] == offset) and (config[2] == offset):
            label.append("mx_py_mz")
        if (config[0] == offset) and (config[4] == offset) and (config[5] == offset):
            label.append("mx_py_pz")
        if (config[3] == offset) and (config[1] == offset) and (config[2] == offset):
            label.append("px_my_mz")
        if (config[3] == offset) and (config[1] == offset) and (config[5] == offset):
            label.append("px_my_pz")
        if (config[3] == offset) and (config[4] == offset) and (config[2] == offset):
            label.append("px_py_mz")
        if (config[3] == offset) and (config[4] == offset) and (config[5] == offset):
            label.append("px_py_pz")
    return label


# %%
