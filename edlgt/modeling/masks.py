"""Boolean masks for selecting lattice regions and staggered sublattices.

This module provides convenience functions to build masks for borders, corners,
staggered subsets, and open-boundary-condition decompositions of hypercubic
lattices.
"""

import numpy as np
from itertools import product
from edlgt.tools import validate_parameters
import logging

logger = logging.getLogger(__name__)

__all__ = ["border_mask", "staggered_mask", "corner_mask", "obc_mask"]


def staggered_mask(lvals, stag_label):
    """Build a mask selecting even or odd staggered lattice sites.

    Parameters
    ----------
    lvals : tuple
        Lattice shape (one entry per lattice axis).
    stag_label : str
        ``"even"`` or ``"odd"``.

    Returns
    -------
    numpy.ndarray
        Boolean mask of shape ``lvals``.
    """
    # Check on parameters
    validate_parameters(lvals=lvals, stag_label=stag_label)
    mask = np.zeros(lvals, dtype=bool)
    for coords in product(*[range(l) for l in lvals]):
        stag = (-1) ** sum(coords)
        if stag_label == "even" and stag > 0:
            mask[coords] = True
        elif stag_label == "odd" and stag < 0:
            mask[coords] = True
    return mask


def border_mask(lvals, border, stag_label=None):
    """Build a mask selecting one lattice border.

    Parameters
    ----------
    lvals : tuple
        Lattice shape (one entry per lattice axis).
    border : str
        Border label (e.g. ``"mx"``, ``"px"``, ``"my"``, ``"py"``).
    stag_label : str, optional
        Optional staggered filter (``"even"`` or ``"odd"``).

    Returns
    -------
    numpy.ndarray
        Boolean mask of shape ``lvals``.
    """
    # Check on parameters
    validate_parameters(lvals=lvals, site_label=border, stag_label=stag_label)
    dim = len(lvals)
    mask = np.zeros(lvals, dtype=bool)
    allowed_borders = [f"{s}{d}" for s in "mp" for d in "xyz"[:dim]]
    if dim == 1:
        if border == "mx":
            mask[0] = True
        elif border == "px":
            mask[-1] = True
        else:
            raise ValueError(f"Expected one of {allowed_borders}: got {border}")
    elif dim == 2:
        if border == "mx":
            mask[0, :] = True
        elif border == "px":
            mask[-1, :] = True
        elif border == "my":
            mask[:, 0] = True
        elif border == "py":
            mask[:, -1] = True
        else:
            raise ValueError(f"Expected one of {allowed_borders}: got {border}")
    elif dim == 3:
        if border == "mx":
            mask[0, :, :] = True
        elif border == "px":
            mask[-1, :, :] = True
        elif border == "my":
            mask[:, 0, :] = True
        elif border == "py":
            mask[:, -1, :] = True
        elif border == "mz":
            mask[:, :, 0] = True
        elif border == "pz":
            mask[:, :, -1] = True
        else:
            raise ValueError(f"Expected one of {allowed_borders}: got {border}")
    if stag_label is None:
        return mask
    else:
        # Apply a staggered mask in addition
        stag_mask = staggered_mask(lvals, stag_label)
        return mask * stag_mask


def corner_mask(lvals, borders, stag_label=None):
    """Build a mask selecting corners defined by a set of borders.

    Parameters
    ----------
    lvals : tuple
        Lattice shape (one entry per lattice axis).
    borders : list[str]
        Border labels that define the corner(s).
    stag_label : str, optional
        Optional staggered filter (``"even"`` or ``"odd"``).

    Returns
    -------
    numpy.ndarray
        Boolean mask of shape ``lvals``.
    """
    mask = np.ones(lvals, dtype=bool)
    for border in borders:
        mask = mask * border_mask(lvals, border, stag_label)
    return mask


def obc_mask(lvals, stag_label=None):
    """Build a dictionary of masks for OBC lattice regions.

    Parameters
    ----------
    lvals : tuple
        Lattice shape (one entry per lattice axis).
    stag_label : str, optional
        Optional staggered filter applied to every mask.

    Returns
    -------
    dict
        Dictionary containing masks for the core, borders, and corners.
    """
    masks = {}
    masks["core"] = np.zeros(lvals, dtype=bool)
    bord_list = []
    for d in "xyz"[: len(lvals)]:
        for s in "mp":
            border = f"{s}{d}"
            bord_list.append(border)
            masks[border] = border_mask(lvals, border)
            # Filter the core mask deleting borders
            masks["core"] = masks["core"] + masks[border]
    masks["core"] = ~masks["core"]
    # Create the corner masks
    for bpair in product(bord_list, bord_list):
        b1, b2 = bpair
        if b1[1] != b2[1] and b1 < b2:
            masks[f"{b1},{b2}"] = corner_mask(lvals, [b1, b2])
    # Filter the borders removing the corners
    # TO BE IMPLEMENTED
    # Filter with staggerization
    if stag_label is not None:
        for mask_name in masks.keys():
            masks[mask_name] = masks[mask_name] * staggered_mask(lvals, stag_label)
    return masks
