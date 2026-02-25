import numpy as np
from itertools import product
from edlgt.tools import validate_parameters
import logging

logger = logging.getLogger(__name__)

__all__ = ["border_mask", "staggered_mask", "corner_mask", "obc_mask"]


def staggered_mask(lvals, stag_label):
    """
    This function provides a d-dimensional array of bools
    corresponding to the sites of the lattice (of size lvals)
    that are respectively site=even or site=odd

    Args:
        lvals (tuple of ints): lattice size

        stag_label (str): It can be "even" or "odd"
    Returns:
        ndarray: mask array of shape=lvals
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
    """
    This function generate the mask d-dimensional array of booleans
    corresponding to specific sites on a certain lattice border.
    Eventually, it can also take into account the staggerization of
    the lattice, acting only on even or odd sites.

    Args:
        lvals (tuple of ints): lattice size

        border (str): one of [mx, px, my, py, mz, pz]

        stag_label (str, optional): It can be "even" or "odd". Defaults to None.
    Returns:
        ndarray: mask array of shape=lvals
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
    if stag_label == None:
        return mask
    else:
        # Apply a staggered mask in addition
        stag_mask = staggered_mask(lvals, stag_label)
        return mask * stag_mask


def corner_mask(lvals, borders, stag_label=None):
    """
    This function generates a d-dimensional boolean mask array identifying
    the corner sites of a lattice based on specified borders. Corner sites
    are those where the indices reach the specified borders in each dimension.

    Args:
        lvals (tuple of ints): Lattice size, with one entry per dimension.

        borders (list of str): List of borders in the form of ["mx", "my", ...]
            specifying the minimum (m) or maximum (p) border in each dimension.

        stag_label (str, optional): Can be "even" or "odd" to further staggerize
            the mask by setting only even or odd sites within the specified corners
            to True. Defaults to None.

    Returns:
        ndarray: Boolean mask array of shape `lvals`, where True represents the
                 corner sites defined by the borders and staggerization, if any.
    """
    mask = np.ones(lvals, dtype=bool)
    for border in borders:
        mask = mask * border_mask(lvals, border, stag_label)
    return mask


def obc_mask(lvals, stag_label=None):
    """
    This function generates a dictionary of boolean masks for a lattice with
    open boundary conditions (OBC), differentiating between the core, border,
    and corner regions of the lattice.

    Args:
        lvals (tuple of ints): Lattice size, with one entry per dimension.

        stag_label (str, optional): Can be "even" or "odd" to staggerize each
            mask in the dictionary, setting only even or odd sites to True.
            Defaults to None.

    Returns:
        dict: A dictionary containing the following keys:
            - "core": Mask of the core region (all sites except borders and corners).
            - Border masks: Masks for each specified border (e.g., "mx", "px", "my").
            - Corner masks: Masks for each unique pair of borders, representing the
                            corners (e.g., "mx,my" for the bottom-left corner in 2D).

            Each mask is an ndarray of shape `lvals`, with True representing the
            sites in the corresponding region (core, border, or corner).
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
