import numpy as np
from itertools import product
from ed_lgt.tools import validate_parameters

__all__ = ["border_mask", "staggered_mask"]


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
