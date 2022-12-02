import numpy as np

__all__ = ["border_mask", "staggered_mask"]


def border_mask(lvals, border):
    """
    Defines the masks for all four sides: top, bottom, left,
    and right as well as the four corners.
    NOTE Rows and Columns of the mask array corresponds to (x,y) coordinates!
    """
    lx = lvals[0]
    ly = lvals[1]
    mask = np.zeros((lx, ly), dtype=bool)
    if border == "my":
        mask[:, 0] = True
    elif border == "py":
        mask[:, -1] = True
    elif border == "mx":
        mask[0, :] = True
    elif border == "px":
        mask[-1, :] = True
    return mask


def staggered_mask(lvals, site):
    lx = lvals[0]
    ly = lvals[1]
    mask = np.zeros((lx, ly), dtype=bool)
    for ii in range(lx):
        for jj in range(ly):
            stag = (-1) ** (ii + jj)
            if site == "even":
                if stag > 0:
                    mask[ii, jj] = True
            elif site == "odd":
                if stag < 0:
                    mask[ii, jj] = True
    return mask
