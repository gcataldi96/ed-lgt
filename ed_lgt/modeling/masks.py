import numpy as np
from itertools import product

__all__ = [
    "border_mask",
    "staggered_mask",
    "get_lattice_borders_labels",
    "LGT_border_configs",
]


def staggered_mask(lvals, site):
    """
    This function provides a d-dimensional array of bools
    corresponding to the sites of the lattice (of size lvals)
    that are respectively site=even or site=odd

    Args:
        lvals (tuple of ints): lattice size
        site (str): It can be "even" or "odd"
    Returns:
        ndarray: mask array of shape=lvals
    """
    # CHECK ON TYPES
    if not isinstance(lvals, list):
        raise TypeError(f"lvals should be a list, not a {type(lvals)}")
    else:
        for ii, ll in enumerate(lvals):
            if not isinstance(ll, int):
                raise TypeError(f"lvals[{ii}] should be INTEGER, not {type(ll)}")
    if not isinstance(site, str):
        raise TypeError(f"site should be STR ('even' / 'odd'), not {type(site)}")
    mask = np.zeros(lvals, dtype=bool)
    for coords in product(*[range(l) for l in lvals]):
        stag = (-1) ** sum(coords)
        if site == "even":
            if stag > 0:
                mask[coords] = True
        elif site == "odd":
            if stag < 0:
                mask[coords] = True
        else:
            raise ValueError(f"Expected one of 'even' or 'odd': got {site}")
    return mask


def border_mask(lvals, border, site=None):
    """
    This function generate the mask d-dimensional array of booleans
    corresponding to specific sites on a certain lattice border.
    Eventually, it can also take into account the staggerization of
    the lattice, acting only on even or odd sites.

    Args:
        lvals (tuple of ints): lattice size
        border (str): one of [mx, px, my, py, mz, pz]
        site (str, optional): It can be "even" or "odd". Defaults to None.
    Returns:
        ndarray: mask array of shape=lvals
    """
    # CHECK ON TYPES
    if not isinstance(lvals, list):
        raise TypeError(f"lvals should be a list, not a {type(lvals)}")
    else:
        for ii, ll in enumerate(lvals):
            if not isinstance(ll, int):
                raise TypeError(f"lvals[{ii}] should be INTEGER, not {type(ll)}")
    if not isinstance(border, str):
        raise TypeError(f"border should be STR, not {type(border)}")
    if site is not None:
        if not isinstance(site, str):
            raise TypeError(f"site should be STR ('even' / 'odd'), not {type(site)}")
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
    if site == None:
        return mask
    # Applied to staggered sites (even or odd)
    else:
        stag_mask = staggered_mask(lvals, site)
        return mask * stag_mask


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
            In QED, where the U(1) Gauge field is truncated with a spin-s representation (integer for the moment) and lives in a gauge Hilbert space of dimension (2s +1), the offset is exactly s.
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
