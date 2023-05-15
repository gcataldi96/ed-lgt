import numpy as np
from scipy.sparse import diags

__all__ = ["get_spin_operators"]


def get_spin_operators(s):
    if not np.isscalar(s):
        raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(s)}")
    """
    This function computes the spin (sparse) matrices: 
    [Sz, Sp=S+, Sm=S-, Sx, Sy, S2=Casimir]
    in any arbitrary spin-s representation

    Args:
        s (scalar, real): spin value, assumed to be integer or semi-integer

    Returns:
        dict: dictionary with the spin matrices
    """
    # Size of the spin matrix
    size = int(2 * s + 1)
    shape = (size, size)
    # Diagonal entries of the Sz matrix
    sz_diag = np.arange(-s, s + 1)[::-1]
    # Diagonal entries of the S+ matrix
    sp_diag = np.sqrt(s * (s + 1) - sz_diag[1:] * (sz_diag[1:] + 1))
    ops = {}
    ops["Sz"] = diags(sz_diag, 0, shape)
    ops["Sp"] = diags(sp_diag, 1, shape)
    ops["Sm"] = ops["Sp"].transpose()
    ops["Sx"] = 0.5 * (ops["Sp"] + ops["Sm"])
    ops["Sy"] = complex(0, -0.5) * (ops["Sp"] - ops["Sm"])
    ops["S2"] = diags([s * (s + 1) for i in range(size)], 0, shape)
    return ops
