import numpy as np
from scipy.sparse import csr_matrix

__all__ = ["get_spin12_operators"]


def get_spin12_operators():
    ops = {}
    data_sx = np.array([1, 1])
    x_sx = np.array([1, 2])
    y_sx = np.array([2, 1])
    ops["sigma_x"] = csr_matrix((data_sx, (x_sx - 1, y_sx - 1)), shape=(2, 2))
    # -------------------------------------------------------
    data_sy = np.array([complex(0, -1), complex(0, 1)])
    x_sy = np.array([1, 2])
    y_sy = np.array([2, 1])
    ops["sigma_y"] = csr_matrix((data_sy, (x_sy - 1, y_sy - 1)), shape=(2, 2))
    # -------------------------------------------------------
    data_sz = np.array([1, -1])
    x_sz = np.array([1, 2])
    y_sz = np.array([1, 2])
    ops["sigma_z"] = csr_matrix((data_sz, (x_sz - 1, y_sz - 1)), shape=(2, 2))
    return ops
