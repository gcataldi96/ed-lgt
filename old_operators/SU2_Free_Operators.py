import numpy as np
from numpy import sqrt
from scipy.sparse import csr_matrix

from .Manage_Data import acquire_data

# ===============================================================================================
# NOTE: OPERATORS FOR THE FREE THEORY:
#       * 9_DIM SINGLE SITE BASIS
#       * NO FERMIONIC MATTER, ONLY GAUGE FIELDS
# ===============================================================================================
def identity():
    data = np.ones(9)
    x = np.arange(1, 10, 1)
    ID = csr_matrix((data, (x - 1, x - 1)), shape=(9, 9))
    return ID


def gamma_operator():
    data = np.array([1, 1, 1, 1, 1, 1, 2, 2])
    x_gamma = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    Gamma = csr_matrix((data, (x_gamma - 1, x_gamma - 1)), shape=(9, 9))
    return Gamma


def plaquette():
    path = "old_operators/Op_SU2_free/"
    # ===================================================================================================
    Bottom_Left = acquire_data(path + "Corner_Bottom_Left.txt")
    # COORDINATES
    x_Bottom_Left = Bottom_Left["0"]
    y_Bottom_Left = Bottom_Left["1"]
    # DATA
    data_Bottom_Left = Bottom_Left["2"]
    # GENERATE MATRIX
    C_Bottom_Left = csr_matrix(
        (data_Bottom_Left, (x_Bottom_Left - 1, y_Bottom_Left - 1)), shape=(9, 9)
    )
    # ===================================================================================================
    Bottom_Right = acquire_data(path + "Corner_Bottom_Right.txt")
    # COORDINATES
    x_Bottom_Right = Bottom_Right["0"]
    y_Bottom_Right = Bottom_Right["1"]
    # DATA
    data_Bottom_Right = Bottom_Right["2"]
    # GENERATE MATRIX
    C_Bottom_Right = csr_matrix(
        (data_Bottom_Right, (x_Bottom_Right - 1, y_Bottom_Right - 1)), shape=(9, 9)
    )
    # ===================================================================================================
    Top_Left = acquire_data(path + "Corner_Top_Left.txt")
    # COORDINATES
    x_Top_Left = Top_Left["0"]
    y_Top_Left = Top_Left["1"]
    # DATA
    data_Top_Left = Top_Left["2"]
    # GENERATE MATRIX
    C_Top_Left = csr_matrix(
        (data_Top_Left, (x_Top_Left - 1, y_Top_Left - 1)), shape=(9, 9)
    )
    # ===================================================================================================
    Top_Right = acquire_data(path + "Corner_Top_Right.txt")
    # COORDINATES
    x_Top_Right = Top_Right["0"]
    y_Top_Right = Top_Right["1"]
    # DATA
    data_Top_Right = Top_Right["2"]
    # GENERATE MATRIX
    C_Top_Right = csr_matrix(
        (data_Top_Right, (x_Top_Right - 1, y_Top_Right - 1)), shape=(9, 9)
    )
    # ===================================================================================================
    return C_Bottom_Left, C_Bottom_Right, C_Top_Left, C_Top_Right


def plaquette_old():
    llambda = 1
    data_plaquette = np.array(
        [
            -1 / sqrt(2),
            -1 / sqrt(2),
            1 / (2 * (llambda**2)),
            1 / (2 * (llambda**2)),
            -(llambda**2) / 2,
            -(llambda**2) / 2,
            -sqrt(6) / 4,
            -sqrt(6) / 4,
            -sqrt(2) / 4,
            -sqrt(2) / 4,
        ]
    )
    x_Bottom_Left = np.array([7, 1, 4, 6, 3, 5, 8, 2, 9, 2])
    y_Bottom_Left = np.array([1, 7, 3, 5, 4, 6, 2, 8, 2, 9])
    C_Bottom_Left = csr_matrix(
        (data_plaquette, (x_Bottom_Left - 1, y_Bottom_Left - 1)), shape=(9, 9)
    )
    # ===================================================================================================
    x_Bottom_Right = np.array([5, 1, 3, 7, 2, 6, 8, 4, 9, 4])
    y_Bottom_Right = np.array([1, 5, 2, 6, 3, 7, 4, 8, 4, 9])
    C_Bottom_Right = csr_matrix(
        (data_plaquette, (x_Bottom_Right - 1, y_Bottom_Right - 1)), shape=(9, 9)
    )
    # ===================================================================================================
    x_Top_Left = np.array([4, 1, 2, 3, 6, 7, 8, 5, 9, 5])
    y_Top_Left = np.array([1, 4, 6, 7, 2, 3, 5, 8, 5, 9])
    C_Top_Left = csr_matrix(
        (-data_plaquette, (x_Top_Left - 1, y_Top_Left - 1)), shape=(9, 9)
    )
    # ===================================================================================================
    x_Top_Right = np.array([2, 1, 5, 6, 3, 4, 8, 7, 9, 7])
    y_Top_Right = np.array([1, 2, 3, 4, 5, 6, 7, 8, 7, 9])
    C_Top_Right = csr_matrix(
        (data_plaquette, (x_Top_Right - 1, y_Top_Right - 1)), shape=(9, 9)
    )
    # ===================================================================================================
    return C_Bottom_Left, C_Bottom_Right, C_Top_Left, C_Top_Right


def W_operators():
    # All these operators are DIAGONAL MATRICES
    x_coords = np.arange(1, 10, 1)
    # ===================================================================================================
    data_Left = np.array([1, 1, -1, 1, -1, 1, -1, -1, -1])
    W_Left = csr_matrix((data_Left, (x_coords - 1, x_coords - 1)), shape=(9, 9))
    # ===================================================================================================
    data_Right = np.array([1, -1, -1, -1, 1, 1, 1, -1, -1])
    W_Right = csr_matrix((data_Right, (x_coords - 1, x_coords - 1)), shape=(9, 9))
    # ===================================================================================================
    data_Bottom = np.array([1, 1, 1, -1, 1, -1, -1, -1, -1])
    W_Bottom = csr_matrix((data_Bottom, (x_coords - 1, x_coords - 1)), shape=(9, 9))
    # ===================================================================================================
    data_Top = np.array([1, -1, 1, 1, -1, -1, 1, -1, -1])
    W_Top = csr_matrix((data_Top, (x_coords - 1, x_coords - 1)), shape=(9, 9))
    # ===================================================================================================
    return W_Left, W_Right, W_Bottom, W_Top


def penalties():
    # Penalties we impose on the corner of the lattice to use
    # OBC and not to consider the action of external Rishon modes
    data_L = np.ones(4)
    x_L = np.array([1, 5, 6, 7])
    P_left = csr_matrix((data_L, (x_L - 1, x_L - 1)), shape=(9, 9))
    # ===================================================================================================
    data_R = np.ones(4)
    x_R = np.array([1, 2, 4, 6])
    P_right = csr_matrix((data_R, (x_R - 1, x_R - 1)), shape=(9, 9))
    # ===================================================================================================
    data_B = np.ones(4)
    x_B = np.array([1, 3, 4, 7])
    P_bottom = csr_matrix((data_B, (x_B - 1, x_B - 1)), shape=(9, 9))
    # ===================================================================================================
    data_T = np.ones(4)
    x_T = np.array([1, 2, 3, 5])
    P_top = csr_matrix((data_T, (x_T - 1, x_T - 1)), shape=(9, 9))
    # ===================================================================================================
    return P_left, P_right, P_bottom, P_top
