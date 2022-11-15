import numpy as np
from numpy import sqrt
from scipy.sparse import csr_matrix
from .Manage_Data import acquire_data

# ===============================================================================================
# NOTE: OPERATORS FOR THE FULL THEORY:
#       * 30_DIM SINGLE SITE BASIS
#       * BOTH FERMIONIC MATTER AND GAUGE FIELDS
# ===============================================================================================
def identity():
    data = np.ones(30)
    x = np.arange(1, 31, 1)
    ID = csr_matrix((data, (x - 1, x - 1)), shape=(30, 30))
    return ID


def matter_operator():
    # * DIAGONAL OPERATOR
    # * The coefficients 2 are referred to DOUBLY OCCUPIED fermionic states
    data_D = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    x_D = np.arange(10, 31, 1)
    D_operator = csr_matrix((data_D, (x_D - 1, x_D - 1)), shape=(30, 30))
    return D_operator


def gamma_operator():
    # * DIAGONAL OPERATOR
    # The first 8 coefficients are for the EMPTY MATTER STATES:
    #   * state 0 is not present as it represent the VACUUM
    #   * state 8 and 9 are TRICKY and this explains the factor 4
    # The second 12 coefficients are for the SINGLY OCCUPIED STATES:
    #   * from state 14 to state 22 we have TRICKY definition and a factor 3
    # The last 9 coefficients are for the DOUBLY OCCUPIED STATES
    #   * state 29 and 30 have TRICKY DEFINITIONS and the factor 4
    data_G = 0.5 * np.array(
        [
            2,
            2,
            2,
            2,
            2,
            2,
            4,
            4,
            1,
            1,
            1,
            1,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            2,
            2,
            2,
            2,
            2,
            2,
            4,
            4,
        ]
    )
    x_G = np.array(
        [
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
        ]
    )
    Gamma = csr_matrix((data_G, (x_G - 1, x_G - 1)), shape=(30, 30))
    return Gamma


def W_operators():
    # * DIAGONAL OPERATORS
    x_coords = np.arange(1, 31, 1)
    # ===================================================================================================
    data_Left = np.array(
        [
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
        ]
    )
    W_Left = csr_matrix((data_Left, (x_coords - 1, x_coords - 1)), shape=(30, 30))
    # ===================================================================================================
    data_Right = np.array(
        [
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
        ]
    )
    W_Right = csr_matrix((data_Right, (x_coords - 1, x_coords - 1)), shape=(30, 30))
    # ===================================================================================================
    data_Bottom = np.array(
        [
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
        ]
    )
    W_Bottom = csr_matrix((data_Bottom, (x_coords - 1, x_coords - 1)), shape=(30, 30))
    # ===================================================================================================
    data_Top = np.array(
        [
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
        ]
    )
    W_Top = csr_matrix((data_Top, (x_coords - 1, x_coords - 1)), shape=(30, 30))
    # ===================================================================================================
    return W_Left, W_Right, W_Bottom, W_Top


def plaquette():
    path = "old_operators/Op_SU2_matter/"
    # ===================================================================================================
    Bottom_Left = acquire_data(path + "Corner_Bottom_Left.txt")
    # COORDINATES
    x_Bottom_Left = Bottom_Left["0"]
    y_Bottom_Left = Bottom_Left["1"]
    # DATA
    data_Bottom_Left = Bottom_Left["2"]
    # GENERATE MATRIX
    C_Bottom_Left = csr_matrix(
        (data_Bottom_Left, (x_Bottom_Left - 1, y_Bottom_Left - 1)), shape=(30, 30)
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
        (data_Bottom_Right, (x_Bottom_Right - 1, y_Bottom_Right - 1)), shape=(30, 30)
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
        (data_Top_Left, (x_Top_Left - 1, y_Top_Left - 1)), shape=(30, 30)
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
        (data_Top_Right, (x_Top_Right - 1, y_Top_Right - 1)), shape=(30, 30)
    )
    # ===================================================================================================
    return C_Bottom_Left, C_Bottom_Right, C_Top_Left, C_Top_Right


def hopping():
    path = "old_operators/Op_SU2_matter/"
    # ===================================================================================================
    Left = acquire_data(path + "Q_Left_dagger.txt")
    # COORDINATES
    x_Left = Left["0"]
    y_Left = Left["1"]
    # DATA
    data_Left = Left["2"]
    # GENERATE MATRIX
    Q_Left_dagger = csr_matrix((data_Left, (x_Left - 1, y_Left - 1)), shape=(30, 30))
    # Q_Left       =csr_matrix((data_Left, (y_Left-1, x_Left-1)), shape=(30, 30))
    # ===================================================================================================
    Right = acquire_data(path + "Q_Right_dagger.txt")
    # COORDINATES
    x_Right = Right["0"]
    y_Right = Right["1"]
    # DATA
    data_Right = Right["2"]
    # GENERATE MATRIX
    Q_Right_dagger = csr_matrix(
        (data_Right, (x_Right - 1, y_Right - 1)), shape=(30, 30)
    )
    # Q_Right       =csr_matrix((data_Right, (y_Right-1, x_Right-1)), shape=(30, 30))
    # ===================================================================================================
    Bottom = acquire_data(path + "Q_Bottom_dagger.txt")
    # COORDINATES
    x_Bottom = Bottom["0"]
    y_Bottom = Bottom["1"]
    # DATA
    data_Bottom = Bottom["2"]
    # GENERATE MATRIX
    Q_Bottom_dagger = csr_matrix(
        (data_Bottom, (x_Bottom - 1, y_Bottom - 1)), shape=(30, 30)
    )
    # Q_Bottom       =csr_matrix((data_Bottom, (y_Bottom-1, x_Bottom-1)), shape=(30, 30))
    # ===================================================================================================
    Top = acquire_data(path + "Q_Top_dagger.txt")
    # COORDINATES
    x_Top = Top["0"]
    y_Top = Top["1"]
    # DATA
    data_Top = Top["2"]
    # GENERATE MATRIX
    Q_Top_dagger = csr_matrix((data_Top, (x_Top - 1, y_Top - 1)), shape=(30, 30))
    # Q_Top       =csr_matrix((data_Top, (y_Top-1, x_Top-1)), shape=(30, 30))
    # ===================================================================================================
    # return Q_Left, Q_Left_dagger, Q_Right, Q_Right_dagger, Q_Bottom, Q_Bottom_dagger, Q_Top, Q_Top_dagger
    return Q_Left_dagger, Q_Right_dagger, Q_Bottom_dagger, Q_Top_dagger


def number_operators():
    data_PAIR = np.ones(9)
    x_PAIR = np.arange(22, 31, 1)
    n_PAIR = csr_matrix((data_PAIR, (x_PAIR - 1, x_PAIR - 1)), shape=(30, 30))
    # ===================================================================================================
    data_SINGLE = np.ones(12)
    x_SINGLE = np.arange(10, 22, 1)
    n_SINGLE = csr_matrix((data_SINGLE, (x_SINGLE - 1, x_SINGLE - 1)), shape=(30, 30))
    # ===================================================================================================
    data_TOTAL = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    )
    x_TOTAL = np.arange(10, 31, 1)
    n_TOTAL = csr_matrix((data_TOTAL, (x_TOTAL - 1, x_TOTAL - 1)), shape=(30, 30))
    # ===================================================================================================
    return n_SINGLE, n_PAIR, n_TOTAL


def penalties():
    # Penalties we impose on the corner of the lattice to use
    # OBC and not to consider the action of external Rishon modes
    data_L = np.ones(13)
    x_L = np.array([1, 5, 6, 7, 11, 12, 13, 20, 21, 22, 26, 27, 28])
    P_left = csr_matrix((data_L, (x_L - 1, x_L - 1)), shape=(30, 30))
    # ===================================================================================================
    data_R = np.ones(13)
    x_R = np.array([1, 2, 4, 6, 10, 11, 13, 16, 17, 22, 23, 25, 27])
    P_right = csr_matrix((data_R, (x_R - 1, x_R - 1)), shape=(30, 30))
    # ===================================================================================================
    data_B = np.ones(13)
    x_B = np.array([1, 3, 4, 7, 10, 12, 13, 18, 19, 22, 24, 25, 28])
    P_bottom = csr_matrix((data_B, (x_B - 1, x_B - 1)), shape=(30, 30))
    # ===================================================================================================
    data_T = np.ones(13)
    x_T = np.array([1, 2, 3, 5, 10, 11, 12, 14, 15, 22, 23, 24, 26])
    P_top = csr_matrix((data_T, (x_T - 1, x_T - 1)), shape=(30, 30))
    # ===================================================================================================
    return P_left, P_right, P_bottom, P_top


def hopping_old():
    # * ALREADY CHECKED!
    # ==========================================================================================================
    data_LEFT = np.array(
        [
            -1,
            1,
            1 / sqrt(2),
            sqrt(3) / 2,
            -1 / 2,
            1 / sqrt(2),
            sqrt(3) / 2,
            -1 / 2,
            1 / sqrt(2),
            -1 / (2 * sqrt(2)),
            -sqrt(3) / (2 * sqrt(2)),
            sqrt(3) / (2 * sqrt(2)),
            -1 / (2 * sqrt(2)),
            -1 / sqrt(2),
            -1 / sqrt(2),
            -1,
            -1 / sqrt(2),
            1,
            1 / (2 * sqrt(2)),
            -sqrt(3) / (2 * sqrt(2)),
            sqrt(3) / (2 * sqrt(2)),
            1 / (2 * sqrt(2)),
            sqrt(3) / 2,
            -1 / 2,
            sqrt(3) / 2,
            -1 / 2,
        ]
    )
    x_LEFT = np.array(
        [
            12,
            15,
            10,
            18,
            19,
            11,
            20,
            21,
            13,
            16,
            17,
            16,
            17,
            24,
            26,
            22,
            28,
            23,
            29,
            30,
            29,
            30,
            25,
            25,
            27,
            27,
        ]
    )
    y_LEFT = np.array(
        [
            1,
            2,
            3,
            4,
            4,
            5,
            6,
            6,
            7,
            8,
            8,
            9,
            9,
            10,
            11,
            12,
            13,
            15,
            16,
            16,
            17,
            17,
            18,
            19,
            20,
            21,
        ]
    )
    Q_LEFT_dag = csr_matrix((data_LEFT, (x_LEFT - 1, y_LEFT - 1)), shape=(30, 30))
    # Q_LEFT    =csr_matrix((data_LEFT, (y_LEFT-1, x_LEFT-1)), shape=(30, 30))
    # ==========================================================================================================
    data_BOTTOM = np.array(
        [
            -1,
            1,
            1,
            1 / sqrt(2),
            1,
            1 / sqrt(2),
            1 / sqrt(2),
            1 / (2 * sqrt(2)),
            -sqrt(3) / (2 * sqrt(2)),
            -sqrt(3) / (2 * sqrt(2)),
            -1 / (2 * sqrt(2)),
            -1 / sqrt(2),
            -1 / sqrt(2),
            -1 / sqrt(2),
            -1,
            -1 / (2 * sqrt(2)),
            sqrt(3) / (2 * sqrt(2)),
            sqrt(3) / (2 * sqrt(2)),
            1 / (2 * sqrt(2)),
            1,
            1,
            1,
        ]
    )
    x_BOTTOM = np.array(
        [
            13,
            17,
            19,
            10,
            21,
            11,
            12,
            14,
            15,
            14,
            15,
            25,
            27,
            28,
            22,
            29,
            30,
            29,
            30,
            23,
            24,
            26,
        ]
    )
    y_BOTTOM = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 11, 12, 13, 14, 14, 15, 15, 17, 19, 21]
    )
    Q_BOTTOM_dag = csr_matrix(
        (data_BOTTOM, (x_BOTTOM - 1, y_BOTTOM - 1)), shape=(30, 30)
    )
    # Q_BOTTOM    =csr_matrix((data_BOTTOM, (y_BOTTOM-1, x_BOTTOM-1)), shape=(30, 30))
    # ==========================================================================================================
    data_RIGHT = np.array(
        [
            -1,
            1 / sqrt(2),
            1 / sqrt(2),
            1 / sqrt(2),
            -sqrt(3) / 2,
            -1 / 2,
            -sqrt(3) / 2,
            -1 / 2,
            -sqrt(3) / 2,
            -1 / 2,
            1 / sqrt(2),
            1 / sqrt(2),
            -1,
            -1 / sqrt(2),
            -1 / sqrt(2),
            -1 / sqrt(2),
            -sqrt(3) / 2,
            -1 / 2,
            -sqrt(3) / 2,
            -1 / 2,
            -sqrt(3) / 2,
            -1 / 2,
            -1 / sqrt(2),
            -1 / sqrt(2),
        ]
    )
    x_RIGHT = np.array(
        [
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            26,
            27,
            27,
            28,
            28,
            29,
            30,
        ]
    )
    y_RIGHT = np.array(
        [
            1,
            2,
            3,
            4,
            5,
            5,
            6,
            6,
            7,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
        ]
    )
    Q_RIGHT_dag = csr_matrix((data_RIGHT, (x_RIGHT - 1, y_RIGHT - 1)), shape=(30, 30))
    # Q_RIGHT    =csr_matrix((data_RIGHT, (y_RIGHT-1, x_RIGHT-1)), shape=(30, 30))
    # ==========================================================================================================
    data_TOP = np.array(
        [
            -1,
            1 / sqrt(2),
            sqrt(3) / 2,
            -1 / 2,
            sqrt(3) / 2,
            -1 / 2,
            1 / sqrt(2),
            1 / sqrt(2),
            -sqrt(3) / 2,
            -1 / 2,
            1 / (2 * sqrt(2)),
            sqrt(3) / (2 * sqrt(2)),
            sqrt(3) / (2 * sqrt(2)),
            -1 / (2 * sqrt(2)),
            -1 / sqrt(2),
            -1,
            -1 / sqrt(2),
            -1 / sqrt(2),
            sqrt(3) / 2,
            -1 / 2,
            sqrt(3) / 2,
            -1 / 2,
            -1 / (2 * sqrt(2)),
            -sqrt(3) / (2 * sqrt(2)),
            -sqrt(3) / (2 * sqrt(2)),
            1 / (2 * sqrt(2)),
            -sqrt(3) / 2,
            -1 / 2,
        ]
    )
    x_TOP = np.array(
        [
            11,
            10,
            14,
            15,
            16,
            17,
            12,
            13,
            20,
            21,
            18,
            19,
            18,
            19,
            23,
            22,
            26,
            27,
            24,
            24,
            25,
            25,
            29,
            30,
            29,
            30,
            28,
            28,
        ]
    )
    y_TOP = np.array(
        [
            1,
            2,
            3,
            3,
            4,
            4,
            5,
            6,
            7,
            7,
            8,
            8,
            9,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            18,
            19,
            19,
            20,
            21,
        ]
    )
    Q_TOP_dag = csr_matrix((data_TOP, (x_TOP - 1, y_TOP - 1)), shape=(30, 30))
    # Q_TOP    =csr_matrix((data_TOP, (y_TOP-1, x_TOP-1)), shape=(30, 30))
    # ==========================================================================================================
    return Q_LEFT_dag, Q_RIGHT_dag, Q_BOTTOM_dag, Q_TOP_dag


def plaquette_old():
    # * Abbiamo posto llambda=1
    # ==========================================================================================================
    # ? DOVREBBE ESSERE CORRETTO (abbiamo corretto il typo 28-22 in 22-28 ultima RIGA)
    data_BOTTOM_LEFT = np.array(
        [
            1 / sqrt(2),
            sqrt(3) / (2 * sqrt(2)),
            1 / (2 * sqrt(2)),
            1 / 2,
            1 / 2,
            1 / 2,
            1 / 2,
            -1 / sqrt(2),
            -sqrt(3) / (2 * sqrt(2)),
            -1 / (2 * sqrt(2)),
            sqrt(3) / (2 * sqrt(2)),
            1 / (2 * sqrt(2)),
            sqrt(3) / (2 * sqrt(2)),
            1 / (2 * sqrt(2)),
            1 / 2,
            1 / 2,
            1 / 2,
            1 / 2,
            1 / 2,
            1 / 2,
            -sqrt(3) / (2 * sqrt(2)),
            -1 / (2 * sqrt(2)),
            -sqrt(3) / (2 * sqrt(2)),
            -1 / (2 * sqrt(2)),
            1 / (sqrt(2)),
            sqrt(3) / (2 * sqrt(2)),
            1 / (2 * sqrt(2)),
            1 / 2,
            1 / 2,
            1 / 2,
            1 / 2,
            -1 / sqrt(2),
            -sqrt(3) / (2 * sqrt(2)),
            -1 / (2 * sqrt(2)),
        ]
    )
    x_BOTTOM_LEFT = np.array(
        [
            7,
            8,
            9,
            4,
            3,
            6,
            5,
            1,
            2,
            2,
            18,
            19,
            20,
            21,
            13,
            12,
            16,
            17,
            14,
            15,
            10,
            10,
            11,
            11,
            28,
            29,
            30,
            25,
            24,
            27,
            26,
            22,
            23,
            23,
        ]
    )
    y_BOTTOM_LEFT = np.array(
        [
            1,
            2,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            10,
            11,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
        ]
    )
    C_BOTTOM_LEFT = csr_matrix(
        (data_BOTTOM_LEFT, (x_BOTTOM_LEFT - 1, y_BOTTOM_LEFT - 1)), shape=(30, 30)
    )
    # ==========================================================================================================
    # ? DOVREBBE ESSERE CORRETTO
    data_TOP_LEFT = np.array(
        [
            1 / sqrt(2),
            1 / 2,
            1 / 2,
            -1 / sqrt(2),
            -1 / sqrt(2),
            1 / 2,
            1 / 2,
            1 / sqrt(2),
            sqrt(3) / (2 * sqrt(2)),
            1 / (2 * sqrt(2)),
            1 / 2,
            1 / 2,
            -1 / sqrt(2),
            -sqrt(3) / (2 * sqrt(2)),
            -1 / (2 * sqrt(2)),
            1 / 2,
            1 / 2,
            1 / 2,
            1 / 2,
            1 / sqrt(2),
            1 / sqrt(2),
            1 / 2,
            1 / 2,
            -1 / sqrt(2),
            -1 / sqrt(2),
            1 / 2,
            1 / 2,
            1 / sqrt(2),
        ]
    )
    x_TOP_LEFT = np.array(
        [
            5,
            3,
            2,
            9,
            1,
            7,
            6,
            4,
            14,
            15,
            12,
            11,
            21,
            10,
            10,
            18,
            19,
            16,
            17,
            13,
            26,
            24,
            23,
            30,
            22,
            28,
            27,
            25,
        ]
    )
    y_TOP_LEFT = np.array(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            9,
            10,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            30,
        ]
    )
    C_TOP_LEFT = csr_matrix(
        (data_TOP_LEFT, (x_TOP_LEFT - 1, y_TOP_LEFT - 1)), shape=(30, 30)
    )
    # ==========================================================================================================
    # ? DOVREBBE ESSERE CORRETTO
    data_TOP_RIGHT = np.array(
        [
            1 / sqrt(2),
            -1 / sqrt(2),
            1 / 2,
            1 / 2,
            1 / 2,
            1 / 2,
            sqrt(3) / (2 * sqrt(2)),
            1 / (2 * sqrt(2)),
            -sqrt(3) / (2 * sqrt(2)),
            -1 / (2 * sqrt(2)),
            1 / 2,
            1 / 2,
            -1 / sqrt(2),
            -1 / sqrt(2),
            1 / sqrt(2),
            1 / sqrt(2),
            1 / 2,
            1 / 2,
            1 / 2,
            1 / 2,
            1 / sqrt(2),
            -1 / sqrt(2),
            1 / 2,
            1 / 2,
            1 / 2,
            1 / 2,
            sqrt(3) / (2 * sqrt(2)),
            1 / (2 * sqrt(2)),
            -sqrt(3) / (2 * sqrt(2)),
            -1 / (2 * sqrt(2)),
        ]
    )
    x_TOP_RIGHT = np.array(
        [
            2,
            1,
            5,
            6,
            3,
            4,
            8,
            9,
            7,
            7,
            11,
            10,
            15,
            17,
            12,
            13,
            20,
            21,
            18,
            19,
            23,
            22,
            26,
            27,
            24,
            25,
            29,
            30,
            28,
            28,
        ]
    )
    y_TOP_RIGHT = np.array(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            15,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            28,
            29,
            30,
        ]
    )
    C_TOP_RIGHT = csr_matrix(
        (data_TOP_RIGHT, (x_TOP_RIGHT - 1, y_TOP_RIGHT - 1)), shape=(30, 30)
    )
    # ==========================================================================================================
    # TODO Ci sono ancora dei dubbi sul segno di alcuni coefficienti scritti da pietro
    data_BOTTOM_RIGHT = np.array(
        [
            -1 / sqrt(2),
            1 / 2,
            1 / 2,
            1 / sqrt(2),
            1 / sqrt(2),
            1 / 2,
            1 / 2,
            -1 / sqrt(2),
            1 / 2,
            sqrt(3) / (2 * sqrt(2)),
            -1 / (2 * sqrt(2)),
            sqrt(3) / (2 * sqrt(2)),
            -1 / (2 * sqrt(2)),
            1 / 2,
            -1 / 4,
            -sqrt(3) / 4,
            sqrt(3) / 4,
            -1 / 4,
            -sqrt(3) / (2 * sqrt(2)),
            1 / (2 * sqrt(2)),
            -sqrt(3) / (2 * sqrt(2)),
            1 / (2 * sqrt(2)),
            -1 / 4,
            sqrt(3) / 4,
            -sqrt(3) / 4,
            -1 / 4,
            -1 / (2 * sqrt(2)),
            1 / 2,
            1 / 2,
            1 / (2 * sqrt(2)),
            1 / (2 * sqrt(2)),
            1 / 2,
            1 / 2,
            -1 / (2 * sqrt(2)),
        ]
    )
    x_BOTTOM_RIGHT = np.array(
        [
            4,
            6,
            7,
            1,
            9,
            2,
            3,
            5,
            13,
            16,
            17,
            18,
            19,
            10,
            20,
            21,
            20,
            21,
            11,
            11,
            12,
            12,
            14,
            15,
            14,
            15,
            25,
            27,
            28,
            22,
            30,
            23,
            24,
            26,
        ]
    )
    y_BOTTOM_RIGHT = np.array(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            9,
            10,
            11,
            11,
            12,
            12,
            13,
            14,
            14,
            15,
            15,
            16,
            17,
            18,
            19,
            20,
            20,
            21,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            30,
        ]
    )
    C_BOTTOM_RIGHT = csr_matrix(
        (data_BOTTOM_RIGHT, (x_BOTTOM_RIGHT - 1, y_BOTTOM_RIGHT - 1)), shape=(30, 30)
    )
    # ==========================================================================================================
    return C_BOTTOM_LEFT, C_BOTTOM_RIGHT, C_TOP_LEFT, C_TOP_RIGHT
