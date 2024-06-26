import numpy as np
from scipy.sparse import csr_matrix


# ========================================================================================
def W_operators():
    data_R = np.array(
        [
            1,
            1,
            -1,
            1,
            -1,
            1,
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
            1,
            1,
            -1,
            1,
            -1,
            1,
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
        ]
    )
    x_R = np.arange(1, 33, 1)
    W_right = csr_matrix((data_R, (x_R - 1, x_R - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    data_L = np.array(
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
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
        ]
    )
    x_L = np.arange(1, 33, 1)
    W_left = csr_matrix((data_L, (x_L - 1, x_L - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    data_T = np.array(
        [
            1,
            1,
            1,
            -1,
            1,
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
            1,
            1,
            1,
            -1,
            1,
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
        ]
    )
    x_T = np.arange(1, 33, 1)
    W_top = csr_matrix((data_T, (x_T - 1, x_T - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    data_B = np.array(
        [
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
        ]
    )
    x_B = np.arange(1, 33, 1)
    W_bottom = csr_matrix((data_B, (x_B - 1, x_B - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    return W_right, W_left, W_top, W_bottom


# ========================================================================================
def Plaquette():
    # -----------------------------------------------------------------
    data_RT = np.array(
        [
            1,
            1,
            -1,
            1,
            -1,
            1,
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
            1,
            1,
            -1,
            1,
            -1,
            1,
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
        ]
    )
    x_RT = np.array(
        [
            7,
            8,
            4,
            3,
            6,
            5,
            1,
            2,
            15,
            16,
            12,
            11,
            14,
            13,
            9,
            10,
            23,
            24,
            20,
            19,
            22,
            21,
            17,
            18,
            31,
            32,
            28,
            27,
            30,
            29,
            25,
            26,
        ]
    )
    y_RT = np.arange(1, 33, 1)
    C_RT = csr_matrix((data_RT, (x_RT - 1, y_RT - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    data_LT = np.array(
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
            -1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
        ]
    )
    x_LT = np.array(
        [
            4,
            6,
            7,
            1,
            8,
            2,
            3,
            5,
            12,
            14,
            15,
            9,
            16,
            10,
            11,
            13,
            20,
            22,
            23,
            17,
            24,
            18,
            19,
            21,
            28,
            30,
            31,
            25,
            32,
            26,
            27,
            29,
        ]
    )
    y_LT = np.arange(1, 33, 1)
    C_LT = csr_matrix((data_LT, (x_LT - 1, y_LT - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    data_BR = np.array(
        [
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
        ]
    )
    x_BR = np.array(
        [
            5,
            3,
            2,
            8,
            1,
            7,
            6,
            4,
            13,
            11,
            10,
            16,
            9,
            15,
            14,
            12,
            21,
            19,
            18,
            24,
            17,
            23,
            22,
            20,
            29,
            27,
            26,
            32,
            25,
            31,
            30,
            28,
        ]
    )
    y_BR = np.arange(1, 33, 1)
    C_BR = csr_matrix((data_BR, (x_BR - 1, y_BR - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    data_LB = np.array(
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
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
        ]
    )
    x_LB = np.array(
        [
            2,
            1,
            5,
            6,
            3,
            4,
            8,
            7,
            10,
            9,
            13,
            14,
            11,
            12,
            16,
            15,
            18,
            17,
            21,
            22,
            19,
            20,
            24,
            23,
            26,
            25,
            29,
            30,
            27,
            28,
            32,
            31,
        ]
    )
    y_LB = np.arange(1, 33, 1)
    C_LB = csr_matrix((data_LB, (x_LB - 1, y_LB - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    return C_RT, C_LT, C_BR, C_LB


# ========================================================================================
def Border_penalties():
    data_R = np.ones(16)
    x_R = np.array([1, 2, 4, 6, 9, 10, 12, 14, 17, 18, 20, 22, 25, 26, 28, 30])
    P_RIGHT = csr_matrix((data_R, (x_R - 1, x_R - 1)), shape=(32, 32))
    # ===============================================================
    data_L = np.ones(16)
    x_L = np.array([1, 5, 6, 7, 10, 11, 12, 16, 18, 19, 20, 24, 25, 29, 30, 31])
    P_LEFT = csr_matrix((data_L, (x_L - 1, x_L - 1)), shape=(32, 32))
    # ===============================================================
    data_T = np.ones(16)
    x_T = np.array([1, 2, 3, 5, 9, 10, 11, 13, 17, 18, 19, 21, 25, 26, 27, 29])
    P_TOP = csr_matrix((data_T, (x_T - 1, x_T - 1)), shape=(32, 32))
    # ===============================================================
    data_B = np.ones(16)
    x_B = np.array([1, 3, 4, 7, 9, 11, 12, 15, 17, 19, 20, 23, 25, 27, 28, 31])
    P_BOTTOM = csr_matrix((data_B, (x_B - 1, x_B - 1)), shape=(32, 32))
    # ===============================================================
    return P_RIGHT, P_LEFT, P_TOP, P_BOTTOM


# ========================================================================================
def Hopping_up():
    data_R = np.array([1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1])
    x_R = np.array([3, 5, 1, 7, 2, 8, 4, 6, 19, 21, 17, 23, 18, 24, 20, 22])
    y_R = np.array([9, 10, 11, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32])
    Q_up_R = csr_matrix((data_R, (x_R - 1, y_R - 1)), shape=(32, 32))
    Q_up_R_dag = csr_matrix((data_R, (y_R - 1, x_R - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    data_L = np.array([-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1])
    x_L = np.array([1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24])
    y_L = np.array([9, 10, 11, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32])
    Q_up_L = csr_matrix((data_L, (x_L - 1, y_L - 1)), shape=(32, 32))
    Q_up_L_dag = csr_matrix((data_L, (y_L - 1, x_L - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    data_T = np.array([1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1])
    x_T = np.array([4, 6, 7, 1, 8, 2, 3, 5, 20, 22, 23, 17, 24, 18, 19, 21])
    y_T = np.array([9, 10, 11, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32])
    Q_up_T = csr_matrix((data_T, (x_T - 1, y_T - 1)), shape=(32, 32))
    Q_up_T_dag = csr_matrix((data_T, (y_T - 1, x_T - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    data_B = np.array([1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1])
    x_B = np.array([2, 1, 5, 6, 3, 4, 8, 7, 18, 17, 21, 22, 19, 20, 24, 23])
    y_B = np.array([9, 10, 11, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32])
    Q_up_B = csr_matrix((data_B, (x_B - 1, y_B - 1)), shape=(32, 32))
    Q_up_B_dag = csr_matrix((data_B, (y_B - 1, x_B - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    return (
        Q_up_R,
        Q_up_R_dag,
        Q_up_L,
        Q_up_L_dag,
        Q_up_T,
        Q_up_T_dag,
        Q_up_B,
        Q_up_B_dag,
    )


# ========================================================================================
def Hopping_down():
    data_R = np.array([1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1])
    x_R = np.array([3, 5, 1, 7, 2, 8, 4, 6, 11, 13, 9, 15, 10, 16, 12, 14])
    y_R = np.arange(17, 33, 1)
    Q_down_R = csr_matrix((data_R, (x_R - 1, y_R - 1)), shape=(32, 32))
    Q_down_R_dag = csr_matrix((data_R, (y_R - 1, x_R - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    data_L = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    x_L = np.arange(1, 17, 1)
    y_L = np.arange(17, 33, 1)
    Q_down_L = csr_matrix((data_L, (x_L - 1, y_L - 1)), shape=(32, 32))
    Q_down_L_dag = csr_matrix((data_L, (y_L - 1, x_L - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    data_T = np.array([1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1])
    x_T = np.array([4, 6, 7, 1, 8, 2, 3, 5, 12, 14, 15, 9, 16, 10, 11, 13])
    y_T = np.arange(17, 33, 1)
    Q_down_T = csr_matrix((data_T, (x_T - 1, y_T - 1)), shape=(32, 32))
    Q_down_T_dag = csr_matrix((data_T, (y_T - 1, x_T - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    data_B = np.array([1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1])
    x_B = np.array([2, 1, 5, 6, 3, 4, 8, 7, 10, 9, 13, 14, 11, 12, 16, 15])
    y_B = np.arange(17, 33, 1)
    Q_down_B = csr_matrix((data_B, (x_B - 1, y_B - 1)), shape=(32, 32))
    Q_down_B_dag = csr_matrix((data_B, (y_B - 1, x_B - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    return (
        Q_down_R,
        Q_down_R_dag,
        Q_down_L,
        Q_down_L_dag,
        Q_down_T,
        Q_down_T_dag,
        Q_down_B,
        Q_down_B_dag,
    )


# ========================================================================================
def Number_operators():
    data_UP = np.ones(16)
    x_UP = np.array([9, 10, 11, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32])
    n_UP = csr_matrix((data_UP, (x_UP - 1, x_UP - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    data_DOWN = np.ones(16)
    x_DOWN = np.arange(17, 33, 1)
    n_DOWN = csr_matrix((data_DOWN, (x_DOWN - 1, x_DOWN - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    data_PAIR = np.ones(8)
    x_PAIR = np.arange(25, 33, 1)
    n_PAIR = csr_matrix((data_PAIR, (x_PAIR - 1, x_PAIR - 1)), shape=(32, 32))
    # -----------------------------------------------------------------
    return n_UP, n_DOWN, n_PAIR


from scipy.sparse import diags, identity, kron


def qmb_op(ops, op_list, add_dagger=False, get_real=False, get_imag=False):
    """
    This function performs the QMB operation of an arbitrary long list
    of operators of arbitrary dimensions.

    Args:
        ops (dict): dictionary storing all the single site operators

        op_list (list): list of the names of the operators involved in the qmb operator
        the list is assumed to be stored according to the zig-zag order on the lattice

        strength (scalar): real/complex coefficient applied in front of the operator

        add_dagger (bool, optional): if true, yields the hermitian conjugate. Defaults to False.

        get_real (bool, optional):  if true, yields only the real part. Defaults to False.

        get_imag (bool, optional): if true, yields only the imaginary part. Defaults to False.
    Returns:
        csr_matrix: QMB sparse operator
    """
    # CHECK ON TYPES
    if not isinstance(ops, dict):
        raise TypeError(f"ops must be a DICT, not a {type(ops)}")
    if not isinstance(op_list, list):
        raise TypeError(f"op_list must be a LIST, not a {type(op_list)}")
    if not isinstance(add_dagger, bool):
        raise TypeError(f"add_dagger should be a BOOL, not a {type(add_dagger)}")
    if not isinstance(get_real, bool):
        raise TypeError(f"get_real should be a BOOL, not a {type(get_real)}")
    if not isinstance(get_imag, bool):
        raise TypeError(f"get_imag should be a BOOL, not a {type(get_imag)}")
    tmp = ops[op_list[0]]
    for op in op_list[1:]:
        tmp = kron(tmp, ops[op])
    if add_dagger:
        tmp = csr_matrix(tmp + tmp.conj().transpose())
    if get_real:
        tmp = csr_matrix(tmp + tmp.conj().transpose()) / 2
    elif get_imag:
        tmp = complex(0.0, -0.5) * (csr_matrix(tmp - tmp.conj().transpose()))
    return tmp


def inner_site_operators():
    ops = {}
    """ 
    --------------------------------------------------------------------------
    Define the generic MATTER FIELD OPERATORS with spin 1/2
    The distinction between the two spin will be specified when considering
    the dressed site operators. 
    --------------------------------------------------------------------------
    """
    ops["psi"] = diags(np.array([1], dtype=float), offsets=1, shape=(2, 2))
    ops["psi_dag"] = ops["psi"].transpose()
    ops["P"] = diags(np.array([1, -1], dtype=float), offsets=0, shape=(2, 2))
    ops["N"] = ops["psi_dag"] * ops["psi"]
    ops["ID"] = identity(2, dtype=float)
    # up & down MATTER OPERATORS
    ops["psi_up"] = qmb_op(ops, ["psi", "ID"])
    ops["psi_down"] = qmb_op(ops, ["P", "psi"])
    ops["N_up"] = qmb_op(ops, ["N", "ID"])
    ops["N_down"] = qmb_op(ops, ["ID", "N"])
    # other number operators
    ops["N_pair"] = ops["N_up"] * ops["N_down"]
    ops["N_tot"] = ops["N_up"] + ops["N_down"]
    ops["N_single"] = ops["N_tot"] - ops["N_pair"]
    # Identity operator for matter site
    ops["ID_m"] = identity(4, dtype=float)
    # Majorana Operator
    ops["zeta"] = ops["psi"] + ops["psi_dag"]
    ops["zeta_x_P"] = ops["zeta"] * ops["P"]
    ops["P_x_zeta"] = ops["P"] * ops["zeta"]
    return ops


def Def_Hubbard_dressed_site_operators():
    # GET INNER SITE OPERATORS
    in_ops = inner_site_operators()
    # ------------------------------------------------------------------------------
    # DICTIONARY FOR DRESSED SITE OPERATORS
    ops = {}
    # MATTER NUMBER OPERATORS
    for op in ["N_up", "N_down", "N_pair", "N_tot", "N_single"]:
        ops[op] = qmb_op(in_ops, [op, "ID", "ID", "ID", "ID"])
    # HOPPING OPERATORS
    for sd in ["mx", "my", "px", "py"]:
        ops[f"Q_{sd}_dag"] = 0
    for s in ["up", "down"]:
        ops[f"Q_{s}_mx_dag"] += qmb_op(
            in_ops, [f"psi_{s}_dag", "zeta", "ID", "ID", "ID"]
        )
        ops[f"Q_{s}_my_dag"] += qmb_op(
            in_ops, [f"psi_{s}_dag", "P", "zeta", "ID", "ID"]
        )
        ops[f"Q_{s}_px_dag"] += qmb_op(in_ops, [f"psi_{s}_dag", "P", "P", "zeta", "ID"])
        ops[f"Q_{s}_py_dag"] += qmb_op(in_ops, [f"psi_{s}_dag", "P", "P", "P", "zeta"])
    # add DAGGER operators
    Qs = {}
    for op in ops:
        dag_op = op.replace("_dag", "")
        Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
    ops |= Qs
    # CORNER OPERATORS
    ops["C_px,py"] += -qmb_op(in_ops, ["ID_m", "ID", "ID", "zeta_x_P", "zeta"])
    ops["C_py,mx"] += qmb_op(in_ops, ["ID_m", "P_x_zeta", "P_z", "P_z", "zeta"])
    ops["C_mx,my"] += qmb_op(in_ops, ["ID_m", "zeta_x_P", "zeta", "ID", "ID"])
    ops["C_my,px"] += qmb_op(in_ops, ["ID_m", "ID", "zeta_x_P", f"zeta", "ID"])
    return ops
