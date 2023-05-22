# %%
import numpy as np
from scipy.sparse import csr_matrix, diags, identity, block_diag, isspmatrix, kron
from scipy.sparse.linalg import norm
from scipy.linalg import null_space

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


def comm(A, B):
    # THIS FUNCTION COMPUTES THE COMMUTATOR of TWO SPARSE MATRICES
    if not isspmatrix(A):
        raise TypeError(f"A should be a csr_matrix, not a {type(A)}")
    if not isspmatrix(B):
        raise TypeError(f"B should be a csr_matrix, not a {type(B)}")
    return A * B - B * A


def anti_comm(A, B):
    # THIS FUNCTION COMPUTES THE ANTI_COMMUTATOR of TWO SPARSE MATRICES
    if not isspmatrix(A):
        raise TypeError(f"A should be a csr_matrix, not a {type(A)}")
    if not isspmatrix(B):
        raise TypeError(f"B should be a csr_matrix, not a {type(B)}")
    return A * B + B * A


def check_matrix(A, B):
    # CHEKS THE DIFFERENCE BETWEEN TWO SPARSE MATRICES
    if not isspmatrix(A):
        raise TypeError(f"A should be a csr_matrix, not a {type(A)}")
    if not isspmatrix(B):
        raise TypeError(f"B should be a csr_matrix, not a {type(B)}")
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch between : A {A.shape} & B: {B.shape}")
    norma = norm(A - B)
    norma_max = max(norm(A + B), norm(A), norm(B))
    ratio = norma / norma_max
    if ratio > 1e-15:
        print("    ERROR: A and B are DIFFERENT MATRICES")
        raise ValueError(f"    NORM {norma}, RATIO {ratio}")


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


def SU2_generators(s):
    if not np.isscalar(s):
        raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(s)}")
    """
    This function computes the SU(2) generators of the Rishon modes adopted
    for the SU2 Lattice Gauge Theory: [Tz, Tp=T+, Tm=T-, Tx, Ty, T2=Casimir]
    in any arbitrary spin-s representation

    Args:
        s (scalar, real): spin value, assumed to be integer or semi-integer

    Returns:
        dict: dictionary with the spin matrices
    """
    matrices = {"Tz": [0], "Tp": [0], "T2": [0]}
    largest_s_size = int(2 * s + 1)
    for s_size in range(1, largest_s_size):
        spin = s_size / 2
        spin_ops = get_spin_operators(spin)
        for op in ["z", "p", "2"]:
            matrices[f"T{op}"] += [spin_ops[f"S{op}"]]
    SU2_gen = {}
    for op in ["Tz", "Tp", "T2"]:
        SU2_gen[op] = block_diag(tuple(matrices[op]), format="csr")
    SU2_gen["Tm"] = SU2_gen["Tp"].transpose()
    SU2_gen["Tx"] = 0.5 * (SU2_gen["Tp"] + SU2_gen["Tm"])
    SU2_gen["Ty"] = complex(0, -0.5) * (SU2_gen["Tp"] - SU2_gen["Tm"])
    return SU2_gen


def chi_function(s, m):
    if not np.isscalar(s):
        raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(s)}")
    "This function computes the factor for SU2 rishon entries"
    return np.sqrt((s - m + 1) / (4 * np.ceil(s) + 2))


def SU2_rishon_operators(s):
    if not np.isscalar(s):
        raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(s)}")
    """
    This function computes the SU2 the Rishon modes adopted
    for the SU2 Lattice Gauge Theory for the chosen spin-s irrepresentation

    Args:
        s (scalar, real): spin value, assumed to be integer or semi-integer

    Returns:
        dict: dictionary with the rishon operators and the parity
    """
    # Compute the dimension of the rishon modes
    largest_s_size = int(2 * s + 1)
    zeta_size = np.sum([s_size for s_size in range(1, largest_s_size + 1)])
    zeta_shape = (zeta_size, zeta_size)
    # List of diagonal entries
    zeta_entries = []
    # List of diagonals
    zeta_diags = []
    # Starting diagonals of the s=0 case
    diag_p = 2
    diag_m = -1
    # Number of zeros at the beginning of the diagonals.
    # It increases with the spin representation
    in_zeros = 0
    for s_size in range(largest_s_size - 1):
        # Obtain spin
        spin = s_size / 2
        # Compute chi & P coefficientes
        sz_diag = np.arange(-spin, spin + 1)[::-1]
        chi_diag_p = (np.vectorize(chi_function)(spin, sz_diag)).tolist()
        chi_diag_m = (-np.vectorize(chi_function)(spin, -sz_diag)).tolist()
        # Fill the diags with zeros according to the lenght of the diag
        out_zeros_p = zeta_size - len(chi_diag_p) - diag_p - in_zeros
        out_zeros_m = zeta_size - len(chi_diag_p) + diag_m - in_zeros
        chi_diag_p = [0] * in_zeros + chi_diag_p + [0] * out_zeros_p
        chi_diag_m = [0] * in_zeros + chi_diag_m + [0] * out_zeros_m
        # Append the diags
        zeta_entries.append(chi_diag_p)
        zeta_diags.append(diag_p)
        zeta_entries.append(chi_diag_m)
        zeta_diags.append(diag_m)
        # Update the diagonals and the number of initial zeros
        diag_p += 1
        diag_m -= 1
        in_zeros += s_size + 1
    # Compose the Rishon operators
    ops = {}
    ops["z_down"] = diags(zeta_entries, zeta_diags, zeta_shape)
    ops["z_up"] = abs(ops["z_down"].transpose())
    # Define the Parity operator
    P_diag = []
    n_diag = []
    for s_size in range(largest_s_size):
        spin = s_size / 2
        P_diag += [((-1) ** s_size)] * (s_size + 1)
        n_diag += [spin * (spin + 1)] * (s_size + 1)
    ops["P_z"] = diags(P_diag, 0, zeta_shape)
    ops["n"] = diags(n_diag, 0, zeta_shape)
    ops["n_square"] = ops["n"] ** 2
    ops["ID_z"] = identity(zeta_size, dtype=float)
    # Useful operators for corner operators
    for sigma in ["up", "down"]:
        ops[f"z_{sigma}_dag"] = ops[f"z_{sigma}"].transpose()
        ops[f"z_{sigma}_x_P"] = ops[f"z_{sigma}"] * ops["P_z"]
        ops[f"P_x_z_{sigma}_dag"] = ops["P_z"] * ops[f"z_{sigma}_dag"]
    return ops


def check_SU2_rishon_algebra(s):
    # Checks that the SU(2) rishon modes satisfies the SU2 algebra
    if not np.isscalar(s):
        raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(s)}")
    # Define the SU2 generators
    T = SU2_generators(s)
    # Define the rishon operators
    z = SU2_rishon_operators(s)
    if check_matrix(2 * comm(z["z_up"], T["Tx"]), z["z_down"]):
        raise ValueError("ERROR")
    if check_matrix(2 * comm(z["z_down"], T["Tx"]), z["z_up"]):
        raise ValueError("ERROR")
    if check_matrix(2 * comm(z["z_up"], T["Ty"]), -complex(0, 1) * z["z_down"]):
        raise ValueError("ERROR")
    if check_matrix(2 * comm(z["z_down"], T["Ty"]), complex(0, 1) * z["z_up"]):
        raise ValueError("ERROR")
    if check_matrix(2 * comm(z["z_up"], T["Tz"]), z["z_up"]):
        raise ValueError("ERROR")
    if check_matrix(2 * comm(z["z_down"], T["Tz"]), -z["z_down"]):
        raise ValueError("ERROR")
    # CHECK RISHON MODES TO BEHAVE LIKE FERMIONS (anticommute with parity)
    for alpha in ["up", "down"]:
        if norm(anti_comm(z[f"z_{alpha}"], z["P_z"])) > 1e-15:
            raise ValueError(f"z_{alpha} is a Fermion and must anticommute with P")


# %%
"""
# TESTS
spin = 2
Z = SU2_rishon_operators(spin)
largest_s_size = int(2 * spin + 1)
zeta_size = np.sum([s_size for s_size in range(1, largest_s_size + 1)])
# Define the Link Casimir Operator
link_casimir = kron(Z["n"], identity(zeta_size)) - kron(identity(zeta_size), Z["n"])
kernel = csr_matrix(null_space(link_casimir.toarray()))
# Parrallel Transporter
U = {
    "up_up": kron((Z["z_up"] * Z["P_z"]), Z["z_up_dag"]),
    "up_down": kron((Z["z_up"] * Z["P_z"]), Z["z_down_dag"]),
    "down_up": kron((Z["z_down"] * Z["P_z"]), Z["z_up_dag"]),
    "down_down": kron((Z["z_down"] * Z["P_z"]), Z["z_down_dag"]),
}

M = kernel * kernel.transpose()
for op in U:
    norma = norm(U[op] * M - M * U[op])
    if norma > 1e-10:
        print(spin, op, norma)
"""


# %%
def inner_site_operators(s):
    if not np.isscalar(s):
        raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(s)}")
    # Define the SU2 Rishon operators
    ops = SU2_rishon_operators(s)
    ops |= SU2_generators(s)
    # --------------------------------------------------------------------------
    # Define the generic MATTER FIELD OPERATORS for both the su2 colors
    # The distinction between the two colors will be specified when considering
    # the dressed site operators.
    ops["psi"] = diags(np.array([1], dtype=float), offsets=1, shape=(2, 2))
    ops["psi_dag"] = ops["psi"].transpose()
    ops["P_psi"] = diags(np.array([1, -1], dtype=float), offsets=0, shape=(2, 2))
    ops["N"] = ops["psi_dag"] * ops["psi"]
    ops["ID"] = identity(2, dtype=float)
    # up & down MATTER OPERATORS
    ops["psi_up"] = qmb_op(ops, ["psi", "ID"])
    ops["psi_down"] = qmb_op(ops, ["P_psi", "psi"])
    ops["N_up"] = qmb_op(ops, ["N", "ID"])
    ops["N_down"] = qmb_op(ops, ["ID", "N"])
    # other number operators
    ops["N_pair"] = ops["N_up"] * ops["N_down"]
    ops["N_tot"] = ops["N_up"] + ops["N_down"]
    ops["N_single"] = ops["N_tot"] - ops["N_pair"]
    # identity on the whole matter site
    ops["ID_psi"] = identity(4, dtype=float)
    for sigma in ["up", "down"]:
        ops[f"psi_{sigma}_dag"] = ops[f"psi_{sigma}"].transpose()
    # Spin matrices for MATTER FIELD OPERATORS
    ops["Sx_psi"] = (
        ops["psi_up_dag"] * ops["psi_down"] - ops["psi_up"] * ops["psi_down_dag"]
    )
    ops["Sy_psi"] = -1j * (
        ops["psi_up_dag"] * ops["psi_down"] + ops["psi_up"] * ops["psi_down_dag"]
    )
    ops["Sz_psi"] = ops["N_up"] - ops["N_down"]
    return ops


# %%
def dressed_site_operators(s, pure=False):
    if not np.isscalar(s):
        raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(s)}")
    # Get inner site operators
    in_ops = inner_site_operators(s)
    # Dictionary for dressed site operators
    ops = {}
    # Rishon NUMBER OPERATORS for the Electric Field
    for op in ["n", "n_square"]:
        ops[f"{op}_mx"] = qmb_op(in_ops, [op, "ID_z", "ID_z", "ID_z"])
        ops[f"{op}_my"] = qmb_op(in_ops, ["ID_z", op, "ID_z", "ID_z"])
        ops[f"{op}_px"] = qmb_op(in_ops, ["ID_z", "ID_z", op, "ID_z"])
        ops[f"{op}_py"] = qmb_op(in_ops, ["ID_z", "ID_z", "ID_z", op])
    # CORNER OPERATORS
    for corner in ["px,py", "py,mx", "mx,my", "my,px"]:
        ops[f"C_{corner}"] = 0
    for sigma in ["up", "down"]:
        ops["C_px,py"] += -qmb_op(
            in_ops, ["ID_z", "ID_z", f"z_{sigma}_x_P", f"z_{sigma}_dag"]
        )
        ops["C_py,mx"] += qmb_op(
            in_ops, [f"P_x_z_{sigma}_dag", "P_z", "P_z", f"z_{sigma}"]
        )
        ops["C_mx,my"] += qmb_op(
            in_ops, [f"z_{sigma}_x_P", f"z_{sigma}_dag", "ID_z", "ID_z"]
        )
        ops["C_my,px"] += qmb_op(
            in_ops, ["ID_z", f"z_{sigma}_x_P", f"z_{sigma}_dag", "ID_z"]
        )
    if pure:
        # DRESSED SITE CASIMIR OPERATOR S^{2}
        ops[f"S2_tot"] = 0
        for d in ["x", "y", "z"]:
            ops[f"S{d}_tot"] = qmb_op(in_ops, [f"T{d}", "ID_z", "ID_z", "ID_z"])
            ops[f"S{d}_tot"] += qmb_op(in_ops, ["ID_z", f"T{d}", "ID_z", "ID_z"])
            ops[f"S{d}_tot"] += qmb_op(in_ops, ["ID_z", "ID_z", f"T{d}", "ID_z"])
            ops[f"S{d}_tot"] += qmb_op(in_ops, ["ID_z", "ID_z", "ID_z", f"T{d}"])
            ops["S2_tot"] += ops[f"S{d}_tot"] ** 2
    else:
        # DRESSED SITE CASIMIR OPERATOR S^{2}
        ops[f"S2_tot"] = 0
        for d in ["x", "y", "z"]:
            ops[f"S{d}_tot"] = qmb_op(
                in_ops, [f"S{d}_psi", "ID_z", "ID_z", "ID_z", "ID_z"]
            )
            ops[f"S{d}_tot"] += qmb_op(
                in_ops, ["ID_psi", f"T{d}", "ID_z", "ID_z", "ID_z"]
            )
            ops[f"S{d}_tot"] += qmb_op(
                in_ops, ["ID_psi", "ID_z", f"T{d}", "ID_z", "ID_z"]
            )
            ops[f"S{d}_tot"] += qmb_op(
                in_ops, ["ID_psi", "ID_z", "ID_z", f"T{d}", "ID_z"]
            )
            ops[f"S{d}_tot"] += qmb_op(
                in_ops, ["ID_psi", "ID_z", "ID_z", "ID_z", f"T{d}"]
            )
            ops["S2_tot"] += ops[f"S{d}_tot"] ** 2
        # ------------------------------------------------------------------------------
        # HOPPING OPERATORS
        for sd in ["mx", "my", "px", "py"]:
            ops[f"Q_{sd}_dag"] = 0
        for sigma in ["up", "down"]:
            ops["Q_mx_dag"] += qmb_op(
                in_ops, [f"psi_{sigma}_dag", f"z_{sigma}", "ID_z", "ID_z", "ID_z"]
            )
            ops["Q_my_dag"] += qmb_op(
                in_ops, [f"psi_{sigma}_dag", "P_z", f"z_{sigma}", "ID_z", "ID_z"]
            )
            ops["Q_px_dag"] += qmb_op(
                in_ops, [f"psi_{sigma}_dag", "P_z", "P_z", f"z_{sigma}", "ID_z"]
            )
            ops["Q_py_dag"] += qmb_op(
                in_ops, [f"psi_{sigma}_dag", "P_z", "P_z", "P_z", f"z_{sigma}"]
            )
        # add DAGGER operators
        Qs = {}
        for op in ops:
            dag_op = op.replace("_dag", "")
            Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
        ops |= Qs
        # Psi NUMBER OPERATORS
        for label in ["up", "down", "tot", "single", "pair"]:
            ops[f"N_{label}"] = qmb_op(
                in_ops, [f"N_{label}", "ID_z", "ID_z", "ID_z", "ID_z"]
            )
        # Rishon NUMBER OPERATORS in case of the Full Theory
        for op in ["n", "n_square"]:
            for sd in ["mx", "my", "px", "py"]:
                ops[f"{op}_{sd}"] = kron(in_ops["ID_psi"], ops[f"{op}_{sd}"])
        # CORNER OPERATORS in case of the Full Theory
        for corner in ["px,py", "py,mx", "mx,my", "my,px"]:
            ops[f"C_{corner}"] = kron(in_ops["ID_psi"], ops[f"C_{corner}"])
    # CASIMIR/ELECTRIC OPERATOR
    ops[f"E_square"] = 0
    for sd in ["mx", "my", "px", "py"]:
        ops[f"E_square"] += 0.5 * ops[f"n_square_{sd}"]
    return ops
