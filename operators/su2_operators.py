import numpy as np
from tools import (
    acquire_data,
    check_matrix,
    anti_commutator as anti_comm,
    commutator as comm,
)
from scipy.sparse import csr_matrix, diags, identity, block_diag
from scipy.sparse.linalg import norm
from modeling import qmb_operator as qmb_op
from .spin_operators import get_spin_operators

__all__ = [
    "get_SU2_operators",
    "get_SU2_Hamiltonian_couplings",
    "get_SU2_surface_operator",
]


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
        ops[f"z_{sigma}_times_P"] = ops[f"z_{sigma}"] * ops["P_z"]
        ops[f"P_times_z_{sigma}_dag"] = ops["P_z"] * ops[f"z_{sigma}_dag"]
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


def inner_site_operators(s):
    if not np.isscalar(s):
        raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(s)}")
    # Define the SU2 Rishon operators
    ops = SU2_rishon_operators(s)
    # --------------------------------------------------------------------------
    # Define the generic MATTER FIELD OPERATORS for both the su2 colors
    # The distinction between the two colors will be specified when considering
    # the dressed site operators.
    ops["psi"] = diags(np.array([1], dtype=float), offsets=1, shape=(2, 2))
    ops["psi_dag"] = ops["psi"]
    ops["P_psi"] = diags(np.array([1, -1], dtype=float), offsets=0, shape=(2, 2))
    ops["N"] = ops["psi_dag"] * ops["psi"]
    # up & down MATTER OPERATORS
    ops["psi_up"] = qmb_op(ops, ["psi", "ID_psi"])
    ops["psi_down"] = qmb_op(ops, ["P_psi", "psi"])
    ops["N_up"] = qmb_op(ops, ["N", "ID_psi"])
    ops["N_down"] = qmb_op(ops, ["ID_psi", "N"])
    # other number operators
    ops["N_pair"] = ops["N_up"] * ops["N_down"]
    ops["N_tot"] = ops["N_up"] + ops["N_down"]
    ops["N_single"] = ops["N_tot"] - ops["N_pair"]
    # identity on the whole matter site
    ops["ID_psi"] = identity(4, dtype=float)
    for sigma in ["up", "down"]:
        ops[f"psi_{sigma}_dag"] = ops[f"psi_{sigma}"].conj().transpose()
    return ops


def dressed_site_operators(s, lattice_dim=2, pure=False):
    if not np.isscalar(s):
        raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(s)}")
    # Get inner site operators
    in_ops = inner_site_operators(s)
    # Define the TOTAL DIMENSION of dressed site operators
    z_size = np.sum([s_size for s_size in range(1, 2 * s + 2)])
    tot_dim = 4 * (z_size) ** (2 * lattice_dim)
    # Dictionary for dressed site operators
    ops = {}
    if not pure:
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
    # Rishon NUMBER OPERATORS
    for op in ["n", "n_square"]:
        ops[f"{op}_mx"] = qmb_op(in_ops, ["ID_psi", op, "ID_z", "ID_z", "ID_z"])
        ops[f"{op}_my"] = qmb_op(in_ops, ["ID_psi", "ID_z", op, "ID_z", "ID_z"])
        ops[f"{op}_px"] = qmb_op(in_ops, ["ID_psi", "ID_z", "ID_z", op, "ID_z"])
        ops[f"{op}_py"] = qmb_op(in_ops, ["ID_psi", "ID_z", "ID_z", "ID_z", op])
    # CASIMIR/ELECTRIC OPERATOR
    ops[f"E_square"] = 0
    for sd in ["mx", "my", "px", "py"]:
        ops[f"E_square"] += 0.5 * ops[f"n_square_{sd}"]
    # CORNER OPERATORS
    for corner in ["px,py", "py,mx", "mx,my", "my,px"]:
        ops[f"C_{corner}"] = 0
    for sigma in ["up", "down"]:
        ops["C_px,py"] += -qmb_op(
            in_ops,
            ["ID_psi", "ID_z", "ID_z", f"z_{sigma}_times_P", f"z_{sigma}_dag"],
        )
        ops["C_py,mx"] += qmb_op(
            in_ops, ["ID_psi", f"P_times_z_{sigma}_dag", "P_z", "P_z", f"z_{sigma}"]
        )
        ops["C_mx,my"] += qmb_op(
            in_ops,
            ["ID_psi", f"z_{sigma}_times_P", f"z_{sigma}_dag", "ID_z", "ID_z"],
        )
        ops["C_my,px"] += qmb_op(
            in_ops,
            ["ID_psi", "ID_z", f"z_{sigma}_times_P", f"z_{sigma}_dag", "ID_z"],
        )
    return ops


# =====================================================================================
# =====================================================================================


def ID(pure_theory):
    ops = {}
    if pure_theory:
        hilb_dim = 9
    else:
        hilb_dim = 30
    ops["ID"] = csr_matrix(identity(hilb_dim))
    return ops


def link_parity(pure_theory):
    ops = {}
    if pure_theory:
        hilb_dim = 9
        path = "operators/su2_operators/pure_operators/"
    else:
        hilb_dim = 30
        path = "operators/su2_operators/full_operators/"
    for axis in ["x", "y"]:
        data = acquire_data(path + f"{axis}_link_parity.txt")
        x = data["0"]
        y = data["1"]
        coeff = data["2"]
        ops[f"p{axis}_link_P"] = csr_matrix(
            (coeff, (x - 1, y - 1)), shape=(hilb_dim, hilb_dim)
        )
    return ops


def gamma_operator(pure_theory):
    ops = {}
    if pure_theory:
        hilb_dim = 9
        path = "operators/su2_operators/pure_operators/"
    else:
        hilb_dim = 30
        path = "operators/su2_operators/full_operators/"
    data = acquire_data(path + f"Gamma.txt")
    x = data["0"]
    y = data["1"]
    coeff = data["2"]
    ops["gamma"] = csr_matrix((coeff, (x, y)), shape=(hilb_dim, hilb_dim))
    return ops


def plaquette(pure_theory):
    ops = {}
    if pure_theory:
        hilb_dim = 9
        path = "operators/su2_operators/pure_operators/"
    else:
        hilb_dim = 30
        path = "operators/su2_operators/full_operators/"

    for corner in ["py_px", "my_px", "py_mx", "my_mx"]:
        data = acquire_data(path + f"new_Corner_{corner}.txt")
        x = data["0"]
        y = data["1"]
        coeff = data["2"]
        ops[f"C_{corner}"] = csr_matrix(
            (coeff, (x - 1, y - 1)), shape=(hilb_dim, hilb_dim)
        )
        ops[f"C_{corner}_dag"] = csr_matrix(ops[f"C_{corner}"].conj().transpose())
    return ops


def W_operators(pure_theory):
    ops = {}
    if pure_theory:
        hilb_dim = 9
        path = "operators/su2_operators/pure_operators/"
    else:
        hilb_dim = 30
        path = "operators/su2_operators/full_operators/"

    for s in ["py", "px", "mx", "my"]:
        data = acquire_data(path + f"W_{s}.txt")
        x = data["0"]
        y = data["1"]
        coeff = data["2"]
        ops[f"W_{s}"] = csr_matrix((coeff, (x, y)), shape=(hilb_dim, hilb_dim))
    return ops


def penalties(pure_theory):
    ops = {}
    # BORDER OPERATORS
    if pure_theory:
        hilb_dim = 9
        data = np.ones(4, dtype=float)
        coords = {
            "mx": np.array([1, 5, 6, 7]),
            "px": np.array([1, 2, 4, 6]),
            "my": np.array([1, 3, 4, 7]),
            "py": np.array([1, 2, 3, 5]),
        }
    else:
        hilb_dim = 30
        data = np.ones(13, dtype=float)
        coords = {
            "mx": np.array([1, 5, 6, 7, 11, 12, 13, 20, 21, 22, 26, 27, 28]),
            "px": np.array([1, 2, 4, 6, 10, 11, 13, 16, 17, 22, 23, 25, 27]),
            "my": np.array([1, 3, 4, 7, 10, 12, 13, 18, 19, 22, 24, 25, 28]),
            "py": np.array([1, 2, 3, 5, 10, 11, 12, 14, 15, 22, 23, 24, 26]),
        }

    for dd in ["mx", "px", "my", "py"]:
        ops[f"P_{dd}"] = csr_matrix(
            (data, (coords[dd] - 1, coords[dd] - 1)), shape=(hilb_dim, hilb_dim)
        )

    # CORNER OPERATORS
    ops["P_mx_my"] = csr_matrix(ops["P_mx"] * ops["P_my"])
    ops["P_px_my"] = csr_matrix(ops["P_px"] * ops["P_my"])
    ops["P_mx_py"] = csr_matrix(ops["P_mx"] * ops["P_py"])
    ops["P_px_py"] = csr_matrix(ops["P_px"] * ops["P_py"])
    return ops


def hopping():
    ops = {}
    path = "operators/su2_operators/full_operators/"
    for side in ["py", "px", "mx", "my"]:
        data = acquire_data(path + f"Q_{side}_dag.txt")
        x = data["0"]
        y = data["1"]
        coeff = data["2"]
        ops[f"Q_{side}_dag"] = csr_matrix((coeff, (x - 1, y - 1)), shape=(30, 30))
        ops[f"Q_{side}"] = csr_matrix((coeff, (y - 1, x - 1)), shape=(30, 30))
    return ops


def matter_operator():
    ops = {}
    path = "operators/su2_operators/full_operators/"
    data = acquire_data(path + f"Mass_op.txt")
    x = data["0"]
    y = data["1"]
    coeff = data["2"]
    ops["mass_op"] = csr_matrix((coeff, (x, y)), shape=(30, 30))
    return ops


def number_operators():
    ops = {}
    data_pair = np.ones(9, dtype=float)
    x_pair = np.arange(22, 31, 1)
    ops["n_pair"] = csr_matrix((data_pair, (x_pair - 1, x_pair - 1)), shape=(30, 30))
    # ===========================================================================
    data_single = np.ones(12, dtype=float)
    x_single = np.arange(10, 22, 1)
    ops["n_single"] = csr_matrix(
        (data_single, (x_single - 1, x_single - 1)), shape=(30, 30)
    )
    # ===========================================================================
    data_tot = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=float
    )
    x_tot = np.arange(10, 31, 1)
    ops["n_tot"] = csr_matrix((data_tot, (x_tot - 1, x_tot - 1)), shape=(30, 30))
    # ===========================================================================
    return ops


def S_Wave_Correlation():
    ops = {}
    data = np.ones(9)
    x = np.arange(1, 10)
    y = np.arange(22, 31, 1)
    ops["Delta"] = csr_matrix((data, (x - 1, y - 1)), shape=(30, 30))
    ops["Delta_dag"] = csr_matrix(ops["Delta"].conj().transpose())
    return ops


def get_SU2_operators(pure_theory):
    ops = {}
    ops |= ID(pure_theory)
    ops |= gamma_operator(pure_theory)
    ops |= link_parity(pure_theory)
    ops |= plaquette(pure_theory)
    ops |= W_operators(pure_theory)
    ops |= penalties(pure_theory)
    if not pure_theory:
        ops |= hopping()
        ops |= matter_operator()
        ops |= number_operators()
        ops |= S_Wave_Correlation()
    return ops


def get_SU2_Hamiltonian_couplings(pure_theory, g, m=None):
    E = 3 * (g**2) / 16  # ELECTRIC FIELD
    B = -4 / (g**2)  # MAGNETIC FIELD
    # DICTIONARY WITH MODEL COEFFICIENTS
    coeffs = {
        "g": g,
        "E": E,  # ELECTRIC FIELD coupling
        "B": B,  # MAGNETIC FIELD coupling
    }
    if pure_theory:
        coeffs["eta"] = -10 * max(E, np.abs(B))
    if not pure_theory:
        coeffs["eta"] = -10 * max(E, np.abs(B), m)
        coeffs |= {
            "m": m,
            "tx_even": -0.5j,  # HORIZONTAL HOPPING (EVEN SITES)
            "tx_odd": -0.5j,  # HORIZONTAL HOPPING (ODD SITES)
            "ty_even": -0.5,  # VERTICAL HOPPING (EVEN SITES)
            "ty_odd": 0.5,  # VERTICAL HOPPING (ODD SITES)
            "m_odd": -m,  # EFFECTIVE MASS for ODD SITES
            "m_even": m,  # EFFECTIVE MASS for EVEN SITES
        }
    return coeffs


def get_SU2_surface_operator(pure_theory, operator, site):
    """
    This function computes the operator in the reduced basis that is appropriate
    for the bordering sites of the lattice
    Args:
        pure_theory (bool): if True, operators describe only SU2 pure theory (no matter). If False,
        operators includes both gauge and dynamical matter d.o.f

        operator (csr_matrix): bulk operator that we want to project in the basis referred to a surface site

        site (str): possible surface site of the lattice.
        It can be a CORNER ["py_px", "my_px", "py_mx", "my_mx"] or a BORDER ["mx", "px", "my", "py"]
    """
    if pure_theory:
        loc_dim = 9
        if site == "py_px":
            coords = np.array([1, 2])
        elif site == "my_px":
            coords = np.array([1, 4])
        elif site == "py_mx":
            coords = np.array([1, 5])
        elif site == "my_mx":
            coords = np.array([1, 6])
        elif site == "mx":
            coords = np.array([1, 5, 6, 7])
        elif site == "px":
            coords = np.array([1, 2, 4, 6])
        elif site == "my":
            coords = np.array([1, 3, 4, 7])
        elif site == "py":
            coords = np.array([1, 2, 3, 5])
    else:
        loc_dim = 30
        if site == "py_px":
            coords = np.array([1, 2, 10, 11, 22, 23])
        elif site == "my_px":
            coords = np.array([1, 4, 10, 13, 22, 25])
        elif site == "py_mx":
            coords = np.array([1, 5, 11, 12, 22, 26])
        elif site == "my_mx":
            coords = np.array([1, 7, 12, 13, 22, 28])
        elif site == "mx":
            coords = np.array([1, 5, 6, 7, 11, 12, 13, 20, 21, 22, 26, 27, 28])
        elif site == "px":
            coords = np.array([1, 2, 4, 6, 10, 11, 13, 16, 17, 22, 23, 25, 27])
        elif site == "my":
            coords = np.array([1, 3, 4, 7, 10, 12, 13, 18, 19, 22, 24, 25, 28])
        elif site == "py":
            coords = np.array([1, 2, 3, 5, 10, 11, 12, 14, 15, 22, 23, 24, 26])
    # CONSTRUCT THE PROJECTOR
    data = np.ones(coords.shape[0], dtype=float)
    P = csr_matrix(
        (data, (coords - 1, np.arange(coords.shape[0]))),
        shape=(loc_dim, coords.shape[0]),
    )
    P_dag = csr_matrix(P.conj().transpose())
    # return the operator projected on the effective basis
    return P_dag * operator * P
