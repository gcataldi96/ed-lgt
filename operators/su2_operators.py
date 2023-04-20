import numpy as np
from tools.manage_data import acquire_data
from scipy.sparse import csr_matrix, diags, identity
from ..modeling.qmb_operations_v2 import qmb_operator as qmb_op

__all__ = [
    "get_su2_operators",
    "get_SU2_Hamiltonian_couplings",
    "get_SU2_surface_operator",
]


def inner_site_operators(su2_irrep):
    # Define the Rishon operators according to the chosen su2 spin irrep
    ops = {}
    if su2_irrep == "1/2":
        xi_dim = 3
        p_diag = np.array([1, -1, -1], dtype=float)
        n_diag = np.array([0, 0.5, 0.5], dtype=float)
        xi = {
            "up": {
                "data": np.array([1, 1], dtype=float) / np.sqrt(2),
                "x": np.array([0, 2], dtype=int),
                "y": np.array([1, 0], dtype=int),
            },
            "down": {
                "data": np.array([1, -1], dtype=float) / np.sqrt(2),
                "x": np.array([0, 1], dtype=int),
                "y": np.array([2, 0], dtype=int),
            },
        }
    elif su2_irrep == "1":
        xi_dim = 6
        p_diag = np.array([1, -1, -1, 1, 1, 1], dtype=float)
        n_diag = np.array([0, 0.5, 0.5, 1, 1, 1], dtype=float)
        xi = {
            "up": {
                "data": np.array(
                    [np.sqrt(3), np.sqrt(2), np.sqrt(3), 1, 1, np.sqrt(2)], dtype=float
                )
                / np.sqrt(6),
                "x": np.array([0, 1, 2, 2, 4, 5], dtype=int),
                "y": np.array([1, 3, 0, 4, 1, 2], dtype=int),
            },
            "down": {
                "data": np.array(
                    [np.sqrt(3), -np.sqrt(3), 1, np.sqrt(2), -np.sqrt(2), 1],
                    dtype=float,
                )
                / np.sqrt(6),
                "x": np.array([0, 1, 1, 2, 3, 4], dtype=int),
                "y": np.array([2, 0, 4, 5, 1, 2], dtype=int),
            },
        }
    else:
        raise Exception("irrep not yet implemented")
    # --------------------------------------------------------------------------
    # Define the RISHON OPERATORS xi according to the chosen spin irrep
    for sigma in ["up", "down"]:
        ops[f"xi_{sigma}"] = csr_matrix(
            (xi[sigma]["data"], (xi[sigma]["x"], xi[sigma]["y"])),
            shape=(xi_dim, xi_dim),
        )
    ops["P_xi"] = diags(p_diag, offsets=0, shape=(xi_dim, xi_dim))
    ops["n_xi"] = diags(n_diag, offsets=0, shape=(xi_dim, xi_dim))
    ops["n_xi_square"] = ops["n_xi"] ** 2
    ops["ID_xi"] = identity(xi_dim, dtype=float)
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


def dressed_site_operators(su2_irrep, lattice_dim=2):
    # Get inner site operators
    in_ops = inner_site_operators(su2_irrep)
    # Define the TOTAL DIMENSION of dressed site operators
    xi_dim = in_ops["xi_up"].shape[0]
    tot_dim = 4 * (xi_dim) ** (2 * lattice_dim)
    # Dictionary for dressed site operators
    ops = {}
    # HOPPING OPERATORS
    for sd in ["mx", "my", "px", "py"]:
        ops[f"Q_{sd}_dag"] = 0
    for sigma in ["up", "down"]:
        ops["Q_mx_dag"] += qmb_op(
            in_ops, [f"psi_{sigma}_dag", f"xi_{sigma}", "ID_xi", "ID_xi", "ID_xi"]
        )
        ops["Q_my_dag"] += qmb_op(
            in_ops, [f"psi_{sigma}_dag", "P_xi", f"xi_{sigma}", "ID_xi", "ID_xi"]
        )
        ops["Q_px_dag"] += qmb_op(
            in_ops, [f"psi_{sigma}_dag", "P_xi", "P_xi", f"xi_{sigma}", "ID_xi"]
        )
        ops["Q_py_dag"] += qmb_op(
            in_ops, [f"psi_{sigma}_dag", "P_xi", "P_xi", "P_xi", f"xi_{sigma}"]
        )
    # add DAGGER operators
    Qs = {}
    for op in ops:
        dag_op = op.replace("_dag", "")
        Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
    ops |= Qs
    # Psi NUMBER OPERATORS
    for sigma in ["up", "down", "tot", "single", "pair"]:
        ops[f"N_{sigma}"] = qmb_op(
            in_ops, [f"N_{sigma}", "ID_xi", "ID_xi", "ID_xi", "ID_xi"]
        )
    # Rishon NUMBER OPERATORS
    for op in ["n", "n_square"]:
        ops[f"{op}_mx"] = qmb_op(in_ops, ["ID_psi", op, "ID_xi", "ID_xi", "ID_xi"])
        ops[f"{op}_my"] = qmb_op(in_ops, ["ID_psi", "ID_xi", op, "ID_xi", "ID_xi"])
        ops[f"{op}_px"] = qmb_op(in_ops, ["ID_psi", "ID_xi", "ID_xi", op, "ID_xi"])
        ops[f"{op}_py"] = qmb_op(in_ops, ["ID_psi", "ID_xi", "ID_xi", "ID_xi", op])
    # CASIMIR/ELECTRIC OPERATOR
    ops[f"E_square"] = 0
    for sd in ["mx", "my", "px", "py"]:
        ops[f"E_square"] += 0.5 * ops[f"n_square_{sd}"]
    # CORNER OPERATORS
    for corner in ["px,py", "py,mx", "mx,my", "my,px"]:
        ops[f"C_{corner}"] = 0
    for sigma in ["up", "down"]:
        ops["C_px,py"] += -qmb_op(
            in_ops, ["ID_psi", "ID_xi", "ID_xi", f"xi_{sigma}", f"xi_{sigma}_dag"]
        )
        ops["C_py,mx"] += qmb_op(
            in_ops, ["ID_psi", f"xi_{sigma}", "P_xi", "P_xi", f"xi_{sigma}"]
        )
        ops["C_mx,my"] += qmb_op(
            in_ops, ["ID_psi", f"xi_{sigma}", f"xi_{sigma}", "ID_xi", "ID_xi"]
        )
        ops["C_my,px"] += qmb_op(
            in_ops, ["ID_psi", "ID_xi", f"xi_{sigma}", f"xi_{sigma}", "ID_xi"]
        )
    return ops


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


# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================


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


def get_su2_operators(pure_theory):
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
        coeffs["eta"] = -20 * max(E, np.abs(B))
    if not pure_theory:
        coeffs["eta"] = -20 * max(E, np.abs(B), m)
        coeffs |= {
            "m": m,
            "tx": -0.5j,  # HORIZONTAL HOPPING
            "tx_dag": 0.5j,  # HORIZONTAL HOPPING DAGGER
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
