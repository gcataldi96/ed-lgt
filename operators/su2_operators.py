import numpy as np
from scipy.sparse import csr_matrix, identity
from tools.manage_data import acquire_data

__all__ = ["get_su2_operators", "get_Hamiltonian_couplings"]


def ID(pure_theory):
    ops = {}
    if pure_theory:
        hilb_dim = 9
    else:
        hilb_dim = 30
    ops["ID"] = csr_matrix(identity(hilb_dim))
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
        data = acquire_data(path + f"Corner_{corner}.txt")
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
    data = np.ones(9)
    x = np.arange(1, 10)
    y = np.arange(22, 31, 1)
    pair = csr_matrix((data, (x - 1, y - 1)), shape=(30, 30))
    Dag_pair = csr_matrix((data, (y - 1, x - 1)), shape=(30, 30))
    return pair, Dag_pair


def get_su2_operators(pure_theory):
    ops = {}
    ops |= ID(pure_theory)
    ops |= gamma_operator(pure_theory)
    ops |= plaquette(pure_theory)
    ops |= W_operators(pure_theory)
    ops |= penalties(pure_theory)
    if not pure_theory:
        ops |= hopping()
        ops |= matter_operator()
        ops |= number_operators()
    return ops


def get_Hamiltonian_couplings(pure_theory, g, m=None):
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
    print(f"PENALTY {coeffs['eta']}")
    return coeffs
