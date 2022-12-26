import numpy as np
from scipy.sparse import csr_matrix, identity
from tools.manage_data import acquire_data

__all__ = ["get_su2_operators", "get_Hamiltonian_couplings", "get_SU2_surface_operator"]


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
