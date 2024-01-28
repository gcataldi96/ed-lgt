import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse import identity
from ed_lgt.modeling import qmb_operator as qmb_op

__all__ = [
    "fermi_operators",
    "bose_operators",
]


def fermi_operators(has_spin, colors=False):
    """
    This functions define the matter field operators of the QED Hamiltonian.
    They are related to spinless Dirac Fermions occupying lattice sites

    Returns:
        dict: dictionary with single site matter field operators
    """
    ops = {}
    # Define the MATTER FIELDS OPERATORS
    ops["psi"] = diags(np.array([1], dtype=float), 1, (2, 2))
    ops["psi_dag"] = ops["psi"].transpose()
    ops["P_psi"] = diags(np.array([1, -1], dtype=float), 0, (2, 2))
    ops["N"] = ops["psi_dag"] * ops["psi"]
    ops["ID_psi"] = identity(2, dtype=float)
    ops["ID"] = identity(2, dtype=float)
    if has_spin:
        # up & down MATTER OPERATORS
        ops["psi_up"] = qmb_op(ops, ["psi", "ID"])
        ops["psi_down"] = qmb_op(ops, ["P_psi", "psi"])
        ops["N_up"] = qmb_op(ops, ["N", "ID"])
        ops["N_down"] = qmb_op(ops, ["ID", "N"])
        # other number operators
        ops["N_pair"] = ops["N_up"] * ops["N_down"]
        ops["N_tot"] = ops["N_up"] + ops["N_down"]
        ops["N_single"] = ops["N_tot"] - 2 * ops["N_pair"]
        # identity on the whole matter site
        ops["ID_psi"] = identity(4, dtype=float)
        for s in ["up", "down"]:
            ops[f"psi_{s}_dag"] = ops[f"psi_{s}"].transpose()
        if colors:
            ops["psi_r"] = ops["psi_up"]
            ops["psi_g"] = ops["psi_down"]
            ops["N_r"] = ops["N_up"]
            ops["N_g"] = ops["N_down"]
            for s in ["r", "g"]:
                ops[f"psi_{s}_dag"] = ops[f"psi_{s}"].transpose()
    return ops


def bose_operators(n_max):
    ops = {}
    entries = np.arange(1, n_max + 1, 1)
    entries = np.sqrt(entries)
    x_coords = np.arange(0, n_max, 1)
    y_coords = np.arange(1, n_max + 1, 1)
    ops["b"] = csr_matrix((entries, (x_coords, y_coords)), shape=(n_max + 1, n_max + 1))
    ops["b_dagger"] = csr_matrix(ops["b"].conj().transpose())
    ops["N"] = ops["b_dagger"] * ops["b"]
    ops["N2"] = ops["N"] ** 2
    return ops
