# %%
import numpy as np
from itertools import product
from scipy.sparse import csr_matrix, diags, kron, identity

__all__ = [
    "gauge_invariant_states",
    "get_QED_operators",
    "get_QED_Hamiltonian_couplings",
]


def qmb_op(ops, list_ops):
    qmb_op = ops[list_ops[0]]
    for op in list_ops[1:]:
        qmb_op = kron(qmb_op, ops[op])
    return csr_matrix(qmb_op)


def gauge_invariant_states(n_rishons, lattice_dim=2):
    single_rishon_configs = np.arange(n_rishons + 1)
    gauge_states = {"odd": [], "even": []}
    gauge_basis = {"odd": [], "even": []}
    # Run over even and odd sites
    for ii, site in enumerate(gauge_states):
        # Define the parity of odd (-1) and even (+1) sites
        parity = -1 if site == "odd" else 1
        # Set raw and col counters
        raw_counter = 0
        col_counter = 0
        # Set a raw list for the basis
        raw = []
        # Possible matter occupation number
        for matter in [0, 1]:
            for n_mx, n_my, n_px, n_py in product(single_rishon_configs, repeat=4):
                # Update raw counter
                raw_counter += 1
                # DEFINE GAUSS LAW
                left = matter + n_mx + n_px + n_my + n_py
                right = lattice_dim * n_rishons + 0.5 * (1 + parity)
                # CHECK GAUSS LAW
                if left == right:
                    # SAVE THE STATE
                    gauge_states[site].append([matter, n_mx, n_my, n_px, n_py, parity])
                    # Update col counter
                    col_counter += 1
                    # FIX raw and col of the site basis
                    raw.append(raw_counter)
        # Build the basis as a sparse matrix
        data = np.ones(col_counter, dtype=float)
        x = np.asarray(raw)
        y = np.arange(col_counter)
        gauge_basis[site] = csr_matrix((data, (x, y)), shape=(raw_counter, col_counter))
        # Save the gauge states as a np.array
        gauge_states[site] = np.asarray(gauge_states[site])
    return gauge_basis, gauge_states


def inner_site_operators(n_rishons):
    # Define the Rishon operators according to the chosen spin representation s (to which correspond n_rishons=2s)
    ops = {}
    shape = (n_rishons + 1, n_rishons + 1)
    if n_rishons == 2:
        data_m = np.array([np.sqrt(2), -np.sqrt(2)], dtype=float)
        p_diag = np.array([1, -1, 1], dtype=float)
    elif n_rishons == 3:
        data_m = np.array([-2 / np.sqrt(3), 4 / 3, -2 / np.sqrt(3)], dtype=float)
        p_diag = np.array([1, -1, 1, -1], dtype=float)
    elif n_rishons == 4:
        data_m = np.array([1, -np.sqrt(6) / 2, np.sqrt(6) / 2, -1], dtype=float)
        p_diag = np.array([1, -1, 1, -1, 1], dtype=float)
    else:
        raise Exception("Not yet implemented")
    data_p = np.ones(n_rishons, dtype=float)
    ops["xi_p"] = diags(data_p, offsets=+1, shape=shape)
    ops["xi_m"] = diags(data_m, offsets=+1, shape=shape)
    ops["psi"] = diags(np.array([1], dtype=float), offsets=1, shape=(2, 2))
    for op in ["xi_p", "xi_m", "psi"]:
        ops[f"{op}_dag"] = ops[op].conj().transpose()
    # Define parity operators
    ops["P_xi"] = diags(p_diag, offsets=0, shape=shape)
    ops["P_psi"] = diags(np.array([1, -1], dtype=float), offsets=0, shape=(2, 2))
    # Single site number operators
    ops["N_psi"] = ops["psi_dag"] * ops["psi"]
    ops["N_xi_m"] = diags(np.arange(n_rishons + 1, dtype=float), offsets=0, shape=shape)
    ops["N_xi_p"] = diags(np.arange(n_rishons + 1), offsets=0, shape=shape)
    # Identities
    ops["ID_psi"] = identity(2)
    ops["ID_xi"] = identity(n_rishons + 1)
    # Operators useful for Electric term and Link symmetries
    for s in ["m", "p"]:
        ops[f"E2_{s}"] = (ops[f"N_xi_{s}"] - n_rishons * identity(n_rishons + 1)) ** 2
        ops[f"E0_{s}"] = (
            ops[f"N_xi_{s}"] - 0.5 * n_rishons * identity(n_rishons + 1)
        ) ** 2
    return ops


def dressed_site_operators(n_rishons):
    # Get inner site operators
    in_ops = inner_site_operators(n_rishons)
    # Dictionary for operators
    ops = {}
    # Hopping operators
    ops["Q_mx_dag"] = qmb_op(in_ops, ["psi_dag", "xi_m", "ID_xi", "ID_xi", "ID_xi"])
    ops["Q_my_dag"] = qmb_op(in_ops, ["psi_dag", "P_xi", "xi_m", "ID_xi", "ID_xi"])
    ops["Q_px_dag"] = qmb_op(in_ops, ["psi_dag", "P_xi", "P_xi", "xi_p", "ID_xi"])
    ops["Q_py_dag"] = qmb_op(in_ops, ["psi_dag", "P_xi", "P_xi", "P_xi", "xi_p"])
    # add dagger operators
    Qs = {}
    for op in ops:
        dag_op = op.replace("_dag", "")
        Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
    ops |= Qs
    # Number operators
    ops["N"] = qmb_op(in_ops, ["N_psi", "ID_xi", "ID_xi", "ID_xi", "ID_xi"])
    ops["n_mx"] = qmb_op(in_ops, ["ID_psi", "N_xi_m", "ID_xi", "ID_xi", "ID_xi"])
    ops["n_my"] = qmb_op(in_ops, ["ID_psi", "ID_xi", "N_xi_m", "ID_xi", "ID_xi"])
    ops["n_px"] = qmb_op(in_ops, ["ID_psi", "ID_xi", "ID_xi", "N_xi_p", "ID_xi"])
    ops["n_py"] = qmb_op(in_ops, ["ID_psi", "ID_xi", "ID_xi", "ID_xi", "N_xi_p"])
    # Corner operators
    in_ops["xi_m_P"] = in_ops["xi_m"] * in_ops["P_xi"]
    in_ops["xi_p_P"] = in_ops["xi_p"] * in_ops["P_xi"]
    in_ops["P_xi_m_dag"] = in_ops["P_xi"] * in_ops["xi_m_dag"]
    ops["C_px,py"] = qmb_op(in_ops, ["ID_psi", "ID_xi", "ID_xi", "xi_p_P", "xi_p_dag"])
    ops["C_py,mx"] = qmb_op(in_ops, ["ID_psi", "P_xi_m_dag", "P_xi", "P_xi", "xi_p"])
    ops["C_mx,my"] = qmb_op(in_ops, ["ID_psi", "xi_m_P", "xi_m_dag", "ID_xi", "ID_xi"])
    ops["C_my,px"] = qmb_op(in_ops, ["ID_psi", "ID_xi", "xi_m_P", "xi_p_dag", "ID_xi"])
    # Electric operator
    ops["E_square"] = qmb_op(in_ops, ["ID_psi", "E2_m", "ID_xi", "ID_xi", "ID_xi"])
    ops["E_square"] += qmb_op(in_ops, ["ID_psi", "ID_xi", "E2_m", "ID_xi", "ID_xi"])
    ops["E_square"] += qmb_op(in_ops, ["ID_psi", "ID_xi", "ID_xi", "E2_p", "ID_xi"])
    ops["E_square"] += qmb_op(in_ops, ["ID_psi", "ID_xi", "ID_xi", "ID_xi", "E2_p"])
    ops["E_square"] = 0.5 * ops["E_square"]
    return ops


def get_QED_operators(n_rishons):
    eff_ops = {"even": {}, "odd": {}}
    # Build the gauge invariant basis (for even and odd sites)
    # in the chosen spin representation s
    gauge_basis, _ = gauge_invariant_states(n_rishons)
    # Acquire dressed site operators
    ops = dressed_site_operators(n_rishons)
    # Project the operators in the subspace with gauge invariant states.
    # Effective operators have the same shape of the local gauge basis
    for site in gauge_basis:
        for op in ops:
            eff_ops[site][op] = csr_matrix(
                gauge_basis[site].conj().transpose() * ops[op] * gauge_basis[site]
            )
    return eff_ops


def get_QED_Hamiltonian_couplings(g, m):
    E = (g**2) / 2  # ELECTRIC FIELD
    B = -1 / (2 * (g**2))  # MAGNETIC FIELD
    # DICTIONARY WITH MODEL COEFFICIENTS
    coeffs = {
        "g": g,
        "m": m,
        "E": E,  # ELECTRIC FIELD coupling
        "B": B,  # MAGNETIC FIELD coupling
        "eta": 20 * max(E, np.abs(B), m),  # PENALTY
        "tx_even": 0.5,  # HORIZONTAL HOPPING
        "tx_odd": 0.5,
        "ty_even": 0.5,  # VERTICAL HOPPING (EVEN SITES)
        "ty_odd": -0.5,  # VERTICAL HOPPING (ODD SITES)
        "m_even": m,
        "m_odd": -m,
    }
    return coeffs
