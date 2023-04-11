# %%
import numpy as np
from itertools import product
from scipy.sparse import csr_matrix, diags, kron, identity
from scipy.sparse.linalg import norm
from simsio import logger

__all__ = [
    "get_QED_operators",
    "get_QED_Hamiltonian_couplings",
    "dressed_site_operators",
    "gauge_invariant_states",
]


def qmb_op(ops, list_ops):
    qmb_op = ops[list_ops[0]]
    for op in list_ops[1:]:
        qmb_op = kron(qmb_op, ops[op])
    return csr_matrix(qmb_op)


def border_configs(config, n_rishons):
    s = int(n_rishons / 2)
    label = []
    if config[1] == s:
        label.append("mx")
    if config[2] == s:
        label.append("my")
    if config[3] == s:
        label.append("px")
    if config[4] == s:
        label.append("py")
    if (config[1] == s) and (config[2] == s):
        label.append("mx_my")
    if (config[1] == s) and (config[4] == s):
        label.append("mx_py")
    if (config[2] == s) and (config[3] == s):
        label.append("my_px")
    if (config[3] == s) and (config[4] == s):
        label.append("px_py")
    return label


def gauge_invariant_states(n_rishons, lattice_dim=2):
    """
    This function generates the gauge invariant basis of a QED LGT
    in a 2D rectangular lattice where gauge and matter degrees of
    freedom are merged in a compact-site notation by exploiting
    a rishon-based quantum link model.
    NOTE: the gague invariant basis is different for even
    and odd sites due to the staggered fermion solution
    NOTE: the function provides also a restricted basis for sites
    on the borderd of the lattice where not all the configurations
    are allowed (the external rishons/gauge fields do not contribute)

    Args:
        n_rishons (int): it corresponds to the spin representation of U(1) s=n_rishons/2

        lattice_dim (int, optional): number of spatial dimensions. Defaults to 2.

    Returns:
        _type_: _description_
    """
    single_rishon_configs = np.arange(n_rishons + 1)
    gauge_states = {"even": [], "odd": []}
    gauge_basis = {}
    # Set a rows list for the basis
    row = {"even": [], "odd": []}
    col_counter = {"even": -1, "odd": -1}
    # Run over even and odd sites
    for ii, site in enumerate(["even", "odd"]):
        for label in ["mx", "my", "px", "py", "mx_my", "mx_py", "my_px", "px_py"]:
            gauge_states[f"{site}_{label}"] = []
            row[f"{site}_{label}"] = []
            col_counter[f"{site}_{label}"] = -1
        # Define the parity of odd (-1) and even (+1) sites
        parity = -1 if site == "odd" else 1
        # Set row and col counters
        row_counter = -1
        # Possible matter occupation number
        for matter in [0, 1]:
            for n_mx, n_my, n_px, n_py in product(single_rishon_configs, repeat=4):
                # Update row counter
                row_counter += 1
                # DEFINE GAUSS LAW
                left = matter + n_mx + n_px + n_my + n_py
                right = lattice_dim * n_rishons + 0.5 * (1 - parity)
                # CHECK GAUSS LAW
                if left == right:
                    # FIX row and col of the site basis
                    row[site].append(row_counter)
                    col_counter[site] += 1
                    # SAVE THE STATE
                    config = [matter, n_mx, n_my, n_px, n_py, parity]
                    gauge_states[site].append(config)
                    # GET THE CONFIG LABEL
                    label = border_configs(config, n_rishons)
                    if label:
                        # save the config state also in the specific subset for the specif border
                        for ll in label:
                            gauge_states[f"{site}_{ll}"].append(config)
                            row[f"{site}_{ll}"].append(row_counter)
                            col_counter[f"{site}_{ll}"] += 1
        # Build the basis as a sparse matrix
        for ll in [
            "",
            "_mx",
            "_my",
            "_px",
            "_py",
            "_mx_my",
            "_mx_py",
            "_my_px",
            "_px_py",
        ]:
            name = f"{site}{ll}"
            data = np.ones(col_counter[name] + 1, dtype=float)
            x = np.asarray(row[name])
            y = np.arange(col_counter[name] + 1)
            gauge_basis[name] = csr_matrix(
                (data, (x, y)), shape=(row_counter + 1, col_counter[name] + 1)
            )
            # Save the gauge states as a np.array
            gauge_states[name] = np.asarray(gauge_states[name])
    return gauge_basis, gauge_states


def inner_site_operators(n_rishons):
    # Define the Rishon operators according to the chosen spin representation s
    # (to which correspond n_rishons=2s)
    ops = {}
    shape = (n_rishons + 1, n_rishons + 1)
    if n_rishons == 2:  # s=1
        data_m = np.array([np.sqrt(2), -np.sqrt(2)], dtype=float)
        p_diag = np.array([1, -1, 1], dtype=float)
    elif n_rishons == 3:  # s=3/2
        data_m = np.array([-2 / np.sqrt(3), 4 / 3, -2 / np.sqrt(3)], dtype=float)
        p_diag = np.array([1, -1, 1, -1], dtype=float)
    elif n_rishons == 4:  # s=2
        data_m = np.array([1, -np.sqrt(6) / 2, np.sqrt(6) / 2, -1], dtype=float)
        p_diag = np.array([1, -1, 1, -1, 1], dtype=float)
    elif n_rishons == 5:  # s=5/2
        data_m = np.array(
            [
                -2 / np.sqrt(5),
                np.sqrt(32) / 5,
                -6 / 5,
                np.sqrt(32) / 5,
                -2 / np.sqrt(5),
            ],
            dtype=float,
        )
        p_diag = np.array([1, -1, 1, -1, 1, -1], dtype=float)
    elif n_rishons == 6:  # s=3
        data_m = np.array(
            [
                np.sqrt(6) / 3,
                -np.sqrt(10) / 3,
                np.sqrt(12) / 3,
                -np.sqrt(12) / 3,
                np.sqrt(10) / 3,
                -np.sqrt(6) / 3,
            ],
            dtype=float,
        )
        p_diag = np.array([1, -1, 1, -1, 1, -1, 1], dtype=float)
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
    ops["N"] = ops["psi_dag"] * ops["psi"]
    ops["n"] = diags(np.arange(n_rishons + 1, dtype=float), offsets=0, shape=shape)
    # Identities
    ops["ID_psi"] = identity(2)
    ops["ID_xi"] = identity(n_rishons + 1)
    # Operators useful for Electric term and Link symmetries
    ops[f"E0"] = ops[f"n"] - 0.5 * n_rishons * identity(n_rishons + 1)
    ops[f"E0_square"] = ops[f"E0"] ** 2
    return ops


def dressed_site_operators(n_rishons, lattice_dim=2):
    tot_dim = 2 * (n_rishons + 1) ** (2 * lattice_dim)
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
    # Psi Number operators
    ops["N"] = qmb_op(in_ops, ["N", "ID_xi", "ID_xi", "ID_xi", "ID_xi"])
    # Corner operators
    in_ops["xi_m_P"] = in_ops["xi_m"] * in_ops["P_xi"]
    in_ops["xi_p_P"] = in_ops["xi_p"] * in_ops["P_xi"]
    in_ops["P_xi_m_dag"] = in_ops["P_xi"] * in_ops["xi_m_dag"]
    ops["C_px,py"] = -qmb_op(in_ops, ["ID_psi", "ID_xi", "ID_xi", "xi_p_P", "xi_p_dag"])
    ops["C_py,mx"] = qmb_op(in_ops, ["ID_psi", "P_xi_m_dag", "P_xi", "P_xi", "xi_p"])
    ops["C_mx,my"] = qmb_op(in_ops, ["ID_psi", "xi_m_P", "xi_m_dag", "ID_xi", "ID_xi"])
    ops["C_my,px"] = qmb_op(in_ops, ["ID_psi", "ID_xi", "xi_m_P", "xi_p_dag", "ID_xi"])
    # Rishon Number operators
    for op in ["E0", "n", "E0_square"]:
        ops[f"{op}_mx"] = qmb_op(in_ops, ["ID_psi", op, "ID_xi", "ID_xi", "ID_xi"])
        ops[f"{op}_my"] = qmb_op(in_ops, ["ID_psi", "ID_xi", op, "ID_xi", "ID_xi"])
        ops[f"{op}_px"] = qmb_op(in_ops, ["ID_psi", "ID_xi", "ID_xi", op, "ID_xi"])
        ops[f"{op}_py"] = qmb_op(in_ops, ["ID_psi", "ID_xi", "ID_xi", "ID_xi", op])
    # E_square operators
    ops["E_square"] = 0
    for s in ["mx", "my", "px", "py"]:
        ops["E_square"] += 0.5 * ops[f"E0_square_{s}"]
    # GAUSS LAW OPERATORS
    gauss_law_ops = {}
    for site in ["even", "odd"]:
        parity_coeff = 0 if site == "even" else +1
        gauss_law_ops[site] = ops["N"] - (
            lattice_dim * n_rishons + parity_coeff
        ) * identity(tot_dim)
        for s in ["mx", "my", "px", "py"]:
            gauss_law_ops[site] += ops[f"n_{s}"]
    # CHECK GAUSS LAW
    gauge_basis, _ = gauge_invariant_states(n_rishons)
    check_gauss_law(gauge_basis, gauss_law_ops)
    return ops


def check_gauss_law(gauge_basis, gauss_law_ops, threshold=1e-10):
    logger.info("CHECK GAUSS LAW")
    M = gauge_basis
    for site in ["even", "odd"]:
        # Get the dimension of the gauge invariant dressed site
        dim_eff_site = M[site].shape[1]
        # Check that the Matrix Basis behave like an isometry
        if norm(M[site].transpose() * M[site] - identity(dim_eff_site)) > threshold:
            raise ValueError(f"The gauge basis M on {site} sites is not an Isometry")
        # Check that M*M^T is a Projector
        Proj = M[site] * M[site].transpose()
        if norm(Proj - Proj**2) > threshold:
            raise ValueError(
                f"The gauge basis M on {site} sites does not provide a projector P=M*M^T"
            )
        # Check that the basis is the one with ALL the states satisfying Gauss law
        if norm(gauss_law_ops[site] * M[site]) > threshold:
            raise ValueError(
                f"The gauge basis states of {site} sites don't satisfy Gauss Law"
            )
        if (
            M[site].shape[0] - np.linalg.matrix_rank(gauss_law_ops[site].todense())
            != M[site].shape[1]
        ):
            logger.info(site)
            logger.info(f"Large dimension {M[site].shape[0]}")
            logger.info(f"Effective dimension {M[site].shape[1]}")
            logger.info(np.linalg.matrix_rank(gauss_law_ops[site].todense()))
            logger.info(f"Some gauge basis states of {site} sites are missing")


def get_QED_operators(n_rishons):
    eff_ops = {}
    # Build the gauge invariant basis (for even and odd sites)
    # in the chosen spin representation s
    gauge_basis, _ = gauge_invariant_states(n_rishons)
    # Acquire dressed site operators
    ops = dressed_site_operators(n_rishons)
    # Project the operators in the subspace with gauge invariant states.
    # Effective operators have the same shape of the local gauge basis
    for site in ["even", "odd"]:
        for op in ops:
            eff_ops[f"{op}_{site}"] = csr_matrix(
                gauge_basis[site].transpose() * ops[op] * gauge_basis[site]
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
        "eta": 10 * max(E, np.abs(B), np.abs(m)),  # PENALTY
        "tx_even": 0.5,  # HORIZONTAL HOPPING
        "tx_odd": 0.5,
        "ty_even": 0.5,  # VERTICAL HOPPING (EVEN SITES)
        "ty_odd": -0.5,  # VERTICAL HOPPING (ODD SITES)
        "m_even": m,
        "m_odd": -m,
    }
    logger.info(f"LINK SYMMETRY PENALTY {coeffs['eta']}")
    return coeffs


"""
n_rishons = 2
M, states = gauge_invariant_states(n_rishons)
in_ops = inner_site_operators(n_rishons)
ops = dressed_site_operators(n_rishons)
eff_ops = get_QED_operators(n_rishons)
"""
