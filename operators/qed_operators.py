import numpy as np
from numpy.linalg import matrix_rank
from itertools import product
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import norm
from simsio import logger
from modeling import qmb_operator as qmb_op
from tools import anti_commutator as anti_comm

__all__ = [
    "get_QED_Hamiltonian_couplings",
    "QED_dressed_site_operators",
    "gauge_invariant_states",
]


def QED_rishon_operators(spin, U="ladder"):
    if not np.isscalar(spin):
        raise TypeError(f"spin must be SCALAR & (semi)INTEGER, not {type(spin)}")
    if not isinstance(U, str):
        raise TypeError(f"U must be str, not {type(U)}")
    """
    This function computes the SU2 the Rishon modes adopted
    for the U(1) Lattice Gauge Theory for the chosen spin-s irrepresentation

    Args:
        spin (scalar, real): spin value, assumed to be integer or semi-integer
        U (str, optional): which version of U you want to use to obtain rishons

    Returns:
        dict: dictionary with the rishon operators and the parity
    """
    # Size of the spin/rishon matrix
    size = int(2 * spin + 1)
    shape = (size, size)
    # Based on the U definition, define the diagonal entries of the rishon modes
    if U == "ladder":
        zm_diag = [(-1) ** (i + 1) for i in range(size - 1)][::-1]
        U_diag = np.ones(size - 1)
    elif U == "spin":
        sz_diag = np.arange(-spin, spin + 1)[::-1]
        U_diag = (np.sqrt(spin * (spin + 1) - sz_diag[:-1] * (sz_diag[:-1] - 1))) / spin
        zm_diag = [U_diag[i] * ((-1) ** (i + 1)) for i in range(size - 1)][::-1]
    else:
        raise ValueError(f"U can only be 'ladder' or 'spin', not {U}")
    ops = {}
    ops["U"] = diags(U_diag, -1, shape)
    # RISHON MODES
    ops["xi_p"] = diags(np.ones(size - 1), 1, shape)
    ops["xi_m"] = diags(zm_diag, 1, shape)
    # PARITY OPERATOR
    ops["P_xi"] = diags([(-1) ** i for i in range(size)], 0, shape)
    # IDENTITY OPERATOR
    ops["ID_xi"] = identity(size)
    # ELECTRIC FIELD OPERATORS
    ops["n"] = diags(np.arange(size), 0, shape)
    ops["E0"] = ops["n"] - 0.5 * (size - 1) * identity(size)
    ops["E0_square"] = ops["E0"] ** 2
    for side in ["p", "m"]:
        # GENERATE THE DAGGER OPERATORS
        ops[f"xi_{side}_dag"] = ops[f"xi_{side}"].transpose()
        # CHECK RISHON MODES TO BEHAVE LIKE FERMIONS (anticommute with parity)
        if norm(anti_comm(ops[f"xi_{side}"], ops["P_xi"])) > 1e-15:
            raise ValueError(f"z_{side} is a Fermion and must anticommute with P")
    return ops


def QED_inner_site_operators(spin, U="ladder"):
    # Define the Rishon operators according to the chosen spin representation s
    ops = QED_rishon_operators(spin, U)
    # Define the MATTER FIELDS OPERATORS
    ops["psi"] = diags(np.array([1], dtype=float), 1, (2, 2))
    ops["psi_dag"] = ops["psi"].transpose()
    ops["P_psi"] = diags(np.array([1, -1], dtype=float), 0, (2, 2))
    ops["N"] = ops["psi_dag"] * ops["psi"]
    ops["ID_psi"] = identity(2)
    return ops


def QED_dressed_site_operators(spin, U="ladder", lattice_dim=2):
    # Size of the spin/rishon matrix
    xi_size = int(2 * spin + 1)
    n_rishons = int(2 * spin)
    tot_dim = 2 * xi_size ** (2 * lattice_dim)
    # Get inner site operators (Rishons + Matter fields)
    in_ops = QED_inner_site_operators(spin, U)
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
        P_coeff = 0 if site == "even" else +1
        gauss_law_ops[site] = ops["N"] - (n_rishons * lattice_dim + P_coeff) * identity(
            tot_dim
        )
        for s in ["mx", "my", "px", "py"]:
            gauss_law_ops[site] += ops[f"n_{s}"]
    # CHECK GAUSS LAW
    if spin < 4:
        gauge_basis, _ = gauge_invariant_states(spin)
        check_QED_gauss_law(gauge_basis, gauss_law_ops)
    return ops


def check_QED_gauss_law(gauge_basis, gauss_law_ops, threshold=1e-15):
    # This functions performs some checks on the QED gauge invariant basis
    logger.info("CHECK GAUSS LAW")
    M = gauge_basis
    for site in ["even", "odd"]:
        # True and the Effective dimensions of the gauge invariant dressed site
        site_dim = M[site].shape[0]
        eff_site_dim = M[site].shape[1]
        # Check that the Matrix Basis behave like an isometry
        if norm(M[site].transpose() * M[site] - identity(eff_site_dim)) > threshold:
            raise ValueError(f"The gauge basis M on {site} sites is not an Isometry")
        # Check that M*M^T is a Projector
        Proj = M[site] * M[site].transpose()
        if norm(Proj - Proj**2) > threshold:
            raise ValueError(
                f"Gauge basis on {site} sites must provide a projector P=M*M^T"
            )
        # Check that the basis is the one with ALL the states satisfying Gauss law
        if norm(gauss_law_ops[site] * M[site]) > threshold:
            raise ValueError(f"Gauss Law not satisfied for {site} sites")
        if site_dim - matrix_rank(gauss_law_ops[site].todense()) != eff_site_dim:
            logger.info(site)
            logger.info(f"Large dimension {site_dim}")
            logger.info(f"Effective dimension {eff_site_dim}")
            logger.info(matrix_rank(gauss_law_ops[site].todense()))
            logger.info(f"Some gauge basis states of {site} sites are missing")


def get_QED_Hamiltonian_couplings(g, m):
    """
    This function provides the QED Hamiltonian coefficients
    starting from the gauge coupling g and the bare mass parameter m
    Args:
        g (scalar): gauge coupling
        m (scalar): bare mass parameter

    Returns:
        dict: dictionary of Hamiltonian coefficients
    """
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
        "ty_odd": -0.5,  # VERTICAL HOPPING (ODD SITES) -1/2
        "m_even": m,
        "m_odd": -m,
    }
    logger.info(f"LINK SYMMETRY PENALTY {coeffs['eta']}")
    return coeffs


def QED_border_configs(config, spin):
    """
    This function fixes the value of the electric field on
    lattices with open boundary conditions (has_obc=True).
    -For integer spin representation, the offset of E is naturally
    the central value assumed by the rishon number.

    -For semi-integer spin representation, there is some freedom
    in the choice of the offset one possible solution is the one
    corresponding to the first negative value of the electric field

    Args:
        config (list of ints): configuration of internal rishons in
        the single dressed site basis, ordered as follows:
        [n_matter, n_mx, n_my, n_px, n_py]

        spin (int): chosen spin representation for U(1)

    Returns:
        list of strings: list of configs corresponding to a border/corner of the lattice
        with a fixed value of the electric field
    """
    n_rishons = int(2 * spin)
    if (n_rishons % 2) == 0:
        # integer spin representation
        off_set = {"p": n_rishons // 2, "m": n_rishons // 2}
    else:
        # semi-integer representation
        off_set = {"p": n_rishons // 2, "m": 1 + (n_rishons // 2)}
    label = []
    if config[1] == off_set["m"]:
        label.append("mx")
    if config[2] == off_set["m"]:
        label.append("my")
    if config[3] == off_set["p"]:
        label.append("px")
    if config[4] == off_set["p"]:
        label.append("py")
    if (config[1] == off_set["m"]) and (config[2] == off_set["m"]):
        label.append("mx_my")
    if (config[1] == off_set["m"]) and (config[4] == off_set["p"]):
        label.append("mx_py")
    if (config[2] == off_set["m"]) and (config[3] == off_set["p"]):
        label.append("my_px")
    if (config[3] == off_set["p"]) and (config[4] == off_set["p"]):
        label.append("px_py")
    return label


def gauge_invariant_states(spin, lattice_dim=2):
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
        spin ((semi)integer): it corresponds to the spin representation of U(1)

        lattice_dim (int, optional): number of spatial dimensions. Defaults to 2.

    Returns:
        _type_: _description_
    """
    rishon_size = int(2 * spin + 1)
    single_rishon_configs = np.arange(rishon_size)
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
                right = lattice_dim * int(2 * spin) + 0.5 * (1 - parity)
                # CHECK GAUSS LAW
                if left == right:
                    # FIX row and col of the site basis
                    row[site].append(row_counter)
                    col_counter[site] += 1
                    # SAVE THE STATE
                    config = [matter, n_mx, n_my, n_px, n_py, parity]
                    gauge_states[site].append(config)
                    # GET THE CONFIG LABEL
                    label = QED_border_configs(config, spin)
                    if label:
                        # save the config state also in the specific subset for the specif border
                        for ll in label:
                            gauge_states[f"{site}_{ll}"].append(config)
                            row[f"{site}_{ll}"].append(row_counter)
                            col_counter[f"{site}_{ll}"] += 1
        # Build the basis as a sparse matrix
        bord = ["", "_mx", "_my", "_px", "_py", "_mx_my", "_mx_py", "_my_px", "_px_py"]
        for ll in bord:
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


def get_QED_operators(spin):
    eff_ops = {}
    # Build the gauge invariant basis (for even and odd sites)
    # in the chosen spin representation s
    gauge_basis, _ = gauge_invariant_states(spin)
    # Acquire dressed site operators
    ops = QED_dressed_site_operators(spin)
    # Project the operators in the subspace with gauge invariant states.
    # Effective operators have the same shape of the local gauge basis
    for site in ["even", "odd"]:
        for op in ops:
            eff_ops[f"{op}_{site}"] = csr_matrix(
                gauge_basis[site].transpose() * ops[op] * gauge_basis[site]
            )
    return eff_ops
