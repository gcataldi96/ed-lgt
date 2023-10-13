import numpy as np
from numpy.linalg import matrix_rank
from itertools import product
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import norm
from modeling import qmb_operator as qmb_op
from tools import anti_commutator as anti_comm

__all__ = [
    "get_QED_Hamiltonian_couplings",
    "QED_dressed_site_operators",
    "gauge_invariant_states",
]


def QED_rishon_operators(spin, U="ladder"):
    """
    This function computes the SU2 the Rishon modes adopted
    for the U(1) Lattice Gauge Theory for the chosen spin-s irrepresentation

    Args:
        spin (scalar, real): spin value, assumed to be integer or semi-integer
        U (str, optional): which version of U you want to use to obtain rishons

    Returns:
        dict: dictionary with the rishon operators and the parity
    """
    if not np.isscalar(spin):
        raise TypeError(f"spin must be SCALAR & (semi)INTEGER, not {type(spin)}")
    if not isinstance(U, str):
        raise TypeError(f"U must be str, not {type(U)}")
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
    ops["z_p"] = diags(np.ones(size - 1), 1, shape)
    ops["z_m"] = diags(zm_diag, 1, shape)
    # PARITY OPERATOR
    ops["P_z"] = diags([(-1) ** i for i in range(size)], 0, shape)
    # IDENTITY OPERATOR
    ops["ID_z"] = identity(size)
    # ELECTRIC FIELD OPERATORS
    ops["n"] = diags(np.arange(size), 0, shape)
    ops["E0"] = ops["n"] - 0.5 * (size - 1) * identity(size)
    ops["E0_square"] = ops["E0"] ** 2
    for side in ["p", "m"]:
        # GENERATE THE DAGGER OPERATORS
        ops[f"z_{side}_dag"] = ops[f"z_{side}"].transpose()
        # CHECK RISHON MODES TO BEHAVE LIKE FERMIONS (anticommute with parity)
        if norm(anti_comm(ops[f"z_{side}"], ops["P_z"])) > 1e-15:
            raise ValueError(f"z_{side} is a Fermion and must anticommute with P")
    return ops


def QED_matter_operators():
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
    ops["ID_psi"] = identity(2)
    return ops


def QED_dressed_site_operators(spin, U="ladder", pure_theory=False, lattice_dim=2):
    """
    This function generates the dressed-site operators of the 2D QED Hamiltonian
    (pure or with matter fields) for any possible trunctation of the spin representation of the gauge fields.

    Args:
        spin (scalar, real): spin value, assumed to be integer or semi-integer

        U (str, optional): which version of U you want to use to obtain rishons. Default to "ladder".

        pure_theory (bool, optional): If true, the dressed site includes matter fields. Defaults to False.

        lattice_dim (int, optional): number of lattice spatial dimensions. Defaults to 2.

    Returns:
        dict: dictionary with all the operators of the QED (pure or full) Hamiltonian
    """
    if not np.isscalar(spin):
        raise TypeError(f"spin must be SCALAR & (semi)INTEGER, not {type(spin)}")
    if not isinstance(U, str):
        raise TypeError(f"U must be str, not {type(U)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    if not np.isscalar(lattice_dim) and not isinstance(lattice_dim, int):
        raise TypeError(
            f"lattice_dim must be SCALAR & INTEGER, not {type(lattice_dim)}"
        )
    # Size of the spin/rishon matrix
    z_size = int(2 * spin + 1)
    n_rishons = int(2 * spin)
    tot_dim = 2 * z_size ** (2 * lattice_dim)
    # Get the Rishon operators according to the chosen spin representation s
    in_ops = QED_rishon_operators(spin, U)
    # Dictionary for operators
    ops = {}
    # Useful operators for Corners
    in_ops["z_m_P"] = in_ops["z_m"] * in_ops["P_z"]
    in_ops["z_p_P"] = in_ops["z_p"] * in_ops["P_z"]
    in_ops["P_z_m_dag"] = in_ops["P_z"] * in_ops["z_m_dag"]
    # Difference between PURE and FULL Theory
    if pure_theory:
        # Rishon Number operators
        for op in ["E0", "n", "E0_square"]:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "ID_z", "ID_z", "ID_z"])
            ops[f"{op}_my"] = qmb_op(in_ops, ["ID_z", op, "ID_z", "ID_z"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["ID_z", "ID_z", op, "ID_z"])
            ops[f"{op}_py"] = qmb_op(in_ops, ["ID_z", "ID_z", "ID_z", op])
        # Corner Operators
        ops["C_px,py"] = -qmb_op(in_ops, ["ID_z", "ID_z", "z_p_P", "z_p_dag"])
        ops["C_py,mx"] = qmb_op(in_ops, ["P_z_m_dag", "P_z", "P_z", "z_p"])
        ops["C_mx,my"] = qmb_op(in_ops, ["z_m_P", "z_m_dag", "ID_z", "ID_z"])
        ops["C_my,px"] = qmb_op(in_ops, ["ID_z", "z_m_P", "z_p_dag", "ID_z"])
    else:
        # Acquire also matter field operators
        in_ops |= QED_matter_operators()
        # Hopping operators
        ops["Q_mx_dag"] = qmb_op(in_ops, ["psi_dag", "z_m", "ID_z", "ID_z", "ID_z"])
        ops["Q_my_dag"] = qmb_op(in_ops, ["psi_dag", "P_z", "z_m", "ID_z", "ID_z"])
        ops["Q_px_dag"] = qmb_op(in_ops, ["psi_dag", "P_z", "P_z", "z_p", "ID_z"])
        ops["Q_py_dag"] = qmb_op(in_ops, ["psi_dag", "P_z", "P_z", "P_z", "z_p"])
        # Add dagger operators
        Qs = {}
        for op in ops:
            dag_op = op.replace("_dag", "")
            Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
        ops |= Qs
        # Psi Number operators
        ops["N"] = qmb_op(in_ops, ["N", "ID_z", "ID_z", "ID_z", "ID_z"])
        # Corner Operators
        ops["C_px,py"] = -qmb_op(in_ops, ["ID_psi", "ID_z", "ID_z", "z_p_P", "z_p_dag"])
        ops["C_py,mx"] = qmb_op(in_ops, ["ID_psi", "P_z_m_dag", "P_z", "P_z", "z_p"])
        ops["C_mx,my"] = qmb_op(in_ops, ["ID_psi", "z_m_P", "z_m_dag", "ID_z", "ID_z"])
        ops["C_my,px"] = qmb_op(in_ops, ["ID_psi", "ID_z", "z_m_P", "z_p_dag", "ID_z"])
        # Rishon Number operators
        for op in ["E0", "n", "E0_square"]:
            ops[f"{op}_mx"] = qmb_op(in_ops, ["ID_psi", op, "ID_z", "ID_z", "ID_z"])
            ops[f"{op}_my"] = qmb_op(in_ops, ["ID_psi", "ID_z", op, "ID_z", "ID_z"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["ID_psi", "ID_z", "ID_z", op, "ID_z"])
            ops[f"{op}_py"] = qmb_op(in_ops, ["ID_psi", "ID_z", "ID_z", "ID_z", op])
    # E_square operators
    ops["E_square"] = 0
    for s in ["mx", "my", "px", "py"]:
        ops["E_square"] += 0.5 * ops[f"E0_square_{s}"]
    # GAUSS LAW OPERATORS
    gauss_law_ops = {}
    if pure_theory:
        gauss_law_ops["site"] = 0
        for s in ["mx", "my", "px", "py"]:
            gauss_law_ops["site"] += ops[f"n_{s}"]
    else:
        for site in ["even", "odd"]:
            P_coeff = 0 if site == "even" else +1
            gauss_law_ops[site] = ops["N"] - (
                n_rishons * lattice_dim + P_coeff
            ) * identity(tot_dim)
            for s in ["mx", "my", "px", "py"]:
                gauss_law_ops[site] += ops[f"n_{s}"]
    # CHECK GAUSS LAW
    if spin < 4:
        gauge_basis, _ = gauge_invariant_states(spin=spin, pure_theory=pure_theory)
        check_QED_gauss_law(gauge_basis, gauss_law_ops, pure_theory)
    return ops


def check_QED_gauss_law(gauge_basis, gauss_law_ops, pure_theory=False, threshold=1e-15):
    """
    This function perform a series of checks to the gauge invariant dressed-site local basis
    of the QED Hamiltonian, in order to verify that Gauss Law is effectively satified.

    Args:
        gauge_basis (dict): It contains the Gauge invarian basis (for each type of lattice site)

        gauss_law_ops (dict): It contains the Gauss Law operators (for each type of lattice site)

        pure_theory (bool, optional): If True the local basis describes gauge invariant states in absence of matter. Defaults to False.

        threshold (scalar & real, optional): numerical threshold for checks. Defaults to 1e-15.

    Raises:
        TypeError: If the input arguments are of incorrect types or formats.

        ValueError: if the gauge basis M does not behave as an Isometry: M^T*M=1

        ValueError: if the gauge basis does not generate a Projector P=M*M^T

        ValueError: if the QED gauss law is not satisfied
    """

    if not isinstance(gauge_basis, dict):
        raise TypeError(f"gauge_basis should be a DICT, not a {type(gauge_basis)}")
    if not isinstance(gauss_law_ops, dict):
        raise TypeError(f"pure_theory should be a DICT, not a {type(gauss_law_ops)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    if not np.isscalar(threshold):
        raise TypeError(f"threshold must be SCALAR, not {type(threshold)}")
    # This functions performs some checks on the QED gauge invariant basis
    print("CHECK GAUSS LAW")
    M = gauge_basis
    if pure_theory:
        site_list = ["site"]
    else:
        site_list = ["even", "odd"]
    for site in site_list:
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
            print(site)
            print(f"Large dimension {site_dim}")
            print(f"Effective dimension {eff_site_dim}")
            print(matrix_rank(gauss_law_ops[site].todense()))
            print(f"Some gauge basis states of {site} sites are missing")


def get_QED_Hamiltonian_couplings(pure_theory, g, m=None):
    """
    This function provides the QED Hamiltonian coefficients
    starting from the gauge coupling g and the bare mass parameter m

    Args:
        pure_theory (bool): True if the theory does not include matter

        g (scalar): gauge coupling

        m (scalar, optional): bare mass parameter

    Returns:
        dict: dictionary of Hamiltonian coefficients
    """
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    if not np.isscalar(g):
        raise TypeError(f"g must be SCALAR, not {type(g)}")
    E = (g**2) / 2  # ELECTRIC FIELD
    B = -1 / (2 * (g**2))  # MAGNETIC FIELD
    # DICTIONARY WITH MODEL COEFFICIENTS
    coeffs = {
        "g": g,
        "E": E,  # ELECTRIC FIELD coupling
        "B": B,  # MAGNETIC FIELD coupling
    }
    if pure_theory:
        coeffs["eta"] = 10 * max(E, np.abs(B))
    else:
        if not np.isscalar(m):
            raise TypeError(f"m must be SCALAR, not {type(m)}")
        coeffs["eta"] = 10 * max(E, np.abs(B), np.abs(m))
        coeffs |= {
            "m": m,
            "tx_even": 0.5,  # HORIZONTAL HOPPING
            "tx_odd": 0.5,
            "ty_even": 0.5,  # VERTICAL HOPPING (EVEN SITES)
            "ty_odd": -0.5,  # VERTICAL HOPPING (ODD SITES) -1/2
            "m_even": m,
            "m_odd": -m,
        }
    print(f"LINK SYMMETRY PENALTY {coeffs['eta']}")
    return coeffs


def QED_border_configs(config, spin, pure_theory=False):
    """
    This function fixes the value of the electric field on
    lattices with open boundary conditions (has_obc=True).

    For integer spin representation, the offset of E is naturally
    the central value assumed by the rishon number.

    For semi-integer spin representation, there is some freedom
    in the choice of the offset one possible solution is the one
    corresponding to the first negative value of the electric field

    Args:
        config (list of ints): configuration of internal rishons in
        the single dressed site basis, ordered as follows:
        [n_matter, n_mx, n_my, n_px, n_py]

        spin (int): chosen spin representation for U(1)

        pure_theory (bool): True if the theory does not include matter

    Returns:
        list of strings: list of configs corresponding to a border/corner of the lattice
        with a fixed value of the electric field
    """
    if not isinstance(config, list):
        raise TypeError(f"config should be a LIST, not a {type(config)}")
    if not np.isscalar(spin):
        raise TypeError(f"spin must be SCALAR & (semi)INTEGER, not {type(spin)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    n_rishons = int(2 * spin)
    if (n_rishons % 2) == 0:
        # integer spin representation
        off_set = {"p": n_rishons // 2, "m": n_rishons // 2}
    else:
        # semi-integer representation
        off_set = {"p": n_rishons // 2, "m": 1 + (n_rishons // 2)}
    label = []
    if not pure_theory:
        config = config[1:]
    if config[0] == off_set["m"]:
        label.append("mx")
    if config[1] == off_set["m"]:
        label.append("my")
    if config[2] == off_set["p"]:
        label.append("px")
    if config[3] == off_set["p"]:
        label.append("py")
    if (config[0] == off_set["m"]) and (config[1] == off_set["m"]):
        label.append("mx_my")
    if (config[0] == off_set["m"]) and (config[3] == off_set["p"]):
        label.append("mx_py")
    if (config[1] == off_set["m"]) and (config[2] == off_set["p"]):
        label.append("my_px")
    if (config[2] == off_set["p"]) and (config[3] == off_set["p"]):
        label.append("px_py")
    return label


def gauge_invariant_states(spin, pure_theory=False, lattice_dim=2):
    """
    This function generates the gauge invariant basis of a QED LGT
    in a 2D rectangular lattice where gauge (and matter) degrees of
    freedom are merged in a compact-site notation by exploiting
    a rishon-based quantum link model.

    NOTE: In presence of matter, the gague invariant basis is different for even
    and odd sites due to the staggered fermion solution

    NOTE: The function provides also a restricted basis for sites
    on the borderd of the lattice where not all the configurations
    are allowed (the external rishons/gauge fields do not contribute)

    Args:
        spin ((semi)integer): it corresponds to the spin representation of U(1)

        pure_theory (bool,optional): if True, the theory does not involve matter fields

        lattice_dim (int, optional): number of spatial dimensions. Defaults to 2.

    Returns:
        (dict, dict): dictionaries with the basis and the states
    """
    if not np.isscalar(spin):
        raise TypeError(f"spin must be SCALAR & (semi)INTEGER, not {type(spin)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    if not np.isscalar(lattice_dim) and not isinstance(lattice_dim, int):
        raise TypeError(
            f"lattice_dim must be SCALAR & INTEGER, not {type(lattice_dim)}"
        )
    rishon_size = int(2 * spin + 1)
    single_rishon_configs = np.arange(rishon_size)
    # List of borders/corners of the lattice
    borders = ["mx", "my", "px", "py", "mx_my", "mx_py", "my_px", "px_py"]
    if pure_theory:
        gauge_states = {"site": []}
        row = {"site": []}
        row_counter = -1
        col_counter = {"site": -1}
        for label in borders:
            gauge_states[f"site_{label}"] = []
            row[f"site_{label}"] = []
            col_counter[f"site_{label}"] = -1
        for n_mx, n_my, n_px, n_py in product(single_rishon_configs, repeat=4):
            # Update row counter
            row_counter += 1
            # DEFINE GAUSS LAW
            left = n_mx + n_px + n_my + n_py
            right = lattice_dim * int(2 * spin)
            # CHECK GAUSS LAW
            if left == right:
                # FIX row and col of the site basis
                row["site"].append(row_counter)
                col_counter["site"] += 1
                # SAVE THE STATE
                config = [n_mx, n_my, n_px, n_py]
                gauge_states["site"].append(config)
                # GET THE CONFIG LABEL
                label = QED_border_configs(config, spin, pure_theory)
                if label:
                    # save the config state also in the specific subset for the specif border
                    for ll in label:
                        gauge_states[f"site_{ll}"].append(config)
                        row[f"site_{ll}"].append(row_counter)
                        col_counter[f"site_{ll}"] += 1
    else:
        # In this case we have a distinction between even and odd sites
        gauge_states = {"even": [], "odd": []}
        # Set a rows list for the basis
        row = {"even": [], "odd": []}
        col_counter = {"even": -1, "odd": -1}
        # Run over even and odd sites
        for site in ["even", "odd"]:
            for label in borders:
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
                        label = QED_border_configs(config, spin, pure_theory)
                        if label:
                            # save the config state also in the specific subset for the specif border
                            for ll in label:
                                gauge_states[f"{site}_{ll}"].append(config)
                                row[f"{site}_{ll}"].append(row_counter)
                                col_counter[f"{site}_{ll}"] += 1
    # Build the basis as a sparse matrix
    gauge_basis = {}
    for name in list(gauge_states.keys()):
        data = np.ones(col_counter[name] + 1, dtype=float)
        x = np.asarray(row[name])
        y = np.arange(col_counter[name] + 1)
        gauge_basis[name] = csr_matrix(
            (data, (x, y)), shape=(row_counter + 1, col_counter[name] + 1)
        )
        # Save the gauge states as a np.array
        gauge_states[name] = np.asarray(gauge_states[name])
    return gauge_basis, gauge_states
