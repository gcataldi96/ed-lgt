import numpy as np
from numpy.linalg import matrix_rank
from itertools import product
from scipy.sparse import csr_matrix, diags, identity, kron
from scipy.sparse.linalg import norm
from ed_lgt.modeling import qmb_operator as qmb_op
from ed_lgt.modeling import get_lattice_borders_labels, LGT_border_configs
from ed_lgt.tools import anti_commutator as anti_comm, validate_parameters
from .bose_fermi_operators import fermi_operators as QED_matter_operators
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "QED_dressed_site_operators",
    "QED_gauge_invariant_states",
    "QED_rishon_operators",
    "QED_check_gauss_law",
]


def QED_rishon_operators(spin, pure_theory, U):
    """
    This function computes the QED Rishon modes adopted
    for the U(1) Lattice Gauge Theory for the chosen spin representation of the Gauge field.

    Args:
        spin (scalar, int): spin representation of the U(1) Gauge field, corresponding to a gauge Hilbert space of dimension (2 spin +1)

        pure_theory (bool): If true, the dressed site includes matter fields

        U (str): which version of U you want to use to obtain rishons: 'ladder', 'spin'

    Returns:
        dict: dictionary with the rishon operators and the parity
    """
    validate_parameters(spin_list=[spin], pure_theory=pure_theory)
    if not isinstance(U, str):
        raise TypeError(f"U must be str, not {type(U)}")
    # Size of rishon operator matrices
    size = int(2 * spin + 1)
    shape = (size, size)
    # Dictionary of Operators
    ops = {}
    # PARITY OPERATOR of RISHON MODES
    ops["P"] = diags([(-1) ** i for i in range(size)], 0, shape)
    # Based on the U definition, define the diagonal entries of the rishon modes
    if U == "ladder":
        zm_diag = [(-1) ** (i + 1) for i in range(size - 1)][::-1]
        U_diag = np.ones(size - 1)
        ops["U"] = diags(U_diag, -1, shape)
        # RISHON MODES
        ops["Zp"] = diags(np.ones(size - 1), 1, shape)
        ops["Zm"] = diags(zm_diag, 1, shape)
    elif U == "spin":
        sz_diag = np.arange(-spin, spin + 1)[::-1]
        U_diag = (np.sqrt(spin * (spin + 1) - sz_diag[:-1] * (sz_diag[:-1] - 1))) / spin
        zm_diag = [U_diag[i] * ((-1) ** (i + 1)) for i in range(size - 1)][::-1]
        ops["U"] = diags(U_diag, -1, shape)
        # RISHON MODES
        ops["Zp"] = diags(np.ones(size - 1), 1, shape)
        ops["Zm"] = diags(zm_diag, 1, shape)
    else:
        raise ValueError(f"U can only be 'ladder', 'spin', not {U}")
    # DAGGER OPERATORS of RISHON MODES
    for s in "pm":
        ops[f"Z{s}_dag"] = ops[f"Z{s}"].transpose()
    # PERFORM CHECKS
    for s in "pm":
        # CHECK RISHON MODES TO BEHAVE LIKE FERMIONS
        # anticommute with parity
        if norm(anti_comm(ops[f"Z{s}"], ops["P"])) > 1e-15:
            raise ValueError(f"Z{s} is a Fermion and must anticommute with P")
    # IDENTITY OPERATOR
    ops["Iz"] = identity(size)
    # Useful operators for Corners
    ops["Zm_P"] = ops["Zm"] * ops["P"]
    ops["Zp_P"] = ops["Zp"] * ops["P"]
    ops["P_Zm_dag"] = ops["P"] * ops["Zm_dag"]
    ops["P_Zp_dag"] = ops["P"] * ops["Zp_dag"]
    # ELECTRIC FIELD OPERATORS
    ops["n"] = diags(np.arange(size), 0, shape)
    ops["E"] = ops["n"] - 0.5 * (size - 1) * identity(size)
    ops["E_square"] = ops["E"] ** 2
    return ops


def QED_dressed_site_operators(spin, pure_theory, U, lattice_dim):
    """
    This function generates the dressed-site operators of the QED Hamiltonian
    in d spatial dimensions for d=1,2,3 (pure or with matter fields)
    for any possible trunctation of the spin representation of the gauge fields.

    Args:
        spin (scalar, int): spin representation of the U(1) Gauge field, corresponding
            to a gauge Hilbert space of dimension (2 spin +1)

        pure_theory (bool): If true, the dressed site includes matter fields

        U (str): which version of U you want to use to obtain rishons: 'ladder', 'spin', 'cyclic'

        lattice_dim (int): number of lattice spatial dimensions

    Returns:
        dict: dictionary with all the operators of the QED (pure or full) Hamiltonian
    """
    validate_parameters(
        spin_list=[spin], pure_theory=pure_theory, lattice_dim=lattice_dim
    )
    if not isinstance(U, str):
        raise TypeError(f"U must be str, not {type(U)}")
    # Lattice Dimensions
    dimensions = "xyz"[:lattice_dim]
    # Get the Rishon operators according to the chosen n truncation
    in_ops = QED_rishon_operators(spin, pure_theory, U)
    # Size of the rishon operators
    z_size = int(2 * spin + 1)
    # Size of the whole dressed site
    tot_dim = z_size ** (2 * lattice_dim)
    if pure_theory:
        core_site_list = ["site"]
        parity = [1]
    else:
        core_site_list = ["even", "odd"]
        parity = [1, -1]
        # Acquire also matter field operators
        tot_dim *= 2
        in_ops |= QED_matter_operators(has_spin=False)
    # Dictionary for operators
    ops = {}
    if lattice_dim == 1:
        # Rishon Electric operators
        for op in ["E", "E_square", "n", "P"]:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "Iz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["Iz", op])
        if not pure_theory:
            # Update Electric operators
            for op in ops.keys():
                ops[op] = kron(in_ops["ID_psi"], ops[op])
            # Add Hopping operators
            ops["Q_mx_dag"] = qmb_op(in_ops, ["psi_dag", "Zm", "Iz"])
            ops["Q_px_dag"] = qmb_op(in_ops, ["psi_dag", "P", "Zp"])
            # and their dagger operators
            Qs = {}
            for op in ops:
                dag_op = op.replace("_dag", "")
                Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
            ops |= Qs
            # Psi Number operators
            ops["N"] = qmb_op(in_ops, ["N", "Iz", "Iz"])
    elif lattice_dim == 2:
        # Rishon Electric operators
        for op in ["E", "E_square", "n", "P"]:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "Iz", "Iz", "Iz"])
            ops[f"{op}_my"] = qmb_op(in_ops, ["Iz", op, "Iz", "Iz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["Iz", "Iz", op, "Iz"])
            ops[f"{op}_py"] = qmb_op(in_ops, ["Iz", "Iz", "Iz", op])
        # Corner Operators
        ops["C_px,py"] = -qmb_op(in_ops, ["Iz", "Iz", "Zp_P", "Zp_dag"])
        ops["C_py,mx"] = qmb_op(in_ops, ["P_Zm_dag", "P", "P", "Zp"])
        ops["C_mx,my"] = qmb_op(in_ops, ["Zm_P", "Zm_dag", "Iz", "Iz"])
        ops["C_my,px"] = qmb_op(in_ops, ["Iz", "Zm_P", "Zp_dag", "Iz"])
        if not pure_theory:
            # Update Electric and Corner operators
            for op in ops.keys():
                ops[op] = kron(in_ops["ID_psi"], ops[op])
            # Hopping operators
            ops["Q_mx_dag"] = qmb_op(in_ops, ["psi_dag", "Zm", "Iz", "Iz", "Iz"])
            ops["Q_my_dag"] = qmb_op(in_ops, ["psi_dag", "P", "Zm", "Iz", "Iz"])
            ops["Q_px_dag"] = qmb_op(in_ops, ["psi_dag", "P", "P", "Zp", "Iz"])
            ops["Q_py_dag"] = qmb_op(in_ops, ["psi_dag", "P", "P", "P", "Zp"])
            # Add dagger operators
            Qs = {}
            for op in ops:
                dag_op = op.replace("_dag", "")
                Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
            ops |= Qs
            # Psi Number operators
            ops["N"] = qmb_op(in_ops, ["N", "Iz", "Iz", "Iz", "Iz"])
    elif lattice_dim == 3:
        # Rishon Electric operators
        for op in ["E", "E_square", "n", "P"]:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "Iz", "Iz", "Iz", "Iz", "Iz"])
            ops[f"{op}_my"] = qmb_op(in_ops, ["Iz", op, "Iz", "Iz", "Iz", "Iz"])
            ops[f"{op}_mz"] = qmb_op(in_ops, ["Iz", "Iz", op, "Iz", "Iz", "Iz"])
            ops[f"{op}_py"] = qmb_op(in_ops, ["Iz", "Iz", "Iz", op, "Iz", "Iz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["Iz", "Iz", "Iz", "Iz", op, "Iz"])
            ops[f"{op}_pz"] = qmb_op(in_ops, ["Iz", "Iz", "Iz", "Iz", "Iz", op])
            # Corner Operators
            # X-Y Plane
            ops["C_px,py"] = -qmb_op(in_ops, ["Iz", "Iz", "Iz", "Zp_P", "Zp_dag", "Iz"])
            ops["C_py,mx"] = qmb_op(in_ops, ["P_Zm_dag", "P", "P", "P", "Zp", "Iz"])
            ops["C_mx,my"] = qmb_op(in_ops, ["Zm_P", "Zm_dag", "Iz", "Iz", "Iz", "Iz"])
            ops["C_my,px"] = qmb_op(in_ops, ["Iz", "Zm_P", "P", "Zp_dag", "Iz", "Iz"])
            # X-Z Plane
            ops["C_px,pz"] = -qmb_op(in_ops, ["Iz", "Iz", "Iz", "Zp_P", "P", "Zp_dag"])
            ops["C_pz,mx"] = qmb_op(in_ops, ["P_Zm_dag", "P", "P", "P", "P", "Zp"])
            ops["C_mx,mz"] = qmb_op(in_ops, ["Zm_P", "P", "Zm_dag", "Iz", "Iz", "Iz"])
            ops["C_mz,px"] = qmb_op(in_ops, ["Iz", "Iz", "Zm_P", "Zp_dag", "Iz", "Iz"])
            # Y_Z Plane
            ops["C_py,pz"] = -qmb_op(in_ops, ["Iz", "Iz", "Iz", "Iz", "Zp_P", "Zp_dag"])
            ops["C_pz,my"] = qmb_op(in_ops, ["Iz", "P_Zm_dag", "P", "P", "P", "Zp"])
            ops["C_my,mz"] = qmb_op(in_ops, ["Iz", "Zm_P", "Zm_dag", "Iz", "Iz", "Iz"])
            ops["C_mz,py"] = qmb_op(in_ops, ["Iz", "Iz", "Zm_P", "P", "Zp_dag", "Iz"])
        if not pure_theory:
            # Update Electric and Corner operators
            for op in ops.keys():
                ops[op] = kron(in_ops["ID_psi"], ops[op])
            # Hopping operators
            ops["Q_mx_dag"] = qmb_op(
                in_ops, ["psi_dag", "Zm", "Iz", "Iz", "Iz", "Iz", "Iz"]
            )
            ops["Q_my_dag"] = qmb_op(
                in_ops, ["psi_dag", "P", "Zm", "Iz", "Iz", "Iz", "Iz"]
            )
            ops["Q_mz_dag"] = qmb_op(
                in_ops, ["psi_dag", "P", "P", "Zm", "Iz", "Iz", "Iz"]
            )
            ops["Q_px_dag"] = qmb_op(
                in_ops, ["psi_dag", "P", "P", "P", "Zp", "Iz", "Iz"]
            )
            ops["Q_py_dag"] = qmb_op(
                in_ops, ["psi_dag", "P", "P", "P", "P", "Zp", "Iz"]
            )
            ops["Q_pz_dag"] = qmb_op(in_ops, ["psi_dag", "P", "P", "P", "P", "P", "Zp"])
            # Add dagger operators
            Qs = {}
            for op in ops:
                dag_op = op.replace("_dag", "")
                Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
            ops |= Qs
            # Psi Number operators
            ops["N"] = qmb_op(in_ops, ["N", "Iz", "Iz", "Iz", "Iz", "Iz", "Iz"])
    # E_square operators
    ops["E_square"] = 0
    for d in dimensions:
        for s in "mp":
            ops["E_square"] += 0.5 * ops[f"E_square_{s}{d}"]
    # Define Gauss Law operators of hard-core lattice sites
    if spin < 4 and lattice_dim < 3:
        # GAUSS LAW OPERATORS
        gauss_law_ops = {}
        for ii, site in enumerate(core_site_list):
            gauss_law_ops[site] = -(
                lattice_dim * (z_size - 1) + 0.5 * (1 - parity[ii])
            ) * identity(tot_dim)
            for d in dimensions:
                for s in "mp":
                    gauss_law_ops[site] += ops[f"n_{s}{d}"]
            if not pure_theory:
                gauss_law_ops[site] += ops["N"]
        QED_check_gauss_law(spin, pure_theory, lattice_dim, gauss_law_ops)
    return ops


def QED_check_gauss_law(spin, pure_theory, lattice_dim, gauss_law_ops, threshold=1e-15):
    """
    This function perform a series of checks to the gauge invariant dressed-site local basis
    of the QED Hamiltonian, in order to verify that Gauss Law is effectively satified.

    Args:
        spin (scalar, int): spin representation of the U(1) Gauge field, corresponding to a gauge Hilbert space of dimension (2 spin +1)

        pure_theory (bool): If True, the local basis describes gauge invariant states in absence of matter. Defaults to False.

        lattice_dim (int): number of lattice spatial dimensions

        gauss_law_ops (dict): It contains the Gauss Law operators (for each type of lattice site)

        threshold (scalar & real, optional): numerical threshold for checks. Defaults to 1e-15.

    Raises:
        TypeError: If the input arguments are of incorrect types or formats.

        ValueError: if the gauge basis M does not behave as an Isometry: M^T*M=1

        ValueError: if the gauge basis does not generate a Projector P=M*M^T

        ValueError: if the QED gauss law is not satisfied
    """
    validate_parameters(
        spin_list=[spin],
        pure_theory=pure_theory,
        lattice_dim=lattice_dim,
        threshold=threshold,
    )
    if not isinstance(gauss_law_ops, dict):
        raise TypeError(f"pure_theory should be a DICT, not a {type(gauss_law_ops)}")
    # This functions performs some checks on the QED gauge invariant basis
    M, _ = QED_gauge_invariant_states(spin, pure_theory, lattice_dim)
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
        norm_GL = norm(gauss_law_ops[site] * M[site])
        if norm_GL > threshold:
            logger.info(f"Norm of GL * Basis: {norm_GL}, expected 0")
            raise ValueError(f"Gauss Law not satisfied for {site} sites")
        if site_dim - matrix_rank(gauss_law_ops[site].todense()) != eff_site_dim:
            logger.info(site)
            logger.info(f"Large dimension {site_dim}")
            logger.info(f"Effective dimension {eff_site_dim}")
            logger.info(matrix_rank(gauss_law_ops[site].todense()))
            logger.info(f"Some gauge basis states of {site} sites are missing")
    logger.info("QED GAUSS LAW SATISFIED")


def QED_gauge_invariant_states(spin, pure_theory, lattice_dim):
    """
    This function generates the gauge invariant basis of a QED LGT
    in a d-dimensional lattice where gauge (and matter) degrees of
    freedom are merged in a compact-site notation by exploiting
    a rishon-based quantum link model.

    NOTE: In presence of matter, the gague invariant basis is different for even
    and odd sites due to the staggered fermion solution

    NOTE: The function provides also a restricted basis for sites
    on the borderd of the lattice where not all the configurations
    are allowed (the external rishons/gauge fields do not contribute)

    Args:
        spin (scalar, int): spin representation of the U(1) Gauge field,
            corresponding to a gauge Hilbert space of dimension (2 spin +1)

        pure_theory (bool,optional): if True, the theory does not involve matter fields

        lattice_dim (int, optional): number of spatial dimensions. Defaults to 2.

    Returns:
        (dict, dict): dictionaries with the basis and the states
    """
    if not np.isscalar(spin) or not isinstance(spin, int):
        raise TypeError(f"spin must be SCALAR & INTEGER, not {type(spin)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    if not np.isscalar(lattice_dim) or not isinstance(lattice_dim, int):
        raise TypeError(
            f"lattice_dim must be SCALAR & INTEGER, not {type(lattice_dim)}"
        )
    rishon_size = int(2 * spin + 1)
    single_rishon_configs = np.arange(rishon_size)
    # List of borders/corners of the lattice
    borders = get_lattice_borders_labels(lattice_dim)
    # List of configurations for each element of the dressed site
    dressed_site_config_list = [single_rishon_configs for i in range(2 * lattice_dim)]
    # Distinction between pure and full theory
    if pure_theory:
        core_labels = ["site"]
        parity = [1]
    else:
        core_labels = ["even", "odd"]
        parity = [1, -1]
        dressed_site_config_list.insert(0, np.arange(2))
    # Define useful quantities
    gauge_states = {}
    row = {}
    col_counter = {}
    for ii, main_label in enumerate(core_labels):
        row_counter = -1
        gauge_states[main_label] = []
        row[main_label] = []
        col_counter[main_label] = -1
        for label in borders:
            gauge_states[f"{main_label}_{label}"] = []
            row[f"{main_label}_{label}"] = []
            col_counter[f"{main_label}_{label}"] = -1
        # Look at all the possible configurations of gauge links and matter fields
        for config in product(*dressed_site_config_list):
            # Update row counter
            row_counter += 1
            # Define Gauss Law
            left = sum(config)
            right = lattice_dim * (rishon_size - 1) + 0.5 * (1 - parity[ii])
            # Check Gauss Law
            if left == right:
                # FIX row and col of the site basis
                row[main_label].append(row_counter)
                col_counter[main_label] += 1
                # Save the gauge invariant state
                gauge_states[main_label].append(config)
                # Get the config labels
                label = LGT_border_configs(config, spin, pure_theory)
                if label:
                    # save the config state also in the specific subset for the specif border
                    for ll in label:
                        gauge_states[f"{main_label}_{ll}"].append(config)
                        row[f"{main_label}_{ll}"].append(row_counter)
                        col_counter[f"{main_label}_{ll}"] += 1
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
