"""Operator factories and gauge-invariant local bases for U(1) (QED) models."""

import numpy as np
from numpy.linalg import matrix_rank
from itertools import product
from scipy.sparse import csr_matrix, diags, identity, kron
from scipy.sparse.linalg import norm
from edlgt.modeling import qmb_operator as qmb_op
from edlgt.modeling import get_lattice_borders_labels, LGT_border_configs
from edlgt.tools import anti_commutator as anti_comm, validate_parameters
from .bose_fermi_operators import fermi_operators as QED_matter_operators
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "QED_dressed_site_operators",
    "QED_gauge_invariant_states",
    "QED_rishon_operators",
    "QED_check_gauss_law",
]


def QED_rishon_operators(spin, pure_theory, U, fermionic=True):
    """
    Build single-rishon operators for a truncated U(1) quantum link model.

    The local basis is the electric-field eigenbasis with dimension "2*spin + 1"
    and eigenvalues "E = -spin, ..., +spin" (in increasing order). The operator
    "U" is implemented as a one-step shift in this basis (chosen so that it
    raises "E" by one unit where defined).

    If "fermionic=True", rishons are treated as fermionic modes via a local
    parity operator "P = (-1)**n" and the convention::

        Zp     = U @ P
        Zm_dag = U
        Zm     = Zm_dag.T
        Zp_dag = Zp.T

    Convenience composites used in corner/plaquette constructions are also
    provided (e.g. "Zp_P", "P_Zm_dag").

    Parameters
    ----------
    spin : int
        Truncation parameter (local dimension ``2*spin + 1``).
    pure_theory : bool
        Included for API consistency and validation.
    U : str
        Shift amplitude convention. Supported values are ``"ladder"`` and
        ``"spin"``.
    fermionic : bool, optional
        If True, include parity and enforce fermionic anticommutation checks.

    Returns
    -------
    dict
        Sparse operator dictionary including ``P``, ``U``, rishon creation and
        annihilation operators, and electric-field operators.
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
    if fermionic:
        ops["P"] = diags([(-1) ** i for i in range(size)], 0, shape, dtype=float)
    else:
        ops["P"] = identity(size, dtype=float)
    # Based on the U definition, define the diagonal entries of the rishon modes
    if U == "ladder":
        S_diag = np.ones(size - 1, dtype=float)
    elif U == "spin":
        # Treat local basis as m = -j,...,j (reversed)
        sz_vals = np.arange(-spin, spin + 1, dtype=float)[::-1]
        S_diag = (np.sqrt(spin * (spin + 1) - sz_vals[1:] * (sz_vals[1:] - 1))) / spin
    else:
        raise ValueError(f"U can only be 'ladder' or 'spin', not {U!r}")
    # Parallel transporter definition
    Uop = diags(S_diag, -1, shape, dtype=float)  # raises E by +1
    ops["U"] = Uop
    # RISHON OPERATORS
    ops["Zp"] = Uop @ ops["P"]  # Zp = S x P
    ops["Zm_dag"] = Uop  # Zm_dag = S
    # DAGGER OPERATORS of RISHON MODES
    ops["Zp_dag"] = ops["Zp"].transpose()  # (S x P)^T = P x S^T since P diagonal
    ops["Zm"] = ops["Zm_dag"].transpose()
    # USEFUL OPERATORS FOR THE CORNER TERMS
    ops["Zm_P"] = ops["Zm"] @ ops["P"]
    ops["Zp_P"] = ops["Zp"] @ ops["P"]
    ops["P_Zm_dag"] = ops["P"] @ ops["Zm_dag"]
    ops["P_Zp_dag"] = ops["P"] @ ops["Zp_dag"]
    # IDENTITY OPERATOR
    ops["Iz"] = identity(size, dtype=float)
    # ELECTRIC FIELD OPERATORS
    ops["E"] = diags(np.arange(-spin, spin + 1, 1, dtype=float), 0, shape, dtype=float)
    ops["E2"] = ops["E"] ** 2
    # In case with dynamical matter, check FERMIONIC RISHONS
    if fermionic:
        for key in ["Zp", "Zm", "Zp_dag", "Zm_dag"]:
            # anticommute with parity
            if norm(anti_comm(ops[key], ops["P"])) > 1e-15:
                raise ValueError(f"{key} is a Fermion and must anticommute with P")
    return ops


def QED_dressed_site_operators(
    spin,
    pure_theory,
    lattice_dim,
    U="ladder",
    fermionic=True,
    background=0,
    check_gauss_law=False,
):
    """Build dressed-site QED operators for 1D, 2D, or 3D lattices.

    Parameters
    ----------
    spin : int
        Gauge-link spin truncation (local link dimension ``2*spin + 1``).
    pure_theory : bool
        If ``True``, build the pure-gauge operator set (no matter fields).
    lattice_dim : int
        Number of spatial lattice dimensions (supported: 1, 2, 3).
    U : str, optional
        Rishon-shift convention passed to :func:`QED_rishon_operators`
        (typically ``"ladder"`` or ``"spin"``).
    fermionic : bool, optional
        If ``True``, use fermionic matter/rishon parity conventions.
    background : int, optional
        Maximum absolute static background charge included at each site. ``0``
        disables the background degree of freedom.
    check_gauss_law : bool, optional
        If ``True``, run internal consistency checks on the constructed local
        Gauss-law operators.

    Returns
    -------
    dict
        Dictionary of dressed-site operators used by the QED model builders.
    """
    validate_parameters(
        spin_list=[spin], pure_theory=pure_theory, lattice_dim=lattice_dim
    )
    if not isinstance(U, str):
        raise TypeError(f"U must be str, not {type(U)}")
    logger.info("----------------------------------------------------")
    logger.info(f"QED OPERATORS s={spin}, bg={background}")
    # Lattice Dimensions
    dimensions = "xyz"[:lattice_dim]
    # Get the Rishon operators according to the chosen n truncation
    in_ops = QED_rishon_operators(spin, pure_theory, U, fermionic)
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
        in_ops |= QED_matter_operators(has_spin=False, fermionic=fermionic)
    if background > 0:
        tot_dim *= 2 * background + 1
    # Dictionary for operators
    ops = {}
    if lattice_dim == 1:
        # Rishon Electric operators
        for op in ["E", "E2", "P"]:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "Iz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["Iz", op])
        if not pure_theory:
            # Update Electric operators
            for op in ops.keys():
                ops[op] = kron(in_ops["ID_psi"], ops[op])
            # Add Hopping operators
            ops["Q_mx_dag"] = qmb_op(in_ops, ["psi_dag_P", "Zm", "Iz"])
            ops["Q_px_dag"] = qmb_op(in_ops, ["psi_dag_P", "P", "Zp"])
            # and their dagger operators
            Qs = {}
            for op_name, op_mat in ops.items():
                if op_name.endswith("_dag"):
                    dag_name = op_name[:-4]  # drop "_dag"
                    Qs[dag_name] = csr_matrix(op_mat.conj().transpose())
            ops |= Qs
            # Psi Number operators
            ops["N"] = qmb_op(in_ops, ["N", "Iz", "Iz"])
            ops["N_zero"] = qmb_op(in_ops, ["N_zero", "Iz", "Iz"])
    elif lattice_dim == 2:
        # Rishon Electric operators
        for op in ["E", "E2", "P"]:
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
            ops["Q_mx_dag"] = qmb_op(in_ops, ["psi_dag_P", "Zm", "Iz", "Iz", "Iz"])
            ops["Q_my_dag"] = qmb_op(in_ops, ["psi_dag_P", "P", "Zm", "Iz", "Iz"])
            ops["Q_px_dag"] = qmb_op(in_ops, ["psi_dag_P", "P", "P", "Zp", "Iz"])
            ops["Q_py_dag"] = qmb_op(in_ops, ["psi_dag_P", "P", "P", "P", "Zp"])
            # Add dagger operators
            Qs = {}
            for op_name, op_mat in ops.items():
                if op_name.endswith("_dag"):
                    dag_name = op_name[:-4]  # drop "_dag"
                    Qs[dag_name] = csr_matrix(op_mat.conj().transpose())
            ops |= Qs
            # Psi Number operators
            ops["N"] = qmb_op(in_ops, ["N", "Iz", "Iz", "Iz", "Iz"])
            ops["N_zero"] = qmb_op(in_ops, ["N_zero", "Iz", "Iz", "Iz", "Iz"])
    elif lattice_dim == 3:
        # Rishon Electric operators
        for op in ["E", "E2", "P"]:  # , "Ep1", "E0", "Em1"]:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "Iz", "Iz", "Iz", "Iz", "Iz"])
            ops[f"{op}_my"] = qmb_op(in_ops, ["Iz", op, "Iz", "Iz", "Iz", "Iz"])
            ops[f"{op}_mz"] = qmb_op(in_ops, ["Iz", "Iz", op, "Iz", "Iz", "Iz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["Iz", "Iz", "Iz", op, "Iz", "Iz"])
            ops[f"{op}_py"] = qmb_op(in_ops, ["Iz", "Iz", "Iz", "Iz", op, "Iz"])
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
            # Theta term corners
            ops["EzC_px,py"] = 1j * (ops["E_pz"] + ops["E_mz"]) @ ops["C_px,py"]
            ops["EyC_px,pz"] = -1j * (ops["E_py"] + ops["E_my"]) @ ops["C_px,pz"]
            ops["ExC_py,pz"] = 1j * (ops["E_px"] + ops["E_mx"]) @ ops["C_py,pz"]
        if not pure_theory:
            # Update Electric and Corner operators
            for op in ops.keys():
                ops[op] = kron(in_ops["ID_psi"], ops[op])
            # Hopping operators
            # ---------------------------------------------------------
            op_list = ["psi_dag", "Zm", "Iz", "Iz", "Iz", "Iz", "Iz"]
            ops["Q_mx_dag"] = qmb_op(in_ops, op_list)
            # ---------------------------------------------------------
            op_list = ["psi_dag", "P", "Zm", "Iz", "Iz", "Iz", "Iz"]
            ops["Q_my_dag"] = qmb_op(in_ops, op_list)
            # ---------------------------------------------------------
            op_list = ["psi_dag", "P", "P", "Zm", "Iz", "Iz", "Iz"]
            ops["Q_mz_dag"] = qmb_op(in_ops, op_list)
            # ---------------------------------------------------------
            op_list = ["psi_dag", "P", "P", "P", "Zp", "Iz", "Iz"]
            ops["Q_px_dag"] = qmb_op(in_ops, op_list)
            # ---------------------------------------------------------
            op_list = ["psi_dag", "P", "P", "P", "P", "Zp", "Iz"]
            ops["Q_py_dag"] = qmb_op(in_ops, op_list)
            # ---------------------------------------------------------
            op_list = ["psi_dag", "P", "P", "P", "P", "P", "Zp"]
            ops["Q_pz_dag"] = qmb_op(in_ops, op_list)
            # Add dagger operators
            Qs = {}
            for op_name, op_mat in ops.items():
                if op_name.endswith("_dag"):
                    dag_name = op_name[:-4]  # drop "_dag"
                    Qs[dag_name] = csr_matrix(op_mat.conj().transpose())
            ops |= Qs
            # Psi Number operators
            ops["N"] = qmb_op(in_ops, ["N", "Iz", "Iz", "Iz", "Iz", "Iz", "Iz"])
            op_list = ["N_zero", "Iz", "Iz", "Iz", "Iz", "Iz", "Iz"]
            ops["N_zero"] = qmb_op(in_ops, op_list)
    # E_square operators
    ops["E2"] = 0
    for d in dimensions:
        for s in "mp":
            ops["E2"] += 0.5 * ops[f"E2_{s}{d}"]
    # -----------------------------------------------------------------------------
    # BACKGROUND FIELD OPERATORS
    if background > 0:
        bg_dim = int(2 * background + 1)
        for op in ops.keys():
            ops[op] = kron(identity(bg_dim), ops[op])
        if pure_theory:
            id_list = ["Iz" for _ in range(2 * lattice_dim)]
        else:
            id_list = ["ID_psi"] + ["Iz" for _ in range(2 * lattice_dim)]
        ops["bg"] = qmb_op(in_ops, ["E"] + id_list)
    # -----------------------------------------------------------------------------
    # # GAUSS LAW OPERATORS on hard-core lattice sites
    if check_gauss_law:
        gauss_law_ops = {}
        for ii, site in enumerate(core_site_list):
            gauss_law_ops[site] = 0
            for d in dimensions:
                gauss_law_ops[site] += ops[f"E_p{d}"] - ops[f"E_m{d}"]
            if not pure_theory:
                stagger_offset = 0.5 * (1 - parity[ii])  # 0 even, 1 odd
                gauss_law_ops[site] -= ops["N"] - stagger_offset * identity(tot_dim)
            if background > 0:
                gauss_law_ops[site] -= ops["bg"]
        QED_check_gauss_law(spin, pure_theory, lattice_dim, gauss_law_ops)
    return ops


def QED_check_gauss_law(
    spin, pure_theory, lattice_dim, gauss_law_ops, background=0, threshold=1e-15
):
    """Validate a local QED gauge-invariant basis against Gauss-law constraints.

    Parameters
    ----------
    spin : int
        Gauge-link spin truncation.
    pure_theory : bool
        If ``True``, check the pure-gauge local basis.
    lattice_dim : int
        Number of spatial lattice dimensions.
    gauss_law_ops : dict
        Dictionary of local Gauss-law operators keyed by site type.
    background : int, optional
        Maximum absolute static background charge included in the local basis.
    threshold : float, optional
        Numerical tolerance used in the consistency checks.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If input argument types are invalid.
    ValueError
        If the basis is not isometric/projective or does not satisfy Gauss law.
    """
    validate_parameters(
        spin_list=[spin],
        pure_theory=pure_theory,
        lattice_dim=lattice_dim,
        threshold=threshold,
    )
    if not isinstance(gauss_law_ops, dict):
        raise TypeError(f"gauss_law_ops should be a DICT, not a {type(gauss_law_ops)}")
    # This functions performs some checks on the QED gauge invariant basis
    gauge_basis, _ = QED_gauge_invariant_states(
        spin, pure_theory, lattice_dim, get_only_bulk=True, background=background
    )
    if pure_theory:
        site_list = ["site"]
    else:
        site_list = ["even", "odd"]
    for site in site_list:
        # True and the Effective dimensions of the gauge invariant dressed site
        site_dim = gauge_basis[site].shape[0]
        eff_site_dim = gauge_basis[site].shape[1]
        # Check that the Matrix Basis behave like an isometry
        projector = gauge_basis[site]
        norm_Pdagger_P = norm(projector.T * projector - identity(eff_site_dim))
        if norm_Pdagger_P > threshold:
            logger.info(f"Norm of P^T*P - I: {norm_Pdagger_P}, expected 0")
            msg = f"The gauge basis on {site} sites does not provide an Isometry"
            raise ValueError(msg)
        # Check that P*P^T is a Projector
        Proj = gauge_basis[site] * gauge_basis[site].T
        if norm(Proj - Proj**2) > threshold:
            msg = f"Gauge basis on {site} sites must provide a projector P*P^T"
            raise ValueError(msg)
        # Check that the basis is the one with ALL the states satisfying Gauss law
        norm_GL = norm(gauss_law_ops[site] * gauge_basis[site])
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


def QED_gauge_invariant_states(
    spin, pure_theory, lattice_dim, background=0, get_only_bulk=False
):
    """Construct local gauge-invariant QED basis states and basis matrices.

    Parameters
    ----------
    spin : int
        Gauge-link spin truncation.
    pure_theory : bool
        If ``True``, exclude matter fields.
    lattice_dim : int
        Number of spatial dimensions.
    background : int, optional
        Maximum absolute static background charge included at the site.
    get_only_bulk : bool, optional
        If ``True``, keep only bulk-compatible site subsets when classifying
        border/corner configurations.

    Returns
    -------
    tuple
        ``(gauge_basis, gauge_states)`` dictionaries. ``gauge_basis`` stores
        sparse basis matrices (full local basis -> gauge-invariant subspace) and
        ``gauge_states`` stores the corresponding local configurations.
    """
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    if not np.isscalar(lattice_dim) or not isinstance(lattice_dim, int):
        msg = f"lattice_dim must be SCALAR & INTEGER, not {type(lattice_dim)}"
        raise TypeError(msg)
    if not np.isscalar(background) or not isinstance(background, int) or background < 0:
        msg = f"background must be a non-negative INTEGER, not {background!r}"
        raise TypeError(msg)
    if not get_only_bulk:
        if not np.isscalar(spin) or not isinstance(spin, int):
            raise TypeError(f"spin must be SCALAR & INTEGER, not {type(spin)}")
    logger.info("----------------------------------------------------")
    logger.info(f"QED GAUGE INVARIANT STATES s={spin}, bg={background}")
    single_rishon_configs = np.arange(-spin, spin + 1, 1, dtype=int)
    # List of borders/corners of the lattice
    borders = get_lattice_borders_labels(lattice_dim)
    # List of configurations for each element of the dressed site
    dressed_site_config_list = [single_rishon_configs for _ in range(2 * lattice_dim)]
    # Distinction between pure and full theory
    if pure_theory:
        core_labels = ["site"]
        parity = [1]
    else:
        core_labels = ["even", "odd"]
        parity = [1, -1]
        # matter occupation (0/1)
        dressed_site_config_list.insert(0, np.arange(2))
    # Add background charge as the leftmost dof (outermost loop in product)
    if background > 0:
        background_values = np.arange(-background, background + 1, dtype=int)
        dressed_site_config_list.insert(0, background_values)
        background_offset = 1
    else:
        background_offset = 0
    # Define useful quantities
    gauge_states = {}
    row = {}
    col_counter = {}
    for ii, main_label in enumerate(core_labels):
        row_counter = -1
        gauge_states[main_label] = []
        row[main_label] = []
        col_counter[main_label] = -1
        for border_label in borders:
            key = f"{main_label}_{border_label}"
            gauge_states[key] = []
            row[key] = []
            col_counter[key] = -1
        # Look at all the possible configurations of gauge links and matter fields
        for config in product(*dressed_site_config_list):
            # Update row counter
            row_counter += 1
            # Split out the background charge if present
            if background_offset:
                q_bg = config[0]
                physical_config = config[1:]  # matter (optional) + gauge fields
            else:
                q_bg = 0
                physical_config = config  # matter (optional) + gauge fields
            if pure_theory:
                matter_charge = 0
                electric_fields_config = physical_config
            else:
                n_psi = int(physical_config[0])  # 0/1
                stagger_offset = int(0.5 * (1 - parity[ii]))  # 0 even, 1 odd
                matter_charge = n_psi - stagger_offset
                electric_fields_config = physical_config[1:]
            # Measure the divergence of the electric field for the given configuration
            divergence_E = 0
            for mu in range(lattice_dim):
                E_m = electric_fields_config[mu]
                E_p = electric_fields_config[mu + lattice_dim]
                divergence_E += E_p - E_m
            # Check Gauss Law: divergence of E must be equal to the total charge (matter + background)
            if divergence_E == (matter_charge + q_bg):
                # FIX row and col of the site basis
                row[main_label].append(row_counter)
                col_counter[main_label] += 1
                # Save the gauge invariant state
                gauge_states[main_label].append(config)
                # Get the config labels: border classification should ignore background charge
                # (and see exactly the same local structure as before, up to the same ordering)
                border_labels = LGT_border_configs(
                    physical_config, 0, pure_theory, get_only_bulk
                )
                if border_labels:
                    # save the config state also in the specific subset for the specif border
                    for border_name in border_labels:
                        border_key = f"{main_label}_{border_name}"
                        gauge_states[border_key].append(config)
                        row[border_key].append(row_counter)
                        col_counter[border_key] += 1
    # Build the basis as a sparse matrix
    gauge_basis = {}
    for name in list(gauge_states.keys()):
        data = np.ones(col_counter[name] + 1, dtype=float)
        x = np.asarray(row[name])
        y = np.arange(col_counter[name] + 1)
        shape = (row_counter + 1, col_counter[name] + 1)
        gauge_basis[name] = csr_matrix((data, (x, y)), shape=shape)
        # Save the gauge states as a np.ndarray
        gauge_states[name] = np.asarray(gauge_states[name])
    return gauge_basis, gauge_states
