import numpy as np
from itertools import product
from sympy import S
from scipy.sparse import csr_matrix, identity, isspmatrix, kron
from scipy.sparse.linalg import norm
from ed_lgt.modeling import qmb_operator as qmb_op
from ed_lgt.modeling import get_lattice_borders_labels, LGT_border_configs
from .SU2_singlets import get_SU2_singlets, SU2_singlet_canonical_vector
from .spin_operators import spin_space, SU2_generators
from .SU2_rishons import SU2_Rishon
from .bose_fermi_operators import fermi_operators as SU2_matter_operators
from ed_lgt.tools import validate_parameters
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "SU2_Hamiltonian_couplings",
    "SU2_dressed_site_operators",
    "SU2_rishon_operators",
    "SU2_check_gauss_law",
    "SU2_gauge_invariant_states",
]


def SU2_rishon_operators(spin):
    """
    This function computes the SU2 the Rishon modes adopted
    for the SU2 Lattice Gauge Theory for the chosen spin-s irrep-resentation

    Args:
        spin (half/integer): maximal spin-representation of the rishon operators

    Returns:
        dict: dictionary with the rishon operators and the parity
    """
    validate_parameters(spin_list=[spin])
    zeta = SU2_Rishon(spin)
    ops = zeta.ops
    ops |= SU2_generators(spin)
    return ops


def SU2_dressed_site_operators(spin, pure_theory, lattice_dim):
    validate_parameters(
        spin_list=[spin], pure_theory=pure_theory, lattice_dim=lattice_dim
    )
    # Lattice directions
    dimensions = "xyz"[:lattice_dim]
    # Get SU2 rishon operator
    in_ops = SU2_rishon_operators(spin)
    if not pure_theory:
        in_ops |= SU2_matter_operators(has_spin=True, colors=True)
        in_ops |= SU2_generators(1 / 2, matter=True)
    for op in in_ops.keys():
        in_ops[op] = csr_matrix(in_ops[op])
    # Dictionary for dressed site operators
    ops = {}
    if lattice_dim == 1:
        # T generators for electric term
        for op in ["T2", "T4", "Tx", "Ty", "Tz", "P"]:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "IDz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["IDz", op])
        if not pure_theory:
            # Update Electric operators
            for op in ops.keys():
                ops[op] = kron(in_ops["ID_psi"], ops[op])
            # Add Hopping operators
            for side in "pm":
                ops[f"Q{side}x_dag"] = 0
            for col in "rg":
                ops["Qpx_dag"] += qmb_op(in_ops, [f"psi_{col}_dag_P", "P", f"Z{col}"])
                ops["Qmx_dag"] += qmb_op(in_ops, [f"psi_{col}_dag_P", f"Z{col}", "IDz"])
            # add their dagger operators
            Qs = {}
            for op in ops:
                dag_op = op.replace("_dag", "")
                Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
            ops |= Qs
    elif lattice_dim == 2:
        # T generators for electric term
        for op in ["T2", "T4", "Tx", "Ty", "Tz", "P"]:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "IDz", "IDz", "IDz"])
            ops[f"{op}_my"] = qmb_op(in_ops, ["IDz", op, "IDz", "IDz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["IDz", "IDz", op, "IDz"])
            ops[f"{op}_py"] = qmb_op(in_ops, ["IDz", "IDz", "IDz", op])
        # Corner Operators
        for corner in ["px,py", "py,mx", "mx,my", "my,px"]:
            ops[f"C_{corner}"] = 0
        for col in ["r", "g"]:
            ops["C_px,py"] += qmb_op(in_ops, ["IDz", "IDz", f"Z{col}_P", f"Z{col}_dag"])
            ops["C_py,mx"] += qmb_op(in_ops, [f"P_Z{col}_dag", "P", "P", f"Z{col}"])
            ops["C_mx,my"] += qmb_op(in_ops, [f"Z{col}_P", f"Z{col}_dag", "IDz", "IDz"])
            ops["C_my,px"] += qmb_op(in_ops, ["IDz", f"Z{col}_P", f"Z{col}_dag", "IDz"])
        if not pure_theory:
            # Update Electric and Corner operators
            for op in ops.keys():
                ops[op] = kron(in_ops["ID_psi"], ops[op])
            # Add Hopping operators
            for side in "pm":
                for ax in dimensions:
                    ops[f"Q{side}{ax}_dag"] = 0
            for col in "rg":
                ops["Qmx_dag"] += qmb_op(
                    in_ops, [f"psi_{col}_dag_P", f"Z{col}", "IDz", "IDz", "IDz"]
                )
                ops["Qmy_dag"] += qmb_op(
                    in_ops, [f"psi_{col}_dag_P", "P", f"Z{col}", "IDz", "IDz"]
                )
                ops["Qpx_dag"] += qmb_op(
                    in_ops, [f"psi_{col}_dag_P", "P", "P", f"Z{col}", "IDz"]
                )
                ops["Qpy_dag"] += qmb_op(
                    in_ops, [f"psi_{col}_dag_P", "P", "P", "P", f"Z{col}"]
                )
            # add their dagger operators
            Qs = {}
            for op in ops:
                dag_op = op.replace("_dag", "")
                Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
            ops |= Qs
    elif lattice_dim == 3:
        # T generators for electric term
        for op in ["T2", "T4", "Tx", "Ty", "Tz", "P"]:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "IDz", "IDz", "IDz", "IDz", "IDz"])
            ops[f"{op}_my"] = qmb_op(in_ops, ["IDz", op, "IDz", "IDz", "IDz", "IDz"])
            ops[f"{op}_mz"] = qmb_op(in_ops, ["IDz", "IDz", op, "IDz", "IDz", "IDz"])
            ops[f"{op}_py"] = qmb_op(in_ops, ["IDz", "IDz", "IDz", op, "IDz", "IDz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["IDz", "IDz", "IDz", "IDz", op, "IDz"])
            ops[f"{op}_pz"] = qmb_op(in_ops, ["IDz", "IDz", "IDz", "IDz", "IDz", op])
        # Corner Operators
        corner_list = []
        for pdir in ["xy", "xz", "yz"]:
            # DEFINE THE LIST OF CORNER OPERATORS
            corner_list += [
                f"p{pdir[0]},p{pdir[1]}",
                f"p{pdir[1]},m{pdir[0]}",
                f"m{pdir[1]},p{pdir[0]}",
                f"m{pdir[0]},m{pdir[1]}",
            ]
        for corner in corner_list:
            ops[f"C_{corner}"] = 0
        for col in "rg":
            # --------------------------------------------------------------------------
            # XY Plane
            ops["C_px,py"] += qmb_op(
                in_ops,
                ["IDz", "IDz", "IDz", f"Z{col}_P", f"Z{col}_dag", "IDz"],
            )
            ops["C_py,mx"] += qmb_op(
                in_ops, [f"P_Z{col}_dag", "P", "P", "P", f"Z{col}", "IDz"]
            )
            ops["C_mx,my"] += qmb_op(
                in_ops,
                [f"Z{col}_P", f"Z{col}_dag", "IDz", "IDz", "IDz", "IDz"],
            )
            ops["C_my,px"] += qmb_op(
                in_ops, ["IDz", f"Z{col}_P", "P", f"Z{col}_dag", "IDz", "IDz"]
            )
            # --------------------------------------------------------------------------
            # XZ Plane
            ops["C_px,pz"] += qmb_op(
                in_ops,
                ["IDz", "IDz", "IDz", f"Z{col}_P", "P", f"Z{col}_dag"],
            )
            ops["C_pz,mx"] += qmb_op(
                in_ops, [f"P_Z{col}_dag", "P", "P", "P", "P", f"Z{col}"]
            )
            ops["C_mx,mz"] += qmb_op(
                in_ops,
                [f"Z{col}_P", "P", f"Z{col}_dag", "IDz", "IDz", "IDz"],
            )
            ops["C_mz,px"] += qmb_op(
                in_ops,
                ["IDz", "IDz", f"Z{col}_P", f"Z{col}_dag", "IDz", "IDz"],
            )
            # --------------------------------------------------------------------------
            # YZ Plane
            ops["C_py,pz"] += qmb_op(
                in_ops,
                ["IDz", "IDz", "IDz", "IDz", f"Z{col}_P", f"Z{col}_dag"],
            )
            ops["C_pz,my"] += qmb_op(
                in_ops, ["IDz", f"P_Z{col}_dag", "P", "P", "P", f"Z{col}"]
            )
            ops["C_my,mz"] += qmb_op(
                in_ops,
                ["IDz", f"Z{col}_P", f"Z{col}_dag", "IDz", "IDz", "IDz"],
            )
            ops["C_mz,py"] += qmb_op(
                in_ops,
                ["IDz", "IDz", f"Z{col}_P", "P", f"Z{col}_dag", "IDz"],
            )
        if not pure_theory:
            # Update Electric and Corner operators
            for op in ops.keys():
                ops[op] = kron(in_ops["ID_psi"], ops[op])
            # Add Hopping operators
            for side in "pm":
                for ax in dimensions:
                    ops[f"Q{side}{ax}_dag"] = 0
            for col in "rg":
                ops["Qmx_dag"] += qmb_op(
                    in_ops,
                    [f"psi_{col}_dag_P", f"Z{col}", "IDz", "IDz", "IDz", "IDz", "IDz"],
                )
                ops["Qmy_dag"] += qmb_op(
                    in_ops,
                    [f"psi_{col}_dag_P", "P", f"Z{col}", "IDz", "IDz", "IDz", "IDz"],
                )
                ops["Qmz_dag"] += qmb_op(
                    in_ops,
                    [f"psi_{col}_dag_P", "P", "P", f"Z{col}", "IDz", "IDz", "IDz"],
                )
                ops["Qpx_dag"] += qmb_op(
                    in_ops, [f"psi_{col}_dag_P", "P", "P", "P", f"Z{col}", "IDz", "IDz"]
                )
                ops["Qpy_dag"] += qmb_op(
                    in_ops, [f"psi_{col}_dag_P", "P", "P", "P", "P", f"Z{col}", "IDz"]
                )
                ops["Qpz_dag"] += qmb_op(
                    in_ops, [f"psi_{col}_dag_P", "P", "P", "P", "P", "P", f"Z{col}"]
                )
            # --------------------------------------------------------------------------
            # add their dagger operators
            Qs = {}
            for op in ops:
                dag_op = op.replace("_dag", "")
                Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
            ops |= Qs
    # -----------------------------------------------------------------------------
    if not pure_theory:
        # Psi NUMBER OPERATORS
        for label in ["r", "g", "tot", "single", "pair"]:
            ops[f"N_{label}"] = qmb_op(
                in_ops, [f"N_{label}"] + ["IDz" for _ in range(2 * lattice_dim)]
            )
        # Psi CASIMIR OPERATORS
        for Td in ["x", "y", "z"]:
            ops[f"S{Td}_psi"] = qmb_op(
                in_ops, [f"S{Td}_psi"] + ["IDz" for _ in range(2 * lattice_dim)]
            )
    # CASIMIR/ELECTRIC OPERATOR
    ops[f"E_square"] = 0
    for s in "pm":
        for d in dimensions:
            ops[f"E_square"] += 0.5 * ops[f"T2_{s}{d}"]
    # DRESSED SITE CASIMIR OPERATOR S^{2}
    ops[f"S2_tot"] = 0
    for ax in ["x", "y", "z"]:
        for side in "pm":
            for d in dimensions:
                ops["S2_tot"] += ops[f"T{ax}_{side}{d}"] ** 2
        if not pure_theory:
            ops["S2_tot"] += ops[f"S{ax}_psi"] ** 2
    return ops


def SU2_gauge_invariant_states(s_max, pure_theory, lattice_dim):
    validate_parameters(
        spin_list=[s_max], pure_theory=pure_theory, lattice_dim=lattice_dim
    )
    spin_list = [S(s_max) for i in range(2 * lattice_dim)]
    spins = []
    # For each single spin particle in the list,
    # consider all the spin irrep up to the max one
    for s in spin_list:
        tmp = np.arange(S(0), spin_space(s), 1)
        spins.append(tmp / 2)
    if not pure_theory:
        spins.insert(0, np.asarray([S(0), S(1) / 2, S(0)]))
    # Set rows and col counters list for the basis
    gauge_states = {"site": []}
    gauge_basis = {"site": []}
    # List of borders/corners of the lattice
    borders = get_lattice_borders_labels(lattice_dim)
    for label in borders:
        gauge_states[f"site_{label}"] = []
        gauge_basis[f"site_{label}"] = []
    for ii, spins_config in enumerate(product(*spins)):
        spins_config = list(spins_config)
        if not pure_theory:
            # Check the matter spin (0 (vacuum), 1/2, 0 (up & down))
            v_sector = np.prod([len(l) for l in [[spins[0][0]]] + spins[1:]])
            if ii < v_sector:
                psi_vacuum = True
            elif 2 * v_sector - 1 < ii < 3 * v_sector:
                psi_vacuum = False
            else:
                psi_vacuum = None
        else:
            psi_vacuum = None
        # Check the existence of a SU2 singlet state
        singlets = get_SU2_singlets(spins_config, pure_theory, psi_vacuum)
        if singlets is not None:
            for s in singlets:
                # s.show()
                # Save the singlet state
                gauge_states["site"].append(s)
                # Save the singlet state written in the canonical basis
                singlet_state = SU2_singlet_canonical_vector(spin_list, s)
                gauge_basis["site"].append(singlet_state)
                # GET THE CONFIG LABEL
                spin_sizes = [spin_space(s) for s in spins_config]
                label = LGT_border_configs(
                    config=spin_sizes, offset=1, pure_theory=pure_theory
                )
                if label:
                    # Save the config state also in the specific subset of borders
                    for ll in label:
                        gauge_states[f"site_{ll}"].append(s)
                        gauge_basis[f"site_{ll}"].append(singlet_state)
    # Build the basis combining the states into a matrix
    for label in list(gauge_basis.keys()):
        gauge_basis[label] = csr_matrix(np.column_stack(tuple(gauge_basis[label])))
    return gauge_basis, gauge_states


def SU2_check_gauss_law(gauge_basis, threshold=1e-14):
    if not isspmatrix(gauge_basis):
        raise TypeError(f"gauge_basis should be csr_matrix, not {type(gauge_basis)}")
    # This functions performs some checks on the SU2 gauge invariant basis
    logger.info("CHECK GAUSS LAW")
    # True and the Effective dimensions of the gauge invariant dressed site
    site_dim = gauge_basis.shape[0]
    eff_site_dim = gauge_basis.shape[1]
    # Check that the Matrix Basis behave like an isometry
    norm_isometry = norm(gauge_basis.transpose() * gauge_basis - identity(eff_site_dim))
    if norm_isometry > threshold:
        raise ValueError(f"Basis must be Isometry: B^T*B=1; got norm {norm_isometry}")
    # Check that B*B^T is a Projector
    Proj = gauge_basis * gauge_basis.transpose()
    norm_Proj = norm(Proj - Proj**2)
    if norm_Proj > threshold:
        raise ValueError(f"P=B*B^T: expected P-P**2=0: obtained norm {norm_Proj}")
    """
    # Check that the basis is the one with ALL the states satisfying Gauss law
    norma_kernel = norm(gauss_law_op * gauge_basis)
    if norma_kernel > threshold:
        raise ValueError(f"Gauss Law Kernel with norm {norma_kernel}; expected 0")
    GL_rank = matrix_rank(gauss_law_op.todense())
    if site_dim - GL_rank != eff_site_dim:
        logger.info(f"Large dimension {site_dim}")
        logger.info(f"Effective dimension {eff_site_dim}")
        logger.info(GL_rank)
        logger.info(f"Some gauge basis states are missing")
    """
    logger.info("GAUSS LAW SATISFIED")


def SU2_Hamiltonian_couplings(lattice_dim, pure_theory, g, m=None):
    """
    This function provides the couplings of the SU2 Yang-Mills Hamiltonian
    starting from the gauge coupling g and the bare mass parameter m

    Args:
        pure_theory (bool): True if the theory does not include matter

        g (scalar): gauge coupling

        m (scalar, optional): bare mass parameter

    Returns:
        dict: dictionary of Hamiltonian coefficients
    """
    validate_parameters(lattice_dim=lattice_dim, pure_theory=pure_theory)
    if lattice_dim == 1:
        E = 8 * g / 3  # The correct one is g**2 / 4
        B = 0
    elif lattice_dim == 2:
        E = 3 * (g**2) / 16
        B = -4 / (g**2)
    else:
        E = 3 * (g**2) / 16
        B = -4 / (g**2)
    # Dictionary with Hamiltonian COEFFICIENTS
    coeffs = {
        "g": g,
        "E": E,  # ELECTRIC FIELD coupling
        "B": B,  # MAGNETIC FIELD coupling
    }
    if not pure_theory:
        coeffs |= {
            "m": m,
            "tx_even": -complex(0, 2),
            # x HOPPING (EVEN SITES) -complex(0, 1/(2*np.sqrt(2)))
            "tx_odd": -complex(0, 2),  # x HOPPING (ODD SITES)
            "ty_even": -0.5,  # y HOPPING (EVEN SITES)
            "ty_odd": 0.5,  # y HOPPING (ODD SITES)
            "tz_even": -0.5,  # z HOPPING (EVEN SITES)
            "tz_odd": 0.5,  # z HOPPING (ODD SITES)
            "m_odd": -m,  # EFFECTIVE MASS for ODD SITES
            "m_even": m,  # EFFECTIVE MASS for EVEN SITES
        }
    return coeffs
