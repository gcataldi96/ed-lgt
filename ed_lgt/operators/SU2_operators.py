import numpy as np
from itertools import product
from sympy import S
from numpy.linalg import matrix_rank
from scipy.sparse import csr_matrix, identity, isspmatrix, kron
from scipy.sparse.linalg import norm
from ed_lgt.modeling import qmb_operator as qmb_op
from ed_lgt.modeling import get_lattice_borders_labels, LGT_border_configs
from .SU2_singlets import get_SU2_singlets, SU2_singlet_canonical_vector
from .spin_operators import spin_space, SU2_generators, get_spin_operators, m_values
from .SU2_rishons import SU2_Rishon, SU2_Rishon_gen
from .bose_fermi_operators import fermi_operators as SU2_matter_operators
from ed_lgt.tools import validate_parameters
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "SU2_dressed_site_operators",
    "SU2_gen_dressed_site_operators",
    "SU2_check_gauss_law",
    "SU2_gauge_invariant_states",
]


def SU2_dressed_site_operators(spin, pure_theory, lattice_dim, background=False):
    validate_parameters(
        spin_list=[spin], pure_theory=pure_theory, lattice_dim=lattice_dim
    )
    # Lattice directions
    dimensions = "xyz"[:lattice_dim]
    # Get SU2 rishon operators
    in_ops = SU2_Rishon(spin).ops
    in_ops |= SU2_generators(spin)
    if not pure_theory:
        in_ops |= SU2_matter_operators(has_spin=True, colors=True)
        in_ops |= SU2_generators(1 / 2, matter=True)
    for op in in_ops.keys():
        in_ops[op] = csr_matrix(in_ops[op])
    # Dictionary for dressed site operators
    ops = {}
    if lattice_dim == 1:
        # T generators for electric term
        for op in ["T2", "P", "Tx", "Ty", "Tz"]:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "Iz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["Iz", op])
        # Operator fot the Polyakov loop
        ops["U"] = 0
        for col in "rg":
            ops["U"] += qmb_op(in_ops, [f"Z{col}_dag_P", f"Z{col}"]) / np.sqrt(2)
        if not pure_theory:
            # Update Electric operators
            for op in ops.keys():
                ops[op] = kron(in_ops["ID_psi"], ops[op])
            # Add Hopping operators
            for side in "pm":
                ops[f"Q{side}x_dag"] = 0
            for col in "rg":
                ops["Qpx_dag"] += qmb_op(in_ops, [f"psi_{col}_dag_P", "P", f"Z{col}"])
                ops["Qmx_dag"] += qmb_op(in_ops, [f"psi_{col}_dag_P", f"Z{col}", "Iz"])
            # add their dagger operators
            Qs = {}
            for op in ops:
                dag_op = op.replace("_dag", "")
                Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
            ops |= Qs
    elif lattice_dim == 2:
        # T generators for electric term
        for op in ["T2", "P", "Tx", "Ty", "Tz"]:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "Iz", "Iz", "Iz"])
            ops[f"{op}_my"] = qmb_op(in_ops, ["Iz", op, "Iz", "Iz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["Iz", "Iz", op, "Iz"])
            ops[f"{op}_py"] = qmb_op(in_ops, ["Iz", "Iz", "Iz", op])
        # Corner Operators
        for corner in ["px,py", "py,mx", "mx,my", "my,px"]:
            ops[f"C_{corner}"] = 0
        for col in ["r", "g"]:
            ops["C_px,py"] += qmb_op(in_ops, ["Iz", "Iz", f"Z{col}_P", f"Z{col}_dag"])
            ops["C_py,mx"] += qmb_op(in_ops, [f"P_Z{col}_dag", "P", "P", f"Z{col}"])
            ops["C_mx,my"] += qmb_op(in_ops, [f"Z{col}_P", f"Z{col}_dag", "Iz", "Iz"])
            ops["C_my,px"] += qmb_op(in_ops, ["Iz", f"Z{col}_P", f"Z{col}_dag", "Iz"])
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
                    in_ops, [f"psi_{col}_dag_P", f"Z{col}", "Iz", "Iz", "Iz"]
                )
                ops["Qmy_dag"] += qmb_op(
                    in_ops, [f"psi_{col}_dag_P", "P", f"Z{col}", "Iz", "Iz"]
                )
                ops["Qpx_dag"] += qmb_op(
                    in_ops, [f"psi_{col}_dag_P", "P", "P", f"Z{col}", "Iz"]
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
        for op in ["T2", "P", "Tx", "Ty", "Tz"]:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "Iz", "Iz", "Iz", "Iz", "Iz"])
            ops[f"{op}_my"] = qmb_op(in_ops, ["Iz", op, "Iz", "Iz", "Iz", "Iz"])
            ops[f"{op}_mz"] = qmb_op(in_ops, ["Iz", "Iz", op, "Iz", "Iz", "Iz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["Iz", "Iz", "Iz", op, "Iz", "Iz"])
            ops[f"{op}_py"] = qmb_op(in_ops, ["Iz", "Iz", "Iz", "Iz", op, "Iz"])
            ops[f"{op}_pz"] = qmb_op(in_ops, ["Iz", "Iz", "Iz", "Iz", "Iz", op])
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
                ["Iz", "Iz", "Iz", f"Z{col}_P", f"Z{col}_dag", "Iz"],
            )
            ops["C_py,mx"] += qmb_op(
                in_ops, [f"P_Z{col}_dag", "P", "P", "P", f"Z{col}", "Iz"]
            )
            ops["C_mx,my"] += qmb_op(
                in_ops,
                [f"Z{col}_P", f"Z{col}_dag", "Iz", "Iz", "Iz", "Iz"],
            )
            ops["C_my,px"] += qmb_op(
                in_ops, ["Iz", f"Z{col}_P", "P", f"Z{col}_dag", "Iz", "Iz"]
            )
            # --------------------------------------------------------------------------
            # XZ Plane
            ops["C_px,pz"] += qmb_op(
                in_ops, ["Iz", "Iz", "Iz", f"Z{col}_P", "P", f"Z{col}_dag"]
            )
            ops["C_pz,mx"] += qmb_op(
                in_ops, [f"P_Z{col}_dag", "P", "P", "P", "P", f"Z{col}"]
            )
            ops["C_mx,mz"] += qmb_op(
                in_ops, [f"Z{col}_P", "P", f"Z{col}_dag", "Iz", "Iz", "Iz"]
            )
            ops["C_mz,px"] += qmb_op(
                in_ops, ["Iz", "Iz", f"Z{col}_P", f"Z{col}_dag", "Iz", "Iz"]
            )
            # --------------------------------------------------------------------------
            # YZ Plane
            ops["C_py,pz"] += qmb_op(
                in_ops, ["Iz", "Iz", "Iz", "Iz", f"Z{col}_P", f"Z{col}_dag"]
            )
            ops["C_pz,my"] += qmb_op(
                in_ops, ["Iz", f"P_Z{col}_dag", "P", "P", "P", f"Z{col}"]
            )
            ops["C_my,mz"] += qmb_op(
                in_ops, ["Iz", f"Z{col}_P", f"Z{col}_dag", "Iz", "Iz", "Iz"]
            )
            ops["C_mz,py"] += qmb_op(
                in_ops, ["Iz", "Iz", f"Z{col}_P", "P", f"Z{col}_dag", "Iz"]
            )
        # THETA TERM
        ops["EzC_px,py"] = 0
        ops["ExC_py,pz"] = 0
        ops["EyC_px,pz"] = 0
        sigma_ops = get_spin_operators(0.5)
        for nu in "xyz":
            Ez = ops[f"T{nu}_pz"] + ops[f"T{nu}_mz"]
            Ey = ops[f"T{nu}_py"] + ops[f"T{nu}_my"]
            Ex = ops[f"T{nu}_px"] + ops[f"T{nu}_mx"]
            for ii, c1 in enumerate("rg"):
                for jj, c2 in enumerate("rg"):
                    factor = sigma_ops[f"S{nu}"].todense()[ii, jj]
                    ops["EzC_px,py"] += qmb_op(
                        in_ops,
                        ["Iz", "Iz", "Iz", f"Z{c1}_P", f"Z{c2}_dag", "Iz"],
                    ) @ (factor * Ez)
                    ops["EyC_px,pz"] += qmb_op(
                        in_ops, ["Iz", "Iz", "Iz", f"Z{c1}_P", "P", f"Z{c2}_dag"]
                    ) @ (factor * Ey)
                    ops["ExC_py,pz"] += qmb_op(
                        in_ops, ["Iz", "Iz", "Iz", "Iz", f"Z{c1}_P", f"Z{c2}_dag"]
                    ) @ (factor * Ex)
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
                    [f"psi_{col}_dag_P", f"Z{col}", "Iz", "Iz", "Iz", "Iz", "Iz"],
                )
                ops["Qmy_dag"] += qmb_op(
                    in_ops,
                    [f"psi_{col}_dag_P", "P", f"Z{col}", "Iz", "Iz", "Iz", "Iz"],
                )
                ops["Qmz_dag"] += qmb_op(
                    in_ops,
                    [f"psi_{col}_dag_P", "P", "P", f"Z{col}", "Iz", "Iz", "Iz"],
                )
                ops["Qpx_dag"] += qmb_op(
                    in_ops, [f"psi_{col}_dag_P", "P", "P", "P", f"Z{col}", "Iz", "Iz"]
                )
                ops["Qpy_dag"] += qmb_op(
                    in_ops, [f"psi_{col}_dag_P", "P", "P", "P", "P", f"Z{col}", "Iz"]
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
        for label in ["r", "g", "tot", "single", "pair", "zero"]:
            ops[f"N_{label}"] = qmb_op(
                in_ops, [f"N_{label}"] + ["Iz" for _ in range(2 * lattice_dim)]
            )
    # CASIMIR/ELECTRIC OPERATOR
    ops[f"E_square"] = 0
    for s in "pm":
        for d in dimensions:
            ops[f"E_square"] += 0.5 * ops[f"T2_{s}{d}"]
    if background > 0:
        bg_len = 0
        j_bg_set = np.arange(0, spin_space(background), 1) / 2
        for irrep in j_bg_set:
            bg_len += len(m_values(irrep))
        for op in ops.keys():
            ops[op] = kron(identity(bg_len), ops[op])
        if pure_theory:
            id_list = ["Iz" for _ in range(2 * lattice_dim)]
        else:
            id_list = ["ID_psi"] + ["Iz" for _ in range(2 * lattice_dim)]
        ops["bg"] = qmb_op(in_ops, ["T2"] + id_list)
    return ops


def SU2_gen_dressed_site_operators(spin, pure_theory, lattice_dim, background=False):
    validate_parameters(
        spin_list=[spin], pure_theory=pure_theory, lattice_dim=lattice_dim
    )
    # Lattice directions
    dimensions = "xyz"[:lattice_dim]
    # Get SU2 rishon operators
    in_ops = SU2_Rishon_gen(spin).ops
    in_ops |= SU2_generators(spin)
    # Add SU2 matter operators / generators
    if not pure_theory:
        in_ops |= SU2_matter_operators(has_spin=True, colors=True)
        in_ops |= SU2_generators(1 / 2, matter=True)
    for op in in_ops.keys():
        in_ops[op] = csr_matrix(in_ops[op])
    # Dictionary for dressed site operators
    ops = {}
    if lattice_dim == 1:
        # T generators for electric term
        for op in ["T2", "P"]:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "Iz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["Iz", op])
        if not pure_theory:
            # Update Electric operators
            for op in ops.keys():
                ops[op] = kron(in_ops["ID_psi"], ops[op])
            # Add Hopping operators
            # ------------------------------------------------------------------
            ops["Q1_mx_dag"] = qmb_op(in_ops, ["psi_r_dag_P", "Zg_dag", "Iz"]) - qmb_op(
                in_ops, ["psi_g_dag_P", "Zr_dag", "Iz"]
            )
            ops["Q1_px_dag"] = qmb_op(in_ops, ["psi_r_dag_P", "P", "Zg_dag"]) - qmb_op(
                in_ops, ["psi_g_dag_P", "P", "Zr_dag"]
            )
            # ------------------------------------------------------------------
            ops["Q2_mx_dag"] = qmb_op(in_ops, ["psi_r_dag_P", "Zr", "Iz"]) + qmb_op(
                in_ops, ["psi_g_dag_P", "Zg", "Iz"]
            )
            ops["Q2_px_dag"] = qmb_op(in_ops, ["psi_r_dag_P", "P", "Zr"]) + qmb_op(
                in_ops, ["psi_g_dag_P", "P", "Zg"]
            )
            # add their dagger operators
            Qs = {}
            for op in ops:
                dag_op = op.replace("_dag", "")
                Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
            ops |= Qs
    elif lattice_dim == 2:
        # T generators for electric term
        for op in ["T2", "P"]:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "Iz", "Iz", "Iz"])
            ops[f"{op}_my"] = qmb_op(in_ops, ["Iz", op, "Iz", "Iz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["Iz", "Iz", op, "Iz"])
            ops[f"{op}_py"] = qmb_op(in_ops, ["Iz", "Iz", "Iz", op])
        # Corner Operators
        for l1, l2 in product(["A", "B"], repeat=2):
            for corner in ["px,py", "py,mx", "mx,my", "my,px"]:
                ops[f"C{l1}{l2}_{corner}"] = 0
            for s in ["r", "g"]:
                ops[f"C{l1}{l2}_px,py"] += qmb_op(
                    in_ops, ["Iz", "Iz", f"Z{l1}_{s}_P", f"Z{l2}_{s}_dag"]
                )
                ops[f"C{l1}{l2}_py,mx"] += qmb_op(
                    in_ops, [f"P_Z{l2}_{s}_dag", "P", "P", f"Z{l1}_{s}"]
                )
                ops[f"C{l1}{l2}_mx,my"] += qmb_op(
                    in_ops, [f"Z{l1}_{s}_P", f"Z{l2}_{s}_dag", "Iz", "Iz"]
                )
                ops[f"C{l1}{l2}_my,px"] += qmb_op(
                    in_ops, ["Iz", f"Z{l1}_{s}_P", f"Z{l2}_{s}_dag", "Iz"]
                )
        if not pure_theory:
            # Update Electric and Corner operators
            for op in ops.keys():
                ops[op] = kron(in_ops["ID_psi"], ops[op])
            # Add Hopping operators
            # ------------------------------------------------------------------
            ops["Q1_mx_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "Zg_dag", "Iz", "Iz", "Iz"]
            ) - qmb_op(in_ops, ["psi_g_dag_P", "Zr_dag", "Iz", "Iz", "Iz"])
            ops["Q1_my_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "P", "Zg_dag", "Iz", "Iz"]
            ) - qmb_op(in_ops, ["psi_g_dag_P", "P", "Zr_dag", "Iz", "Iz"])
            ops["Q1_px_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "P", "P", "Zg_dag", "Iz"]
            ) - qmb_op(in_ops, ["psi_g_dag_P", "P", "P", "Zr_dag", "Iz"])
            ops["Q1_py_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "P", "P", "P", "Zg_dag"]
            ) - qmb_op(in_ops, ["psi_g_dag_P", "P", "P", "P", "Zr_dag"])
            # ------------------------------------------------------------------
            ops["Q2_mx_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "Zr", "Iz", "Iz", "Iz"]
            ) + qmb_op(in_ops, ["psi_g_dag_P", "Zg", "Iz", "Iz", "Iz"])
            ops["Q2_my_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "P", "Zr", "Iz", "Iz"]
            ) + qmb_op(in_ops, ["psi_g_dag_P", "P", "Zg", "Iz", "Iz"])
            ops["Q2_px_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "P", "P", "Zr", "Iz"]
            ) + qmb_op(in_ops, ["psi_g_dag_P", "P", "P", "Zg", "Iz"])
            ops["Q2_py_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "P", "P", "P", "Zr"]
            ) + qmb_op(in_ops, ["psi_g_dag_P", "P", "P", "P", "Zg"])
            # add their dagger operators
            Qs = {}
            for op in ops:
                dag_op = op.replace("_dag", "")
                Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
            ops |= Qs
    elif lattice_dim == 3:
        # T generators for electric term
        for op in ["T2", "P"]:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "Iz", "Iz", "Iz", "Iz", "Iz"])
            ops[f"{op}_my"] = qmb_op(in_ops, ["Iz", op, "Iz", "Iz", "Iz", "Iz"])
            ops[f"{op}_mz"] = qmb_op(in_ops, ["Iz", "Iz", op, "Iz", "Iz", "Iz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["Iz", "Iz", "Iz", op, "Iz", "Iz"])
            ops[f"{op}_py"] = qmb_op(in_ops, ["Iz", "Iz", "Iz", "Iz", op, "Iz"])
            ops[f"{op}_pz"] = qmb_op(in_ops, ["Iz", "Iz", "Iz", "Iz", "Iz", op])
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
        for l1, l2 in product(["A", "B"], repeat=2):
            for corner in corner_list:
                ops[f"C{l1}{l2}_{corner}"] = 0
            for s in ["r", "g"]:
                # XY Plane
                ops[f"C{l1}{l2}_px,py"] += qmb_op(
                    in_ops,
                    ["Iz", "Iz", "Iz", f"Z{l1}_{s}_P", f"Z{l2}_{s}_dag", "Iz"],
                )
                ops[f"C{l1}{l2}_py,mx"] += qmb_op(
                    in_ops, [f"P_Z{l2}_{s}_dag", "P", "P", "P", f"Z{l1}_{s}", "Iz"]
                )
                ops[f"C{l1}{l2}_mx,my"] += qmb_op(
                    in_ops,
                    [f"Z{l1}_{s}_P", f"Z{l2}_{s}_dag", "Iz", "Iz", "Iz", "Iz"],
                )
                ops[f"C{l1}{l2}_my,px"] += qmb_op(
                    in_ops, ["Iz", f"Z{l1}_{s}_P", "P", f"Z{l2}_{s}_dag", "Iz", "Iz"]
                )
                # XZ Plane
                ops[f"C{l1}{l2}_px,pz"] += qmb_op(
                    in_ops,
                    ["Iz", "Iz", "Iz", f"Z{l1}_{s}_P", "P", f"Z{l2}_{s}_dag"],
                )
                ops[f"C{l1}{l2}_pz,mx"] += qmb_op(
                    in_ops, [f"P_Z{l2}_{s}_dag", "P", "P", "P", "P", f"Z{l1}_{s}"]
                )
                ops[f"C{l1}{l2}_mx,mz"] += qmb_op(
                    in_ops,
                    [f"Z{l1}_{s}_P", "P", f"Z{l2}_{s}_dag", "Iz", "Iz", "Iz"],
                )
                ops[f"C{l1}{l2}_mz,px"] += qmb_op(
                    in_ops,
                    ["Iz", "Iz", f"Z{l1}_{s}_P", f"Z{l2}_{s}_dag", "Iz", "Iz"],
                )
                # YZ Plane
                ops[f"C{l1}{l2}_py,pz"] += qmb_op(
                    in_ops,
                    ["Iz", "Iz", "Iz", "Iz", f"Z{l1}_{s}_P", f"Z{l2}_{s}_dag"],
                )
                ops[f"C{l1}{l2}_pz,my"] += qmb_op(
                    in_ops, ["Iz", f"P_Z{l2}_{s}_dag", "P", "P", "P", f"Z{l1}_{s}"]
                )
                ops[f"C{l1}{l2}_my,mz"] += qmb_op(
                    in_ops,
                    ["Iz", f"Z{l1}_{s}_P", f"Z{l2}_{s}_dag", "Iz", "Iz", "Iz"],
                )
                ops[f"C{l1}{l2}_mz,py"] += qmb_op(
                    in_ops,
                    ["Iz", "Iz", f"Z{l1}_{s}_P", "P", f"Z{l2}_{s}_dag", "Iz"],
                )
        if not pure_theory:
            # Update Electric and Corner operators
            for op in ops.keys():
                ops[op] = kron(in_ops["ID_psi"], ops[op])
            # Add Hopping operators
            ops["Q1_mx_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "Zg_dag", "Iz", "Iz", "Iz", "Iz", "Iz"]
            ) - qmb_op(in_ops, ["psi_g_dag_P", "Zr_dag", "Iz", "Iz", "Iz", "Iz", "Iz"])
            ops["Q1_my_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "P", "Zg_dag", "Iz", "Iz", "Iz", "Iz"]
            ) - qmb_op(in_ops, ["psi_g_dag_P", "P", "Zr_dag", "Iz", "Iz", "Iz", "Iz"])
            ops["Q1_mz_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "P", "P", "Zg_dag", "Iz", "Iz", "Iz"]
            ) - qmb_op(in_ops, ["psi_g_dag_P", "P", "P", "Zr_dag", "Iz", "Iz", "Iz"])
            ops["Q1_px_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "P", "P", "P", "Zg_dag", "Iz", "Iz"]
            ) - qmb_op(in_ops, ["psi_g_dag_P", "P", "P", "P", "Zr_dag", "Iz", "Iz"])
            ops["Q1_py_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "P", "P", "P", "P", "Zg_dag", "Iz"]
            ) - qmb_op(in_ops, ["psi_g_dag_P", "P", "P", "P", "P", "Zr_dag", "Iz"])
            ops["Q1_pz_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "P", "P", "P", "P", "P", "Zg_dag"]
            ) - qmb_op(in_ops, ["psi_g_dag_P", "P", "P", "P", "P", "P", "Zr_dag"])
            # --------------------------------------------------------------------------
            ops["Q2_mx_dag"] = qmb_op(
                in_ops,
                ["psi_r_dag_P", "Zr", "Iz", "Iz", "Iz", "Iz", "Iz"],
            ) + qmb_op(
                in_ops,
                ["psi_g_dag_P", "Zg", "Iz", "Iz", "Iz", "Iz", "Iz"],
            )
            ops["Q2_my_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "P", "Zr", "Iz", "Iz", "Iz", "Iz"]
            ) + qmb_op(
                in_ops,
                ["psi_g_dag_P", "P", "Zg", "Iz", "Iz", "Iz", "Iz"],
            )
            ops["Q2_mz_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "P", "P", "Zr", "Iz", "Iz", "Iz"]
            ) + qmb_op(
                in_ops,
                ["psi_g_dag_P", "P", "P", "Zg", "Iz", "Iz", "Iz"],
            )
            ops["Q2_px_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "P", "P", "P", "Zr", "Iz", "Iz"]
            ) + qmb_op(in_ops, ["psi_g_dag_P", "P", "P", "P", "Zg", "Iz", "Iz"])
            ops["Q2_py_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "P", "P", "P", "P", "Zr", "Iz"]
            ) + qmb_op(in_ops, ["psi_g_dag_P", "P", "P", "P", "P", "Zg", "Iz"])
            ops["Q2_pz_dag"] = qmb_op(
                in_ops, ["psi_r_dag_P", "P", "P", "P", "P", "P", "Zr"]
            ) + qmb_op(in_ops, ["psi_g_dag_P", "P", "P", "P", "P", "P", "Zg"])
            # add their dagger operators
            Qs = {}
            for op in ops:
                dag_op = op.replace("_dag", "")
                Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
            ops |= Qs
    # -----------------------------------------------------------------------------
    if not pure_theory:
        # Psi NUMBER OPERATORS
        for label in ["r", "g", "tot", "single", "pair", "zero"]:
            ops[f"N_{label}"] = qmb_op(
                in_ops, [f"N_{label}"] + ["Iz" for i in range(2 * lattice_dim)]
            )
    # CASIMIR/ELECTRIC OPERATOR
    ops[f"E_square"] = 0
    for s in "pm":
        for d in dimensions:
            ops[f"E_square"] += 0.5 * ops[f"T2_{s}{d}"]
    if background > 0:
        bg_len = 0
        j_bg_set = np.arange(0, spin_space(background), 1) / 2
        for irrep in j_bg_set:
            bg_len += len(m_values(irrep))
        for op in ops.keys():
            ops[op] = kron(identity(bg_len), ops[op])
        if pure_theory:
            id_list = ["Iz" for _ in range(2 * lattice_dim)]
        else:
            id_list = ["ID_psi"] + ["Iz" for _ in range(2 * lattice_dim)]
        ops["bg"] = qmb_op(in_ops, ["T2"] + id_list)
    return ops


def SU2_gauge_invariant_states(s_max, pure_theory, lattice_dim, background=0):
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
    if background != 0:
        bg_tmp = np.arange(S(0), spin_space(background), 1)
        spins.insert(0, bg_tmp / 2)
    vind = 0 if not background else 1
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
        j2 = [int(2 * dd) for dd in spins_config]
        total = sum(j2)
        # 2) parity
        if total & 1:
            continue
        # 3) single‐largest test
        max_j2 = max(j2)
        if max_j2 > total - max_j2:
            continue
        if not pure_theory:
            # Check the matter spin (0 (vacuum), 1/2, 0 (up & down))
            matter_sector = (ii // np.prod([len(l) for l in spins[vind + 1 :]])) % 3
            if matter_sector == 0:
                psi_vacuum = True
            elif matter_sector == 2:
                psi_vacuum = False
            else:
                psi_vacuum = None
        else:
            psi_vacuum = None
        # Check the existence of a SU2 singlet state
        singlets = get_SU2_singlets(spins_config, pure_theory, psi_vacuum, background)
        if singlets is not None:
            for s in singlets:
                # Save the singlet state
                gauge_states["site"].append(s)
                # Save the singlet state written in the canonical basis
                singlet_state = SU2_singlet_canonical_vector(spin_list, s, background)
                gauge_basis["site"].append(singlet_state)
                # GET THE CONFIG LABEL
                spin_sizes = [spin_space(ss) for ss in spins_config[vind:]]
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


def SU2_check_gauss_law(
    gauge_basis: csr_matrix, gauss_law_op: csr_matrix, threshold=1e-14
):
    if not isspmatrix(gauge_basis):
        raise TypeError(f"gauge_basis should be csr_matrix, not {type(gauge_basis)}")
    if not isspmatrix(gauss_law_op):
        raise TypeError(f"gauss_law_op shoul be csr_matrix, not {type(gauss_law_op)}")
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
    logger.info("GAUSS LAW SATISFIED")
