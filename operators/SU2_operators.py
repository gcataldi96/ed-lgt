import numpy as np
from itertools import product
from sympy import S
from numpy.linalg import matrix_rank
from scipy.sparse import csr_matrix, diags, identity, block_diag, isspmatrix, kron
from scipy.sparse.linalg import norm
from simsio import logger
from modeling import qmb_operator as qmb_op
from tools import commutator as comm
from tools import anti_commutator as anti_comm
from tools import check_matrix
from .SU2_singlets import (
    m_values,
    spin_space,
    get_spin_operators,
    SU2_singlet,
    get_SU2_singlets,
)

__all__ = [
    "SU2_Hamiltonian_couplings",
    "SU2_dressed_site_operators",
    "SU2_gauge_basis",
]


def chi_function(s, color, m):
    if not np.isscalar(s):
        raise TypeError(f"s must be scalar (int or real), not {type(s)}")
    if not np.isscalar(m):
        raise TypeError(f"m must be scalar (int or real), not {type(m)}")
    "This function computes the factor for SU2 rishon entries"
    if color == "up":
        return np.sqrt((s + m + 1) / np.sqrt((2 * s + 1) * (2 * s + 2)))
    elif color == "down":
        return np.sqrt((s - m + 1) / np.sqrt((2 * s + 1) * (2 * s + 2)))
    else:
        raise ValueError(f"color can be only 'up' or 'down': got {color}")


def SU2_generators(s, matter=False):
    if not np.isscalar(s):
        raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(s)}")
    if not isinstance(matter, bool):
        raise TypeError(f"matter should be a BOOL, not a {type(matter)}")
    """
    This function computes the generators of the group for the SU2 Lattice Gauge Theory: 
    [Tz, Tp=T+, Tm=T-, Tx, Ty, T2=Casimir] in any arbitrary spin-s representation

    Args:
        s (scalar, real): spin value, assumed to be integer or semi-integer
        matter (bool, optional): 
            if true, it yields the SU2 generators of flavorless SU(2) 1/2 particles
            if false it yields the SU2 generators of Rishon modes in the LGT

    Returns:
        dict: dictionary with the spin matrices
    """
    largest_s_size = int(2 * s + 1)
    matrices = {"Tz": [0], "Tp": [0], "T2": [0]}
    if not matter:
        for s_size in range(1, largest_s_size):
            spin = s_size / 2
            spin_ops = get_spin_operators(spin)
            for op in ["z", "p", "2"]:
                matrices[f"T{op}"] += [spin_ops[f"S{op}"]]
        SU2_gen = {}
        for op in ["Tz", "Tp", "T2"]:
            SU2_gen[op] = block_diag(tuple(matrices[op]), format="csr")
        SU2_gen["Tm"] = SU2_gen["Tp"].transpose()
        SU2_gen["Tx"] = 0.5 * (SU2_gen["Tp"] + SU2_gen["Tm"])
        SU2_gen["Ty"] = complex(0, -0.5) * (SU2_gen["Tp"] - SU2_gen["Tm"])
        SU2_gen["T4"] = SU2_gen["T2"] ** 2
        # Introduce the effective Casimir operator on which a single rishon is acting
        gen_size = SU2_gen["T2"].shape[0]
        ID = identity(gen_size)
        SU2_gen["T2_root"] = 0.5 * (csr_matrix(ID + 4 * SU2_gen["T2"]).sqrt() - ID)
    else:
        spin_ops = get_spin_operators(1 / 2)
        for op in ["z", "p", "2"]:
            matrices[f"T{op}"] += [spin_ops[f"S{op}"]]
            matrices[f"T{op}"] += [0]
        SU2_gen = {}
        for op in ["z", "p", "2"]:
            SU2_gen[f"S{op}_psi"] = block_diag(tuple(matrices[f"T{op}"]), format="csr")
        SU2_gen["Sm_psi"] = SU2_gen["Sp_psi"].transpose()
        SU2_gen["Sx_psi"] = 0.5 * (SU2_gen["Sp_psi"] + SU2_gen["Sm_psi"])
        SU2_gen["Sy_psi"] = complex(0, -0.5) * (SU2_gen["Sp_psi"] - SU2_gen["Sm_psi"])
    return SU2_gen


class SU2_Rishon:
    def __init__(self, spin, color):
        if not np.isscalar(spin):
            raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(spin)}")
        # Maximal spin rep and color
        self.s = spin
        self.color = color
        # Compute the dimension of the rishon mode
        self.largest_s_size = spin_space(spin)
        self.size = np.sum([s_size for s_size in range(1, self.largest_s_size + 1)])
        self.shape = (self.size, self.size)
        # List of diagonal entries
        self.entries = []
        # List of diagonals
        self.diagonals = []
        # Starting diagonals of the s=0 case
        if color == "up":
            self.diag = 1
        elif color == "down":
            self.diag = 2
        else:
            raise ValueError(f"color can be only 'up' or 'down': got {color}")
        # Number of zeros at the beginning of the diagonals.
        # It increases with the spin representation
        self.in_zeros = 0

    def construct_rishon(self):
        for s_size in range(self.largest_s_size - 1):
            # Obtain spin
            spin = s_size / 2
            # Compute chi & P coefficientes
            sz_diag = m_values(spin)
            chi_diags = (np.vectorize(chi_function)(spin, self.color, sz_diag)).tolist()
            # Fill the diags with zeros according to the lenght of the diag
            out_zeros = self.size - len(chi_diags) - self.diag - self.in_zeros
            chi_diags = [0] * self.in_zeros + chi_diags + [0] * out_zeros
            # Append the diags
            self.entries.append(chi_diags)
            self.diagonals.append(self.diag)
            # Update the diagonals and the number of initial zeros
            self.diag += 1
            self.in_zeros += s_size + 1
        # Compose the Rishon operators
        self.zeta = diags(self.entries, self.diagonals, self.shape)

    def correlated_rishon_operators(self):
        ops = {}
        ops[f"zb_{self.color}"] = self.zeta
        # Define the Hermitian conjugate
        ops[f"zb_{self.color}_dag"] = ops[f"zb_{self.color}"].transpose()
        # Define the opposite p Rishon
        if self.color == "up":
            new_color = "down"
            ops[f"za_{new_color}"] = -ops[f"zb_{self.color}"].transpose()
        elif self.color == "down":
            new_color = "up"
            ops[f"za_{new_color}"] = ops[f"zb_{self.color}"].transpose()
        else:
            raise ValueError(f"color can be only 'up' or 'down': got {self.color}")
        ops[f"za_{new_color}_dag"] = ops[f"za_{new_color}"].transpose()
        # Define the Parity
        P_diag = []
        for s_size in range(self.largest_s_size):
            P_diag += [((-1) ** s_size)] * (s_size + 1)
        ops["P_z"] = diags(P_diag, 0, self.shape)
        ops["IDz"] = identity(self.size, dtype=float)
        # Useful operators for corner operators
        ops[f"zb_{self.color}_P"] = ops[f"zb_{self.color}"] * ops["P_z"]
        ops[f"za_{new_color}_P"] = ops[f"za_{new_color}"] * ops["P_z"]
        ops[f"P_zb_{self.color}_dag"] = ops["P_z"] * ops[f"zb_{self.color}_dag"]
        ops[f"P_za_{new_color}_dag"] = ops["P_z"] * ops[f"za_{new_color}_dag"]
        # SU2 generators
        ops |= SU2_generators(self.s, matter=False)
        return ops


def SU2_rishon_operators(s):
    if not np.isscalar(s):
        raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(s)}")
    """
    This function computes the SU2 the Rishon modes adopted
    for the SU2 Lattice Gauge Theory for the chosen spin-s irrepresentation
    Args:
        s (scalar, real): spin value, assumed to be integer or semi-integer

    Returns:
        dict: dictionary with the rishon operators and the parity
    """
    ops = {}
    for color in ["up", "down"]:
        zeta = SU2_Rishon(s, color)
        zeta.construct_rishon()
        ops |= zeta.correlated_rishon_operators()
        ops |= SU2_generators(s)
    return ops


def check_SU2_rishon_algebra(s):
    # Checks that the SU(2) rishon modes satisfies the SU2 algebra
    if not np.isscalar(s):
        raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(s)}")
    # Define the Rishon operators
    ops = SU2_rishon_operators(s)
    logger.info("CHECK SU2 RISHON ALGEBRA")
    # For each type of rishon operators (+ a, - b) check the following properties of the algebra
    for ii, kind in enumerate(["a", "b"]):
        sign = (-1) ** ii
        if check_matrix(2 * comm(ops[f"z{kind}_up"], ops["Tx"]), ops[f"z{kind}_down"]):
            raise ValueError("ERROR")
        if check_matrix(2 * comm(ops[f"z{kind}_down"], ops["Tx"]), ops[f"z{kind}_up"]):
            raise ValueError("ERROR")
        if check_matrix(
            2 * comm(ops[f"z{kind}_up"], ops["Ty"]),
            -complex(0, 1) * ops[f"z{kind}_down"],
        ):
            raise ValueError("ERROR")
        if check_matrix(
            2 * comm(ops[f"z{kind}_down"], ops["Ty"]),
            complex(0, 1) * ops[f"z{kind}_up"],
        ):
            raise ValueError("ERROR")
        if check_matrix(2 * comm(ops[f"z{kind}_up"], ops["Tz"]), ops[f"z{kind}_up"]):
            raise ValueError("ERROR")
        if check_matrix(
            2 * comm(ops[f"z{kind}_down"], ops["Tz"]), -ops[f"z{kind}_down"]
        ):
            raise ValueError("ERROR")
        for color in ["up", "down"]:
            # CHECK RISHON MODES TO BEHAVE LIKE FERMIONS (anticommute with parity)
            if norm(anti_comm(ops[f"z{kind}_{color}"], ops["P_z"])) > 1e-15:
                raise ValueError(
                    f"z{kind}_{color} is a Fermion and must anticommute with P"
                )
            # CHECK THE ACTION OF THE RISHONS ON THE CASIMIR OPERATOR
            if check_matrix(
                2 * comm(ops["T2_root"], ops[f"z{kind}_{color}"]),
                sign * ops[f"z{kind}_{color}"],
            ):
                raise ValueError(
                    f"z{kind}_{color} has a wrong action onto the Casimir operator"
                )


# CONSTRUCT THE DRESSED SITE BASIS
def inner_site_operators(s, pure_theory=False):
    if not np.isscalar(s):
        raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(s)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    # Define the SU2 Rishon operators
    ops = SU2_rishon_operators(s)
    # Check rishon Algebra
    check_SU2_rishon_algebra(s)
    if not pure_theory:
        # --------------------------------------------------------------------------
        # Define the generic MATTER FIELD OPERATORS for both the su2 colors
        # The distinction between the two colors will be specified when considering
        # the dressed site operators.
        ops["psi"] = diags(np.array([1], dtype=float), offsets=1, shape=(2, 2))
        ops["psi_dag"] = ops["psi"].transpose()
        ops["P_psi"] = diags(np.array([1, -1], dtype=float), offsets=0, shape=(2, 2))
        ops["N"] = ops["psi_dag"] * ops["psi"]
        ops["ID"] = diags(np.array([1, 1], dtype=float), offsets=0, shape=(2, 2))
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
        # Spin matrices for MATTER FIELD OPERATORS
        ops |= SU2_generators(1 / 2, matter=True)
    return ops


def SU2_dressed_site_operators(s, pure_theory=False):
    if not np.isscalar(s):
        raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(s)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    # Get inner site operators in the full or pure theory
    in_ops = inner_site_operators(s, pure_theory=False)
    # Dictionary for dressed site operators
    ops = {}
    # ---------------------------------------------------------------------------------
    # Rishon NUMBER OPERATORS for the Electric Field
    for op in ["T2", "T4"]:
        ops[f"{op}_mx"] = qmb_op(in_ops, [op, "IDz", "IDz", "IDz"])
        ops[f"{op}_my"] = qmb_op(in_ops, ["IDz", op, "IDz", "IDz"])
        ops[f"{op}_px"] = qmb_op(in_ops, ["IDz", "IDz", op, "IDz"])
        ops[f"{op}_py"] = qmb_op(in_ops, ["IDz", "IDz", "IDz", op])
    if not pure_theory:
        # Rishon NUMBER OPERATORS in case of the Full Theory
        for op in ["T2", "T4"]:
            for sd in ["mx", "my", "px", "py"]:
                ops[f"{op}_{sd}"] = kron(in_ops["ID_psi"], ops[f"{op}_{sd}"])
    # ---------------------------------------------------------------------------------
    # CASIMIR/ELECTRIC OPERATOR
    ops[f"E_square"] = 0
    for sd in ["mx", "my", "px", "py"]:
        ops[f"E_square"] += 0.5 * ops[f"T2_{sd}"]
    # ---------------------------------------------------------------------------------
    # DRESSED SITE CASIMIR OPERATOR S^{2}
    if pure_theory:
        ops[f"S2_tot"] = 0
        for d in ["x", "y", "z"]:
            ops[f"S{d}_tot"] = qmb_op(in_ops, [f"T{d}", "IDz", "IDz", "IDz"])
            ops[f"S{d}_tot"] += qmb_op(in_ops, ["IDz", f"T{d}", "IDz", "IDz"])
            ops[f"S{d}_tot"] += qmb_op(in_ops, ["IDz", "IDz", f"T{d}", "IDz"])
            ops[f"S{d}_tot"] += qmb_op(in_ops, ["IDz", "IDz", "IDz", f"T{d}"])
            ops["S2_tot"] += ops[f"S{d}_tot"] ** 2
    else:
        ops[f"S2_tot"] = 0
        for d in ["x", "y", "z"]:
            ops[f"S{d}_tot"] = qmb_op(in_ops, [f"S{d}_psi", "IDz", "IDz", "IDz", "IDz"])
            ops[f"S{d}_tot"] += qmb_op(in_ops, ["ID_psi", f"T{d}", "IDz", "IDz", "IDz"])
            ops[f"S{d}_tot"] += qmb_op(in_ops, ["ID_psi", "IDz", f"T{d}", "IDz", "IDz"])
            ops[f"S{d}_tot"] += qmb_op(in_ops, ["ID_psi", "IDz", "IDz", f"T{d}", "IDz"])
            ops[f"S{d}_tot"] += qmb_op(in_ops, ["ID_psi", "IDz", "IDz", "IDz", f"T{d}"])
            ops["S2_tot"] += ops[f"S{d}_tot"] ** 2
        # Matter Casimir Operator
        ops[f"S2_psi"] = qmb_op(in_ops, [f"S2_psi", "IDz", "IDz", "IDz", "IDz"])
    # ---------------------------------------------------------------------------------
    # CORNER OPERATORS
    for l1, l2 in product(["a", "b"], repeat=2):
        for corner in ["px,py", "py,mx", "mx,my", "my,px"]:
            ops[f"C{l1}{l2}_{corner}"] = 0
    for l1, l2 in product(["a", "b"], repeat=2):
        for s in ["up", "down"]:
            ops[f"C{l1}{l2}_px,py"] += -qmb_op(
                in_ops, ["IDz", "IDz", f"z{l1}_{s}_P", f"z{l2}_{s}_dag"]
            )
            ops[f"C{l1}{l2}_py,mx"] += qmb_op(
                in_ops, [f"P_z{l1}_{s}_dag", "P_z", "P_z", f"z{l2}_{s}"]
            )
            ops[f"C{l1}{l2}_mx,my"] += qmb_op(
                in_ops, [f"z{l1}_{s}_P", f"z{l2}_{s}_dag", "IDz", "IDz"]
            )
            ops[f"C{l1}{l2}_my,px"] += qmb_op(
                in_ops, ["IDz", f"z{l1}_{s}_P", f"z{l2}_{s}_dag", "IDz"]
            )
    if not pure_theory:
        for l1, l2 in product(["a", "b"], repeat=2):
            for corner in ["px,py", "py,mx", "mx,my", "my,px"]:
                ops[f"C{l1}{l2}_{corner}"] = kron(
                    in_ops["ID_psi"], ops[f"C{l1}{l2}_{corner}"]
                )
    # ---------------------------------------------------------------------------------
    if not pure_theory:
        # HOPPING OPERATORS
        for ll in ["a", "b"]:
            for sd in ["mx", "my", "px", "py"]:
                ops[f"Q{ll}_{sd}_dag"] = 0
        for ll in ["a", "b"]:
            for s in ["up", "down"]:
                ops[f"Q{ll}_mx_dag"] += qmb_op(
                    in_ops, [f"psi_{s}_dag", f"z{ll}_{s}", "IDz", "IDz", "IDz"]
                )
                ops[f"Q{ll}_my_dag"] += qmb_op(
                    in_ops, [f"psi_{s}_dag", "P_z", f"z{ll}_{s}", "IDz", "IDz"]
                )
                ops[f"Q{ll}_px_dag"] += qmb_op(
                    in_ops, [f"psi_{s}_dag", "P_z", "P_z", f"z{ll}_{s}", "IDz"]
                )
                ops[f"Q{ll}_py_dag"] += qmb_op(
                    in_ops, [f"psi_{s}_dag", "P_z", "P_z", "P_z", f"z{ll}_{s}"]
                )
        # add DAGGER operators
        Qs = {}
        for op in ops:
            dag_op = op.replace("_dag", "")
            Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
        ops |= Qs
        # Psi NUMBER OPERATORS
        for label in ["up", "down", "tot", "single", "pair"]:
            ops[f"N_{label}"] = qmb_op(
                in_ops, [f"N_{label}", "IDz", "IDz", "IDz", "IDz"]
            )
    return ops


# SU2 BASIS
def SU2_border_configs(config, pure_theory=False):
    """
    This function fixes the value of the SU2 irrep = 0 on the borders
    of 2D lattices with open boundary conditions (has_obc=True).

    Args:
        config (list of spins): spin configuration of internal rishons
        in the single dressed site basis, ordered as follows:

        pure_theory (bool):
        if True, then config=[j_mx, j_my, j_px, j_py]
        if False, then config=[j_matter, j_mx, j_my, j_px, j_py]

    Returns:
        list of strings: list of configs corresponding to a border/corner
        of the 2D lattice with 0 SU(2)-irrep
    """
    if not pure_theory:
        config = config[1:]
    # List of the size of the Hilbert spaces
    tmp = [spin_space(s) for s in config]
    label = []
    if tmp[0] == 1:
        label.append("mx")
    if tmp[1] == 1:
        label.append("my")
    if tmp[2] == 1:
        label.append("px")
    if tmp[3] == 1:
        label.append("py")
    if (tmp[0] == 1) and (tmp[1] == 1):
        label.append("mx_my")
    if (tmp[0] == 1) and (tmp[3] == 1):
        label.append("mx_py")
    if (tmp[1] == 1) and (tmp[2] == 1):
        label.append("my_px")
    if (tmp[2] == 1) and (tmp[3] == 1):
        label.append("px_py")
    return label


def SU2_gauge_basis(s_max, pure_theory=True, dim=2):
    if not np.isscalar(s_max):
        raise TypeError(f"s_max must be SCALAR & (semi)INTEGER, not {type(s_max)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    spin_list = [S(s_max) for i in range(2 * dim)]
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
    # Run over lattice borders:
    borders = ["mx", "my", "px", "py", "mx_my", "mx_py", "my_px", "px_py"]
    for label in borders:
        gauge_states[f"site_{label}"] = []
        gauge_basis[f"site_{label}"] = []
    for ii, spins_config in enumerate(product(*spins)):
        spins_config = list(spins_config)
        if not pure_theory:
            # Check the matter spin (0 (vacuum),1/2,0 (up & down))
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
                singlet_state = canonical_vector(spin_list, s)
                gauge_basis["site"].append(singlet_state)
                # GET THE CONFIG LABEL
                label = SU2_border_configs(spins_config)
                if label:
                    # Save the config state also in the specific subset of borders
                    for ll in label:
                        gauge_states[f"site_{ll}"].append(s)
                        gauge_basis[f"site_{ll}"].append(singlet_state)
    # Build the basis combining the states into a matrix
    for label in list(gauge_basis.keys()):
        gauge_basis[label] = csr_matrix(np.column_stack(tuple(gauge_basis[label])))
    return gauge_basis, gauge_states


def get_list(spins, pure_theory=True):
    if not isinstance(spins, list):
        raise TypeError(f"spins must be a list, not {type(spins)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    spin_list = []
    # For each single spin particle in the list,
    # consider all the spin irrep up to the max one
    for s in spins:
        tmp = np.arange(S(0), spin_space(s), 1)
        spin_list.append(tmp / 2)
    m_list = []
    j_list = []
    for s in spin_list:
        m_set = []
        j_set = []
        for ss in s:
            m_set += list(m_values(ss))
            j_set += [ss for i in m_values(ss)]
        m_list.append(m_set)
        j_list.append(j_set)
    if not pure_theory:
        j_list.insert(0, ["V", S(1) / 2, S(1) / 2, "P"])
        m_list.insert(0, [S(0), S(1) / 2, -S(1) / 2, S(0)])
    return j_list, m_list


def canonical_vector(spin_list, singlet):
    if not isinstance(spin_list, list):
        raise TypeError(f"spin_list must be a list, not {type(spin_list)}")
    if not isinstance(singlet, SU2_singlet):
        raise TypeError(f"singlet is not {SU2_singlet}, but {type(singlet)}")
    # Acquire the list of matter and rishons
    j_list, m_list = get_list(spin_list, singlet.pure_theory)
    len_basis = len(list(product(*m_list)))
    state = np.zeros(len_basis)
    for ii, (j_config, m_config) in enumerate(zip(product(*j_list), product(*m_list))):
        JM_config = list(j_config) + list(m_config)
        if JM_config in singlet.JM_configs:
            index = singlet.JM_configs.index(JM_config)
            state[ii] = singlet.CG_values[index]
    # Check the norm of the state in the canonical basis
    state_norm = np.sum(state**2)
    if np.abs(state_norm - 1) > 1e-10:
        raise ValueError(f"The state is not normalized: norm {state_norm}")
    return state


# CHECK GAUSS LAW
def check_SU2_gauss_law(basis, gauss_law_op, threshold=1e-14):
    if not isspmatrix(basis):
        raise TypeError(f"basis should be csr_matrix, not {type(basis)}")
    if not isspmatrix(gauss_law_op):
        raise TypeError(f"gauss_law_op shoul be csr_matrix, not {type(gauss_law_op)}")
    # This functions performs some checks on the SU2 gauge invariant basis
    logger.info("CHECK GAUSS LAW")
    # True and the Effective dimensions of the gauge invariant dressed site
    site_dim = basis.shape[0]
    eff_site_dim = basis.shape[1]
    # Check that the Matrix Basis behave like an isometry
    norm_isometry = norm(basis.transpose() * basis - identity(eff_site_dim))
    if norm_isometry > threshold:
        raise ValueError(f"Basis must be Isometry: B^T*B=1; got norm {norm_isometry}")
    # Check that B*B^T is a Projector
    Proj = basis * basis.transpose()
    norm_Proj = norm(Proj - Proj**2)
    if norm_Proj > threshold:
        raise ValueError(f"P=B*B^T: expected P-P**2=0: obtained norm {norm_Proj}")
    # Check that the basis is the one with ALL the states satisfying Gauss law
    norma_kernel = norm(gauss_law_op * basis)
    if norma_kernel > threshold:
        raise ValueError(f"Gauss Law Kernel with norm {norma_kernel}; expected 0")
    GL_rank = matrix_rank(gauss_law_op.todense())
    if site_dim - GL_rank != eff_site_dim:
        logger.info(f"Large dimension {site_dim}")
        logger.info(f"Effective dimension {eff_site_dim}")
        logger.info(GL_rank)
        logger.info(f"Some gauge basis states are missing")
    logger.info("GAUSS LAW SATISFIED")


def SU2_Hamiltonian_couplings(pure_theory, g, m=None):
    E = 3 * (g**2) / 16  # ELECTRIC FIELD
    B = -4 / (g**2)  # MAGNETIC FIELD
    # DICTIONARY WITH MODEL COEFFICIENTS
    coeffs = {
        "g": g,
        "E": E,  # ELECTRIC FIELD coupling
        "B": B,  # MAGNETIC FIELD coupling
    }
    if pure_theory:
        coeffs["eta"] = 10 * max(E, np.abs(B))
    if not pure_theory:
        coeffs["eta"] = 10 * max(E, np.abs(B), m)
        coeffs |= {
            "m": m,
            "tx_even": -0.5j,  # HORIZONTAL HOPPING (EVEN SITES)
            "tx_odd": -0.5j,  # HORIZONTAL HOPPING (ODD SITES)
            "ty_even": -0.5,  # VERTICAL HOPPING (EVEN SITES)
            "ty_odd": 0.5,  # VERTICAL HOPPING (ODD SITES)
            "m_odd": -m,  # EFFECTIVE MASS for ODD SITES
            "m_even": m,  # EFFECTIVE MASS for EVEN SITES
        }
    return coeffs


# %%
"""
# CHECK THE GAUGE BASIS
pure = False
j1 = 1 / 2
# Compute the SU2 gauge invariant basis
basis, states = SU2_gauge_basis(S(j1), pure_theory=pure, dim=2)
for s in states["site"]:
    s.show()
logger.info(basis["site"].shape)
# Acquire dressed site operators
ops = SU2_dressed_site_operators(j1, pure_theory=pure)
# Check gauss law
check_SU2_gauss_law(basis["site"], ops["S2_tot"])
"""
