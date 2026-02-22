import numpy as np
from scipy.sparse import diags, identity, csr_matrix
from scipy.sparse.linalg import norm
from ed_lgt.tools import (
    commutator as comm,
    anti_commutator as anti_comm,
    check_matrix as check_m,
    validate_parameters,
)
from .spin_operators import m_values, spin_space, SU2_generators
import logging

logger = logging.getLogger(__name__)

__all__ = ["SU2_Rishon", "SU2_Rishon_gen"]


class SU2_Rishon:
    def __init__(self, spin):
        if not np.isscalar(spin):
            raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(spin)}")
        # Maximal spin rep and color
        self.s = spin
        # Compute the dimension of the rishon mode
        self.largest_s_size = spin_space(spin)
        self.size = np.sum([s_size for s_size in range(1, self.largest_s_size + 1)])
        self.shape = (self.size, self.size)
        self.construct_rishons()
        self.SU2_check_rishon_algebra()

    def construct_rishons(self):
        # Define dictionary for operators
        self.ops = {}
        # ---------------------------------------------------------------------
        # Define SU2 Parity and Identity
        P_diag = []
        for s_size in range(self.largest_s_size):
            P_diag += [(-1) ** s_size] * (s_size + 1)
        self.ops["P"] = diags(P_diag, 0, self.shape, dtype=np.float64, format="csr")
        self.ops["Iz"] = identity(self.size, dtype=float)
        # In the special case of s=1/2, the shape of rishon is simpler
        cf = 1 / (2 ** (0.25))
        if self.s == 1 / 2:
            self.ops["Zr"] = cf * csr_matrix(([1, 1], ([0, 2], [1, 0])), shape=(3, 3))
            self.ops["Zg"] = cf * csr_matrix(([1, -1], ([0, 1], [2, 0])), shape=(3, 3))
            for color in "rg":
                self.ops[f"Z{color}_dag"] = self.ops[f"Z{color}"].transpose()
                # Useful operators for corner operators
                self.ops[f"Z{color}_P"] = self.ops[f"Z{color}"] * self.ops["P"]
                self.ops[f"P_Z{color}_dag"] = self.ops["P"] * self.ops[f"Z{color}_dag"]
                self.ops[f"Z{color}_dag_P"] = self.ops[f"Z{color}_dag"] * self.ops["P"]
        # ---------------------------------------------------------------------
        else:
            raise ValueError("For the moment it works only with j=1/2")
        # Add SU2 generators
        self.ops |= SU2_generators(self.s)

    def SU2_check_rishon_algebra(self):
        check_m(2 * comm(self.ops["Zr"], self.ops["Tx"]), self.ops["Zg"])
        check_m(2 * comm(self.ops["Zg"], self.ops["Tx"]), self.ops["Zr"])
        check_m(comm(self.ops["Zr"], self.ops["Ty"]), -complex(0, 0.5) * self.ops["Zg"])
        check_m(comm(self.ops["Zg"], self.ops["Ty"]), complex(0, 0.5) * self.ops["Zr"])
        check_m(2 * comm(self.ops["Zr"], self.ops["Tz"]), self.ops["Zr"])
        check_m(2 * comm(self.ops["Zg"], self.ops["Tz"]), -self.ops["Zg"])
        for color in ["r", "g"]:
            # CHECK RISHON MODES TO BEHAVE LIKE FERMIONS (anticommute with parity)
            if norm(anti_comm(self.ops[f"Z{color}"], self.ops["P"])) > 1e-15:
                raise ValueError(f"Z{color} is a Fermion and must anticommute with P")
            """
            # CHECK THE ACTION OF THE RISHONS ON THE CASIMIR OPERATOR
            if check_m(
                2 * comm(self.ops["T2_root"], self.ops[f"Z{color}"]),
                self.ops[f"Z{color}"],
            ):
                raise ValueError(
                    f"Z{color} has a wrong action onto the Casimir operator"
                )
            """
        logger.info("SU2 RISHON ALGEBRA SATISFIED")


class SU2_Rishon_gen:
    def __init__(self, spin):
        # Check spin
        validate_parameters(spin_list=[spin])
        # Maximal spin rep and color
        self.s = spin
        # Compute the dimension of the rishon mode
        self.largest_s_size = spin_space(spin)
        self.size = np.sum([s_size for s_size in range(1, self.largest_s_size + 1)])
        self.shape = (self.size, self.size)
        self.construct_rishons()
        self.SU2_check_rishon_algebra()

    def construct_rishons(self):
        # Define dictionary for operators
        self.ops = {}
        # ---------------------------------------------------------------------
        # Define SU2 Parity and Identity
        P_diag = []
        for s_size in range(self.largest_s_size):
            P_diag += [(-1) ** s_size] * (s_size + 1)
        self.ops["P"] = diags(P_diag, 0, self.shape, dtype=np.float64, format="csr")
        self.ops["Iz"] = identity(self.size, dtype=float)
        # ---------------------------------------------------------------------
        # Starting diagonals of the s=0 case for the red and green rishon
        initial_diags = [1, 2]
        # Define the Rishons Z_red and Z_green
        for ii, color in enumerate(["r", "g"]):
            # List of diagonal entries
            entries = []
            # List of diagonals
            diagonals = []
            # Number of zeros at the beginning of the diagonals.
            # It increases with the spin representation
            in_zeros = 0
            # Select the initial diag
            diag = initial_diags[ii]
            # Run over the sizes of the sectors with increasing spin
            for s_size in range(self.largest_s_size - 1):
                # Obtain spin
                spin = s_size / 2
                # Compute chi & P coefficients
                sz_diag = m_values(spin)
                chi_diags = (
                    np.vectorize(self.chi_function)(spin, color, sz_diag)
                ).tolist()
                # Fill the diags with zeros according to the len of the diag
                out_zeros = self.size - len(chi_diags) - diag - in_zeros
                chi_diags = [0] * in_zeros + chi_diags + [0] * out_zeros
                # Append the diags
                entries.append(chi_diags)
                diagonals.append(diag)
                # Update the diagonals and the number of initial zeros
                diag += 1
                in_zeros += s_size + 1
            # Compose the Rishon operators
            self.ops[f"Z{color}"] = diags(entries, diagonals, self.shape)
            self.ops[f"Z{color}_dag"] = self.ops[f"Z{color}"].transpose()
            self.ops[f"ZB_{color}"] = self.ops[f"Z{color}"]
            self.ops[f"ZB_{color}_dag"] = self.ops[f"Z{color}_dag"]
            # ---------------------------------------------------------------------
            # Useful operators for corner operators
            self.ops[f"ZB_{color}_P"] = self.ops[f"ZB_{color}"] * self.ops["P"]
            self.ops[f"P_ZB_{color}_dag"] = self.ops["P"] * self.ops[f"ZB_{color}_dag"]
        # Define eventually the two species of rishons
        self.ops["ZA_r"] = self.ops["ZB_g_dag"]
        self.ops["ZA_g"] = -self.ops["ZB_r_dag"]
        for ii, color in enumerate(["r", "g"]):
            self.ops[f"ZA_{color}_dag"] = self.ops[f"ZA_{color}"].transpose()
            # Useful operators for corner operators
            self.ops[f"ZA_{color}_P"] = self.ops[f"ZA_{color}"] * self.ops["P"]
            self.ops[f"P_ZA_{color}_dag"] = self.ops["P"] * self.ops[f"ZA_{color}_dag"]
        # Add SU2 generators
        self.ops |= SU2_generators(self.s)

    def SU2_check_rishon_algebra(self):
        logger.info("CHECK SU2 RISHON ALGEBRA")
        check_m(2 * comm(self.ops["Zr"], self.ops["Tx"]), self.ops["Zg"])
        check_m(2 * comm(self.ops["Zg"], self.ops["Tx"]), self.ops["Zr"])
        check_m(comm(self.ops["Zr"], self.ops["Ty"]), -complex(0, 0.5) * self.ops["Zg"])
        check_m(comm(self.ops["Zg"], self.ops["Ty"]), complex(0, 0.5) * self.ops["Zr"])
        check_m(2 * comm(self.ops["Zr"], self.ops["Tz"]), self.ops["Zr"])
        check_m(2 * comm(self.ops["Zg"], self.ops["Tz"]), -self.ops["Zg"])
        for color in ["r", "g"]:
            # CHECK RISHON MODES TO BEHAVE LIKE FERMIONS (anticommute with parity)
            if norm(anti_comm(self.ops[f"Z{color}"], self.ops["P"])) > 1e-15:
                raise ValueError(f"Z{color} is a Fermion and must anticommute with P")
        logger.info("SU2 RISHON ALGEBRA SATISFIED")

    @staticmethod
    def chi_function(s, color, m):
        """This function computes the factor for SU2 rishon entries"""
        validate_parameters(spin_list=[s], sz_list=[m])
        if color == "r":
            return np.sqrt((s + m + 1) / np.sqrt((2 * s + 1) * (2 * s + 2)))
        elif color == "g":
            return np.sqrt((s - m + 1) / np.sqrt((2 * s + 1) * (2 * s + 2)))
        else:
            raise ValueError(f"color can be only 'r' (red) or 'g'(green): got {color}")
