import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import norm
from ed_lgt.tools import commutator as comm
from ed_lgt.tools import anti_commutator as anti_comm
from ed_lgt.tools import check_matrix as check_m
from .SU2_singlets import m_values, spin_space
from .SU2_singlets import SU2_generators


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

    def construct_rishons(self):
        # Define dictionary for operators
        self.ops = {}
        # ---------------------------------------------------------------------
        # Define SU2 Parity and Identity
        P_diag = []
        for s_size in range(self.largest_s_size):
            P_diag += [((-1) ** s_size)] * (s_size + 1)
        self.ops["P"] = diags(P_diag, 0, self.shape)
        self.ops["IDz"] = identity(self.size, dtype=float)
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
        print("CHECK SU2 RISHON ALGEBRA")
        check_m(2 * comm(self.ops[f"Zr"], self.ops["Tx"]), self.ops[f"Zg"])
        check_m(2 * comm(self.ops[f"Zg"], self.ops["Tx"]), self.ops[f"Zr"])
        check_m(
            comm(self.ops[f"Zr"], self.ops["Ty"]), -complex(0, 0.5) * self.ops[f"Zg"]
        )
        check_m(
            comm(self.ops[f"Zg"], self.ops["Ty"]), complex(0, 0.5) * self.ops[f"Zr"]
        )
        check_m(2 * comm(self.ops[f"Zr"], self.ops["Tz"]), self.ops[f"Zr"])
        check_m(2 * comm(self.ops[f"Zg"], self.ops["Tz"]), -self.ops[f"Zg"])
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
        print("SU2 RISHON ALGEBRA SATISFIED")

    @staticmethod
    def chi_function(s, color, m):
        """This function computes the factor for SU2 rishon entries"""
        if not np.isscalar(s):
            raise TypeError(f"s must be scalar (int or real), not {type(s)}")
        if not np.isscalar(m):
            raise TypeError(f"m must be scalar (int or real), not {type(m)}")
        if color == "r":
            return np.sqrt((s + m + 1) / np.sqrt((2 * s + 1) * (2 * s + 2)))
        elif color == "g":
            return np.sqrt((s - m + 1) / np.sqrt((2 * s + 1) * (2 * s + 2)))
        else:
            raise ValueError(f"color can be only 'r' (red) or 'g'(green): got {color}")
