import numpy as np
from scipy.sparse import diags, identity, csr_matrix
from scipy.sparse.linalg import norm
from ed_lgt.tools import commutator as comm
from ed_lgt.tools import anti_commutator as anti_comm
from ed_lgt.tools import check_matrix as check_m
from .spin_operators import m_values, spin_space, SU2_generators
import logging

logger = logging.getLogger(__name__)

__all__ = ["SU2_Rishon"]


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
            P_diag += [((-1) ** s_size)] * (s_size + 1)
        self.ops["P"] = diags(P_diag, 0, self.shape)
        self.ops["IDz"] = identity(self.size, dtype=float)
        # In the special case of s=1/2, the shape of rishon is simpler
        if self.s == 1 / 2:
            self.ops["Zr"] = csr_matrix(([1, 1], ([0, 2], [1, 0])), shape=(3, 3))
            self.ops["Zg"] = csr_matrix(([1, -1], ([0, 1], [2, 0])), shape=(3, 3))
            for color in "rg":
                self.ops[f"Z{color}_dag"] = self.ops[f"Z{color}"].transpose()
                # Useful operators for corner operators
                self.ops[f"Z{color}_P"] = self.ops[f"Z{color}"] * self.ops["P"]
                self.ops[f"P_Z{color}_dag"] = self.ops["P"] * self.ops[f"Z{color}_dag"]
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
