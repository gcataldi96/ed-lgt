import numpy as np
from scipy.sparse import isspmatrix_csr, isspmatrix, csr_matrix
from tools import zig_zag, inverse_zig_zag
from modeling import two_body_op

__all__ = ["TwoBodyTerm2D"]


class TwoBodyTerm2D:
    def __init__(self, axis, Op1, Op2, Op1_name, Op2_name):
        # CHECK ON TYPES
        if not isinstance(axis, str):
            raise TypeError(f"axis should be a STRING, not a {type(axis)}")
        if not isspmatrix_csr(Op1):
            raise TypeError(f"Op1 should be CSR_MATRIX, not {type(Op1)}")
        if not isspmatrix_csr(Op2):
            raise TypeError(f"Op2 should be CSR_MATRIX, not {type(Op2)}")
        if not isinstance(Op1_name, str):
            raise TypeError(f"Op1_name should be a STRING, not a {type(Op1_name)}")
        if not isinstance(Op2_name, str):
            raise TypeError(f"Op2_name should be a STRING, not a {type(Op2_name)}")
        self.axis = axis
        if axis == "x":
            self.Left = Op1
            self.Right = Op2
        elif axis == "y":
            self.Bottom = Op1
            self.Top = Op2
        else:
            raise ValueError(f"{axis} can be only x or y")

    def get_Hamiltonian(
        self,
        nx,
        ny,
        strength,
        has_obc=True,
        add_dagger=False,
    ):
        # CHECK ON TYPES
        if not np.isscalar(nx) and not isinstance(nx, int):
            raise TypeError(f"nx must be a SCALAR & INTEGER, not a {type(nx)}")
        if not np.isscalar(ny) and not isinstance(ny, int):
            raise TypeError(f"ny must be a SCALAR & INTEGER, not a {type(ny)}")
        if not np.isscalar(strength):
            raise TypeError(f"strength must be SCALAR, not {type(strength)}")
        if not isinstance(has_obc, bool):
            raise TypeError(f"has_obc must be a BOOL, not a {type(has_obc)}")
        if not isinstance(add_dagger, bool):
            raise TypeError(f"add_dagger must be a BOOL, not a {type(add_dagger)}")
        # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        # Make two lists for Single-Site Operators involved in TwoBody Operators
        if self.axis == "x":
            op_list = [self.Left, self.Right]
        elif self.axis == "y":
            op_list = [self.Bottom, self.Top]
        # Hamiltonian
        self.Ham = 0
        # Run over all the single lattice sites, ordered with the ZIG ZAG CURVE
        for ii in range(self.n):
            # Compute the corresponding (x,y) coords
            x, y = zig_zag(nx, ny, ii)
            # HORIZONTAL 2BODY HAMILTONIAN
            if self.axis == "x":
                if x < nx - 1:
                    sites_list = [ii + 1, ii + 2]
                else:
                    # PERIODIC BOUNDARY CONDITIONS
                    if not has_obc:
                        jj = inverse_zig_zag(nx, ny, 0, y)
                        sites_list = [ii + 1, jj + 1]
                    else:
                        continue
            # VERTICAL 2BODY HAMILTONIAN
            elif self.axis == "x":
                if y < ny - 1:
                    sites_list = [ii + 1, ii + nx + 1]
                else:
                    # PERIODIC BOUNDARY CONDITIONS
                    if not has_obc:
                        jj = inverse_zig_zag(nx, ny, x, 0)
                        sites_list = [ii + 1, jj + 1]
                    else:
                        continue
            # Add the term to the Hamiltonian
            self.Ham = self.Ham + two_body_op(op_list, sites_list, self.n)
        if not isspmatrix(self.Ham):
            self.Ham = csr_matrix(self.Ham)
        if add_dagger:
            self.Ham = self.Ham + csr_matrix(self.Ham.conj().transpose())
