import numpy as np
from scipy.sparse import isspmatrix_csr, isspmatrix, csr_matrix
from tools import zig_zag, inverse_zig_zag
from modeling import four_body_op

__all__ = ["PlaquetteTerm2D"]


class PlaquetteTerm2D:
    def __init__(self, Op_BL, Op_BR, Op_TL, Op_TR):
        # CHECK ON TYPES
        if not isspmatrix_csr(Op_BL):
            raise TypeError(f"Op_BL should be CSR_MATRIX, not {type(Op_BL)}")
        if not isspmatrix_csr(Op_BR):
            raise TypeError(f"Op_BR should be CSR_MATRIX, not {type(Op_BR)}")
        if not isspmatrix_csr(Op_TL):
            raise TypeError(f"Op_TL should be an CSR_MATRIX, not a {type(Op_TL)}")
        if not isspmatrix_csr(Op_TR):
            raise TypeError(f"Op_TR should be an CSR_MATRIX, not a {type(Op_TR)}")
        self.BL = Op_BL
        self.BR = Op_BR
        self.TL = Op_TL
        self.TR = Op_TR

    def add_Op_names(self, Op_BL_name, Op_BR_name, Op_TL_name, Op_TR_name):
        # CHECK ON TYPES
        if not isinstance(Op_TR_name, str):
            raise TypeError(f"Op_BL_name should be a STRING, not {type(Op_BL_name)}")
        if not isinstance(Op_BR_name, str):
            raise TypeError(f"Op_BR_name should be a STRING, not {type(Op_BR_name)}")
        if not isinstance(Op_TL_name, str):
            raise TypeError(f"Op_TL_name should be a STRING, not a {type(Op_TL_name)}")
        if not isinstance(Op_TR_name, str):
            raise TypeError(f"Op_TR_name should be a STRING, not a {type(Op_TR_name)}")
        self.BL_name = Op_TR_name
        self.BR_name = Op_BR_name
        self.TL_name = Op_TL_name
        self.TR_name = Op_TR_name

    def get_Hamiltonian(self, nx, ny, strength, has_obc=True, add_dagger=False):
        # CHECK ON TYPES
        if not np.isscalar(nx) and not isinstance(nx, int):
            raise TypeError(f"nx must be SCALAR & INTEGER, not {type(nx)}")
        if not np.isscalar(ny) and not isinstance(ny, int):
            raise TypeError(f"ny must be SCALAR & INTEGER, not {type(ny)}")
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
        # Define a list with the Four Operators involved in the Plaquette:
        op_list = [self.BL, self.BR, self.TL, self.TR]
        # Define the Hamiltonian
        self.Ham = 0
        for ii in range(self.n):
            # Compute the corresponding (x,y) coords
            x, y = zig_zag(nx, ny, ii)
            if x < nx - 1 and y < ny - 1:
                # List of Sites where to apply Operators
                sites_list = [ii + 1, ii + 2, ii + nx + 1, ii + nx + 2]
            else:
                if not has_obc:
                    # PERIODIC BOUNDARY CONDITIONS
                    if x < nx - 1 and y == ny - 1:
                        # UPPER BORDER
                        jj = inverse_zig_zag(nx, ny, x, 0)
                        # List of Sites where to apply Operators
                        sites_list = [ii + 1, ii + 2, jj + 1, jj + 2]
                    elif x == nx - 1 and y < ny - 1:
                        # RIGHT BORDER
                        # List of Sites where to apply Operators
                        sites_list = [ii + 1, ii + 2 - nx, ii + nx + 1, ii + 2]
                    else:
                        # UPPER RIGHT CORNER
                        # List of Sites where to apply Operators
                        sites_list = [ii + 1, ii + 2 - nx, nx, 1]
                else:
                    continue
            # Add the Plaquette to the Hamiltonian
            self.Ham = self.Ham + four_body_op(op_list, sites_list, self.n)
        if not isspmatrix(self.Ham):
            self.Ham = csr_matrix(self.Ham)
        if add_dagger:
            self.Ham = self.Ham + csr_matrix(self.Ham.conj().transpose())
        # multiply by the strenght of the Hamiltonian term
        self.Ham = strength * self.Ham
