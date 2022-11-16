import numpy as np
from scipy.sparse import isspmatrix_csr
from simsio import logger
from tools import zig_zag
from modeling import local_op

__all__ = ["LocalTerm"]


class LocalTerm:
    def __init__(self, operator, op_name):
        # CHECK ON TYPES
        if not isspmatrix_csr(operator):
            raise TypeError(f"operator should be CSR_MATRIX, not {type(operator)}")
        if not isinstance(op_name, str):
            raise TypeError(f"op_name should be a STRING, not a {type(op_name)}")
        self.Op = operator
        self.Op_name = op_name

    def get_Hamiltonian(self, nx, ny, strength, mask=None):
        # CHECK ON TYPES
        if not np.isscalar(nx) and not isinstance(nx, int):
            raise TypeError(f"nx must be SCALAR & INTEGER, not {type(nx)}")
        if not np.isscalar(ny) and not isinstance(ny, int):
            raise TypeError(f"ny must be SCALAR & INTEGER, not {type(ny)}")
        if not np.isscalar(strength):
            raise TypeError(f"strength must be SCALAR not a {type(strength)}")
        # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
        n = nx * ny
        self.strength = strength
        # LOCAL HAMILTONIAN
        H_Local = 0
        for ii in range(n):
            if mask is not None:
                self.mask = mask
                x, y = zig_zag(nx, ny, ii)
                if mask[x, y]:
                    H_Local = H_Local + local_op(self.Op, ii + 1, n)
            else:
                H_Local = H_Local + local_op(self.Op, ii + 1, n)
        self.Ham = self.strength * H_Local
