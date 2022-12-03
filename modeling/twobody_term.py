import numpy as np
from scipy.sparse import isspmatrix_csr, isspmatrix, csr_matrix
from tools import zig_zag, inverse_zig_zag
from .qmb_operations import two_body_op
from simsio import logger

__all__ = ["TwoBodyTerm2D"]


class TwoBodyTerm2D:
    def __init__(self, axis, op_list, op_name_list):
        # CHECK ON TYPES
        if not isinstance(axis, str):
            raise TypeError(f"axis should be a STRING, not a {type(axis)}")
        if not isinstance(op_list, list):
            raise TypeError(f"op_list should be a list, not a {type(op_list)}")
        else:
            for ii, op in enumerate(op_list):
                if not isspmatrix_csr(op):
                    raise TypeError(
                        f"op_list[{ii}] should be a CSR_MATRIX, not {type(op)}"
                    )
        if not isinstance(op_name_list, list):
            raise TypeError(
                f"op_name_list should be a list, not a {type(op_name_list)}"
            )
        else:
            for ii, name in enumerate(op_name_list):
                if not isinstance(name, str):
                    raise TypeError(
                        f"op_name_list[{ii}] should be a STRING, not {type(name)}"
                    )
        self.axis = axis
        if axis == "x":
            self.op_list = op_list
            self.Left = op_list[0]
            self.Right = op_list[1]
        elif axis == "y":
            self.op_list = op_list
            self.Bottom = op_list[0]
            self.Top = op_list[1]
        else:
            raise ValueError(f"{axis} can be only x or y")

    def get_Hamiltonian(
        self, lvals, strength, has_obc=True, add_dagger=False, mask=None
    ):
        # CHECK ON TYPES
        if not isinstance(lvals, list):
            raise TypeError(f"lvals should be a list, not a {type(lvals)}")
        else:
            for ii, ll in enumerate(lvals):
                if not isinstance(ll, int):
                    raise TypeError(f"lvals[{ii}] should be INTEGER, not {type(ll)}")
        if not np.isscalar(strength):
            raise TypeError(f"strength must be SCALAR, not {type(strength)}")
        if not isinstance(has_obc, bool):
            raise TypeError(f"has_obc must be a BOOL, not a {type(has_obc)}")
        if not isinstance(add_dagger, bool):
            raise TypeError(f"add_dagger must be a BOOL, not a {type(add_dagger)}")
        # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
        nx = lvals[0]
        ny = lvals[1]
        n = nx * ny
        # Hamiltonian
        H_twobody = 0
        # Run over all the single lattice sites, ordered with the ZIG ZAG CURVE
        for ii in range(n):
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
            elif self.axis == "y":
                if y < ny - 1:
                    sites_list = [ii + 1, ii + nx + 1]
                else:
                    # PERIODIC BOUNDARY CONDITIONS
                    if not has_obc:
                        jj = inverse_zig_zag(nx, ny, x, 0)
                        sites_list = [ii + 1, jj + 1]
                    else:
                        continue
            if mask is not None:
                if mask[x, y]:
                    # Add the term to the Hamiltonian
                    H_twobody += two_body_op(self.op_list, sites_list, n)
            else:
                # Add the term to the Hamiltonian
                H_twobody += strength * two_body_op(self.op_list, sites_list, n)
        if not isspmatrix(H_twobody):
            H_twobody = csr_matrix(H_twobody)
        if add_dagger:
            H_twobody += csr_matrix(H_twobody.conj().transpose())
        return H_twobody

    def get_expval(self, psi, lvals, has_obc=False):
        # CHECK ON TYPES
        if not isinstance(psi, np.ndarray):
            raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
        if not isinstance(lvals, list):
            raise TypeError(f"lvals should be a list, not a {type(lvals)}")
        else:
            for ii, ll in enumerate(lvals):
                if not isinstance(ll, int):
                    raise TypeError(f"lvals[{ii}] should be INTEGER, not {type(ll)}")
        if not isinstance(has_obc, bool):
            raise TypeError(f"has_obc should be a BOOL, not a {type(has_obc)}")
        # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
        nx = lvals[0]
        ny = lvals[1]
        n = nx * ny
        # Compute the complex_conjugate of the ground state psi
        psi_dag = np.conjugate(psi)
        # Create an array to store the correlator
        self.corr = np.zeros((nx, ny, nx, ny))
        # RUN OVER THE LATTICE SITES
        for ii in range(n):
            x1, y1 = zig_zag(nx, ny, ii)
            for jj in range(n):
                x2, y2 = zig_zag(nx, ny, jj)
                # AVOID SELF CORRELATIONS
                if ii != jj:
                    self.corr[x1, y1, x2, y2] = np.real(
                        np.dot(
                            psi_dag,
                            two_body_op(self.op_list, [ii + 1, jj + 1], n).dot(psi),
                        )
                    )
                else:
                    self.corr[x1, y1, x2, y2] = 0

    def check_link_symm(self, value=1, threshold=1e-10, has_obc=True):
        logger.info(f" CHECK LINK SYMMETRIES")
        # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
        nx = self.corr.shape[0]
        ny = self.corr.shape[1]
        n = nx * ny
        if self.axis == "x":
            for y in range(ny):
                for x in range(nx):
                    if x == nx - 1:
                        if not has_obc:
                            if np.abs(self.corr[x, y, 0, y] - value) > threshold:
                                raise ValueError(
                                    f"W{self.axis}_({x},{y})-({0},{y})={self.corr[x,y,0,y]}: expected {value}"
                                )
                        else:
                            continue
                    else:
                        if np.abs(self.corr[x, y, x + 1, y] - value) > threshold:
                            raise ValueError(
                                f"W{self.axis}_({x},{y})-({x+1},{y})={self.corr[x,y,x+1,y]}: expected {value}"
                            )
        if self.axis == "y":
            for x in range(nx):
                for y in range(ny):
                    if y == ny - 1:
                        if not has_obc:
                            if np.abs(self.corr[x, y, x, 0] - value) > threshold:
                                raise ValueError(
                                    f"W{self.axis}_({x},{y})-({x},{0})={self.corr[x,y,x,0]}: expected {value}"
                                )
                        else:
                            continue
                    else:
                        if np.abs(self.corr[x, y, x, y + 1] - value) > threshold:
                            raise ValueError(
                                f"W{self.axis}_({x},{y})-({x},{y+1})={self.corr[x,y,x,y+1]}: expected {value}"
                            )
        logger.info(f" All the {self.axis} Link Symmetries are satisfied")
