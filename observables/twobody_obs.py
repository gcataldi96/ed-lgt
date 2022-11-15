import numpy as np
from simsio import logger
from tools import zig_zag
from modeling import two_body_op

__all__ = ["TwoBody_Obs2D"]


class TwoBody_Obs2D:
    def __init__(self, psi, nx, ny, axis, Corr, has_obc=False):
        if not isinstance(psi, np.ndarray):
            raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
        if not np.isscalar(nx) and not isinstance(nx, int):
            raise TypeError(f"nx must be a SCALAR & INTEGER, not a {type(nx)}")
        if not np.isscalar(ny) and not isinstance(ny, int):
            raise TypeError(f"ny must be a SCALAR & INTEGER, not a {type(ny)}")
        if not isinstance(has_obc, bool):
            raise TypeError(f"has_obc should be a BOOL, not a {type(has_obc)}")
        # Compute the total number of particles
        n = nx * ny
        self.nx = nx
        self.ny = ny
        if axis == "x":
            Op_list = [Corr.Left, Corr.Right]
        elif axis == "y":
            Op_list = [Corr.Bottom, Corr.Top]
        else:
            raise ValueError("axis is neither x nor y")
        # Compute the complex_conjugate of the ground state psi
        psi_dag = np.conjugate(psi)
        # Create an array to store the correlator
        self.corr = np.zeros((nx, ny, nx, ny))
        # RUN OVER THE LATTICE SITES
        for ii in range(n):
            x1, y1 = zig_zag(nx, ny, ii)
            for jj in range(n):
                x2, y2 = zig_zag(nx, ny, jj)
                self.corr[x1, y1, x2, y2] = np.real(
                    np.dot(psi_dag, two_body_op(Op_list, [ii + 1, jj + 1], n).dot(psi))
                )

    def check_link_symm(self, axis, value=1, threshold=1e-10, has_obc=True):
        if axis == "x":
            for y in range(self.ny):
                for x in range(self.nx):
                    if x == self.nx - 1:
                        if not has_obc:
                            if np.abs(self.corr[x, y, 0, y] - value) > threshold:
                                logger.info(
                                    f"W{axis}_({x},{y})-({0},{y})={self.corr[x,y,0,y]}"
                                )
                        else:
                            continue
                    else:
                        if np.abs(self.corr[x, y, x + 1, y] - value) > threshold:
                            logger.info(
                                f"W{axis}_({x},{y})-({x+1},{y})={self.corr[x,y,x+1,y]}"
                            )
        if axis == "y":
            for x in range(self.nx):
                for y in range(self.ny):
                    if y == self.ny - 1:
                        if not has_obc:
                            if np.abs(self.corr[x, y, x, 0] - value) > threshold:
                                logger.info(
                                    f"W{axis}_({x},{y})-({x},{0})={self.corr[x,y,x,0]}"
                                )
                        else:
                            continue
                    else:
                        if np.abs(self.corr[x, y, x, y + 1] - value) > threshold:
                            logger.info(
                                f"W{axis}_({x},{y})-({x},{y+1})={self.corr[x,y,x,y+1]}"
                            )
