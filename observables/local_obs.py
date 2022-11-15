import numpy as np

from simsio import logger
from tools import zig_zag
from modeling import local_op

__all__ = ["LocalObs"]

class LocalObs:
    def __init__(self, psi, operator, nx, ny):
        self.operator = operator
        # CHECK ON TYPES
        if not isinstance(psi, np.ndarray):
            raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
        if not np.isscalar(nx) and not isinstance(nx, int):
            raise TypeError(f"nx must be a SCALAR & INTEGER, not a {type(nx)}")
        if not np.isscalar(ny) and not isinstance(ny, int):
            raise TypeError(f"ny must be a SCALAR & INTEGER, not a {type(ny)}")
        # Compute the total number of particles
        n = nx * ny
        self.nx = nx
        self.ny = ny
        # Compute the complex_conjugate of the ground state psi
        psi_dag = np.conjugate(psi)
        # Create an array to store the abg local observable
        self.obs = np.zeros((nx, ny))
        # RUN OVER THE LATTICE SITES
        for ii in range(n):
            # Given the 1D point on the lattice, get the corresponding (x,y) coords
            x, y = zig_zag(nx, ny, ii)
            # Compute the average value in the site x,y
            self.obs[x, y] = np.dot(psi_dag, local_op(operator, ii + 1, n).dot(psi))
            if np.imag(self.obs[x, y]) > 1e-10:
                raise ValueError(f"Local Obs expected to be REAL")
            self.obs[x, y] = np.real(self.obs[x, y])

    def get_avg(self, staggered=False):
        n = self.nx * self.ny
        # DEFINE A VECTOR FOR THE STORED VALUES
        if staggered:
            avg_odd = 0
            avg_even = 0
            for ii in range(n):
                # Given the 1D point on the lattice, get the corresponding (x,y) coords
                x, y = zig_zag(self.nx, self.ny, ii)
                staggered_factor = (-1) ** (x + y)
                if staggered_factor < 1:
                    avg_odd += 2 * self.obs[x, y] / n
                else:
                    avg_even += 2 * self.obs[x, y] / n
            return avg_even, avg_odd
        else:
            return np.sum(self.obs) / n

    def check_border(self, border, value=0, threshold=1e-10):
        if border == "mx":
            if np.any(np.abs(self.operator[0, :] - value) > threshold):
                logger.info(self.operator[0, :])
                raise ValueError(f"{border} border penalty not satisfied")
        elif border == "px":
            if np.any(np.abs(self.operator[-1, :] - value) > threshold):
                logger.info(self.operator[-1, :])
                raise ValueError(f"{border} border penalty not satisfied")
        if border == "my":
            if np.any(np.abs(self.operator[:, 0] - value) > threshold):
                logger.info(self.operator[:, 0])
                raise ValueError(f"{border} border penalty not satisfied")
        elif border == "py":
            if np.any(np.abs(self.operator[:, -1] - value) > threshold):
                logger.info(self.operator[:,-1])
                raise ValueError(f"{border} border penalty not satisfied")
        else:
            raise ValueError(f"border must be in (mx, px, my, py), not {border}")