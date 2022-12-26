import numpy as np
from scipy.sparse import isspmatrix_csr
from simsio import logger
from tools import zig_zag
from .qmb_operations import local_op

__all__ = ["LocalTerm2D"]


class LocalTerm2D:
    def __init__(self, operator, op_name):
        # CHECK ON TYPES
        if not isspmatrix_csr(operator):
            raise TypeError(f"operator should be CSR_MATRIX, not {type(operator)}")
        if not isinstance(op_name, str):
            raise TypeError(f"op_name should be a STRING, not a {type(op_name)}")
        self.Op = operator
        self.Op_name = op_name

    def get_Hamiltonian(self, lvals, strength, mask=None):
        # CHECK ON TYPES
        if not isinstance(lvals, list):
            raise TypeError(f"lvals should be a list, not a {type(lvals)}")
        else:
            for ii, ll in enumerate(lvals):
                if not isinstance(ll, int):
                    raise TypeError(f"lvals[{ii}] should be INTEGER, not {type(ll)}")
        if not np.isscalar(strength):
            raise TypeError(f"strength must be SCALAR not a {type(strength)}")
        # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
        nx = lvals[0]
        ny = lvals[1]
        n = nx * ny
        # LOCAL HAMILTONIAN
        H_Local = 0
        for ii in range(n):
            x, y = zig_zag(nx, ny, ii)
            # ADD THE TERM TO THE HAMILTONIAN
            if mask is not None:
                if mask[x, y]:
                    H_Local += local_op(self.Op, ii + 1, n)
            else:
                H_Local += local_op(self.Op, ii + 1, n)
        return strength * H_Local

    def get_loc_expval(self, psi, lvals, staggered=False):
        # CHECK ON TYPES
        if not isinstance(psi, np.ndarray):
            raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
        if not isinstance(lvals, list):
            raise TypeError(f"lvals should be a list, not a {type(lvals)}")
        else:
            for ii, ll in enumerate(lvals):
                if not isinstance(ll, int):
                    raise TypeError(f"lvals[{ii}] should be INTEGER, not {type(ll)}")
        if not isinstance(staggered, bool):
            raise TypeError(f"staggered should be a BOOL, not a {type(staggered)}")
        # PRINT OBSERVABLE NAME
        logger.info(f"-----------------------")
        logger.info(f" {self.Op_name}")
        logger.info(f"----------------------")
        # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
        nx = lvals[0]
        ny = lvals[1]
        n = nx * ny
        # Compute the complex_conjugate of the ground state psi
        psi_dag = np.conjugate(psi)
        # Create an array to store the abg local observable
        self.obs = np.zeros((nx, ny))
        # DEFINE A VECTOR FOR THE STORED VALUES
        if staggered:
            avg_odd = 0
            avg_even = 0
        # RUN OVER THE LATTICE SITES
        for ii in range(n):
            # Given the 1D point on the lattice, get the corresponding (x,y) coords
            x, y = zig_zag(nx, ny, ii)
            # Compute the average value in the site x,y
            self.obs[x, y] = np.real(
                np.dot(psi_dag, local_op(self.Op, ii + 1, n).dot(psi))
            )
            logger.info(f" ({x+1},{y+1}) {format(self.obs[x,y], '.12f')}")
            if staggered:
                if (-1) ** (x + y) < 1:
                    avg_odd += 2 * self.obs[x, y] / n
                else:
                    avg_even += 2 * self.obs[x, y] / n
        if staggered:
            return avg_even, avg_odd
        else:
            return np.sum(self.obs) / n

    def get_fluctuations(self, psi, lvals, staggered=False):
        # CHECK ON TYPES
        if not isinstance(psi, np.ndarray):
            raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
        if not isinstance(lvals, list):
            raise TypeError(f"lvals should be a list, not a {type(lvals)}")
        else:
            for ii, ll in enumerate(lvals):
                if not isinstance(ll, int):
                    raise TypeError(f"lvals[{ii}] should be INTEGER, not {type(ll)}")
        if not isinstance(staggered, bool):
            raise TypeError(f"staggered should be a BOOL, not a {type(staggered)}")
        # PRINT OBSERVABLE NAME
        logger.info(f"-----------------------")
        logger.info(f" {self.Op_name}")
        logger.info(f"----------------------")
        # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
        nx = lvals[0]
        ny = lvals[1]
        n = nx * ny
        # Compute the complex_conjugate of the ground state psi
        psi_dag = np.conjugate(psi)
        # Create array to store fluctuations: (<O^{2}>-<O>^{2})^{1/2}
        self.var = np.zeros((nx, ny))
        # DEFINE A VECTOR FOR THE STORED VALUES
        if staggered:
            var_odd = 0
            var_even = 0
        # RUN OVER THE LATTICE SITES
        for ii in range(n):
            # Given the 1D point on the lattice, get the corresponding (x,y) coords
            x, y = zig_zag(nx, ny, ii)
            # Compute the standard deviation
            self.var[x, y] = (
                np.real(np.dot(psi_dag, (local_op(self.Op, ii + 1, n) ** 2).dot(psi)))
                - self.obs[x, y] ** 2
            )
            if staggered:
                if (-1) ** (x + y) < 1:
                    var_odd += 2 * self.var[x, y] / n
                else:
                    var_even += 2 * self.var[x, y] / n
        if staggered:
            return np.sqrt(var_even), np.sqrt(var_odd)
        else:
            return np.sqrt(np.sum(self.var) / n)

    def check_on_borders(self, border, value=1, threshold=1e-10):
        logger.info(f" CHECK BORDER PENALTIES")
        if border == "mx":
            if np.any(np.abs(self.obs[0, :] - value) > threshold):
                logger.info(self.obs[0, :])
                raise ValueError(f" {border} border penalty not satisfied")
        elif border == "px":
            if np.any(np.abs(self.obs[-1, :] - value) > threshold):
                logger.info(self.obs[-1, :])
                raise ValueError(f" {border} border penalty not satisfied")
        elif border == "my":
            if np.any(np.abs(self.obs[:, 0] - value) > threshold):
                logger.info(self.obs[:, 0])
                raise ValueError(f" {border} border penalty not satisfied")
        elif border == "py":
            if np.any(np.abs(self.obs[:, -1] - value) > threshold):
                logger.info(self.obs[:, -1])
                raise ValueError(f" {border} border penalty not satisfied")
        else:
            raise ValueError(f" border must be in (mx, px, my, py), not {border}")
        logger.info(f" {border}-border penalties are satisfied")
