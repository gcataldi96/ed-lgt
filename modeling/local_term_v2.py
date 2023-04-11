import numpy as np
from scipy.sparse import isspmatrix_csr
from simsio import logger
from tools import zig_zag
from .qmb_operations_v2 import local_op

__all__ = ["LocalTerm2D"]


class LocalTerm2D:
    def __init__(self, operator, op_name, staggered_basis=False, site_basis=None):
        # CHECK ON TYPES
        if not isspmatrix_csr(operator):
            raise TypeError(f"operator should be CSR_MATRIX, not {type(operator)}")
        if not isinstance(op_name, str):
            raise TypeError(f"op_name should be a STRING, not a {type(op_name)}")
        self.op = operator
        self.op_name = op_name
        logger.info(f"local-term {self.op_name}")
        self.stag_basis = staggered_basis
        self.site_basis = site_basis

    def get_Hamiltonian(self, lvals, has_obc, strength, mask=None):
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
            if mask is None:
                mask_conditions = True
            else:
                if mask[x, y] == True:
                    mask_conditions = True
                else:
                    mask_conditions = False
            if mask_conditions:
                H_Local += local_op(
                    operator=self.op,
                    op_1D_site=ii,
                    lvals=lvals,
                    has_obc=has_obc,
                    staggered_basis=self.stag_basis,
                    site_basis=self.site_basis,
                )
        return strength * H_Local

    def get_loc_expval(self, psi, lvals, has_obc, site=None):
        # CHECK ON TYPES
        if not isinstance(psi, np.ndarray):
            raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
        if not isinstance(lvals, list):
            raise TypeError(f"lvals should be a list, not a {type(lvals)}")
        else:
            for ii, ll in enumerate(lvals):
                if not isinstance(ll, int):
                    raise TypeError(f"lvals[{ii}] should be INTEGER, not {type(ll)}")
        # PRINT OBSERVABLE NAME
        logger.info(f"----------------------")
        logger.info(f"{self.op_name}")
        # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
        nx = lvals[0]
        ny = lvals[1]
        n = nx * ny
        # Compute the complex_conjugate of the ground state psi
        psi_dag = np.conjugate(psi)
        # AVERAGE EXP VAL
        avg = 0.0
        counter = 0
        # RUN OVER THE LATTICE SITES
        for ii in range(n):
            # Given the 1D point on the lattice, get the corresponding (x,y) coords
            x, y = zig_zag(nx, ny, ii)
            stag = (-1) ** (x + y)
            mask_conditions = [
                site is None,
                ((site == "even") and (stag > 0)),
                ((site == "odd") and (stag < 0)),
            ]
            if any(mask_conditions):
                counter += 1
                # Compute the average value in the site x,y
                val = np.real(
                    np.dot(
                        psi_dag,
                        (
                            local_op(
                                self.op,
                                ii,
                                lvals,
                                has_obc,
                                self.stag_basis,
                                self.site_basis,
                            ).dot(psi)
                        ),
                    )
                )
                logger.info(f"({x+1},{y+1}) {format(val, '.12f')}")
                avg += val
        return avg / counter

    # TODO: adjust according to the new changes
    def get_fluctuations(self, psi, lvals, has_obc, site=None):
        # CHECK ON TYPES
        if not isinstance(psi, np.ndarray):
            raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
        if not isinstance(lvals, list):
            raise TypeError(f"lvals should be a list, not a {type(lvals)}")
        else:
            for ii, ll in enumerate(lvals):
                if not isinstance(ll, int):
                    raise TypeError(f"lvals[{ii}] should be INTEGER, not {type(ll)}")
        # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
        nx = lvals[0]
        ny = lvals[1]
        n = nx * ny
        # Compute the complex_conjugate of the ground state psi
        psi_dag = np.conjugate(psi)
        # Create array to store fluctuations: (<O^{2}>-<O>^{2})^{1/2}
        self.var = np.zeros((nx, ny))
        # Define a variable for the average fluctuation
        var = 0
        # RUN OVER THE LATTICE SITES
        for ii in range(n):
            # Given the 1D point on the lattice, get the corresponding (x,y) coords
            x, y = zig_zag(nx, ny, ii)
            # Compute the staggered factor
            stag = (-1) ** (x + y)
            # Compute the variance
            self.var[x, y] = (
                np.real(np.dot(psi_dag, (local_op(self.op, ii + 1, n) ** 2).dot(psi)))
                - self.obs[x, y] ** 2
            )
            if site == "odd" and stag < 0:
                var += 2 * self.var[x, y] / n
            elif site == "even" and stag > 0:
                var += 2 * self.var[x, y] / n
            elif site == None:
                var += self.var[x, y] / n
        else:
            return np.sqrt(var)  # standard deviation
