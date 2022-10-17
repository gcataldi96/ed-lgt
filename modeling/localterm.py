import numpy as np
from Hamitonian_Functions.QMB_Operations.Mappings_1D_2D import *
from modeling.qmb_operations import local_op


class LocalTerm:
    """
    Hamiltonian term associated to the action of a Local Operator
    on every (or some) single lattice site.
    NOTE It works only for 1D and 2D lattices

    **Arguments**

    operator : str
        String identifier for the operator. Before launching the simulation,
        the python API will check that the operator is defined.

    strength : str, callable, numeric (optional)
        Defines the coupling strength of the local terms. It can
        be parameterized via a value in the dictionary for the
        simulation or a function.
        Default to 1.

    mask : callable or ``None``, optional
        The true-false-mask allows to apply the local Hamiltonians
        only to specific sites, i.e., with true values. The function
        takes the dictionary with the simulation parameters as an
        argument.
        Default to ``None`` (all sites have a local term)

    """

    def __init__(self, operator, strength, mask=None):
        if not np.isscalar(strength):
            raise TypeError(f"strength must be SCALAR not a {type(strength)}")
        self.operator = operator
        self.strength = strength
        self.mask = mask

    def get_interactions(self, lvals):
        if np.isscalar(lvals):
            self.lvals = [lvals] * 2
        # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
        nx = self.lvals[0]
        ny = self.lvals[1]
        n = nx * ny
        # LOCAL HAMILTONIAN
        H_Local = 0
        for ii in range(n):
            if self.mask is not None:
                # Compute the corresponding (x,y) coords
                x, y = zig_zag(nx, ny, ii)
                # Check if mask(x,y) is True and eventually apply the operator
                if self.mask[x, y] is True:
                    H_Local += local_op(self.operator, ii + 1, n)
                else:
                    continue
            else:
                H_Local += local_op(self.operator, ii + 1, n)
        return self.strength * H_Local
