import numpy as np
from math import prod
from scipy.sparse import isspmatrix
from tools import zig_zag
from .qmb_operations import local_op
from .qmb_state import expectation_value as exp_val

__all__ = ["LocalTerm"]


class LocalTerm:
    def __init__(
        self, operator, op_name, lvals, has_obc, staggered_basis=False, site_basis=None
    ):
        """
        This function provides methods for computing local terms in a d-dimensional lattice model.
        It takes an operator matrix, its name, the lattice dimension and its topology/boundary conditions,
        and provides methods to compute the Local Hamiltonian Term and expectation values.

        Args:
            operator (scipy.sparse): A single site sparse operator matrix.

            op_name (str): Operator name

            lvals (list of ints): Dimensions (# of sites) of a d-dimensional hypercubic lattice

            has_obc (bool): It specifies the type of boundary conditions. If False, the topology is a thorus

            staggered_basis (bool, optional): Whether the lattice has staggered basis. Defaults to False.

            site_basis (dict, optional): Dictionary of Basis Projectors (sparse matrices) for lattice sites (corners, borders, lattice core, even/odd sites). Defaults to None.

        Raises:
            TypeError: If the input arguments are of incorrect types or formats.
        """
        # CHECK ON TYPES
        if not isspmatrix(operator):
            raise TypeError(f"operator should be SPARSE, not {type(operator)}")
        if not isinstance(op_name, str):
            raise TypeError(f"op_name should be a STRING, not a {type(op_name)}")
        if not isinstance(lvals, list):
            raise TypeError(f"lvals should be a list, not a {type(lvals)}")
        elif not all(np.isscalar(dim) and isinstance(dim, int) for dim in lvals):
            raise TypeError("All items of lvals must be scalar integers.")
        if not isinstance(has_obc, bool):
            raise TypeError(f"has_obc must be a BOOL, not a {type(has_obc)}")
        self.op = operator
        self.op_name = op_name
        self.lvals = lvals
        self.has_obc = has_obc
        self.stag_basis = staggered_basis
        self.site_basis = site_basis

    def get_Hamiltonian(self, strength, mask=None):
        """
        The function calculates the Local Hamiltonian by summing up local terms for each lattice site,
        potentially with some sites excluded based on the mask.
        The result is scaled by the strength parameter before being returned.

        Args:
            strength (scalar): Coupling of the Hamiltonian term.

            mask (np.ndarray, optional): d-dimensional array with bool variables specifying (if True) where to apply the local term. Defaults to None.

        Raises:
            TypeError: If the input arguments are of incorrect types or formats.

        Returns:
            scipy.sparse: Local Hamiltonian term ready to be used for exact diagonalization/expectation values.
        """
        # CHECK ON TYPES
        if not np.isscalar(strength):
            raise TypeError(f"strength must be SCALAR not a {type(strength)}")
        # LOCAL HAMILTONIAN
        H_Local = 0
        for ii in range(prod(self.lvals)):
            coords = zig_zag(self.lvals, ii)
            # ADD THE TERM TO THE HAMILTONIAN
            if mask is None:
                mask_conditions = True
            else:
                if mask[coords] == True:
                    mask_conditions = True
                else:
                    mask_conditions = False
            if mask_conditions:
                H_Local += local_op(
                    operator=self.op,
                    op_1D_site=ii,
                    lvals=self.lvals,
                    has_obc=self.has_obc,
                    staggered_basis=self.stag_basis,
                    site_basis=self.site_basis,
                )
        return strength * H_Local

    def get_expval(self, psi, site=None):
        """
        The function calculates the expectation value (and it variance) of the Local Hamiltonian
        and is averaged over all the lattice sites.

        Args:
            psi (numpy.ndarray): QMB state where the expectation value has to be computed

            site (str, optional): if odd/even, then the expectation value is performed only on that kind of sites. Defaults to None.

        Raises:
            TypeError: If the input arguments are of incorrect types or formats.
        """
        # CHECK ON TYPES
        if not isinstance(psi, np.ndarray):
            raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
        if site is not None:
            if not isinstance(site, str):
                raise TypeError(f"site should be STR ('even' / 'odd'), not {type(str)}")
        # PRINT OBSERVABLE NAME
        print(f"----------------------------------------------------")
        if site is None:
            print(f"{self.op_name}")
        else:
            print(f"{self.op_name} {site}")
        # AVERAGE EXP VAL <O> & STD DEVIATION (<O^{2}>-<O>^{2})^{1/2}
        self.obs = np.zeros(self.lvals)
        self.avg = 0.0
        self.std = 0.0
        counter = 0
        # RUN OVER THE LATTICE SITES
        for ii in range(prod(self.lvals)):
            # Given the 1D point on the d-dimensional lattice, get the corresponding coords
            coords = zig_zag(self.lvals, ii)
            stag = (-1) ** (sum(coords))
            mask_conditions = [
                site is None,
                ((site == "even") and (stag > 0)),
                ((site == "odd") and (stag < 0)),
            ]
            # Compute the average value in the site x,y
            exp_obs = exp_val(
                psi,
                local_op(
                    operator=self.op,
                    op_1D_site=ii,
                    lvals=self.lvals,
                    has_obc=self.has_obc,
                    staggered_basis=self.stag_basis,
                    site_basis=self.site_basis,
                ),
            )
            # Compute the corresponding quantum fluctuation
            exp_var = (
                exp_val(
                    psi,
                    local_op(
                        operator=self.op**2,
                        op_1D_site=ii,
                        lvals=self.lvals,
                        has_obc=self.has_obc,
                        staggered_basis=self.stag_basis,
                        site_basis=self.site_basis,
                    ),
                )
                - exp_obs**2
            )
            if any(mask_conditions):
                print(f"{coords} {format(exp_obs, '.12f')}")
                counter += 1
                self.obs[coords] = exp_obs
                self.avg += exp_obs
                self.std += exp_var
        self.avg = self.avg / counter
        self.std = np.sqrt(np.abs(self.std) / counter)
        print(f"{format(self.avg, '.10f')} +/- {format(self.std, '.10f')}")

    """
    def check_on_borders(self, border, value=1, threshold=1e-10):
        This function checks the value of an observable on a specific border of the lattice

        Args:
            border (str): lattice border where the expectation value of the observable (mx, px, my, py)

            value (int, optional): Default value which is expected for the observable. Defaults to 1.
            
            threshold (scalar, real, optional): Tolerance for the checks. Defaults to 1e-10.

        Raises:
            ValueError: If any site of the chosen border has not the expected value of the observable
        if border == "mx":
            if np.any(np.abs(self.obs[0, :] - value) > threshold):
                print(self.obs[0, :])
                raise ValueError(f"{border} border penalty not satisfied")
        elif border == "px":
            if np.any(np.abs(self.obs[-1, :] - value) > threshold):
                print(self.obs[-1, :])
                raise ValueError(f"{border} border penalty not satisfied")
        elif border == "my":
            if np.any(np.abs(self.obs[:, 0] - value) > threshold):
                print(self.obs[:, 0])
                raise ValueError(f"{border} border penalty not satisfied")
        elif border == "py":
            if np.any(np.abs(self.obs[:, -1] - value) > threshold):
                print(self.obs[:, -1])
                raise ValueError(f"{border} border penalty not satisfied")
        else:
            raise ValueError(f"border must be in (mx, px, my, py), not {border}")
        print(f"{border}-BORDER PENALTIES SATISFIED")
    """
