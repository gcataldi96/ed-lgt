import numpy as np
from math import prod
from ed_lgt.tools import validate_parameters
from .qmb_operations import local_op
from .qmb_state import QMB_state
from .lattice_geometry import get_close_sites_along_direction
from .lattice_mappings import zig_zag

__all__ = ["LocalTerm", "check_link_symmetry"]


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

            site_basis (dict, optional): Dictionary of Basis Projectors (sparse matrices)
                for lattice sites (corners, borders, lattice core, even/odd sites). Defaults to None.

        Raises:
            TypeError: If the input arguments are of incorrect types or formats.
        """
        # Validate type of parameters
        validate_parameters(
            op_list=[operator],
            op_names_list=[op_name],
            lvals=lvals,
            has_obc=has_obc,
            staggered_basis=staggered_basis,
            site_basis=site_basis,
        )
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

    def get_expval(self, psi, stag_label=None):
        """
        The function calculates the expectation value (and it variance) of the Local Hamiltonian
        and is averaged over all the lattice sites.

        Args:
            psi (instance of QMB_state class): QMB state where the expectation value has to be computed

            stag_label (str, optional): if odd/even, then the expectation value is performed only on that kind of sites. Defaults to None.

        Raises:
            TypeError: If the input arguments are of incorrect types or formats.
        """
        # Check on parameters
        if not isinstance(psi, QMB_state):
            raise TypeError(f"psi must be instance of class:QMB_state not {type(psi)}")
        validate_parameters(stag_label=stag_label)
        # PRINT OBSERVABLE NAME
        print(f"----------------------------------------------------")
        print(f"{self.op_name}") if stag_label is None else print(
            f"{self.op_name} {stag_label}"
        )
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
                stag_label is None,
                ((stag_label == "even") and (stag > 0)),
                ((stag_label == "odd") and (stag < 0)),
            ]
            # Compute the average value in the site x,y
            exp_obs = psi.expectation_value(
                local_op(
                    operator=self.op,
                    op_1D_site=ii,
                    lvals=self.lvals,
                    has_obc=self.has_obc,
                    staggered_basis=self.stag_basis,
                    site_basis=self.site_basis,
                )
            )
            # Compute the corresponding quantum fluctuation
            exp_var = (
                psi.expectation_value(
                    local_op(
                        operator=self.op**2,
                        op_1D_site=ii,
                        lvals=self.lvals,
                        has_obc=self.has_obc,
                        staggered_basis=self.stag_basis,
                        site_basis=self.site_basis,
                    )
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


def check_link_symmetry(axis, loc_op1, loc_op2, value=0, sign=1):
    if not isinstance(loc_op1, LocalTerm):
        raise TypeError(f"loc_op1 should be instance of LocalTerm, not {type(loc_op1)}")
    if not isinstance(loc_op2, LocalTerm):
        raise TypeError(f"loc_op2 should be instance of LocalTerm, not {type(loc_op2)}")
    for ii in range(prod(loc_op1.lvals)):
        coords = zig_zag(loc_op1.lvals, ii)
        coords_list, sites_list = get_close_sites_along_direction(
            coords, loc_op1.lvals, axis, loc_op1.has_obc
        )
        if sites_list is None:
            continue
        else:
            c1 = coords_list[0]
            c2 = coords_list[1]
            tmp = loc_op1.obs[c1]
            tmp += sign * loc_op2.obs[c2]
        if np.abs(tmp - value) > 1e-10:
            print(loc_op1.obs[c1], loc_op2.obs[c2])
            raise ValueError(f"{axis}-Link Symmetry is violated at index {ii}")
    print(f"{axis}-LINK SYMMETRY IS SATISFIED")
