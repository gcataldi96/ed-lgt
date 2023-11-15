import numpy as np
from math import prod
from copy import deepcopy
from scipy.sparse import isspmatrix, csr_matrix
from ed_lgt.tools import zig_zag, inverse_zig_zag
from .qmb_operations import two_body_op
from .qmb_state import expectation_value as exp_val

__all__ = ["TwoBodyTerm"]


class TwoBodyTerm:
    def __init__(
        self,
        axis,
        op_list,
        op_name_list,
        lvals,
        has_obc,
        staggered_basis=False,
        site_basis=None,
    ):
        """
        This function provides methods for computing twobody terms in a d-dimensional lattice model along a certain axis.
        It takes a list of two operators, their names, the lattice dimension and its topology/boundary conditions,
        and provides methods to compute the Local Hamiltonian Term and expectation values.

        Args:
            axis (str): axis along which the 2Body Term is performed

            op_list (list of 2 scipy.sparse.matrices): list of the two operators involved in the 2Body Term

            op_name_list (list of 2 str): list of the names of the two operators

            lvals (list of ints): Dimensions (# of sites) of a d-dimensional hypercubic lattice

            has_obc (bool): It specifies the type of boundary conditions. If False, the topology is a thorus

            staggered_basis (bool, optional): Whether the lattice has staggered basis. Defaults to False.

            site_basis (dict, optional): Dictionary of Basis Projectors (sparse matrices) for lattice sites
            (corners, borders, lattice core, even/odd sites). Defaults to None.

        Raises:
            TypeError: If the input arguments are of incorrect types or formats.
        """
        # CHECK ON TYPES
        if not isinstance(axis, str):
            raise TypeError(f"axis should be a STRING, not a {type(axis)}")
        if not isinstance(op_list, list):
            raise TypeError(f"op_list should be a list, not a {type(op_list)}")
        else:
            for ii, op in enumerate(op_list):
                if not isspmatrix(op):
                    raise TypeError(f"op_list[{ii}] should be SPARSE, not {type(op)}")
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
        if not isinstance(lvals, list):
            raise TypeError(f"lvals should be a list, not a {type(lvals)}")
        elif not all(np.isscalar(dim) and isinstance(dim, int) for dim in lvals):
            raise TypeError("All items of lvals must be scalar integers.")
        if not isinstance(has_obc, bool):
            raise TypeError(f"has_obc must be a BOOL, not a {type(has_obc)}")
        dimensions = "xyz"[: len(lvals)]
        if axis not in dimensions:
            raise ValueError(f"axis should be in {dimensions}: got {axis}")
        else:
            self.axis = axis
        self.op_list = op_list
        self.op_name_list = op_name_list
        self.lvals = lvals
        self.dimensions = dimensions
        self.has_obc = has_obc
        self.stag_basis = staggered_basis
        self.site_basis = site_basis
        print(f"twobody-term {self.op_name_list[0]}-{self.op_name_list[1]}")

    def get_Hamiltonian(self, strength, add_dagger=False, mask=None):
        """
        The function calculates the TwoBody Hamiltonian by summing up 2body terms for each lattice site,
        potentially with some sites excluded based on the mask.
        The result is scaled by the strength parameter before being returned.
        Eventually, it is possible to sum also the dagger part of the Hamiltonian.

        Args:
            strength (scalar): Coupling of the Hamiltonian term.

            add_dagger (bool, optional): If true, it add the hermitian conjugate of the resulting Hamiltonian. Defaults to False.

            mask (np.ndarray, optional): 2D array with bool variables specifying (if True) where to apply the local term. Defaults to None.

        Raises:
            TypeError: If the input arguments are of incorrect types or formats.

        Returns:
            scipy.sparse: TwoBody Hamiltonian term ready to be used for exact diagonalization/expectation values.
        """
        # CHECK ON TYPES
        if not np.isscalar(strength):
            raise TypeError(f"strength must be SCALAR, not {type(strength)}")
        if not isinstance(add_dagger, bool):
            raise TypeError(f"add_dagger must be a BOOL, not a {type(add_dagger)}")
        # Hamiltonian
        H_twobody = 0
        # Run over all the single lattice sites, ordered with the ZIG ZAG CURVE
        for ii in range(prod(self.lvals)):
            # Compute the corresponding coords
            coords = zig_zag(self.lvals, ii)
            # Check if it admits a twobody term according to the lattice geometry
            coords_list, sites_list = self.get_twobodyterm_sites(coords)
            if sites_list is None:
                continue
            # Check Mask condition on that site
            if mask is None:
                mask_conditions = True
            else:
                if mask[coords] == True:
                    mask_conditions = True
                else:
                    mask_conditions = False
            # ADD THE TERM TO THE HAMILTONIAN
            if mask_conditions:
                H_twobody += strength * two_body_op(
                    op_list=self.op_list,
                    op_sites_list=sites_list,
                    lvals=self.lvals,
                    has_obc=self.has_obc,
                    staggered_basis=self.stag_basis,
                    site_basis=self.site_basis,
                )
        if not isspmatrix(H_twobody):
            H_twobody = csr_matrix(H_twobody)
        if add_dagger:
            H_twobody += csr_matrix(H_twobody.conj().transpose())
        return H_twobody

    def get_expval(self, psi, site=None):
        """
        The function calculates the expectation value (and it variance) of the TwoBody Hamiltonian
        and its average over all the lattice sites.

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
        # Create an array to store the correlator
        self.corr = np.zeros(self.lvals + self.lvals)
        # RUN OVER THE LATTICE SITES
        for ii in range(prod(self.lvals)):
            coords1 = zig_zag(self.lvals, ii)
            for jj in range(prod(self.lvals)):
                coords2 = zig_zag(self.lvals, jj)
                # AVOID SELF CORRELATIONS
                if ii != jj:
                    self.corr[coords1 + coords2] = exp_val(
                        psi,
                        two_body_op(
                            op_list=self.op_list,
                            op_sites_list=[ii, jj],
                            lvals=self.lvals,
                            has_obc=self.has_obc,
                            staggered_basis=self.stag_basis,
                            site_basis=self.site_basis,
                        ),
                    )

    def get_twobodyterm_sites(self, coords):
        coords1 = list(coords)
        i1 = inverse_zig_zag(self.lvals, coords1)
        coords2 = deepcopy(coords1)
        # Check if the site admits a neighbor where to apply the twobody term
        # Look at the specific index of the axis
        indx = self.dimensions.index(self.axis)
        # If along that axis, there is space for a twobody term:
        if coords1[indx] < self.lvals[indx] - 1:
            coords2[indx] += 1
            i2 = inverse_zig_zag(self.lvals, coords2)
            sites_list = [i1, i2]
            coords_list = [tuple(coords1), tuple(coords2)]
        else:
            # PERIODIC BOUNDARY CONDITIONS
            if not self.has_obc:
                coords2[indx] = 0
                i2 = inverse_zig_zag(self.lvals, coords2)
                sites_list = [i1, i2]
                coords_list = [tuple(coords1), tuple(coords2)]
            else:
                sites_list = None
                coords_list = None
        return coords_list, sites_list


"""
    def check_link_symm(self, value=1, threshold=1e-10, has_obc=True):
        This function checks the value of a 2body operator along the self.axis and compare it with an expected value.

        Args:
            value (int, optional): Default value which is expected for the observable. Defaults to 1.
            threshold (scalar, real, optional): Tolerance for the checks. Defaults to 1e-10.
            has_obc (bool): It specifies the type of boundary conditions. If False, the topology is a thorus and there are more links to be checked

        Raises:
            ValueError: If any site of the chosen border has not the expected value of the observable
        # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
        nx = self.corr.shape[0]
        ny = self.corr.shape[1]
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
        else:
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
        print(f"{self.axis} LINK SYMMETRIES SATISFIED")
"""
