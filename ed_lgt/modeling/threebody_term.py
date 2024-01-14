import numpy as np
from math import prod
from scipy.sparse import isspmatrix
from ed_lgt.tools import zig_zag
from .qmb_operations import three_body_op
from .qmb_state import expectation_value as exp_val

__all__ = ["ThreeBodyTerm"]


class ThreeBodyTerm:
    def __init__(
        self,
        op_list,
        op_name_list,
        lvals,
        has_obc,
        staggered_basis=False,
        site_basis=None,
    ):
        """
        This function provides methods for computing 3body terms in a d-dimensional lattice model along a certain axis.
        It takes a list of 3 operators, their names, the lattice dimension and its topology/boundary conditions,
        and provides methods to compute the Local Hamiltonian Term and expectation values.

        Args:

            op_list (list of 3 scipy.sparse.matrices): list of the 3 operators involved in the 2Body Term

            op_name_list (list of 3 str): list of the names of the 3 operators

            lvals (list of ints): Dimensions (# of sites) of a d-dimensional hypercubic lattice

            has_obc (bool): It specifies the type of boundary conditions. If False, the topology is a thorus

            staggered_basis (bool, optional): Whether the lattice has staggered basis. Defaults to False.

            site_basis (dict, optional): Dictionary of Basis Projectors (sparse matrices) for lattice sites
            (corners, borders, lattice core, even/odd sites). Defaults to None.

        Raises:
            TypeError: If the input arguments are of incorrect types or formats.
        """
        # CHECK ON TYPES
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
        self.op_list = op_list
        self.op_name_list = op_name_list
        self.lvals = lvals
        self.dimensions = dimensions
        self.has_obc = has_obc
        self.stag_basis = staggered_basis
        self.site_basis = site_basis
        print(
            f"Threebody-term {self.op_name_list[0]}-{self.op_name_list[1]}-{self.op_name_list[2]}"
        )

    def get_expval(self, psi, site=None):
        """
        The function calculates the expectation value (and it variance) of the ThreeBody Hamiltonian
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
        self.corr = np.zeros(self.lvals + self.lvals + self.lvals)
        # RUN OVER THE LATTICE SITES
        for j1 in range(prod(self.lvals)):
            coords1 = zig_zag(self.lvals, j1)
            for j2 in range(prod(self.lvals)):
                coords2 = zig_zag(self.lvals, j2)
                for j3 in range(prod(self.lvals)):
                    coords3 = zig_zag(self.lvals, j3)
                    if all[j1 != j2, j1 != j3, j2 != j3]:
                        self.corr[coords1 + coords2 + coords3] = exp_val(
                            psi,
                            three_body_op(
                                op_list=self.op_list,
                                op_sites_list=[j1, j2, j3],
                                lvals=self.lvals,
                                has_obc=self.has_obc,
                                site_basis=self.site_basis,
                            ),
                        )
