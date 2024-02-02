import numpy as np
from math import prod
from itertools import product, chain
from .lattice_mappings import zig_zag
from .qmb_state import QMB_state
from .symmetries import nbody_sector

__all__ = ["NBodyTerm"]


class NBodyTerm:
    def __init__(
        self,
        op_list,
        op_names_list,
        lvals,
        has_obc,
        staggered_basis=False,
        site_basis=None,
        sector_indices=None,
        sector_basis=None,
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

            has_obc (list of bool): true for OBC, false for PBC along each direction

            staggered_basis (bool, optional): Whether the lattice has staggered basis. Defaults to False.

            site_basis (dict, optional): Dictionary of Basis Projectors (sparse matrices) for lattice sites
            (corners, borders, lattice core, even/odd sites). Defaults to None.

        Raises:
            TypeError: If the input arguments are of incorrect types or formats.
        """
        # Initialize base class with the common parameters
        self.op_list = op_list
        self.op_name_list = op_names_list
        self.lvals = lvals
        self.dimensions = "xyz"[: len(lvals)]
        self.has_obc = has_obc
        self.stag_basis = staggered_basis
        self.site_basis = site_basis
        self.sector_indices = sector_indices
        self.sector_basis = sector_basis
        print(" N_body-term_" + "_".join(op_names_list))

    def get_expval(self, psi):
        """
        The function calculates the expectation value (and it variance) of the TwoBody Hamiltonian
        and its average over all the lattice sites.

        Args:
            psi (numpy.ndarray): QMB state where the expectation value has to be computed

            stag_label (str, optional): if odd/even, then the expectation value is performed only on that kind of sites. Defaults to None.

        Raises:
            TypeError: If the input arguments are of incorrect types or formats.
        """
        # Check on parameters
        if not isinstance(psi, QMB_state):
            raise TypeError(f"psi must be instance of class:QMB_state not {type(psi)}")
        # Create an array to store the correlator
        self.corr = np.zeros(
            list(chain(*[self.lvals for _ in range(len(self.op_list))]))
        )
        # RUN OVER THE LATTICE SITES
        for indxs in product(range(prod(self.lvals)), repeat=len(self.op_list)):
            indxs = list(indxs)
            coords = [zig_zag(self.lvals, ii) for ii in indxs]
            # AVOID SELF CORRELATIONS
            if len(set(indxs)) == len(indxs):
                # GET THE EXPVAL ON THE SYMMETRY SECTOR
                if self.sector_basis is not None:
                    self.corr[list(chain(*coords))] = psi.expectation_value(
                        nbody_sector(
                            op_list=self.op_list,
                            op_sites_list=indxs,
                            sector_basis=self.sector_basis,
                        )
                    )
