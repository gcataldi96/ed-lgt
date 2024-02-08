import numpy as np
from math import prod
from itertools import product
from scipy.sparse import isspmatrix, csr_matrix
from .lattice_mappings import zig_zag
from .qmb_operations import two_body_op
from .lattice_geometry import get_neighbor_sites
from .qmb_state import QMB_state
from .symmetries import nbody_sector
from ed_lgt.tools import validate_parameters
import logging

logger = logging.getLogger(__name__)

__all__ = ["TwoBodyTerm"]


class TwoBodyTerm:
    def __init__(
        self,
        axis,
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
        # CHECK ON TYPES
        validate_parameters(
            axes=[axis],
            op_list=op_list,
            op_names_list=op_names_list,
            lvals=lvals,
            has_obc=has_obc,
        )
        dimensions = "xyz"[: len(lvals)]
        if axis not in dimensions:
            raise ValueError(f"axis should be in {dimensions}: got {axis}")
        else:
            self.axis = axis
        self.op_list = op_list
        self.op_name_list = op_names_list
        self.lvals = lvals
        self.dimensions = dimensions
        self.has_obc = has_obc
        self.stag_basis = staggered_basis
        self.site_basis = site_basis
        self.sector_indices = sector_indices
        self.sector_basis = sector_basis
        logger.info(f"twobody-term {self.op_name_list[0]}-{self.op_name_list[1]}")

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
        validate_parameters(add_dagger=add_dagger)
        # Hamiltonian
        H_twobody = 0
        # Run over all the single lattice sites, ordered with the ZIG ZAG CURVE
        for ii in range(prod(self.lvals)):
            # Compute the corresponding coords
            coords = zig_zag(self.lvals, ii)
            # Check if it admits a twobody term according to the lattice geometry
            _, sites_list = get_neighbor_sites(
                coords, self.lvals, self.axis, self.has_obc
            )
            if sites_list is None:
                continue
            # CHECK MASK CONDITION ON THE SITE
            mask_conditions = (
                True if mask is None or all([mask is not None, mask[coords]]) else False
            )
            # ADD THE TERM TO THE HAMILTONIAN
            if mask_conditions:
                if self.sector_basis is None:
                    H_twobody += strength * two_body_op(
                        op_list=self.op_list,
                        op_sites_list=sites_list,
                        lvals=self.lvals,
                        has_obc=self.has_obc,
                        staggered_basis=self.stag_basis,
                        site_basis=self.site_basis,
                    )
                else:
                    # GET ONLY THE SYMMETRY SECTOR of THE HAMILTONIAN TERM
                    H_twobody += strength * nbody_sector(
                        op_list=self.op_list,
                        op_sites_list=sites_list,
                        sector_basis=self.sector_basis,
                    )
        if not isspmatrix(H_twobody):
            H_twobody = csr_matrix(H_twobody)
        if add_dagger:
            H_twobody += csr_matrix(H_twobody.conj().transpose())
        return H_twobody

    def get_expval(self, psi, stag_label=None):
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
        validate_parameters(stag_label=stag_label)
        # Create an array to store the correlator
        self.corr = np.zeros(self.lvals + self.lvals)
        # RUN OVER THE LATTICE SITES
        for ii, jj in product(range(prod(self.lvals)), repeat=2):
            coords1 = zig_zag(self.lvals, ii)
            coords2 = zig_zag(self.lvals, jj)
            # AVOID SELF CORRELATIONS
            if ii != jj:
                if self.sector_basis is None:
                    self.corr[coords1 + coords2] = psi.expectation_value(
                        two_body_op(
                            op_list=self.op_list,
                            op_sites_list=[ii, jj],
                            lvals=self.lvals,
                            has_obc=self.has_obc,
                            staggered_basis=self.stag_basis,
                            site_basis=self.site_basis,
                            sector_indices=self.sector_indices,
                        )
                    )
                else:
                    # GET THE EXPVAL ON THE SYMMETRY SECTOR
                    self.corr[coords1 + coords2] = psi.expectation_value(
                        nbody_sector(
                            op_list=self.op_list,
                            op_sites_list=[ii, jj],
                            sector_basis=self.sector_basis,
                        )
                    )
