import numpy as np
from math import prod
from itertools import product
from scipy.sparse import isspmatrix, csr_matrix
from .lattice_geometry import get_neighbor_sites
from .lattice_mappings import zig_zag
from .qmb_operations import two_body_op
from .qmb_state import QMB_state
from .qmb_term import QMBTerm
from ed_lgt.tools import validate_parameters
from ed_lgt.symmetries import nbody_term
import logging

logger = logging.getLogger(__name__)

__all__ = ["TwoBodyTerm"]


class TwoBodyTerm(QMBTerm):
    def __init__(self, axis, op_list, op_names_list, **kwargs):
        """
        This function provides methods for computing twobody Hamiltonian terms
        in a d-dimensional lattice model along a certain axis.

        Args:
            axis (str): axis along which the 2Body Term is performed

            op_list (list of 2 scipy.sparse.matrices): list of the two operators
                involved in the 2Body Term

            op_name_list (list of 2 str): list of the names of the two operators
        """
        # CHECK ON TYPES
        validate_parameters(
            axes=[axis],
            op_list=op_list,
            op_names_list=op_names_list,
        )
        # Preprocess arguments
        super().__init__(op_list=op_list, op_names_list=op_names_list, **kwargs)
        if axis not in self.dimensions:
            raise ValueError(f"axis should be in {self.dimensions}: got {axis}")
        else:
            self.axis = axis

    def get_Hamiltonian(self, strength, add_dagger=False, mask=None):
        """
        The function calculates the TwoBody Hamiltonian by summing up 2body terms
        for each lattice site, potentially with some sites excluded based on the mask.
        The result is scaled by the strength parameter before being returned.
        Eventually, it is possible to sum also the dagger part of the Hamiltonian.

        Args:
            strength (scalar): Coupling of the Hamiltonian term.

            add_dagger (bool, optional): If true, it add the hermitian conjugate
                of the resulting Hamiltonian. Defaults to False.

            mask (np.ndarray, optional): 2D array with bool variables specifying
                (if True) where to apply the local term. Defaults to None.

        Returns:
            scipy.sparse: TwoBody Hamiltonian term ready to be used for exact diagonalization/
                expectation values.
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
            if self.get_mask_conditions(coords, mask):
                # ADD THE TERM TO THE HAMILTONIAN
                if self.sector_configs is None:
                    H_twobody += strength * two_body_op(
                        op_list=self.op_list,
                        op_sites_list=sites_list,
                        **self.def_params,
                    )
                else:
                    # GET ONLY THE SYMMETRY SECTOR of THE HAMILTONIAN TERM
                    H_twobody += strength * nbody_term(
                        op_list=self.sym_ops,
                        op_sites_list=np.array(sites_list),
                        sector_configs=self.sector_configs,
                    )
        if not isspmatrix(H_twobody):
            H_twobody = csr_matrix(H_twobody)
        if add_dagger:
            H_twobody += csr_matrix(H_twobody.conj().transpose())
        return H_twobody

    def get_expval(self, psi):
        """
        The function calculates the expectation value (and it variance) of the TwoBody Hamiltonian
        and its average over all the lattice sites.

        Args:
            psi (numpy.ndarray): QMB state where the expectation value has to be computed
        """
        # Check on parameters
        if not isinstance(psi, QMB_state):
            raise TypeError(f"psi must be instance of class:QMB_state not {type(psi)}")
        # PRINT OBSERVABLE NAME
        logger.info(f"----------------------------------------------------")
        logger.info(f"{'-'.join(self.op_names_list)}")
        # Create an array to store the correlator
        self.corr = np.zeros(self.lvals + self.lvals)
        # RUN OVER THE LATTICE SITES
        for ii, jj in product(range(prod(self.lvals)), repeat=2):
            coords1 = zig_zag(self.lvals, ii)
            coords2 = zig_zag(self.lvals, jj)
            # AVOID SELF CORRELATIONS
            if ii != jj:
                if self.sector_configs is None:
                    self.corr[coords1 + coords2] = psi.expectation_value(
                        two_body_op(
                            op_list=self.op_list,
                            op_sites_list=[ii, jj],
                            **self.def_params,
                        )
                    )
                else:
                    # GET THE EXPVAL ON THE SYMMETRY SECTOR
                    self.corr[coords1 + coords2] = psi.expectation_value(
                        nbody_term(
                            op_list=self.sym_ops,
                            op_sites_list=np.array([ii, jj]),
                            sector_configs=self.sector_configs,
                        )
                    )

    def print_nearest_neighbors(self):
        for ii in range(prod(self.lvals)):
            # Compute the corresponding coords
            coords = zig_zag(self.lvals, ii)
            # Check if it admits a twobody term according to the lattice geometry
            coords_list, sites_list = get_neighbor_sites(
                coords, self.lvals, self.axis, self.has_obc
            )
            if sites_list is None:
                continue
            else:
                c1 = coords_list[0]
                c2 = coords_list[1]
                logger.info(f"{c1}-{c2} {self.corr[c1 + c2]}")
