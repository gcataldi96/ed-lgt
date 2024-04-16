import numpy as np
from math import prod
from itertools import product, chain
from .lattice_mappings import zig_zag
from .qmb_state import QMB_state
from .qmb_term import QMBTerm
from ed_lgt.tools import validate_parameters
from ed_lgt.symmetries import nbody_term
import logging

logger = logging.getLogger(__name__)

__all__ = ["NBodyTerm"]


class NBodyTerm(QMBTerm):
    def __init__(self, op_list, op_names_list, **kwargs):
        """
        This function provides methods for computing Nbody terms in a d-dimensional lattice model along a certain axis.

        Args:

            op_list (list of N scipy.sparse.matrices): list of the operators involved in the NBody Term

            op_name_list (list of N str): list of the names of the operators
        """
        # CHECK ON TYPES
        validate_parameters(
            op_list=op_list,
            op_names_list=op_names_list,
        )
        # Preprocess arguments
        super().__init__(op_list=op_list, op_names_list=op_names_list, **kwargs)
        logger.info(" N_body-term_" + "_".join(op_names_list))

    def get_expval(self, psi):
        """
        The function calculates the expectation value (and it variance) of the
         NBody Hamiltonian nd its average over all the lattice sites.

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
                if self.sector_configs is not None:
                    self.corr[list(chain(*coords))] = psi.expectation_value(
                        nbody_term(
                            op_list=self.sym_ops,
                            op_sites_list=np.array(indxs),
                            sector_configs=self.sector_configs,
                        )
                    )
