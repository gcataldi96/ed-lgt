import numpy as np
from math import prod
from .lattice_mappings import zig_zag
from .lattice_geometry import get_neighbor_sites
from .qmb_operations import local_op
from .qmb_state import QMB_state
from .qmb_term import QMBTerm
from ed_lgt.tools import validate_parameters, get_time
from ed_lgt.symmetries import nbody_term
import logging

logger = logging.getLogger(__name__)

__all__ = ["LocalTerm", "check_link_symmetry"]


class LocalTerm(QMBTerm):
    def __init__(self, operator, op_name, **kwargs):
        """
        This function provides methods for computing local Hamiltonian terms in
        a d-dimensional lattice model.

        Args:
            operator (scipy.sparse): A single site sparse operator matrix.

            op_name (str): Operator name

            **kwargs: Additional keyword arguments for QMBTerm.
        """
        # Validate type of parameters
        validate_parameters(op_list=[operator], op_names_list=[op_name])
        # Preprocess arguments
        super().__init__(operator=operator, op_name=op_name, **kwargs)

    @get_time
    def get_Hamiltonian(self, strength, mask=None):
        """
        The function calculates the Local Hamiltonian by summing up local terms
        for each lattice site, potentially with some sites excluded based on the mask.
        The result is scaled by the strength parameter before being returned.

        Args:
            strength (scalar): Coupling of the Hamiltonian term.

            mask (np.ndarray, optional): d-dimensional array with bool variables
                specifying (if True) where to apply the local term. Defaults to None.

        Raises:
            TypeError: If the input arguments are of incorrect types or formats.

        Returns:
            scipy.sparse: Local Hamiltonian term ready to be used for exact diagonalization/
                expectation values.
        """
        # CHECK ON TYPES
        if not np.isscalar(strength):
            raise TypeError(f"strength must be SCALAR not a {type(strength)}")
        # LOCAL HAMILTONIAN
        H_Local = 0
        for ii in range(prod(self.lvals)):
            coords = zig_zag(self.lvals, ii)
            # CHECK MASK CONDITION ON THE SITE
            if self.get_mask_conditions(coords, mask):
                if self.sector_configs is None:
                    H_Local += local_op(operator=self.op, op_site=ii, **self.def_params)
                else:
                    # GET ONLY THE SYMMETRY SECTOR of THE HAMILTONIAN TERM
                    H_Local += nbody_term(
                        self.sym_ops,
                        np.array([ii]),
                        self.sector_configs,
                        self.momentum_basis,
                        self.momentum_k,
                    )
        return strength * H_Local

    def get_expval(self, psi, stag_label=None):
        """
        The function calculates the expectation value (and it variance) of the Local Hamiltonian
        and is averaged over all the lattice sites.

        Args:
            psi (instance of QMB_state class): QMB state where the expectation value has to be computed

            stag_label (str, optional): if odd/even, then the expectation value
                is performed only on that kind of sites. Defaults to None.

        Raises:
            TypeError: If the input arguments are of incorrect types or formats.
        """
        # Check on parameters
        if not isinstance(psi, QMB_state):
            raise TypeError(f"psi must be instance of class:QMB_state not {type(psi)}")
        validate_parameters(stag_label=stag_label)
        # PRINT OBSERVABLE NAME
        logger.info(f"----------------------------------------------------")
        (
            logger.info(f"{self.op_name}")
            if stag_label is None
            else logger.info(f"{self.op_name} {stag_label}")
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
            # Compute the average value in the site x,y
            if self.sector_configs is None:
                exp_obs = psi.expectation_value(
                    local_op(operator=self.op, op_site=ii, **self.def_params)
                )
                # Compute the corresponding quantum fluctuation
                exp_var = (
                    psi.expectation_value(
                        local_op(operator=self.op**2, op_site=ii, **self.def_params)
                    )
                    - exp_obs**2
                )
            else:
                # GET THE EXPVAL ON THE SYMMETRY SECTOR
                exp_obs = psi.expectation_value(
                    nbody_term(
                        self.sym_ops,
                        np.array([ii]),
                        self.sector_configs,
                        self.momentum_basis,
                        self.momentum_k,
                    )
                )
                # Compute the corresponding quantum fluctuation
                exp_var = (
                    psi.expectation_value(
                        nbody_term(
                            self.sym_ops,
                            np.array([ii]),
                            self.sector_configs,
                            self.momentum_basis,
                            self.momentum_k,
                        )
                        ** 2
                    )
                    - exp_obs**2
                )
            if self.get_staggered_conditions(coords, stag_label):
                logger.info(f"{coords} {format(exp_obs, '.12f')}")
                counter += 1
                self.obs[coords] = exp_obs
                self.avg += exp_obs
                self.std += exp_var
        self.avg = self.avg / counter
        self.std = np.sqrt(np.abs(self.std) / counter)
        logger.info(f"{format(self.avg, '.10f')} +/- {format(self.std, '.10f')}")


def check_link_symmetry(axis, loc_op1, loc_op2, value=0, sign=1):
    if not isinstance(loc_op1, LocalTerm):
        raise TypeError(f"loc_op1 should be instance of LocalTerm, not {type(loc_op1)}")
    if not isinstance(loc_op2, LocalTerm):
        raise TypeError(f"loc_op2 should be instance of LocalTerm, not {type(loc_op2)}")
    for ii in range(prod(loc_op1.lvals)):
        coords = zig_zag(loc_op1.lvals, ii)
        coords_list, sites_list = get_neighbor_sites(
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
            logger.info(f"{loc_op1.obs[c1]}")
            logger.info(f"{loc_op2.obs[c2]}")
            raise ValueError(f"{axis}-Link Symmetry is violated at index {ii}")
    logger.info("----------------------------------------------------")
    logger.info(f"{axis}-LINK SYMMETRY IS SATISFIED")
