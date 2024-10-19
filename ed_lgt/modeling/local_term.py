import numpy as np
from math import prod
from numba import njit, prange
from .lattice_mappings import zig_zag
from .lattice_geometry import get_neighbor_sites
from .qmb_operations import local_op
from .qmb_state import QMB_state
from .qmb_term import QMBTerm
from ed_lgt.tools import validate_parameters, get_time
from ed_lgt.symmetries import nbody_term, nbody_data_par, exp_val_numba
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
        n_sites = prod(self.lvals)
        if self.sector_configs is None:
            # Stores the expectation values <O>
            self.obs = np.zeros(n_sites, dtype=float)
            # Stores the variances <O^2> - <O>^2
            self.var = np.zeros(n_sites, dtype=float)
            for ii in range(prod(self.lvals)):
                self.obs[ii] = psi.expectation_value(
                    local_op(operator=self.op, op_site=ii, **self.def_params)
                )
                self.var[ii] = (
                    psi.expectation_value(
                        local_op(operator=self.op**2, op_site=ii, **self.def_params)
                    )
                    - self.obs[ii] ** 2
                )
        else:
            self.obs, self.var = lattice_local_exp_val(
                psi.psi, n_sites, self.sector_configs, self.sym_ops
            )
        # CHECK STAGGERED CONDITION AND PRINT VALUES
        self.avg = 0.0
        self.std = 0.0
        counter = 0
        for ii in range(n_sites):
            # Given the 1D point on the d-dimensional lattice, get the corresponding coords
            coords = zig_zag(self.lvals, ii)
            if self.get_staggered_conditions(coords, stag_label):
                logger.info(f"{coords} {format(self.obs[ii], '.12f')}")
                counter += 1
                self.avg += self.obs[ii]
                self.std += self.var[ii]
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


@njit(parallel=True)
def lattice_local_exp_val(psi, n_sites, sector_configs, sym_ops):
    """
    Computes the expectation value <O> and the variance <O^2> - <O>^2
    for a local operator O on all lattice sites in parallel.

    Args:
        psi (np.ndarray): The quantum state (wavefunction) in the form of a vector.
        n_sites (int): The total number of lattice sites on which the local operator acts.
        sector_configs (np.ndarray): Array representing the symmetry sectors or configurations for the system.
        sym_ops (np.ndarray): Symmetry operators for the local operator O, represented as a set of nonzero matrix elements.

    Returns:
        obs (np.ndarray): The expectation values <O> for the local operator O on each lattice site.
        var (np.ndarray): The variances <O^2> - <O>^2 for the local operator O on each lattice site.

    Notes:
        - The expectation value <O> is computed for each site using matrix-vector multiplication.
        - The variance <O^2> - <O>^2 is also computed for each site. For local operators,
          squaring the non-zero entries in `value_list` (the matrix elements of O) is
          sufficient to compute O^2.
        - The function runs in parallel over all lattice sites using `prange` for performance optimization.
    """
    # Initialize arrays for storing expectation values and variances for all sites
    obs = np.zeros(n_sites, dtype=float)  # Stores the expectation values <O>
    var = np.zeros(n_sites, dtype=float)  # Stores the variances <O^2> - <O>^2
    # Loop over each lattice site in parallel using prange
    for ii in prange(n_sites):
        # Compute the n-body operator's non-zero elements for site 'ii'
        row_list, col_list, value_list = nbody_data_par(
            sym_ops, np.array([ii]), sector_configs
        )
        # Compute the expectation value <O> for site 'ii'
        obs[ii] = exp_val_numba(psi, row_list, col_list, value_list)
        # For local operators, the variance <O^2> - <O>^2 can be computed by squaring value_list
        var[ii] = exp_val_numba(psi, row_list, col_list, value_list**2) - obs[ii] ** 2
    return obs, var
