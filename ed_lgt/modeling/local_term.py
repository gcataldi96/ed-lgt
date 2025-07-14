import numpy as np
from math import prod
from numba import njit, prange
from .lattice_mappings import zig_zag, inverse_zig_zag
from .lattice_geometry import get_neighbor_sites
from .qmb_operations import local_op
from .qmb_state import QMB_state, exp_val_data
from .qmb_term import QMBTerm
from ed_lgt.tools import validate_parameters, get_time
from ed_lgt.symmetries import nbody_term, localbody_data_par, nbody_data_momentum_1site
import logging

logger = logging.getLogger(__name__)

__all__ = ["LocalTerm", "check_link_symmetry"]


class LocalTerm(QMBTerm):
    def __init__(self, operator: np.ndarray, op_name: str, **kwargs):
        """
        This function provides methods for computing local Hamiltonian terms in
        a d-dimensional lattice model.

        Args:
            operator (scipy.sparse): A single site sparse operator matrix.

            op_name (str): Operator name

            **kwargs: Additional keyword arguments for QMBTerm.
        """
        logger.info(f"LocalTerm: {op_name}")
        # Validate type of parameters
        validate_parameters(op_list=[operator], op_names_list=[op_name])
        # Preprocess arguments
        super().__init__(operator=operator, op_name=op_name, **kwargs)
        # Number of lattice sites
        self.n_sites = prod(self.lvals)
        # Local dimensions of the operator
        if self.sector_configs is not None:
            self.max_loc_dim = self.op.shape[2]

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
        if self.sector_configs is None:
            H_Local = 0
            for ii in range(self.n_sites):
                coords = zig_zag(self.lvals, ii)
                # CHECK MASK CONDITION ON THE SITE
                if self.get_mask_conditions(coords, mask):
                    H_Local += local_op(operator=self.op, op_site=ii, **self.def_params)
            return strength * H_Local
        else:
            # Initialize lists for nonzero entries
            all_row_list = []
            all_col_list = []
            all_value_list = []
            for ii in range(self.n_sites):
                coords = zig_zag(self.lvals, ii)
                # CHECK MASK CONDITION ON THE SITE
                if self.get_mask_conditions(coords, mask):
                    if len(self.sector_configs) > 2**24.5:
                        logger.info(f"Site {ii}")
                    # GET ONLY THE SYMMETRY SECTOR of THE HAMILTONIAN TERM
                    row_list, col_list, value_list = nbody_term(
                        self.sym_ops,
                        np.array([ii]),
                        self.sector_configs,
                        self.momentum_basis,
                    )
                    all_row_list.append(row_list)
                    all_col_list.append(col_list)
                    all_value_list.append(value_list)
            # Concatenate global lists
            row_list = np.concatenate(all_row_list)
            col_list = np.concatenate(all_col_list)
            value_list = np.concatenate(all_value_list) * strength
            return row_list, col_list, value_list

    @get_time
    def get_Hamiltonian_v2(self, strength, mask=None):
        """
        Calculate the Local Hamiltonian using local terms for each lattice site,
        applying the operator only at those sites that satisfy the mask condition.
        In the presence of a symmetry sector (self.sector_configs is not None), the
        Hamiltonian is built by computing the nonzero entries of the local operator
        (diagonal in this case) using the symmetry sector.

        Args:
            strength (scalar): The coupling constant.
            mask (np.ndarray, optional): A d-dimensional boolean array that specifies
                which lattice sites to include.

        Returns:
            Either a full matrix (if no symmetry sector is used) or a tuple of three
            arrays (row_list, col_list, value_list) representing the sparse Hamiltonian.
        """
        if not np.isscalar(strength):
            raise TypeError(f"strength must be SCALAR not a {type(strength)}")

        # Case 1: No symmetry sector: build the Hamiltonian normally.
        if self.sector_configs is None:
            H_Local = 0
            for ii in range(self.n_sites):
                coords = zig_zag(self.lvals, ii)
                if self.get_mask_conditions(coords, mask):
                    H_Local += local_op(operator=self.op, op_site=ii, **self.def_params)
            return strength * H_Local

        # Case 2: With symmetry sector: build the sparse Hamiltonian using the symmetry sector.
        else:
            # Instead of looping over all sites and concatenating many small arrays,
            # first determine which lattice sites satisfy the mask.
            valid_sites_list = []
            for ii in range(self.n_sites):
                coords = zig_zag(self.lvals, ii)
                if self.get_mask_conditions(coords, mask):
                    valid_sites_list.append(int(ii))
            if len(valid_sites_list) == 0:
                raise ValueError("No lattice sites satisfy the mask condition.")

            # Now, call a version of localbody_data_par that processes a set of sites.
            # Here we assume that self.op is a diagonal operator such that its action at a site
            # is given by op[site]. The function localbody_data_par (modified to accept a vector
            # of sites) will compute for each valid site a contribution (a diagonal element)
            # and return three arrays (row indices, col indices, and values).
            row_list, col_list, value_list = localbody_data_par(
                self.sym_ops[0], np.array(valid_sites_list), self.sector_configs
            )
            # Multiply the nonzero values by the strength
            return row_list, col_list, value_list * strength

    def get_expval(self, psi, stag_label=None, print_values=True):
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
        if print_values:
            msg = "" if stag_label is None else f"{stag_label}"
            logger.info(f"----------------------------------------------------")
            logger.info(f"{self.op_name} {msg}")
        # Distinguish between the two cases: with and without symmetry sector
        if self.sector_configs is None:
            # Stores the expectation values <O>
            self.obs = np.zeros(self.n_sites, dtype=float)
            # Stores the variances <O^2> - <O>^2
            self.var = np.zeros(self.n_sites, dtype=float)
            for ii in range(self.n_sites):
                self.obs[ii] = psi.expectation_value(
                    local_op(operator=self.op, op_site=ii, **self.def_params)
                )
                self.var[ii] = psi.expectation_value(
                    local_op(operator=self.op**2, op_site=ii, **self.def_params)
                )
                self.var[ii] -= self.obs[ii] ** 2
        else:
            # Compute the operator for the variance
            shape = (1, self.n_sites, self.max_loc_dim, self.max_loc_dim)
            opvar = np.zeros(shape, dtype=float)
            for ii in range(self.n_sites):
                opvar[0, ii] = np.dot(self.sym_ops[0, ii], self.sym_ops[0, ii])
            # GET THE EXPVAL ON THE SYMMETRY SECTOR
            if self.momentum_basis is not None:
                self.obs, self.var = lattice_local_exp_val_mom(
                    psi.psi,
                    self.n_sites,
                    self.sector_configs,
                    self.sym_ops,
                    self.momentum_basis,
                    opvar,
                )
            else:
                self.obs, self.var = lattice_local_exp_val(
                    psi.psi, self.n_sites, self.sector_configs, self.sym_ops
                )
        # CHECK STAGGERED CONDITION AND PRINT VALUES
        self.avg = 0.0
        self.std = 0.0
        counter = 0
        for ii in range(self.n_sites):
            # Given the 1D point on the d-dimensional lattice, get the corresponding coords
            coords = zig_zag(self.lvals, ii)
            if self.get_staggered_conditions(coords, stag_label):
                if print_values:
                    logger.info(f"{coords} {format(self.obs[ii], '.12f')}")
                counter += 1
                self.avg += self.obs[ii]
                self.std += self.var[ii]
        self.avg = self.avg / counter
        self.std = np.sqrt(np.abs(self.std) / counter)
        if print_values:
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
            jj = inverse_zig_zag(loc_op1.lvals, coords_list[1])
            tmp = loc_op1.obs[ii] + sign * loc_op2.obs[jj]
        if np.abs(tmp - value) > 1e-10:
            logger.info(f"{loc_op1.obs[ii]}")
            logger.info(f"{loc_op2.obs[jj]}")
            raise ValueError(f"{axis}-Link Symmetry is violated at index {ii}")
    logger.info("----------------------------------------------------")
    logger.info(f"{axis}-LINK SYMMETRY IS SATISFIED")


@njit(parallel=True, cache=True)
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
    chunk_size = n_sites if n_sites < 11 else 4
    # Divide the sites into chunks
    num_chunks = (n_sites + chunk_size - 1) // chunk_size
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, n_sites)
        # Process the current chunk in parallel
        for ii in prange(start, end):
            row_list, col_list, value_list = localbody_data_par(
                sym_ops[0], ii, sector_configs
            )
            obs[ii] = exp_val_data(psi, row_list, col_list, value_list)
            # var[ii] = exp_val_data(psi, row_list, col_list, value_list**2)
            # var[ii] =- obs[ii] ** 2
    return obs, var


@njit(parallel=True, cache=True)
def lattice_local_exp_val_mom(
    psi, n_sites, sector_configs, sym_ops, momentum_basis, opvar
):
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
        row_list, col_list, value_list = nbody_data_momentum_1site(
            sym_ops, np.array([ii]), sector_configs, momentum_basis
        )
        # Compute the expectation value <O> for site 'ii'
        obs[ii] = exp_val_data(psi, row_list, col_list, value_list)
        # For local operators, the variance <O^2> - <O>^2 can be computed by squaring value_list
        row_list1, col_list1, value_list1 = nbody_data_momentum_1site(
            opvar, np.array([ii]), sector_configs, momentum_basis
        )
        var[ii] = exp_val_data(psi, row_list1, col_list1, value_list1) - obs[ii] ** 2
    return obs, var
