import numpy as np
from math import prod
from numba import njit, prange
from scipy.sparse import isspmatrix, csr_matrix
from .lattice_geometry import get_neighbor_sites
from .lattice_mappings import zig_zag, inverse_zig_zag
from .qmb_operations import two_body_op
from .qmb_state import QMB_state, exp_val_data
from .qmb_term import QMBTerm
from ed_lgt.tools import validate_parameters
from ed_lgt.symmetries import nbody_term, nbody_data_par
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
        logger.info(f"TwoBodyTerm: {op_names_list}")
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
        # Case with no symmetry sector
        if self.sector_configs is None:
            H_twobody = 0
            for ii in range(prod(self.lvals)):
                coords = zig_zag(self.lvals, ii)
                _, sites_list = get_neighbor_sites(
                    coords, self.lvals, self.axis, self.has_obc
                )
                if sites_list is None:
                    continue
                if self.get_mask_conditions(coords, mask):
                    H_twobody += strength * two_body_op(
                        op_list=self.op_list,
                        op_sites_list=sites_list,
                        **self.def_params,
                    )
            if not isspmatrix(H_twobody):
                H_twobody = csr_matrix(H_twobody)
            if add_dagger:
                H_twobody += csr_matrix(H_twobody.conj().transpose())
            return H_twobody
        else:
            # Case with symmetry sector
            all_row_list = []
            all_col_list = []
            all_value_list = []
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
                    # GET ONLY THE SYMMETRY SECTOR of THE HAMILTONIAN TERM
                    row_list, col_list, value_list = nbody_term(
                        op_list=self.sym_ops,
                        op_sites_list=np.array(sites_list),
                        sector_configs=self.sector_configs,
                        momentum_basis=self.momentum_basis,
                        k=self.momentum_k,
                    )
                    all_row_list.append(row_list)
                    all_col_list.append(col_list)
                    all_value_list.append(value_list)
            row_list = np.concatenate(all_row_list)
            col_list = np.concatenate(all_col_list)
            value_list = np.concatenate(all_value_list) * strength
            if add_dagger:
                # Add Hermitian conjugate after the full term construction
                dagger_row_list = col_list
                dagger_col_list = row_list
                dagger_value_list = np.conjugate(value_list)
                # Concatenate the dagger part to the original term
                row_list = np.concatenate([row_list, dagger_row_list])
                col_list = np.concatenate([col_list, dagger_col_list])
                value_list = np.concatenate([value_list, dagger_value_list])
            return row_list, col_list, value_list

    def get_expval(self, psi):
        """
        The function calculates the expectation value (and it variance) of the
        TwoBody Hamiltonian and its average over all the lattice sites.

        Args:
            psi (numpy.ndarray): QMB state where the expectation value has to be computed
        """
        # Check on parameters
        if not isinstance(psi, QMB_state):
            raise TypeError(f"psi must be instance of class:QMB_state not {type(psi)}")
        # PRINT OBSERVABLE NAME
        logger.info(f"----------------------------------------------------")
        logger.info(f"{'-'.join(self.op_names_list)}")
        n_sites = prod(self.lvals)
        # IN CASE OF NO SYMMETRY SECTOR
        if self.sector_configs is None:
            # Create an array to store the correlator
            self.corr = np.zeros((n_sites, n_sites), dtype=float)
            for ii in prange(n_sites):
                for jj in range(ii + 1, n_sites):
                    self.corr[ii, jj] = psi.expectation_value(
                        two_body_op(
                            op_list=self.op_list,
                            op_sites_list=[ii, jj],
                            **self.def_params,
                        )
                    )
                    self.corr[jj, ii] = self.corr[ii, jj]
        else:
            # GET THE EXPVAL ON THE SYMMETRY SECTOR
            self.corr = lattice_twobody_exp_val(
                psi.psi, n_sites, self.sector_configs, self.sym_ops
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
                i1 = inverse_zig_zag(self.lvals, c1)
                i2 = inverse_zig_zag(self.lvals, c2)
                logger.info(f"{c1}-{c2} {self.corr[i1,i2]}")


@njit(parallel=True, cache=True)
def lattice_twobody_exp_val(psi, n_sites, sector_configs, sym_ops):
    """
    Computes the expectation value <O> for a two-body operator O over all pairs of lattice sites in parallel.

    Args:
        psi (np.ndarray): The quantum state (wavefunction) in the form of a vector.
        n_sites (int): The total number of lattice sites in the system.
        sector_configs (np.ndarray): Array representing the symmetry sectors or configurations for the system.
        sym_ops (np.ndarray): Symmetry operators for the two-body operator O, represented as
                              a set of nonzero matrix elements.

    Returns:
        corr (np.ndarray): A 2D array where corr[ii, jj] stores the expectation values <O> for the two-body operator between site ii and site jj.
        The result is symmetric, i.e., corr[jj, ii] = corr[ii, jj].

    Notes:
        - The expectation value <O> is computed for each pair of sites using matrix-vector multiplication.
        - The function runs in parallel over all pairs of lattice sites using `prange` for performance optimization.
    """
    # Initialize a 2D array to store expectation values for all pairs (ii, jj)
    corr = np.zeros((n_sites, n_sites), dtype=float)
    # Loop over all unique pairs of lattice sites in parallel using prange
    for ii in prange(n_sites):
        for jj in range(ii + 1, n_sites):
            # Compute the n-body operator's non-zero elements for the pair (ii, jj)
            row_list, col_list, value_list = nbody_data_par(
                sym_ops, np.array([ii, jj]), sector_configs
            )
            # Compute the expectation value <O> for the pair (ii, jj)
            exp_value = exp_val_data(psi, row_list, col_list, value_list)
            # Assign the result symmetrically
            corr[ii, jj] = exp_value
            # Mirror the value for (jj, ii)
            corr[jj, ii] = exp_value
    return corr
