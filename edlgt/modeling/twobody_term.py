"""Two-body lattice interaction terms and correlation measurements.

This module provides :class:`TwoBodyTerm`, a ``QMBTerm`` subclass for
constructing nearest-neighbor two-body Hamiltonian contributions and measuring
two-point correlators on lattice models.
"""

import numpy as np
from math import prod
from numba import njit, prange
from scipy.sparse import isspmatrix, csr_matrix
from .lattice_geometry import get_neighbor_sites
from .lattice_mappings import zig_zag, inverse_zig_zag
from .qmb_operations import two_body_op
from .qmb_state import QMB_state, exp_val_data
from .qmb_term import QMBTerm
from edlgt.tools import validate_parameters
from edlgt.symmetries import nbody_term, nbody_data_2sites
import logging

logger = logging.getLogger(__name__)

__all__ = ["TwoBodyTerm"]


class TwoBodyTerm(QMBTerm):
    """Nearest-neighbor two-body term along a selected lattice axis."""

    def __init__(self, axis, op_list, op_names_list, **kwargs):
        """Initialize a two-body term definition.

        Parameters
        ----------
        axis : str
            Lattice axis along which neighboring pairs are selected.
        op_list : list
            Two local operators defining the two-body term.
        op_names_list : list
            Human-readable names corresponding to ``op_list``.
        **kwargs
            Additional keyword arguments forwarded to ``QMBTerm``.

        Returns
        -------
        None
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
        """Build the two-body Hamiltonian contribution.

        Parameters
        ----------
        strength : scalar
            Coupling constant multiplying the two-body term.
        add_dagger : bool, optional
            If ``True``, add the Hermitian conjugate of the constructed term.
        mask : numpy.ndarray, optional
            Boolean lattice mask selecting the sites where the term is applied.

        Returns
        -------
        scipy.sparse.spmatrix or tuple
            Return type depends on the current workflow:

            - if ``self.sector_configs is None``: sparse matrix Hamiltonian term;
            - otherwise: ``(row_list, col_list, value_list)`` as three NumPy arrays in the
              symmetry-reduced basis.

        Raises
        ------
        TypeError
            If ``strength`` is not scalar.
        NotImplementedError
            If ``add_dagger=True`` is requested in an unsupported momentum-basis
            pair mode.
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
                    if len(self.sector_configs) > 2**24.5:
                        logger.info(f"Sites {sites_list}")
                    # GET ONLY THE SYMMETRY SECTOR of THE HAMILTONIAN TERM
                    row_list, col_list, value_list = nbody_term(
                        op_list=self.sym_ops,
                        op_sites_list=np.array(sites_list),
                        sector_configs=self.sector_configs,
                        momentum_basis=self.momentum_basis,
                    )
                    all_row_list.append(row_list)
                    all_col_list.append(col_list)
                    all_value_list.append(value_list)
            row_list = np.concatenate(all_row_list)
            col_list = np.concatenate(all_col_list)
            value_list = np.concatenate(all_value_list) * strength
            if add_dagger:
                if self.momentum_basis is not None and self.momentum_basis["pair_mode"]:
                    raise NotImplementedError("Add the explicit HC term!")
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
        """Compute the two-point correlator matrix for the selected operator pair.

        Parameters
        ----------
        psi : QMB_state
            Quantum many-body state used for the measurement.

        Returns
        -------
        None
            The correlator matrix is stored in ``self.corr``.

        Raises
        ------
        TypeError
            If ``psi`` is not a ``QMB_state`` instance.
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
        """Log nearest-neighbor correlator values along the configured axis.

        Returns
        -------
        None
        """
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
            row_list, col_list, value_list = nbody_data_2sites(
                sym_ops, np.array([ii, jj]), sector_configs
            )
            # Compute the expectation value <O> for the pair (ii, jj)
            exp_value = exp_val_data(psi, row_list, col_list, value_list)
            # Assign the result symmetrically
            corr[ii, jj] = float(np.real(exp_value))
            # Mirror the value for (jj, ii)
            corr[jj, ii] = float(np.real(exp_value))
    return corr
