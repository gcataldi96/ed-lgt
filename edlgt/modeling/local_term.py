"""Local lattice terms and local-observable measurements.

This module provides :class:`LocalTerm`, a ``QMBTerm`` subclass for building
single-site Hamiltonian contributions and measuring local observables across a
lattice. It also includes a helper to validate link-symmetry relations between
two measured local observables.
"""

import numpy as np
from math import prod
from .lattice_mappings import zig_zag, inverse_zig_zag
from .lattice_geometry import get_neighbor_sites
from .qmb_operations import local_op
from .qmb_state import QMB_state, exp_val_data
from .qmb_term import QMBTerm
from edlgt.tools import validate_parameters, get_time
from edlgt.symmetries import nbody_term
import logging

logger = logging.getLogger(__name__)

__all__ = ["LocalTerm", "check_link_symmetry"]


class LocalTerm(QMBTerm):
    """Single-site term on a lattice model."""

    def __init__(self, operator: np.ndarray, op_name: str, **kwargs):
        """Initialize a local term definition.

        Parameters
        ----------
        operator : scipy.sparse.spmatrix
            Single-site operator.
        op_name : str
            Human-readable operator name.
        **kwargs
            Additional keyword arguments forwarded to ``QMBTerm``.

        Returns
        -------
        None
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
        """Build the local Hamiltonian contribution.

        Parameters
        ----------
        strength : scalar
            Coupling constant multiplying the local term.
        mask : numpy.ndarray, optional
            Boolean lattice mask selecting the sites where the term is applied.

        Returns
        -------
        scipy.sparse.spmatrix or tuple
            Return type depends on the current workflow:

            - if ``self.sector_configs is None``: sparse matrix Hamiltonian term;
            - otherwise: ``(r_list, c_list, v_list)`` as three NumPy arrays in the
              symmetry-reduced basis.

        Raises
        ------
        TypeError
            If ``strength`` is not scalar.
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
            all_r_list = []
            all_c_list = []
            all_v_list = []
            for ii in range(self.n_sites):
                coords = zig_zag(self.lvals, ii)
                # CHECK MASK CONDITION ON THE SITE
                if self.get_mask_conditions(coords, mask):
                    if len(self.sector_configs) > 2**24.5:
                        logger.info(f"Site {ii}")
                    # GET ONLY THE SYMMETRY SECTOR of THE HAMILTONIAN TERM
                    r_list, c_list, v_list = nbody_term(
                        self.sym_ops,
                        np.array([ii]),
                        self.sector_configs,
                        self.momentum_basis,
                    )
                    all_r_list.append(r_list)
                    all_c_list.append(c_list)
                    all_v_list.append(v_list)
            # Concatenate global lists
            r_list = np.concatenate(all_r_list)
            c_list = np.concatenate(all_c_list)
            v_list = np.concatenate(all_v_list) * strength
            return r_list, c_list, v_list

    def get_expval(self, psi, stag_label=None, print_values=True, get_variance=False):
        """Compute local expectation values (and optionally variances) on all sites.

        Parameters
        ----------
        psi : QMB_state
            Quantum many-body state used for the measurement.
        stag_label : str, optional
            Optional staggered-site selector (``"even"`` or ``"odd"``).
        print_values : bool, optional
            If ``True``, log per-site values and final averages.
        get_variance : bool, optional
            If ``True``, also compute local variances when supported.

        Returns
        -------
        None
            Results are stored on the instance attributes ``obs``, ``var`` (if
            requested), ``avg``, and ``std``.

        Raises
        ------
        TypeError
            If ``psi`` is not a ``QMB_state`` instance.

        Notes
        -----
        For local operators without momentum basis, squaring the non-zero matrix
        entries is sufficient to compute the variance contribution. In momentum
        basis, the variance requires constructing a dedicated operator.
        """
        # Check on parameters
        if not isinstance(psi, QMB_state):
            raise TypeError(f"psi must be instance of class:QMB_state not {type(psi)}")
        validate_parameters(stag_label=stag_label)
        # Initialize arrays for storing expectation values and variances for all sites
        self.obs = np.zeros(self.n_sites, dtype=float)  # Stores exp values <O>
        if get_variance:
            self.std = 0.0
            self.var = np.zeros(self.n_sites, dtype=float)  # Stores var <O^2> - <O>^2
        # PRINT OBSERVABLE NAME
        if print_values:
            msg = "" if stag_label is None else f"{stag_label}"
            logger.info(f"----------------------------------------------------")
            logger.info(f"{self.op_name} {msg}")
        # =============================================================================
        if self.sector_configs is None:
            for ii in range(self.n_sites):
                exp_val = psi.expectation_value(
                    local_op(operator=self.op, op_site=ii, **self.def_params)
                )
                self.obs[ii] = float(np.real(exp_val))
                if get_variance:
                    var_val = psi.expectation_value(
                        local_op(operator=self.op**2, op_site=ii, **self.def_params)
                    )
                    self.var[ii] = float(np.real(var_val))
                    self.var[ii] -= self.obs[ii] ** 2
        # =============================================================================
        else:
            # COMPUTE THE LOCAL OBSERVABLE
            for ii in range(self.n_sites):
                r_list, c_list, v_list = nbody_term(
                    self.sym_ops,
                    np.array([ii]),
                    self.sector_configs,
                    self.momentum_basis,
                )
                exp_val = exp_val_data(psi.psi, r_list, c_list, v_list)
                self.obs[ii] = float(np.real(exp_val))
                if get_variance and self.momentum_basis is None:
                    # Without momentum basis, <O^2> - <O>^2 CAN be computed by squaring v_list
                    var_val = exp_val_data(psi.psi, r_list, c_list, v_list**2)
                    self.var[ii] = float(np.real(var_val))
                    self.var[ii] -= self.obs[ii] ** 2
            # COMPUTE THE VARIANCE within MOMENTUM BASIS
            if get_variance and self.momentum_basis is not None:
                # Compute the operator for the variance
                shape = (1, self.n_sites, self.max_loc_dim, self.max_loc_dim)
                opvar = np.zeros(shape, dtype=float)
                for ii in range(self.n_sites):
                    opvar[0, ii] = np.dot(self.sym_ops[0, ii], self.sym_ops[0, ii])
                # Compute the variance
                for ii in range(self.n_sites):
                    # In momentum basis, the variance <O^2> - <O>^2
                    # can NOT be computed by squaring v_list
                    r_list1, c_list1, v_list1 = nbody_term(
                        opvar,
                        np.array([ii]),
                        self.sector_configs,
                        self.momentum_basis,
                    )
                    var_val = exp_val_data(psi.psi, r_list1, c_list1, v_list1)
                    self.var[ii] = float(np.real(var_val))
                    self.var[ii] -= self.obs[ii] ** 2
        # =============================================================================
        # CHECK STAGGERED CONDITION AND PRINT VALUES
        self.avg = 0.0
        self.std = 0.0
        counter = 0
        for ii in range(self.n_sites):
            # Given the 1D point on the d-dimensional lattice, get the corresponding coords
            coords = zig_zag(self.lvals, ii)
            if self.get_staggered_conditions(coords, stag_label):
                if print_values:
                    logger.info(f"{coords} {format(self.obs[ii], '.16f')}")
                counter += 1
                self.avg += self.obs[ii]
                if get_variance:
                    self.std += self.var[ii]
        self.avg = self.avg / counter
        if get_variance:
            self.std = np.sqrt(np.abs(self.std) / counter)
        if print_values:
            if get_variance:
                msg = f"{format(self.avg, '.16f')} +/- {format(self.std, '.16f')}"
            else:
                msg = f"{format(self.avg, '.16f')}"
            logger.info(msg)


def check_link_symmetry(axis, loc_op1, loc_op2, value=0, sign=1):
    """Check a link-symmetry relation between two measured local observables.

    Parameters
    ----------
    axis : str
        Lattice axis along which neighboring sites are paired.
    loc_op1, loc_op2 : LocalTerm
        Local-term objects whose ``obs`` arrays are compared.
    value : float, optional
        Expected value of ``loc_op1.obs[i] + sign * loc_op2.obs[j]``.
    sign : int, optional
        Relative sign used in the comparison (typically ``+1`` or ``-1``).

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``loc_op1`` or ``loc_op2`` is not a ``LocalTerm`` instance.
    ValueError
        If the symmetry relation is violated beyond the tolerance.
    """
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
