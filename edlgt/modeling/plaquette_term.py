"""Plaquette interaction terms and plaquette observables on lattice models.

This module provides :class:`PlaquetteTerm`, a ``QMBTerm``
subclass for constructing four-body plaquette Hamiltonian contributions and for
measuring plaquette expectation values on quantum many-body states.
"""

import numpy as np
from math import prod
from scipy.sparse import isspmatrix, csr_matrix
from .lattice_geometry import get_plaquette_neighbors
from .lattice_mappings import zig_zag
from .qmb_operations import four_body_op
from .qmb_state import QMB_state
from .qmb_term import QMBTerm
from edlgt.tools import validate_parameters
from edlgt.symmetries import nbody_term
import logging

logger = logging.getLogger(__name__)

__all__ = ["PlaquetteTerm"]


class PlaquetteTerm(QMBTerm):
    """Four-body plaquette term on a lattice.

    The class supports both:

    - direct sparse-matrix construction (no symmetry-sector reduction), and
    - symmetry-sector workflows where Hamiltonian contributions are returned as
      ``(rows, cols, vals)`` triplets.
    """

    def __init__(self, axes, op_list, op_names_list, print_plaq=True, **kwargs):
        """Initialize a plaquette term definition.

        Parameters
        ----------
        axes : list
            Two lattice axes defining the plaquette plane (for example
            ``["x", "y"]``).
        op_list : list
            Operators used to build the plaquette term.
        op_names_list : list
            Human-readable names corresponding to ``op_list``.
        print_plaq : bool, optional
            If ``True``, print plaquette values when calling
            :meth:`get_expval`.
        **kwargs
            Additional keyword arguments forwarded to :class:`QMBTerm`.

        Returns
        -------
        None
        """
        validate_parameters(
            axes=axes,
            op_list=op_list,
            op_names_list=op_names_list,
            print_plaq=print_plaq,
        )
        # Preprocess arguments
        super().__init__(op_list=op_list, op_names_list=op_names_list, **kwargs)
        self.axes = axes
        self.print_plaq = print_plaq
        # Number of lattice sites
        self.n_sites = prod(self.lvals)
        axes_name = "".join(self.axes)
        logger.info(f"PlaqTerm {axes_name}: {' '.join(op_names_list)}")

    def get_Hamiltonian(self, strength, add_dagger=False, mask=None):
        """Build the plaquette Hamiltonian contribution.

        The plaquette term is summed over all lattice sites where a plaquette is
        defined (and where ``mask`` allows it), then multiplied by ``strength``.

        Parameters
        ----------
        strength : scalar
            Coupling constant multiplying the plaquette term.
        add_dagger : bool, optional
            If ``True``, add the Hermitian conjugate of the constructed term.
        mask : numpy.ndarray, optional
            Boolean mask controlling where the local term is applied.

        Returns
        -------
        scipy.sparse.csr_matrix or tuple
            Return type depends on the current workflow:

            - if ``self.sector_configs is None``: sparse matrix Hamiltonian term;
            - otherwise: ``(row_list, col_list, val_list)`` as three NumPy arrays in the
              symmetry-reduced basis.

        Raises
        ------
        TypeError
            If ``strength`` is not scalar or ``add_dagger`` is invalid.
        """
        if not np.isscalar(strength):
            raise TypeError(f"strength must be scalar, not {type(strength)}")
        validate_parameters(add_dagger=add_dagger)
        # --------------------------------------------------------------------
        # 1) No sector_configs ⇒ build one big sparse matrix
        if self.sector_configs is None:
            H = 0
            for ii in range(self.n_sites):
                coords = zig_zag(self.lvals, ii)
                _, sites = get_plaquette_neighbors(
                    coords, self.lvals, self.axes, self.has_obc
                )
                if sites is None or not self.get_mask_conditions(coords, mask):
                    continue
                H += strength * four_body_op(
                    op_list=self.op_list, op_sites_list=sites, **self.def_params
                )
            if not isspmatrix(H):
                H = csr_matrix(H)
            if add_dagger:
                H = H + csr_matrix(H.conj().T)
            return H

        # --------------------------------------------------------------------
        # 2) With sector_configs ⇒ collect (r,c,v) arrays
        all_r = []
        all_c = []
        all_v = []
        for ii in range(self.n_sites):
            coords = zig_zag(self.lvals, ii)
            _, sites = get_plaquette_neighbors(
                coords, self.lvals, self.axes, self.has_obc
            )
            # logger.info(f"coords: {ii} {coords}, sites: {sites}")
            if sites is None or not self.get_mask_conditions(coords, mask):
                continue
            if len(self.sector_configs) > 2**24.5:
                logger.info(f"Sites {sites}")
            # this gives three 1D arrays for this plaquette
            sites_array = np.array(sites, dtype=np.int32)
            r, c, v = nbody_term(
                op_list=self.sym_ops,
                op_sites_list=sites_array,
                sector_configs=self.sector_configs,
                momentum_basis=self.momentum_basis,
            )
            all_r.append(r)
            all_c.append(c)
            all_v.append(v)
        # merge them
        row_list = np.concatenate(all_r)
        col_list = np.concatenate(all_c)
        val_list = np.concatenate(all_v) * strength
        if add_dagger:
            # Add Hermitian conjugate after the full term construction
            dagger_row_list = col_list
            dagger_col_list = row_list
            dagger_val_list = np.conjugate(val_list)
            # Concatenate the dagger part to the original term
            row_list = np.concatenate([row_list, dagger_row_list])
            col_list = np.concatenate([col_list, dagger_col_list])
            val_list = np.concatenate([val_list, dagger_val_list])
        return row_list, col_list, val_list

    def get_expval(self, psi, component: str = "real", stag_label: str | None = None):
        """Compute plaquette expectation values site by site and aggregate statistics.

        Parameters
        ----------
        psi : QMB_state
            Quantum many-body state on which the expectation values are
            evaluated.
        component : str, optional
            Component of the plaquette operator to measure. Allowed values are
            ``"real"`` (Hermitian part) and ``"imag"`` (anti-Hermitian part).
        stag_label : str, optional
            Optional staggered-site selector passed through the common validation
            logic. Allowed values are ``"even"`` and ``"odd"``. If ``None``,
            all plaquettes are considered.

        Returns
        -------
        None
            Results are stored on the instance attributes ``obs``, ``var``,
            ``avg``, and ``std``.

        Raises
        ------
        TypeError
            If ``psi`` is not a ``QMB_state`` instance.
        ValueError
            If ``component`` is not ``"real"`` or ``"imag"``.

        Notes
        -----
        In symmetry-sector mode, variances are currently not computed.
        """
        # Check on parameters
        if not isinstance(psi, QMB_state):
            raise TypeError(f"psi must be instance of class:QMB_state not {type(psi)}")
        validate_parameters(stag_label=stag_label)
        if component not in ["real", "imag"]:
            raise ValueError(f"component must be 'real' or 'imag': got {component}")
        # ADVERTISE OF THE CHOSEN PART OF THE PLAQUETTE YOU WANT TO COMPUTE
        if self.print_plaq:
            logger.info(f"----------------------------------------------------")
            if stag_label is None:
                logger.info(f"PLAQUETTE: {' '.join(self.op_names_list)}")
            else:
                logger.info(f"PLAQUETTE: {' '.join(self.op_names_list)}")
            logger.info(f"----------------------------------------------------")
        # MEASURE NUMBER OF PLAQUETTES:
        list_of_plaq_sites = []
        list_of_plaq_strings = []
        for ii in range(self.n_sites):
            # Compute the corresponding coords of the BL site of the Plaquette
            coords = zig_zag(self.lvals, ii)
            coords_list, sites_list = get_plaquette_neighbors(
                coords, self.lvals, self.axes, self.has_obc
            )
            if sites_list is None:
                continue
            else:
                list_of_plaq_sites.append(sites_list)
                list_of_plaq_strings.append([f"{c}" for c in coords_list])
        self.obs = np.zeros(len(list_of_plaq_sites), dtype=float)
        self.var = np.zeros(len(list_of_plaq_sites), dtype=float)
        # IN CASE OF NO SYMMETRY SECTOR
        if self.sector_configs is None:
            for ii, sites_list in enumerate(list_of_plaq_sites):
                plaq_op = four_body_op(
                    op_list=self.op_list,
                    op_sites_list=sites_list,
                    **self.def_params,
                )
                if component == "real":
                    obs_op = 0.5 * (plaq_op + plaq_op.getH())
                elif component == "imag":
                    obs_op = (-0.5j) * (plaq_op - plaq_op.getH())
                self.obs[ii] = np.real(np.vdot(psi.psi, obs_op.dot(psi.psi)))
                tmp = obs_op.dot(psi.psi)
                self.var[ii] = np.real(np.vdot(tmp, tmp))
                self.var[ii] -= self.obs[ii] ** 2
                if self.print_plaq:
                    self.print_Plaquette(list_of_plaq_strings[ii], self.obs[ii])
        else:
            # GET THE EXPVAL ON THE SYMMETRY SECTOR
            for ii, sites_list in enumerate(list_of_plaq_sites):
                rows, cols, vals = nbody_term(
                    op_list=self.sym_ops,
                    op_sites_list=np.array(sites_list),
                    sector_configs=self.sector_configs,
                    momentum_basis=self.momentum_basis,
                )
                self.obs[ii] = psi.expectation_value((rows, cols, vals), component)
                # for the moment, variance is not computed in symmetry sectors
                if self.print_plaq:
                    self.print_Plaquette(list_of_plaq_strings[ii], self.obs[ii])
        # GET STATISTICS
        self.avg = np.mean(self.obs)
        self.std = np.sqrt(np.mean(np.abs(self.var)))
        if self.print_plaq:
            logger.info(f"{format(self.avg, '.10f')} +/- {format(self.std, '.10f')}")

    def print_Plaquette(self, sites_list, value):
        """Log a formatted ASCII representation of a plaquette value.

        Parameters
        ----------
        sites_list : list
            List of four coordinate labels describing the plaquette corners.
        value : float
            Plaquette value to print.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If ``sites_list`` is not a list or ``value`` is not a float.
        ValueError
            If ``sites_list`` does not contain exactly four entries.
        """
        if not isinstance(sites_list, list):
            raise TypeError(f"sites_list should be a LIST, not a {type(sites_list)}")
        if len(sites_list) != 4:
            raise ValueError(f"sites_list has 4 elements, not {str(len(sites_list))}")
        if not isinstance(value, float):
            raise TypeError(f"sites_list should be FLOAT, not a {type(value)}")
        if value > 0:
            value = format(value, ".10f")
        else:
            if np.abs(value) < 10 ** (-10):
                value = format(np.abs(value), ".10f")
            else:
                value = format(value, ".9f")
        logger.info(f"{sites_list[2]}------------{sites_list[3]}")
        logger.info(f"  |                  |")
        logger.info(f"  |   {value}   |")
        logger.info(f"  |                  |")
        logger.info(f"{sites_list[0]}------------{sites_list[1]}")
        logger.info("")
