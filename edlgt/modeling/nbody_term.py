"""N-body interaction terms on hypercubic lattices.

This module defines :class:`NBodyTerm`, which applies an ordered list of local
operators to a site and its displaced neighbors, then sums the resulting term
over the lattice.
"""

import numpy as np
from math import prod
from scipy.sparse import isspmatrix, csr_matrix
from .lattice_mappings import zig_zag, inverse_zig_zag
from .qmb_operations import construct_operator_list, qmb_operator
from .qmb_state import QMB_state
from .qmb_term import QMBTerm
from edlgt.tools import validate_parameters
from edlgt.symmetries import nbody_term
import logging

logger = logging.getLogger(__name__)

__all__ = ["NBodyTerm"]


class NBodyTerm(QMBTerm):
    def __init__(self, op_list, op_names_list, distances, **kwargs):
        """Initialize an N-body lattice term.

        Parameters
        ----------
        op_list : list
            Operators participating in the N-body term.
        op_names_list : list[str]
            Labels corresponding to ``op_list``.
        distances : list[tuple]
            Relative lattice displacements from the starting site to each
            additional operator. Its length must be ``len(op_list) - 1``.
        **kwargs
            Additional arguments forwarded to :class:`~edlgt.modeling.qmb_term.QMBTerm`.

        Raises
        ------
        ValueError
            If ``len(distances) != len(op_list) - 1``.
        """
        # CHECK ON TYPES
        validate_parameters(op_list=op_list, op_names_list=op_names_list)
        # Store the distances and the starting site for operator application
        if not len(distances) == len(op_list) - 1:
            msg = f"The distances length should be len(op_list)-1, not {len(distances)}"
            raise ValueError(msg)
        # Preprocess arguments and initialize the base class
        super().__init__(op_list=op_list, op_names_list=op_names_list, **kwargs)
        if not all(
            isinstance(dist, (list, tuple))
            and len(dist) == len(self.lvals)
            and all(isinstance(delta, int) for delta in dist)
            for dist in distances
        ):
            raise TypeError(
                "Each entry in distances must be a list/tuple of ints with "
                f"length len(lvals)={len(self.lvals)}."
            )
        self.distances = distances
        logger.info("N_bodyterm_" + "_".join(op_names_list))

    def get_Hamiltonian(self, strength, add_dagger=False, mask=None):
        """Assemble the lattice-summed N-body Hamiltonian term.

        Parameters
        ----------
        strength : scalar
            Coupling constant multiplying the term.
        add_dagger : bool, optional
            If ``True``, add the Hermitian conjugate of the assembled term.
        mask : numpy.ndarray, optional
            Boolean mask selecting the starting sites where the term is applied.

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
            If ``add_dagger=True`` is requested in unsupported momentum-basis
            pair mode.
        """
        if not np.isscalar(strength):
            raise TypeError(f"strength must be SCALAR, not {type(strength)}")
        validate_parameters(add_dagger=add_dagger)
        n_sites = prod(self.lvals)
        # --------------------------------------------------------------------
        # 1) No sector_configs -> build one sparse matrix
        if self.sector_configs is None:
            H_nbody = 0
            for ii in range(n_sites):
                coords = zig_zag(self.lvals, ii)
                if not self.get_mask_conditions(coords, mask):
                    continue
                # Get neighboring sites based on the distances and the lattice geometry
                _, neighbor_sites = self.get_nbody_neighbors(coords)
                # Skip if neighbor sites are not valid / compatible with masks
                if neighbor_sites is None:
                    continue
                ops, op_names = construct_operator_list(
                    op_list=self.op_list,
                    op_sites_list=neighbor_sites,
                    **self.def_params,
                )
                H_nbody += strength * qmb_operator(ops, op_names)
            if not isspmatrix(H_nbody):
                H_nbody = csr_matrix(H_nbody)
            if add_dagger:
                H_nbody += csr_matrix(H_nbody.conj().transpose())
            return H_nbody

        # --------------------------------------------------------------------
        # 2) With sector_configs -> collect (row, col, value) arrays
        all_row_list = []
        all_col_list = []
        all_value_list = []
        for ii in range(n_sites):
            coords = zig_zag(self.lvals, ii)
            if not self.get_mask_conditions(coords, mask):
                continue
            _, neighbor_sites = self.get_nbody_neighbors(coords)
            if neighbor_sites is None:
                continue
            if len(self.sector_configs) > 2**24.5:
                logger.info(f"sites {neighbor_sites}")
            row_list, col_list, value_list = nbody_term(
                op_list=self.sym_ops,
                op_sites_list=np.array(neighbor_sites, dtype=np.int32),
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
            pair_mode = self.momentum_basis.get("pair_mode", False)
            if self.momentum_basis is not None and pair_mode:
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

    def get_expval(self, psi, component: str = "real"):
        """Compute site-resolved expectation values of the N-body term.

        Parameters
        ----------
        psi : edlgt.modeling.qmb_state.QMB_state
            Quantum state used to evaluate the term.
        component : str, optional
            Output component selector: ``"real"`` or ``"imag"``.

        Returns
        -------
        None
            Results are stored in ``self.obs`` as a 1D NumPy array.

        Raises
        ------
        TypeError
            If ``psi`` is not a :class:`~edlgt.modeling.qmb_state.QMB_state`.
        """
        # Check on parameters
        if not isinstance(psi, QMB_state):
            raise TypeError(f"psi must be instance of class:QMB_state not {type(psi)}")
        # PRINT OBSERVABLE NAME
        logger.info(f"----------------------------------------------------")
        logger.info(f"{'-'.join(self.op_names_list)}")
        n_sites = prod(self.lvals)
        # Store the expectation values for each N-body term in a 1D array
        self.obs = []
        # Loop over all lattice sites (zig-zag order)
        for ii in range(n_sites):
            coords = zig_zag(self.lvals, ii)
            # Get neighboring sites based on the distances and the lattice geometry
            _, neighbor_sites = self.get_nbody_neighbors(coords)
            # Skip if neighbor sites are not valid / compatible with masks
            if neighbor_sites is None:
                continue
            if self.sector_configs is None:
                ops, op_names = construct_operator_list(
                    op_list=self.op_list,
                    op_sites_list=neighbor_sites,
                    **self.def_params,
                )
                operator = qmb_operator(ops, op_names)
            else:
                operator = nbody_term(
                    op_list=self.sym_ops,
                    op_sites_list=np.array(neighbor_sites, dtype=np.int32),
                    sector_configs=self.sector_configs,
                    momentum_basis=self.momentum_basis,
                )
            exp_value = psi.expectation_value(operator, component=component)
            # Store the result in the self.obs list
            self.obs.append(exp_value)
            logger.info(f"{coords} {format(exp_value, '.10f')}")
        # Finalize by converting the list to a 1D numpy array
        self.obs = np.array(self.obs)

    def get_nbody_neighbors(self, coords):
        """Compute the ordered lattice sites touched by the N-body pattern.

        Parameters
        ----------
        coords : tuple
            Coordinates of the starting site.

        Returns
        -------
        tuple
            ``(neighbor_coords, neighbor_sites)``. If the pattern exits the
            lattice under open boundaries, returns ``(None, None)``.

        Notes
        -----
        Distances are interpreted as displacements from the starting site
        ``coords`` (not cumulative displacements between successive operators).
        """
        neighbor_coords = [coords]
        # Iterate through each distance and compute the new coordinates
        for dist in self.distances:
            new_coords = list(coords)
            for jj in range(len(self.lvals)):
                new_coords[jj] += dist[jj]
                # Apply periodic boundary conditions (PBC)
                if not self.has_obc[jj]:
                    new_coords[jj] %= self.lvals[jj]
            # Check if new_coords are within the lattice bounds for OBC
            valid = all(
                0 <= new_coords[kk] < self.lvals[kk] for kk in range(len(self.lvals))
            )
            if valid:
                neighbor_coords.append(tuple(new_coords))
            else:
                # Invalid neighbors, skip this N-body term
                return None, None
        # Convert coordinates to lattice indices using inverse_zig_zag
        neighbor_sites = [
            inverse_zig_zag(self.lvals, coords) for coords in neighbor_coords
        ]
        return neighbor_coords, neighbor_sites
