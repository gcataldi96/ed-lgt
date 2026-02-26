"""N-body interaction terms on hypercubic lattices.

This module defines :class:`NBodyTerm`, which applies an ordered list of local
operators to a site and its displaced neighbors, then sums the resulting term
over the lattice.
"""

import numpy as np
from math import prod
from scipy.sparse import isspmatrix, csr_matrix
from .lattice_mappings import zig_zag, inverse_zig_zag
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
            Relative lattice displacements between consecutive operators. Its
            length must be ``len(op_list) - 1``.
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
            raise ValueError(
                f"The length of distances should be len(op_list)-1, not {len(distances)}"
            )
        self.distances = distances
        # Preprocess arguments and initialize the base class
        super().__init__(op_list=op_list, op_names_list=op_names_list, **kwargs)
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
        scipy.sparse.spmatrix
            Sparse Hamiltonian contribution.

        Raises
        ------
        TypeError
            If ``strength`` is not scalar.
        NotImplementedError
            If called without a symmetry-sector basis.
        """
        if not np.isscalar(strength):
            raise TypeError(f"strength must be SCALAR, not {type(strength)}")
        validate_parameters(add_dagger=add_dagger)
        H_nbody = 0
        n_sites = prod(self.lvals)
        # GET ONLY THE SYMMETRY SECTOR of THE HAMILTONIAN TERM
        if self.sector_configs is not None:
            # Loop over all lattice sites (zig-zag order)
            for ii in range(n_sites):
                coords = zig_zag(self.lvals, ii)
                if not self.get_mask_conditions(coords, mask):
                    continue
                # Get neighboring sites based on the distances and the lattice geometry
                _, neighbor_sites = self.get_nbody_neighbors(coords)
                # Skip if neighbor sites are not valid / compatible with masks
                if neighbor_sites is None:
                    continue
                logger.info(f"sites {neighbor_sites}")
                H_nbody += strength * nbody_term(
                    op_list=self.sym_ops,
                    op_sites_list=np.array(neighbor_sites),
                    sector_configs=self.sector_configs,
                    momentum_basis=self.momentum_basis,
                    k=self.momentum_k,
                )
        else:
            raise NotImplementedError(
                "The case without symmetry sector has to be implemented"
            )
        if not isspmatrix(H_nbody):
            H_nbody = csr_matrix(H_nbody)
        if add_dagger:
            H_nbody += csr_matrix(H_nbody.conj().transpose())
        return H_nbody

    def get_expval(self, psi):
        """Compute site-resolved expectation values of the N-body term.

        Parameters
        ----------
        psi : edlgt.modeling.qmb_state.QMB_state
            Quantum state used to evaluate the term.

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
        # IN CASE OF SYMMETRY SECTOR
        if self.sector_configs is not None:
            # Loop over all lattice sites (zig-zag order)
            for ii in range(n_sites):
                coords = zig_zag(self.lvals, ii)
                # Get neighboring sites based on the distances and the lattice geometry
                _, neighbor_sites = self.get_nbody_neighbors(coords)
                # Skip if neighbor sites are not valid / compatible with masks
                if neighbor_sites is None:
                    continue
                exp_value = psi.expectation_value(
                    nbody_term(
                        op_list=self.sym_ops,
                        op_sites_list=np.array(neighbor_sites),
                        sector_configs=self.sector_configs,
                        momentum_basis=self.momentum_basis,
                        k=self.momentum_k,
                    )
                )
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
