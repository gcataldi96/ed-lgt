"""Legacy XYZ spin model helper.

This module appears to follow an older API and may require adaptation before
use with the current :class:`edlgt.models.quantum_model.QuantumModel`
interface.
"""

import numpy as np
from edlgt.modeling import TwoBodyTerm, QMB_hamiltonian
from edlgt.operators import get_Pauli_operators
from .quantum_model import QuantumModel

__all__ = ["XYZModel"]


class XYZModel(QuantumModel):
    """Legacy helper for an anisotropic XYZ spin model."""

    def __init__(self, params):
        """Initialize the legacy XYZ model helper.

        Parameters
        ----------
        params : object
            Legacy parameter container used by the historical implementation.

        Notes
        -----
        This constructor uses an older calling convention and may need updates
        to match the current ``QuantumModel`` base class.
        """
        # Initialize base class with the common parameters
        super().__init__(params)
        # Initialize specific attributes for XYZModel
        self.loc_dims = np.array([2 for _ in range(self.n_sites)])

    def get_operators(self, sparse=True):
        """Load Pauli operators for the model.

        Parameters
        ----------
        sparse : bool, optional
            If ``False``, convert operators to dense arrays.
        """
        self.ops = get_Pauli_operators()
        if not sparse:
            for op in self.ops.keys():
                self.ops[op] = self.ops[op].toarray()

    def build_Hamiltonian(self):
        """Assemble the legacy XYZ Hamiltonian using nearest-neighbor couplings."""
        # CONSTRUCT THE HAMILTONIAN
        self.H = QMB_hamiltonian(0, self.lvals, self.loc_dims)
        h_terms = {}
        # ---------------------------------------------------------------------------
        # NEAREST NEIGHBOR INTERACTION
        twobody_terms = [["Sx", "Sx"], ["Sy", "Sy"], ["Sz", "Sz"]]
        twobody_coeffs = [1, 1, self.coeffs["Delta"]]
        for ii, op_names_list in enumerate(twobody_terms):
            for d in self.directions:
                op_list = [self.ops[op] for op in op_names_list]
                # Define the Hamiltonian term
                h_term_name = f"{d}_" + "_".join(op_names_list)
                h_terms[h_term_name] = TwoBodyTerm(
                    axis=d,
                    op_list=op_list,
                    op_names_list=op_names_list,
                    lvals=self.lvals,
                    has_obc=self.has_obc,
                    sector_indices=self.sector_indices,
                    sector_basis=self.sector_basis,
                )
                self.H.Ham += h_terms[h_term_name].get_Hamiltonian(
                    strength=twobody_coeffs[ii]
                )
