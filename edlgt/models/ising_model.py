import numpy as np
from edlgt.modeling import LocalTerm, TwoBodyTerm, QMB_hamiltonian
from edlgt.operators import get_Pauli_operators
from .quantum_model import QuantumModel

__all__ = ["IsingModel"]


class IsingModel(QuantumModel):
    def __init__(self, **kwargs):
        # Initialize base class with the common parameters
        super().__init__(**kwargs)
        # Initialize specific attributes for IsingModel
        self.loc_dims = np.array([2 for _ in range(self.n_sites)], dtype=np.uint8)
        # Acquire operators
        self.ops = get_Pauli_operators()
        # Acquire local dimension and lattice label
        self.get_local_site_dimensions()

    def build_Hamiltonian(self, coeffs):
        # Hamiltonian Coefficients
        self.coeffs = coeffs
        # CONSTRUCT THE HAMILTONIAN
        h_terms = {}
        # ---------------------------------------------------------------------------
        # NEAREST NEIGHBOR INTERACTION
        for d in self.directions:
            op_names_list = ["Sx", "Sx"]
            op_list = [self.ops[op] for op in op_names_list]
            # Define the Hamiltonian term
            h_terms[f"NN_{d}"] = TwoBodyTerm(
                axis=d, op_list=op_list, op_names_list=op_names_list, **self.def_params
            )
            self.H.Ham += h_terms[f"NN_{d}"].get_Hamiltonian(strength=-self.coeffs["J"])
        # EXTERNAL MAGNETIC FIELD
        op_name = "Sz"
        h_terms[op_name] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
        self.H.Ham += h_terms[op_name].get_Hamiltonian(strength=-self.coeffs["h"])
