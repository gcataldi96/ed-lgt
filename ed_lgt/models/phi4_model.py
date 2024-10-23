import numpy as np
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, QMB_hamiltonian

# from ed_lgt.operators import get_Pauli_operators
from ed_lgt.operators import bose_fermi_operators
from .quantum_model import QuantumModel

__all__ = ["IsingModel"]


class Phi4Model(QuantumModel):
    def __init__(self, n_max, **kwargs):
        # Initialize base class with the common parameters
        super().__init__(**kwargs)
        # Initialize specific attributes for IsingModel
        self.loc_dims = np.array(
            [n_max + 1 for _ in range(self.n_sites)], dtype=np.uint8
        )
        # Acquire operators
        # self.ops = get_Pauli_operators()
        self.ops = bose_fermi_operators.bose_operators(n_max=n_max)
        self.ops["phi"] = (1 / np.sqrt(2)) * (self.ops["b"] + self.ops["b_dagger"])
        self.ops["pi"] = (1j / np.sqrt(2)) * (self.ops["b_dagger"] - self.ops["b"])
        # Acquire local dimension and lattice label
        self.get_local_site_dimensions()

    def build_Hamiltonian(self, coeffs):
        # Hamiltonian Coefficients
        self.coeffs = coeffs
        # CONSTRUCT THE HAMILTONIAN
        h_terms = {}
        # ---------------------------------------------------------------------------
        
        # NEAREST NEIGHBOR INTERACTION
        op_names_list = ["Sx", "Sx"]
        op_list = [self.ops[op] for op in op_names_list]
        for d in self.directions:
            # Define the Hamiltonian term
            h_terms[f"NN_{d}"] = TwoBodyTerm(
                axis=d, op_list=op_list, op_names_list=op_names_list, **self.def_params
           )
            self.H.Ham += h_terms[f"NN_{d}"].get_Hamiltonian(strength=-self.coeffs["J"])
        
        
        # SINGLE BODY TERM
        op_name="Sz"
        h_terms[op_name] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
        self.H.Ham += h_terms[op_name].get_Hamiltonian(strength=-self.coeffs["h"])


a = Phi4Model(lvals=[2], has_obc=[False], n_max=3)
