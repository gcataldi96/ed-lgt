import numpy as np
from scipy.sparse import csc_matrix

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
        self.ops = bose_fermi_operators.bose_operators(n_max=n_max)
        self.ops["phi"] = (1 / np.sqrt(2)) * (self.ops["b"] + self.ops["b_dagger"])
        self.ops["pi"] = (1j / np.sqrt(2)) * (self.ops["b_dagger"] - self.ops["b"])

        self.ops["phi2"] = self.ops["phi"] @ self.ops["phi"]
        self.ops["phi4"] = self.ops["phi2"] @ self.ops["phi2"]
        self.ops["pi2"] = self.ops["pi"] @ self.ops["pi"]
        self.ops["Id"] = csc_matrix(np.identity(n_max+1))

        # Acquire local dimension and lattice label
        self.get_local_site_dimensions()

    def build_Hamiltonian(self, coeffs):
        # Hamiltonian Coefficients
        self.coeffs = coeffs
        # CONSTRUCT THE HAMILTONIAN
        h_terms = {}
        # ---------------------------------------------------------------------------

        # NEAREST NEIGHBOR INTERACTION
        op_names_list = ["Id", "phi2"]
        op_list = [self.ops[op] for op in op_names_list]
        for d in self.directions:
            # Define the Hamiltonian term
            h_terms[f"NN_{d}"] = TwoBodyTerm(
                axis=d, op_list=op_list, op_names_list=op_names_list, **self.def_params
            )
            self.H.Ham += h_terms[f"NN_{d}"].get_Hamiltonian(strength=0.5)

        op_names_list = ["phi", "phi"]
        op_list = [self.ops[op] for op in op_names_list]
        for d in self.directions:
            # Define the Hamiltonian term
            h_terms[f"NN_{d}"] = TwoBodyTerm(
                axis=d, op_list=op_list, op_names_list=op_names_list, **self.def_params
            )
            self.H.Ham += h_terms[f"NN_{d}"].get_Hamiltonian(strength=1)

        op_names_list = ["phi2", "Id"]
        op_list = [self.ops[op] for op in op_names_list]
        for d in self.directions:
            # Define the Hamiltonian term
            h_terms[f"NN_{d}"] = TwoBodyTerm(
                axis=d, op_list=op_list, op_names_list=op_names_list, **self.def_params
            )
            self.H.Ham += h_terms[f"NN_{d}"].get_Hamiltonian(strength=0.5)

        # SINGLE BODY TERM
        op_name = "pi2"
        h_terms[op_name] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
        self.H.Ham += h_terms[op_name].get_Hamiltonian(strength=0.5)

        op_name = "phi2"
        h_terms[op_name] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
        self.H.Ham += h_terms[op_name].get_Hamiltonian(
            strength=0.5 * self.coeffs["mu2"]
        )

        op_name = "phi4"
        h_terms[op_name] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
        self.H.Ham += h_terms[op_name].get_Hamiltonian(
            strength=self.coeffs["lambda"] / (24)
        )

