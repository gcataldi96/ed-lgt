import numpy as np
from edlgt.modeling import LocalTerm, TwoBodyTerm, QMB_hamiltonian
from edlgt.operators import bose_operators
from .quantum_model import QuantumModel

__all__ = ["BoseHubbard_Model"]


class BoseHubbard_Model(QuantumModel):
    def __init__(self, n_max, sectors, **kwargs):
        # Initialize base class with the common parameters
        super().__init__(**kwargs)
        # Initialize specific attributes for BOSE HUBBARD model
        self.n_max = n_max
        self.loc_dims = np.array(
            [self.n_max + 1 for _ in range(self.n_sites)], dtype=np.uint8
        )
        # Acquire operators
        self.ops = bose_operators(self.n_max)
        # Acquire lattice label
        self.get_local_site_dimensions()
        # GLOBAL SYMMETRY
        global_ops = [self.ops["N"]]
        global_sectors = sectors
        # GET SYMMETRY SECTOR
        self.get_abelian_symmetry_sector(global_ops, global_sectors)
        self.default_params()

    def build_Hamiltonian(self, coeffs):
        # Hamiltonian Coefficients
        self.coeffs = coeffs
        # CONSTRUCT THE HAMILTONIAN
        h_terms = {}
        # ---------------------------------------------------------------------------
        # NEAREST NEIGHBOR INTERACTION
        for d in self.directions:
            op_names_list = ["b_dagger", "b"]
            op_list = [self.ops[op] for op in op_names_list]
            # Define the Hamiltonian term
            h_terms[f"NN_{d}"] = TwoBodyTerm(
                axis=d, op_list=op_list, op_names_list=op_names_list, **self.def_params
            )
            self.H.Ham += h_terms[f"NN_{d}"].get_Hamiltonian(
                strength=-self.coeffs["t"], add_dagger=True
            )
        # SINGLE SITE POTENTIAL
        op_name = "N2"
        h_terms[op_name] = LocalTerm(
            operator=self.ops[op_name], op_name=op_name, **self.def_params
        )
        self.H.Ham += h_terms[op_name].get_Hamiltonian(strength=0.5 * self.coeffs["U"])
        op_name = "N"
        h_terms[op_name] = LocalTerm(
            operator=self.ops[op_name], op_name=op_name, **self.def_params
        )
        self.H.Ham += h_terms[op_name].get_Hamiltonian(strength=-0.5 * self.coeffs["U"])
        # ADD SINGLE SITE NOISE
        noise = np.random.rand(self.n_sites)
        for ii in range(self.n_sites):
            mask = np.zeros(self.n_sites, dtype=bool)
            mask[ii] = True
            self.H.Ham += h_terms["N"].get_Hamiltonian(strength=noise[ii], mask=mask)
