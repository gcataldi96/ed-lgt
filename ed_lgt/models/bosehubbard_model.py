import numpy as np
from ed_lgt.modeling import abelian_sector_indices
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, QMB_hamiltonian
from ed_lgt.operators import bose_operators
from .quantum_model import QuantumModel

__all__ = ["BoseHubbard_Model"]


class BoseHubbard_Model(QuantumModel):
    def __init__(self, params):
        # Initialize base class with the common parameters
        super().__init__(params)
        # Initialize specific attributes for BOSE HUBBARD model
        self.n_max = params["n_max"]
        self.loc_dims = np.array([self.n_max + 1 for _ in range(self.n_sites)])

    def get_operators(self, sparse=True):
        self.ops = bose_operators()
        if not sparse:
            for op in self.ops.keys():
                self.ops[op] = self.ops[op].toarray()

    def build_Hamiltonian(self):
        # CONSTRUCT THE HAMILTONIAN
        self.H = QMB_hamiltonian(0, self.lvals, self.loc_dims)
        h_terms = {}
        # ---------------------------------------------------------------------------
        # NEAREST NEIGHBOR INTERACTION
        for d in self.directions:
            op_names_list = ["b_dagger", "b"]
            op_list = [self.ops[op] for op in op_names_list]
            # Define the Hamiltonian term
            h_terms[f"NN_{d}"] = TwoBodyTerm(
                axis=d,
                op_list=op_list,
                op_names_list=op_names_list,
                lvals=self.lvals,
                has_obc=self.has_obc,
                sector_basis=self.sector_basis,
                sector_indices=self.sector_indices,
            )
            self.H.Ham += h_terms[f"NN_{d}"].get_Hamiltonian(
                strength=-self.coeffs["t"], add_dagger=True
            )
        # SINGLE SITE POTENTIAL
        op_name = "N2"
        h_terms[op_name] = LocalTerm(
            operator=self.ops[op_name],
            op_name=op_name,
            lvals=self.lvals,
            has_obc=self.has_obc,
            sector_basis=self.sector_basis,
            sector_indices=self.sector_indices,
        )
        self.H.Ham += h_terms[op_name].get_Hamiltonian(strength=0.5 * self.coeffs["U"])
        op_name = "N"
        h_terms[op_name] = LocalTerm(
            operator=self.ops[op_name],
            op_name=op_name,
            lvals=self.lvals,
            has_obc=self.has_obc,
            sector_basis=self.sector_basis,
            sector_indices=self.sector_indices,
        )
        self.H.Ham += h_terms[op_name].get_Hamiltonian(strength=-0.5 * self.coeffs["U"])
        # ADD SINGLE SITE NOISE
        noise = np.random.rand(self.n_sites)
        for ii in self.n_sites:
            mask = np.zeros(dtype=bool)
            mask[ii] = True
            self.H.Ham += h_terms["N"].get_Hamiltonian(strength=noise[ii], mask=mask)
