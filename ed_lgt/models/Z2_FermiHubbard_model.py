from ed_lgt.operators import (
    Z2_FermiHubbard_dressed_site_operators,
    Z2_FermiHubbard_gauge_invariant_states,
)
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, QMB_hamiltonian
from ed_lgt.modeling import check_link_symmetry
from .quantum_model import QuantumModel


__all__ = ["Z2_FermiHubbard_Model"]


class Z2_FermiHubbard_Model(QuantumModel):
    def __init__(self, **kwargs):
        # Initialize base class with the common parameters
        super().__init__(**kwargs)
        # Acquire operators
        self.ops = Z2_FermiHubbard_dressed_site_operators(lattice_dim=self.dim)
        # Acquire gauge invariant basis and states
        self.gauge_basis, self.gauge_states = Z2_FermiHubbard_gauge_invariant_states(
            lattice_dim=self.dim
        )
        # Acquire local dimension and lattice label
        self.get_local_site_dimensions()

    def build_Hamiltonian(self, coeffs):
        # Hamiltonian Coefficients
        self.coeffs = coeffs
        # CONSTRUCT THE HAMILTONIAN
        self.H = QMB_hamiltonian(0, self.lvals, self.loc_dims)
        h_terms = {}
        # -------------------------------------------------------------------------------
        # COULOMB POTENTIAL
        op_name = "N_pair_half"
        h_terms["V"] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
        self.H.Ham += h_terms["V"].get_Hamiltonian(strength=self.coeffs["U"])
        # -------------------------------------------------------------------------------
        # HOPPING
        for d in self.directions:
            for s in ["up", "down"]:
                # Define the list of the 2 non trivial operators
                op_names_list = [f"Q{s}_p{d}_dag", f"Q{s}_m{d}"]
                op_list = [self.ops[op] for op in op_names_list]
                # Define the Hamiltonian term
                h_terms[f"{d}_hop_{s}"] = TwoBodyTerm(
                    d, op_list, op_names_list, **self.def_params
                )
                self.H.Ham += h_terms[f"{d}_hop_{s}"].get_Hamiltonian(
                    strength=self.coeffs["t"], add_dagger=True
                )

    def check_symmetries(self):
        # CHECK LINK SYMMETRIES
        for ax in self.directions:
            check_link_symmetry(
                ax,
                self.obs_list[f"n_p{ax}"],
                self.obs_list[f"n_m{ax}"],
                value=0,
                sign=-1,
            )
