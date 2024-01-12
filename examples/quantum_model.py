import numpy as np
from math import prod
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm, QMB_hamiltonian


class Quantum_Model:
    def __init__(self, params):
        self.lvals = params["lvals"]
        self.dim = len(self.lvals)
        self.directions = "xyz"[: self.dim]
        self.n_sites = prod(self.lvals)
        self.has_obc = params["has_obc"]
        self.coeffs = params["coeffs"]
        self.n_eigs = params["n_eigs"]
        self.site_basis = params["site_basis"]

    def build_Hamiltonian(self):
        # CONSTRUCT THE HAMILTONIAN
        self.H = QMB_hamiltonian(0, self.lvals, self.loc_dims)
        h_terms = {}
        # ---------------------------------------------------------------------------
        # NEAREST NEIGHBOR INTERACTION
        for d in self.directions:
            op_names_list = ["Sx", "Sx"]
            op_list = [self.ops[op] for op in op_names_list]
            # Define the Hamiltonian term
            h_terms[f"NN_{d}"] = TwoBodyTerm(
                axis=d,
                op_list=op_list,
                op_names_list=op_names_list,
                lvals=self.lvals,
                has_obc=self.has_obc,
            )
            self.H.Ham += h_terms[f"NN_{d}"].get_Hamiltonian(strength=-self.coeffs["J"])
        # EXTERNAL MAGNETIC FIELD
        op_name = "Sz"
        h_terms[op_name] = LocalTerm(
            self.ops[op_name], op_name, lvals=self.lvals, has_obc=self.has_obc
        )
        self.H.Ham += h_terms[op_name].get_Hamiltonian(strength=-self.coeffs["h"])

    def get_observables(self, local_obs, twobody_obs, plaq_obs):
        self.obs_list = {}
        # ---------------------------------------------------------------------------
        # LIST OF LOCAL OBSERVABLES
        for obs in local_obs:
            self.obs_list[obs] = LocalTerm(
                self.ops[obs], obs, lvals=self.lvals, has_obc=self.has_obc
            )
        # ---------------------------------------------------------------------------
        # LIST OF TWOBODY CORRELATORS
        for obs1, obs2 in twobody_obs:
            op_list = [self.ops[obs1], self.ops[obs2]]
            self.obs_list[f"{obs1}_{obs2}"] = TwoBodyTerm(
                axis="x",
                op_list=op_list,
                op_names_list=[obs1, obs2],
                lvals=self.lvals,
                has_obc=self.has_obc,
            )
        # ---------------------------------------------------------------------------
        # LIST OF PLAQUETTE CORRELATORS
        for plaq_names_list in plaq_obs:
            op_list = [self.ops[op] for op in plaq_names_list]
            self.obs_list["_".join(plaq_names_list)] = PlaquetteTerm(
                axes=["x", "y"],
                op_list=op_list,
                op_names_list=plaq_names_list,
                lvals=self.lvals,
                has_obc=self.has_obc,
                site_basis=self.site_basis,
            )
        # ---------------------------------------------------------------------------
        # COMPUTE EXPECTATION VALUES

    def get_qmb_state_properties(
        self,
        state_configs=False,
        entanglement_entropy=False,
        reduced_density_matrix=False,
    ):
        for ii in range(self.n_eigs):
            # PRINT ENERGY
            self.H.print_energy(ii)
            # STATE CONFIGURATIONS
            if state_configs:
                self.H.Npsi[ii].get_state_configurations(threshold=1e-3)
            # ENTANGLEMENT ENTROPY
            if entanglement_entropy:
                self.H.Npsi[ii].entanglement_entropy(int(self.n_sites / 2))
            # REDUCED DENSITY MATRIX EIGVALS
            if reduced_density_matrix:
                self.H.Npsi[ii].reduced_density_matrix(0)
