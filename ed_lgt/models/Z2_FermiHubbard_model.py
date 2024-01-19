from math import prod
from scipy.sparse import identity as ID
from ed_lgt.operators import (
    Z2_FermiHubbard_dressed_site_operators,
    Z2_FermiHubbard_gauge_invariant_states,
)
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm, QMB_hamiltonian
from ed_lgt.modeling import (
    check_link_symmetry,
    lattice_base_configs,
)


__all__ = ["Z2_FermiHubbard_Model"]


class Z2_FermiHubbard_Model:
    def __init__(self, params):
        self.lvals = params["lvals"]
        self.dim = len(self.lvals)
        self.directions = "xyz"[: self.dim]
        self.n_sites = prod(self.lvals)
        self.has_obc = params["has_obc"]
        self.coeffs = params["coeffs"]
        self.n_eigs = params["n_eigs"]
        self.sector = params["sector"]
        # ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
        self.ops = Z2_FermiHubbard_dressed_site_operators(lattice_dim=self.dim)
        # ACQUIRE BASIS AND GAUGE INVARIANT SITES
        self.basis, _ = Z2_FermiHubbard_gauge_invariant_states(lattice_dim=self.dim)
        # ACQUIRE LOCAL DIMENSION OF EVERY SINGLE SITE
        lattice_base, loc_dims = lattice_base_configs(
            self.basis, self.lvals, self.has_obc
        )
        self.loc_dims = loc_dims.transpose().reshape(self.n_sites)
        self.lattice_base = lattice_base.transpose().reshape(self.n_sites)
        self.res = {}

    def build_Hamiltonian(self):
        # CONSTRUCT THE HAMILTONIAN
        self.H = QMB_hamiltonian(0, self.lvals, self.loc_dims)
        h_terms = {}
        # ---------------------------------------------------------------------------
        # LINK PENALTIES & Border penalties
        for d in self.directions:
            op_names_list = [f"n_p{d}", f"n_m{d}"]
            op_list = [self.ops[op] for op in op_names_list]
            # Define the Hamiltonian term
            h_terms[f"W{d}"] = TwoBodyTerm(
                axis=d,
                op_list=op_list,
                op_names_list=op_names_list,
                lvals=self.lvals,
                has_obc=self.has_obc,
                site_basis=self.basis,
            )
            self.H.Ham += h_terms[f"W{d}"].get_Hamiltonian(
                strength=-2 * self.coeffs["eta"]
            )
        # SINGLE SITE OPERATORS needed for the LINK SYMMETRY/OBC PENALTIES
        op_name = f"n_total"
        h_terms[op_name] = LocalTerm(
            self.ops[op_name],
            op_name,
            lvals=self.lvals,
            has_obc=self.has_obc,
            site_basis=self.basis,
        )
        self.H.Ham += h_terms[op_name].get_Hamiltonian(strength=self.coeffs["eta"])
        # -------------------------------------------------------------------------------
        # COULOMB POTENTIAL
        op_name = "N_pair_half"
        h_terms["V"] = LocalTerm(
            self.ops[op_name],
            op_name,
            lvals=self.lvals,
            has_obc=self.has_obc,
            site_basis=self.basis,
        )
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
                    d,
                    op_list,
                    op_names_list,
                    lvals=self.lvals,
                    has_obc=self.has_obc,
                    site_basis=self.basis,
                )
                self.H.Ham += h_terms[f"{d}_hop_{s}"].get_Hamiltonian(
                    strength=self.coeffs["t"], add_dagger=True
                )
        # ===========================================================================
        # SYMMETRY SECTOR
        if self.sector is not None:
            op_name = f"N_tot"
            h_terms[op_name] = LocalTerm(
                self.ops[op_name],
                op_name,
                lvals=self.lvals,
                has_obc=self.has_obc,
                site_basis=self.basis,
            )
            self.H.Ham += (
                0.5
                * self.coeffs["eta"]
                * (
                    h_terms[op_name].get_Hamiltonian(strength=1)
                    - self.sector * ID(prod(self.loc_dims))
                )
                ** 2
            )
        # DIAGONALIZE THE HAMILTONIAN
        self.H.diagonalize(self.n_eigs)
        self.res["energies"] = self.H.Nenergies
        if self.n_eigs > 1:
            self.res["true_gap"] = self.H.Nenergies[1] - self.H.Nenergies[0]

    def get_observables(self, local_obs, twobody_obs, plaquette_obs):
        self.local_obs = local_obs
        self.twobody_obs = twobody_obs
        self.plaquette_obs = plaquette_obs
        self.obs_list = {}
        # ---------------------------------------------------------------------------
        # LIST OF LOCAL OBSERVABLES
        for obs in local_obs:
            self.obs_list[obs] = LocalTerm(
                self.ops[obs],
                obs,
                lvals=self.lvals,
                has_obc=self.has_obc,
                site_basis=self.basis,
            )
        # ---------------------------------------------------------------------------
        # LIST OF TWOBODY CORRELATORS
        for op_names_list in twobody_obs:
            obs = "_".join(op_names_list)
            op_list = [self.ops[op] for op in op_names_list]
            self.obs_list[obs] = TwoBodyTerm(
                axis="x",
                op_list=op_list,
                op_names_list=op_names_list,
                lvals=self.lvals,
                has_obc=self.has_obc,
                site_basis=self.basis,
            )
        # ---------------------------------------------------------------------------
        # LIST OF PLAQUETTE CORRELATORS
        for op_names_list in plaquette_obs:
            obs = "_".join(op_names_list)
            op_list = [self.ops[op] for op in op_names_list]
            self.obs_list[obs] = PlaquetteTerm(
                axis=["x", "y"],
                op_list=op_list,
                op_names_list=op_names_list,
                lvals=self.lvals,
                has_obc=self.has_obc,
                site_basis=self.basis,
            )

    def measure_observables(self, state_number):
        for obs in self.local_obs:
            self.obs_list[obs].get_expval(self.H.Npsi[state_number])
            self.res[obs] = self.obs_list[obs].obs
        for op_names_list in self.twobody_obs:
            obs = "_".join(op_names_list)
            self.obs_list[obs].get_expval(self.H.Npsi[state_number])
            self.res[obs] = self.obs_list[obs].corr

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
