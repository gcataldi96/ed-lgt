from math import prod
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm, NBodyTerm
from ed_lgt.modeling import QMB_hamiltonian
from ed_lgt.modeling import abelian_sector_indices

__all__ = ["QuantumModel"]


class QuantumModel:
    def __init__(self, params):
        # Lattice parameters
        self.lvals = params["lvals"]
        self.dim = len(self.lvals)
        self.directions = "xyz"[: self.dim]
        self.n_sites = prod(self.lvals)
        self.has_obc = params["has_obc"]
        # Hamiltonian parameters
        self.n_eigs = params["n_eigs"]
        self.coeffs = params["coeffs"]
        # Symmetry sector indices
        self.sector_basis = None
        self.sector_indices = None
        # Site Basis
        self.site_basis = None
        # Staggered Basis
        self.staggered_basis = False
        # Dictionary for results
        self.res = {}

    def get_abelian_symmetry_sector(self, op_names_list, op_sectors_list, sym_type):
        self.sector_indices, self.sector_basis = abelian_sector_indices(
            self.loc_dims,
            [self.ops[op] for op in op_names_list],
            op_sectors_list,
            sym_type,
        )

    def diagonalize_Hamiltonian(self):
        if isinstance(self.H, QMB_hamiltonian):
            # DIAGONALIZE THE HAMILTONIAN
            self.H.diagonalize(self.n_eigs)
            self.res["energies"] = self.H.Nenergies
            if self.n_eigs > 1:
                self.res["true_gap"] = self.H.Nenergies[1] - self.H.Nenergies[0]

    def get_observables(
        self, local_obs=[], twobody_obs=[], plaquette_obs=[], nbody_obs=[]
    ):
        self.local_obs = local_obs
        self.twobody_obs = twobody_obs
        self.plaquette_obs = plaquette_obs
        self.nbody_obs = nbody_obs
        self.obs_list = {}
        # ---------------------------------------------------------------------------
        # LIST OF LOCAL OBSERVABLES
        for obs in local_obs:
            self.obs_list[obs] = LocalTerm(
                operator=self.ops[obs],
                op_name=obs,
                lvals=self.lvals,
                has_obc=self.has_obc,
                site_basis=self.site_basis,
                sector_indices=self.sector_indices,
                sector_basis=self.sector_basis,
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
                site_basis=self.site_basis,
                sector_indices=self.sector_indices,
                sector_basis=self.sector_basis,
            )
        # ---------------------------------------------------------------------------
        # LIST OF PLAQUETTE CORRELATORS
        for op_names_list in plaquette_obs:
            obs = "_".join(op_names_list)
            op_list = [self.ops[op] for op in op_names_list]
            self.obs_list[obs] = PlaquetteTerm(
                axes=["x", "y"],
                op_list=op_list,
                op_names_list=op_names_list,
                lvals=self.lvals,
                has_obc=self.has_obc,
                site_basis=self.site_basis,
            )
        # ---------------------------------------------------------------------------
        # LIST OF NBODY CORRELATORS
        for op_names_list in nbody_obs:
            obs = "_".join(op_names_list)
            op_list = [self.ops[op] for op in op_names_list]
            self.obs_list[obs] = NBodyTerm(
                op_list=op_list,
                op_names_list=op_names_list,
                lvals=self.lvals,
                has_obc=self.has_obc,
                site_basis=self.site_basis,
                sector_indices=self.sector_indices,
                sector_basis=self.sector_basis,
            )

    def measure_observables(self, state_number):
        for obs in self.local_obs:
            self.obs_list[obs].get_expval(self.H.Npsi[state_number])
            self.res[obs] = self.obs_list[obs].obs
        for op_names_list in self.twobody_obs:
            obs = "_".join(op_names_list)
            self.obs_list[obs].get_expval(self.H.Npsi[state_number])
            self.res[obs] = self.obs_list[obs].corr
        for op_names_list in self.nbody_obs:
            obs = "_".join(op_names_list)
            self.obs_list[obs].get_expval(self.H.Npsi[state_number])
            self.res[obs] = self.obs_list[obs].corr
