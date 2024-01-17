from math import prod
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm

__all__ = ["Quantum_Model"]


class Quantum_Model:
    def __init__(self, params):
        self.lvals = params["lvals"]
        self.dim = len(self.lvals)
        self.directions = "xyz"[: self.dim]
        self.n_sites = prod(self.lvals)
        self.has_obc = params["has_obc"]
        self.n_eigs = params["n_eigs"]
        self.basis = None
        self.res = {}

    def get_observables(self, local_obs=[], twobody_obs=[], plaquette_obs=[]):
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
