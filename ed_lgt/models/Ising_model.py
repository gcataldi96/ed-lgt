import numpy as np
from math import prod
from itertools import product
from scipy.linalg import eigh
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, QMB_hamiltonian
from ed_lgt.operators import get_Pauli_operators

__all__ = ["Ising_Model", "get_N_operator", "get_M_operator", "get_Q_operator"]


class Ising_Model:
    def __init__(self, params):
        self.lvals = params["lvals"]
        self.dim = len(self.lvals)
        self.directions = "xyz"[: self.dim]
        self.n_sites = prod(self.lvals)
        self.has_obc = params["has_obc"]
        self.coeffs = params["coeffs"]
        self.n_eigs = params["n_eigs"]
        self.ops = get_Pauli_operators()
        self.loc_dims = np.array([int(2 * 0.5 + 1) for i in range(prod(self.lvals))])
        self.res = {}

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
            )

    def measure_observables(self, state_number):
        for obs in self.local_obs:
            self.obs_list[obs].get_expval(self.H.Npsi[state_number])
            self.res[obs] = self.obs_list[obs].obs
        for op_names_list in self.twobody_obs:
            obs = "_".join(op_names_list)
            self.obs_list[obs].get_expval(self.H.Npsi[state_number])
            self.res[obs] = self.obs_list[obs].corr

    def get_energy_gap(self):
        N = get_N_operator(self.lvals, self.res)
        M = get_M_operator(self.lvals, self.has_obc, self.res, self.coeffs)
        self.res["th_gap"] = eigh(a=M, b=N, eigvals_only=True)[0]


def get_N_operator(lvals, obs):
    n_sites = prod(lvals)
    N = np.zeros((n_sites, n_sites), dtype=float)
    for ii in range(n_sites):
        N[ii, ii] += obs["Sz"][ii]
    return N


def get_M_operator(lvals, has_obc, obs, coeffs):
    n_sites = prod(lvals)
    M = np.zeros((n_sites, n_sites), dtype=complex)
    for ii, jj in product(range(n_sites), repeat=2):
        nn_condition = [
            all([ii > 0, jj == ii - 1]),
            all([ii < n_sites - 1, jj == ii + 1]),
            all([not has_obc, ii == 0, jj == n_sites - 1]),
            all([not has_obc, ii == n_sites - 1, jj == 0]),
        ]
        if any(nn_condition):
            M[ii, jj] += coeffs["J"] * obs["Sz_Sz"][ii, jj]
        elif jj == ii:
            M[ii, jj] += 2 * coeffs["h"] * obs["Sz"][ii]
            if 0 < ii < n_sites - 1 or all(
                [(ii == 0 or ii == n_sites - 1), not has_obc]
            ):
                M[ii, jj] += complex(0, 0.5 * coeffs["J"]) * (
                    obs["Sm_Sx"][ii, (ii + 1) % n_sites]
                    - obs["Sp_Sx"][ii, (ii + 1) % n_sites]
                    + obs["Sx_Sm"][(ii - 1) % n_sites, ii]
                    - obs["Sx_Sp"][(ii - 1) % n_sites, ii]
                )
    return M


def get_Q_operator(lvals, has_obc, obs):
    n_sites = prod(lvals)
    Q = np.zeros((2, 2), dtype=object)
    for alpha, beta in product(range(2), repeat=2):
        Q[alpha, beta] = np.zeros((n_sites, n_sites), dtype=float)
        if alpha == beta == 0:
            return get_N_operator(lvals, obs)
        elif alpha == 1 and beta == 0:
            for ii, jj in product(range(n_sites), repeat=2):
                if jj == ii - 2 and any([ii > 1, ii <= 1 and not has_obc]):
                    Q[alpha, beta][ii, jj] += obs["Sp_Sp_Sz"][
                        (ii - 2) % n_sites, (ii - 1) % n_sites, ii
                    ]
        elif alpha == 1 and beta == 1:
            for ii, jj in product(range(n_sites), repeat=2):
                if ii == jj and any(
                    [ii < n_sites - 2, ii >= n_sites - 2 and not has_obc]
                ):
                    Q[alpha, beta][ii, jj] += obs["Sz_Sz_Sz"][
                        ii, (ii + 1) % n_sites, (ii + 2) % n_sites
                    ]
    Q[0, 1] = Q[1, 0].transpose()
    return Q
