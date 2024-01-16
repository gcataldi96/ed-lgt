import numpy as np
from math import prod
from itertools import product
from scipy.linalg import eigh
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, QMB_hamiltonian
from ed_lgt.operators import get_Pauli_operators


class Ising_Model:
    def __init__(self, params):
        self.lvals = params["lvals"]
        self.dim = len(self.lvals)
        self.directions = "xyz"[: self.dim]
        self.n_sites = prod(self.lvals)
        self.has_obc = params["has_obc"]
        self.coeffs = params["coeffs"]
        self.n_eigs = params["n_eigs"]
        self.loc_dims = np.array([int(2 * 0.5 + 1) for i in range(prod(self.lvals))])
        self.ops = get_Pauli_operators(0.5)

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

    def get_observables(self, local_obs, twobody_obs, state_number):
        self.obs_list = {}
        # ---------------------------------------------------------------------------
        # LIST OF LOCAL OBSERVABLES
        for obs in local_obs:
            self.obs_list[obs] = LocalTerm(
                self.ops[obs], obs, lvals=self.lvals, has_obc=self.has_obc
            )
            self.obs_list[obs].get_expval(self.H.Npsi[state_number])
        # ---------------------------------------------------------------------------
        # LIST OF TWOBODY CORRELATORS
        for op_name_list in twobody_obs:
            op_list = [self.ops[op] for op in op_name_list]
            self.obs_list["_".join(op_name_list)] = TwoBodyTerm(
                axis="x",
                op_list=op_list,
                op_name_list=op_name_list,
                lvals=self.lvals,
                has_obc=self.has_obc,
            )
            self.obs_list[obs].get_expval(self.H.Npsi[state_number])

    def get_qmb_state_properties(
        self,
        state_number,
        state_configs=False,
        entanglement_entropy=False,
        reduced_density_matrix=False,
    ):
        # PRINT ENERGY
        self.H.print_energy(state_number)
        # STATE CONFIGURATIONS
        if state_configs:
            self.H.Npsi[state_number].get_state_configurations(threshold=1e-3)
        # ENTANGLEMENT ENTROPY
        if entanglement_entropy:
            self.H.Npsi[state_number].entanglement_entropy(int(self.n_sites / 2))
        # REDUCED DENSITY MATRIX EIGVALS
        if reduced_density_matrix:
            self.H.Npsi[state_number].reduced_density_matrix(0)

    def get_energy_gap(self):
        N = get_N_operator(self.lvals, self.obs_list)
        M = get_M_operator(self.lvals, self.has_obc, self.obs_list, self.coeffs)
        return eigh(a=M, b=N, eigvals_only=True)[0]


def get_N_operator(lvals, obs):
    n_sites = prod(lvals)
    N = np.zeros((n_sites, n_sites), dtype=float)
    for ii in range(n_sites):
        N[ii, ii] += obs["Sz"].obs[ii]
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
            M[ii, jj] += coeffs["J"] * obs["Sz_Sz"].corr[ii, jj]
        elif jj == ii:
            M[ii, jj] += 2 * coeffs["h"] * obs["Sz"].obs[ii]
            if 0 < ii < n_sites - 1 or all(
                [(ii == 0 or ii == n_sites - 1), not has_obc]
            ):
                M[ii, jj] += complex(0, 0.5 * coeffs["J"]) * (
                    obs["Sm_Sx"].corr[ii, (ii + 1) % n_sites]
                    - obs["Sp_Sx"].corr[ii, (ii + 1) % n_sites]
                    + obs["Sx_Sm"].corr[(ii - 1) % n_sites, ii]
                    - obs["Sx_Sp"].corr[(ii - 1) % n_sites, ii]
                )
    return M
