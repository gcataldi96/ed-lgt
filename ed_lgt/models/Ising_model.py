import numpy as np
from math import prod
from itertools import product
from scipy.linalg import eigh
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, QMB_hamiltonian
from ed_lgt.operators import get_Pauli_operators
from .quantum_model import QuantumModel

__all__ = ["IsingModel", "get_N_operator", "get_M_operator", "get_Q_operator"]


class IsingModel(QuantumModel):
    def __init__(self, params):
        # Initialize base class with the common parameters
        super().__init__(params)
        # Initialize specific attributes for IsingModel
        self.loc_dims = np.array([2 for _ in range(self.n_sites)])

    def get_operators(self, sparse=True):
        self.ops = get_Pauli_operators()
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
            op_names_list = ["Sx", "Sx"]
            op_list = [self.ops[op] for op in op_names_list]
            # Define the Hamiltonian term
            h_terms[f"NN_{d}"] = TwoBodyTerm(
                axis=d,
                op_list=op_list,
                op_names_list=op_names_list,
                lvals=self.lvals,
                has_obc=self.has_obc,
                sector_indices=self.sector_indices,
                sector_basis=self.sector_basis,
            )
            self.H.Ham += h_terms[f"NN_{d}"].get_Hamiltonian(strength=-self.coeffs["J"])
        # EXTERNAL MAGNETIC FIELD
        op_name = "Sz"
        h_terms[op_name] = LocalTerm(
            self.ops[op_name],
            op_name,
            lvals=self.lvals,
            has_obc=self.has_obc,
            sector_indices=self.sector_indices,
            sector_basis=self.sector_basis,
        )
        self.H.Ham += h_terms[op_name].get_Hamiltonian(strength=-self.coeffs["h"])

    def get_energy_gap(self):
        N = get_N_operator(self.lvals, self.res)
        M = get_M_operator(self.lvals, self.has_obc[0], self.res, self.coeffs)
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
