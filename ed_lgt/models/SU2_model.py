import numpy as np
from math import prod
from itertools import product
from scipy.linalg import eigh
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, QMB_hamiltonian
from ed_lgt.operators import (
    SU2_dressed_site_operators,
    SU2_gauge_invariant_states,
    SU2_Hamiltonian_couplings,
)
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm, QMB_hamiltonian
from ed_lgt.modeling import (
    check_link_symmetry,
    diagonalize_density_matrix,
    staggered_mask,
    truncation,
    lattice_base_configs,
)

__all__ = ["SU2_Model"]


class SU2_Model:
    def __init__(self, params):
        self.lvals = params["lvals"]
        self.dim = len(self.lvals)
        self.directions = "xyz"[: self.dim]
        self.n_sites = prod(self.lvals)
        self.has_obc = params["has_obc"]
        self.coeffs = params["coeffs"]
        self.n_eigs = params["n_eigs"]
        self.spin = params["spin"]
        self.pure_theory = params["pure_theory"]
        staggered_basis = False if self.pure_theory else True
        # ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
        self.ops = SU2_dressed_site_operators(
            self.spin, self.pure_theory, lattice_dim=self.dim
        )
        # ACQUIRE BASIS AND GAUGE INVARIANT SITES
        M, _ = SU2_gauge_invariant_states(
            self.spin, self.pure_theory, lattice_dim=self.dim
        )
        # ACQUIRE LOCAL DIMENSION OF EVERY SINGLE SITE
        lattice_base, loc_dims = lattice_base_configs(
            M, self.lvals, self.has_obc, staggered=staggered_basis
        )
        self.loc_dims = loc_dims.transpose().reshape(self.n_sites)
        self.lattice_base = lattice_base.transpose().reshape(self.n_sites)
