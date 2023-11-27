# %%
import numpy as np
from math import prod
from ed_lgt.operators import (
    SU2_Hamiltonian_couplings,
    SU2_rishon_operators,
    SU2_dressed_site_operators,
    SU2_gauge_invariant_states,
    SU2_check_gauss_law,
)
from ed_lgt.modeling import Ground_State, LocalTerm, TwoBodyTerm, PlaquetteTerm
from ed_lgt.modeling import (
    entanglement_entropy,
    get_reduced_density_matrix,
    diagonalize_density_matrix,
    staggered_mask,
    get_state_configurations,
    truncation,
    lattice_base_configs,
)

# N eigenvalues
n_eigs = 1
# LATTICE DIMENSIONS
lvals = [2, 2]
dim = len(lvals)
directions = "xyz"[:dim]
n_sites = prod(lvals)
# BOUNDARY CONDITIONS
has_obc = False
# DEFINE the maximal truncation of the gauge link
j_max = 3 / 2
# PURE or FULL THEORY
pure_theory = False
# GET g COUPLING
g = 0.1
if pure_theory:
    m = None
else:
    m = 0.1
# ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
in_ops = SU2_rishon_operators(j_max)
ops = SU2_dressed_site_operators(j_max, pure_theory, lattice_dim=dim)
# Acquire SU2 Basis and gauge invariant states
M, states = SU2_gauge_invariant_states(j_max, pure_theory, lattice_dim=dim)
for s in states["site"]:
    s.show()
# ACQUIRE LOCAL DIMENSION OF EVERY SINGLE SITE
lattice_base, loc_dims = lattice_base_configs(M, lvals, has_obc, staggered=False)
loc_dims = loc_dims.transpose().reshape(n_sites)
lattice_base = lattice_base.transpose().reshape(n_sites)
print("local dimensions:", loc_dims)
# SU2_check_gauss_law(basis=M["site"], gauss_law_op=ops["S2_tot"])
# ACQUIRE HAMILTONIAN COEFFICIENTS
coeffs = SU2_Hamiltonian_couplings(pure_theory, g, m)
# %%
