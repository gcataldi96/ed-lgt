# %%
import numpy as np
from math import prod
from ed_lgt.operators import get_spin_operators
from ed_lgt.modeling import Ground_State, LocalTerm, TwoBodyTerm, PlaquetteTerm
from ed_lgt.modeling import (
    entanglement_entropy,
    get_reduced_density_matrix,
    diagonalize_density_matrix,
    get_state_configurations,
    truncation,
    lattice_base_configs,
)
from ed_lgt.tools import check_hermitian

# Spin representation
spin = 1 / 2
# N eigenvalues
n_eigs = 4
# LATTICE DIMENSIONS
lvals = [4, 4]
dim = len(lvals)
directions = "xyz"[:dim]
n_sites = prod(lvals)
# BOUNDARY CONDITIONS
has_obc = True
# ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
ops = get_spin_operators(spin)
loc_dims = np.array([int(2 * spin + 1) for i in range(n_sites)])
print("local dimensions:", loc_dims)
# ACQUIRE HAMILTONIAN COEFFICIENTS
coeffs = {"J": 1, "h": 10}
# CONSTRUCT THE HAMILTONIAN
H = 0
h_terms = {}
# -------------------------------------------------------------------------------
# NEAREST NEIGHBOR INTERACTION
for d in directions:
    op_name_list = ["Sx", "Sx"]
    op_list = [ops[op] for op in op_name_list]
    # Define the Hamiltonian term
    h_terms[f"NN_{d}"] = TwoBodyTerm(
        axis=d, op_list=op_list, op_name_list=op_name_list, lvals=lvals, has_obc=has_obc
    )
    H += h_terms[f"NN_{d}"].get_Hamiltonian(strength=coeffs["J"])
# EXTERNAL MAGNETIC FIELD
op_name = "Sz"
h_terms[op_name] = LocalTerm(ops[op_name], op_name, lvals=lvals, has_obc=has_obc)
H += h_terms[op_name].get_Hamiltonian(strength=coeffs["h"])
# ===========================================================================
# CHECK THAT THE HAMILTONIAN IS HERMITIAN
check_hermitian(H)
# DIAGONALIZE THE HAMILTONIAN
GS = Ground_State(H, n_eigs)
# Dictionary for results
res = {}
res["energy"] = GS.Nenergies
# ===========================================================================
# DEFINE THE OBSERVABLE LIST
res["entropy"] = []
res["rho_eigvals"] = []
obs_list = ["Sx", "Sz"]
for obs in obs_list:
    res[obs] = []
# ===========================================================================
for ii in range(n_eigs):
    print("====================================================")
    print(f"{ii} ENERGY: {format(res['energy'][ii], '.9f')}")
    # GET STATE CONFIGURATIONS
    get_state_configurations(truncation(GS.Npsi[:, ii], 1e-10), loc_dims, lvals)
    # ===========================================================================
    # MEASURE OBSERVABLES:
    for obs in obs_list:
        h_terms[obs] = LocalTerm(ops[op_name], op_name, lvals=lvals, has_obc=has_obc)
        h_terms[obs].get_expval(GS.Npsi[:, ii])
        res[obs].append(h_terms[obs].avg)

# %%
