# %%
import numpy as np
from math import prod
from ed_lgt.operators import get_Pauli_operators
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, QMB_hamiltonian

# N eigenvalues
n_eigs = 2
# LATTICE DIMENSIONS
lvals = [10]
dim = len(lvals)
directions = "xyz"[:dim]
n_sites = prod(lvals)
# BOUNDARY CONDITIONS
has_obc = False
# ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
ops = get_Pauli_operators()
loc_dims = np.array([2 for i in range(n_sites)])
# ACQUIRE HAMILTONIAN COEFFICIENTS
coeffs = {"J": 1, "h": 10}
# CONSTRUCT THE HAMILTONIAN
H = QMB_hamiltonian(0, lvals, loc_dims)
h_terms = {}
# ---------------------------------------------------------------------------
# NEAREST NEIGHBOR INTERACTION
for d in directions:
    op_names_list = ["Sx", "Sx"]
    op_list = [ops[op] for op in op_names_list]
    # Define the Hamiltonian term
    h_terms[f"NN_{d}"] = TwoBodyTerm(
        axis=d,
        op_list=op_list,
        op_names_list=op_names_list,
        lvals=lvals,
        has_obc=has_obc,
    )
    H.Ham += h_terms[f"NN_{d}"].get_Hamiltonian(strength=-coeffs["J"])
# EXTERNAL MAGNETIC FIELD
op_name = "Sz"
h_terms[op_name] = LocalTerm(ops[op_name], op_name, lvals=lvals, has_obc=has_obc)
H.Ham += h_terms[op_name].get_Hamiltonian(strength=-coeffs["h"])
# ===========================================================================
# DIAGONALIZE THE HAMILTONIAN
H.diagonalize(n_eigs)
# Dictionary for results
res = {}
res["energy"] = H.Nenergies
# ===========================================================================
# LIST OF LOCAL OBSERVABLES
loc_obs = ["Sx", "Sz"]
for obs in loc_obs:
    res[obs] = []
    h_terms[obs] = LocalTerm(ops[obs], obs, lvals=lvals, has_obc=has_obc)
# LIST OF TWOBODY CORRELATORS
twobody_obs = [["Sz", "Sz"], ["Sx", "Sm"], ["Sx", "Sp"], ["Sp", "Sx"], ["Sm", "Sx"]]
for obs1, obs2 in twobody_obs:
    op_list = [ops[obs1], ops[obs2]]
    h_terms[f"{obs1}_{obs2}"] = TwoBodyTerm(
        axis="x",
        op_list=op_list,
        op_names_list=[obs1, obs2],
        lvals=lvals,
        has_obc=has_obc,
    )
# ===========================================================================
for ii in range(n_eigs):
    print("====================================================")
    print(f"{ii} ENERGY: {format(res['energy'][ii], '.9f')}")
    if ii > 0:
        res["DeltaE"] = res["energy"][ii] - res["energy"][0]
    # GET STATE CONFIGURATIONS
    H.Npsi[ii].get_state_configurations(threshold=1e-3)
    # =======================================================================
    # MEASURE LOCAL OBSERVABLES:
    for obs in loc_obs:
        h_terms[obs].get_expval(H.Npsi[ii])
        res[obs].append(h_terms[obs].avg)
    # MEASURE TWOBODY OBSERVABLES:
    for obs1, obs2 in twobody_obs:
        print("----------------------------------------------------")
        print(f"{obs1}_{obs2}")
        print("----------------------------------------------------")
        h_terms[f"{obs1}_{obs2}"].get_expval(H.Npsi[ii])
# %%
