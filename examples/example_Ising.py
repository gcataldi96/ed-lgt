# %%
import numpy as np
from math import prod
from ed_lgt.operators import get_Pauli_operators, energy_gap
from ed_lgt.modeling import (
    Ground_State,
    LocalTerm,
    TwoBodyTerm,
    ThreeBodyTerm,
    FourBodyTerm,
)
from ed_lgt.modeling import get_state_configurations, truncation
from ed_lgt.tools import check_hermitian


# N EIGENVALUES
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
# GET LOCAL DIMENSIONS
spin = 1 / 2
loc_dims = np.array([int(2 * spin + 1) for i in range(n_sites)])
# ACQUIRE HAMILTONIAN COEFFICIENTS
coeffs = {"J": 1, "h": 10}
# CONSTRUCT THE HAMILTONIAN
H = 0
h_terms = {}
# ---------------------------------------------------------------------------
# NEAREST NEIGHBOR INTERACTION
for d in directions:
    op_name_list = ["Sx", "Sx"]
    op_list = [ops[op] for op in op_name_list]
    # Define the Hamiltonian term
    h_terms[f"NN_{d}"] = TwoBodyTerm(
        axis=d, op_list=op_list, op_name_list=op_name_list, lvals=lvals, has_obc=has_obc
    )
    H += h_terms[f"NN_{d}"].get_Hamiltonian(strength=-coeffs["J"])
# EXTERNAL MAGNETIC FIELD
op_name = "Sz"
h_terms[op_name] = LocalTerm(ops[op_name], op_name, lvals=lvals, has_obc=has_obc)
H += h_terms[op_name].get_Hamiltonian(strength=-coeffs["h"])
# ===========================================================================
# CHECK THAT THE HAMILTONIAN IS HERMITIAN
check_hermitian(H)
# DIAGONALIZE THE HAMILTONIAN
GS = Ground_State(H, n_eigs)
# Dictionary for results
res = {}
res["energy"] = GS.Nenergies
# ===========================================================================
# LIST OF LOCAL OBSERVABLES
loc_obs = ["Sx", "Sz"]
for obs in loc_obs:
    res[obs] = []
    h_terms[obs] = LocalTerm(ops[obs], obs, lvals=lvals, has_obc=has_obc)
# ---------------------------------------------------------------------------
# LIST OF TWOBODY CORRELATORS
twobody_obs = [
    ["Sz", "Sz"],
    ["Sx", "Sm"],
    ["Sx", "Sp"],
    ["Sp", "Sx"],
    ["Sm", "Sx"],
    ["Sz", "Sm"],
    ["Sm", "Sz"],
    ["Sp", "Sz"],
    ["Sz", "Sp"],
    ["Sp", "Sp"],
    ["Sm", "Sm"],
]
for op_name_list in twobody_obs:
    op_list = [ops[op] for op in op_name_list]
    h_terms["_".join(op_name_list)] = TwoBodyTerm(
        axis="x",
        op_list=op_list,
        op_name_list=op_name_list,
        lvals=lvals,
        has_obc=has_obc,
    )
# ---------------------------------------------------------------------------
# LIST OF THREEBODY CORRELATORS
threebody_obs = [
    ["Sm", "Sz", "Sp"],
    ["Sp", "Sz", "Sm"],
    ["Sx", "Sp", "Sp"],
    ["Sx", "Sm", "Sm"],
    ["Sp", "Sz", "Sz"],
    ["Sm", "Sz", "Sz"],
    ["Sp", "Sz", "Sx"],
    ["Sm", "Sz", "Sx"],
    ["Sp", "Sp", "Sz"],
    ["Sm", "Sm", "Sz"],
    ["Sx", "Sp", "Sz"],
    ["Sx", "Sm", "Sz"],
    ["Sz", "Sp", "Sx"],
    ["Sz", "Sp", "Sm"],
    ["Sz", "Sp", "Sp"],
    ["Sz", "Sm", "Sx"],
    ["Sz", "Sm", "Sp"],
    ["Sz", "Sm", "Sm"],
    ["Sp", "Sm", "Sz"],
    ["Sm", "Sp", "Sz"],
]
for op_name_list in threebody_obs:
    op_list = [ops[op] for op in op_name_list]
    h_terms["_".join(op_name_list)] = ThreeBodyTerm(
        op_list=op_list,
        op_name_list=op_name_list,
        lvals=lvals,
        has_obc=has_obc,
    )
# ---------------------------------------------------------------------------
# LIST OF FOURBODY CORRELATORS
fourbody_obs = [
    ["Sp", "Sz", "Sz", "Sm"],
    ["Sm", "Sz", "Sz", "Sp"],
    ["Sp", "Sz", "Sz", "Sx"],
    ["Sm", "Sz", "Sz", "Sx"],
    ["Sx", "Sz", "Sz", "Sm"],
    ["Sx", "Sz", "Sz", "Sp"],
]
for op_name_list in fourbody_obs:
    op_list = [ops[op] for op in op_name_list]
    h_terms["_".join(op_name_list)] = FourBodyTerm(
        op_list=op_list,
        op_name_list=op_name_list,
        lvals=lvals,
        has_obc=has_obc,
    )
# ===========================================================================
for ii in range(n_eigs):
    print("====================================================")
    print(f"{ii} ENERGY: {format(res['energy'][ii], '.9f')}")
    # GET STATE CONFIGURATIONS
    get_state_configurations(truncation(GS.Npsi[:, ii], 1e-3), loc_dims, lvals)
    # =======================================================================
    # MEASURE LOCAL-BODY OBSERVABLES:
    for obs in loc_obs:
        h_terms[obs].get_expval(GS.Npsi[:, ii])
        res[obs].append(h_terms[obs].avg)
    # MEASURE TWO-BODY OBSERVABLES:
    for op_name_list in twobody_obs:
        obs_name = "_".join(op_name_list)
        print("----------------------------------------------------")
        print(obs_name)
        print("----------------------------------------------------")
        h_terms[obs_name].get_expval(GS.Npsi[:, ii])
    # MEASURE THREE-BODY OBSERVABLES:
    for op_name_list in threebody_obs:
        obs_name = "_".join(op_name_list)
        print("----------------------------------------------------")
        print(obs_name)
        print("----------------------------------------------------")
        h_terms[obs_name].get_expval(GS.Npsi[:, ii])
    # MEASURE FOUR-BODY OBSERVABLES:
    for op_name_list in fourbody_obs:
        obs_name = "_".join(op_name_list)
        print("----------------------------------------------------")
        print(obs_name)
        print("----------------------------------------------------")
        h_terms[obs_name].get_expval(GS.Npsi[:, ii])
    # SAVE TRUE ENERGY GAP
    if ii == 1:
        res["DeltaE"] = res["energy"][ii] - res["energy"][0]
        # print(res["DeltaE"])
    # COMPUTE THE ENERGY GAP WITH THE NEW METHOD
    elif ii == 0:
        res["gap"] = energy_gap(lvals, has_obc, h_terms, coeffs)

if n_eigs > 1:
    print((res["gap"] - res["DeltaE"]) / (-res["DeltaE"]))

# %%
