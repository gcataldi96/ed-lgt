# %%
import numpy as np
from math import prod
from scipy.linalg import eigh
from ed_lgt.operators import get_spin_operators
from ed_lgt.modeling import (
    Ground_State,
    LocalTerm,
    TwoBodyTerm,
    ThreeBodyTerm,
    PlaquetteTerm,
)
from ed_lgt.modeling import get_state_configurations, truncation
from ed_lgt.tools import check_hermitian
from .ising_gaps import get_M_operator, get_N_operator, get_P_operator, get_Q_operator

# Spin representation
spin = 1 / 2
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
ops = get_spin_operators(spin)
for op in ["Sz", "Sx", "Sy"]:
    ops[op] = 2 * ops[op]
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
res["DeltaE"] = []
# ===========================================================================
# LIST OF LOCAL OBSERVABLES
loc_obs = ["Sx", "Sz"]
for obs in loc_obs:
    res[obs] = []
    h_terms[obs] = LocalTerm(ops[obs], obs, lvals=lvals, has_obc=has_obc)
# ---------------------------------------------------------------------------
# LIST OF TWOBODY CORRELATORS
twobody_obs = [["Sz", "Sz"], ["Sx", "Sm"], ["Sx", "Sp"], ["Sp", "Sx"], ["Sm", "Sx"]]
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
    ["Sm", "Sz", "Sz"],
    ["Sp", "Sz", "Sz"],
    ["Sx", "Sp", "Sp"],
    ["Sx", "Sm", "Sm"],
    ["Sz", "Sm", "Sm"],
    ["Sz", "Sp", "Sp"],
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
    ["Sm", "Sz", "Sz"],
    ["Sp", "Sz", "Sz"],
    ["Sx", "Sp", "Sp"],
    ["Sx", "Sm", "Sm"],
    ["Sz", "Sm", "Sm"],
    ["Sz", "Sp", "Sp"],
]
op_list = [ops[op] for op in op_name_list]
h_terms["_".join(op_name_list)] = PlaquetteTerm(
    axes=["x", "y"],
    op_list=op_list,
    op_name_list=op_name_list,
    lvals=lvals,
    has_obc=has_obc,
)
# ===========================================================================
for ii in range(n_eigs):
    print("====================================================")
    print(f"{ii} ENERGY: {format(res['energy'][ii], '.9f')}")
    if ii > 0:
        res["DeltaE"].append(res["energy"][ii] - res["energy"][0])
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
