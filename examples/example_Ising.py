# %%
import numpy as np
from math import prod
from scipy.linalg import eigh
from ed_lgt.operators import get_spin_operators
from ed_lgt.modeling import Ground_State, LocalTerm, TwoBodyTerm
from ed_lgt.modeling import get_state_configurations, truncation
from ed_lgt.tools import check_hermitian

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
# LIST OF TWOBODY CORRELATORS
twobody_obs = [["Sz", "Sz"], ["Sx", "Sm"], ["Sx", "Sp"], ["Sp", "Sx"], ["Sm", "Sx"]]
for obs1, obs2 in twobody_obs:
    op_list = [ops[obs1], ops[obs2]]
    h_terms[f"{obs1}_{obs2}"] = TwoBodyTerm(
        axis="x",
        op_list=op_list,
        op_name_list=[obs1, obs2],
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
    # MEASURE LOCAL OBSERVABLES:
    for obs in loc_obs:
        h_terms[obs].get_expval(GS.Npsi[:, ii])
        res[obs].append(h_terms[obs].avg)
    # MEASURE TWOBODY OBSERVABLES:
    for obs1, obs2 in twobody_obs:
        print("----------------------------------------------------")
        print(f"{obs1}_{obs2}")
        print("----------------------------------------------------")
        h_terms[f"{obs1}_{obs2}"].get_expval(GS.Npsi[:, ii])
        # print(h_terms[f"{obs1}_{obs2}"].corr)

if n_eigs > 1:
    print(f"Energy Gaps {res['DeltaE']}")
# %%
from scipy.sparse import csr_matrix

# Compute the Matrices for the generalized Eigenvalue Problem: Mx = w Nx
M = np.zeros((n_sites, n_sites), dtype=complex)
N = np.zeros((n_sites, n_sites), dtype=complex)

for ii in range(n_sites):
    for jj in range(n_sites):
        nn_condition = [
            (ii > 0) and (jj == ii - 1),
            (ii < n_sites - 1) and (jj == ii + 1),
            np.all([not has_obc, ii == 0, jj == n_sites - 1]),
            np.all([not has_obc, ii == n_sites - 1, jj == 0]),
        ]
        if np.any(nn_condition):
            M[ii, jj] += coeffs["J"] * h_terms["Sz_Sz"].corr[ii, jj]
        elif jj == ii:
            # ---------------------------------------------------
            N[ii, jj] += h_terms["Sz"].obs[ii]
            # ---------------------------------------------------
            M[ii, jj] += 2 * coeffs["h"] * h_terms["Sz"].obs[ii]
            if ii < n_sites - 1:
                M[ii, jj] += (
                    complex(0, 0.5)
                    * coeffs["J"]
                    * (
                        h_terms["Sm_Sx"].corr[ii, ii + 1]
                        - h_terms["Sp_Sx"].corr[ii, ii + 1]
                    )
                )
            else:
                if not has_obc:
                    M[ii, jj] += (
                        complex(0, 0.5)
                        * coeffs["J"]
                        * (h_terms["Sm_Sx"].corr[ii, 0] - h_terms["Sp_Sx"].corr[ii, 0])
                    )
            if ii > 0:
                M[ii, jj] += (
                    complex(0, 0.5)
                    * coeffs["J"]
                    * (
                        h_terms["Sx_Sm"].corr[ii - 1, ii]
                        - h_terms["Sx_Sp"].corr[ii - 1, ii]
                    )
                )
            else:
                if not has_obc:
                    M[ii, jj] += (
                        complex(0, 0.5)
                        * coeffs["J"]
                        * (
                            h_terms["Sx_Sm"].corr[n_sites - 1, ii]
                            - h_terms["Sx_Sp"].corr[n_sites - 1, ii]
                        )
                    )
print(csr_matrix(M))
# %%
# Solve the generalized egenvalue problem
w = eigh(a=M, b=N, eigvals_only=True)
gap = np.mean(w)
# %%
