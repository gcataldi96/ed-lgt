# %%
import numpy as np
from math import prod
from ed_lgt.operators import bose_operators
from ed_lgt.modeling import abelian_sector_indices
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, QMB_hamiltonian

# N eigenvalues
n_eigs = 1
# LATTICE GEOMETRY
lvals = [10]
dim = len(lvals)
directions = "xyz"[:dim]
n_sites = prod(lvals)
has_obc = [False]
# LOCAL SITE DIMENSION
n_max = 3
loc_dims = np.array([n_max + 1 for i in range(n_sites)])
# ACQUIRE HAMILTONIAN COEFFICIENTS
coeffs = {"t": 1, "U": 1}
# ACQUIRE OPERATORS
ops = bose_operators(n_max)
# SYMMETRY SECTOR
sector = True
if sector:
    for op in ops.keys():
        ops[op] = ops[op].toarray()
    sector_indices, sector_basis = abelian_sector_indices(
        loc_dims, [ops["N"]], [3], sym_type="U"
    )
    print(np.prod(loc_dims))
    print(sector_indices.shape[0])
else:
    sector_indices = None
    sector_basis = None
# CONSTRUCT THE HAMILTONIAN
H = QMB_hamiltonian(0, lvals, loc_dims)
h_terms = {}
# ---------------------------------------------------------------------------
# NEAREST NEIGHBOR INTERACTION
for d in directions:
    op_names_list = ["b_dagger", "b"]
    op_list = [ops[op] for op in op_names_list]
    # Define the Hamiltonian term
    h_terms[f"NN_{d}"] = TwoBodyTerm(
        axis=d,
        op_list=op_list,
        op_names_list=op_names_list,
        lvals=lvals,
        has_obc=has_obc,
        sector_basis=sector_basis,
        sector_indices=sector_indices,
    )
    H.Ham += h_terms[f"NN_{d}"].get_Hamiltonian(strength=-coeffs["t"], add_dagger=True)
# SINGLE SITE POTENTIAL
op_name = "N2"
h_terms[op_name] = LocalTerm(
    operator=ops[op_name],
    op_name=op_name,
    lvals=lvals,
    has_obc=has_obc,
    sector_basis=sector_basis,
    sector_indices=sector_indices,
)
H.Ham += h_terms[op_name].get_Hamiltonian(strength=0.5 * coeffs["U"])
op_name = "N"
h_terms[op_name] = LocalTerm(
    operator=ops[op_name],
    op_name=op_name,
    lvals=lvals,
    has_obc=has_obc,
    sector_basis=sector_basis,
    sector_indices=sector_indices,
)
H.Ham += h_terms[op_name].get_Hamiltonian(strength=-0.5 * coeffs["U"])
# ADD SINGLE SITE NOISE
noise = np.random.rand(n_sites)
for ii in n_sites:
    mask = np.zeros(dtype=bool)
    mask[ii] = True
    H.Ham += h_terms["N"].get_Hamiltonian(strength=noise[ii], mask=mask)
# ===========================================================================
# DIAGONALIZE THE HAMILTONIAN
H.diagonalize(n_eigs)
# Dictionary for results
res = {}
res["energy"] = H.Nenergies
# ===========================================================================
# LIST OF LOCAL OBSERVABLES
loc_obs = ["N"]
for obs in loc_obs:
    res[obs] = []
    h_terms[obs] = LocalTerm(
        ops[obs],
        obs,
        lvals=lvals,
        has_obc=has_obc,
        sector_basis=sector_basis,
        sector_indices=sector_indices,
    )
# LIST OF TWOBODY CORRELATORS
twobody_obs = []
for obs1, obs2 in twobody_obs:
    op_list = [ops[obs1], ops[obs2]]
    h_terms[f"{obs1}_{obs2}"] = TwoBodyTerm(
        axis="x",
        op_list=op_list,
        op_names_list=[obs1, obs2],
        lvals=lvals,
        has_obc=has_obc,
        sector_basis=sector_basis,
        sector_indices=sector_indices,
    )
# ===========================================================================
for ii in range(n_eigs):
    print("====================================================")
    print(f"{ii} ENERGY: {format(res['energy'][ii], '.9f')}")
    if ii > 0:
        res["DeltaE"] = res["energy"][ii] - res["energy"][0]
    # GET STATE CONFIGURATIONS
    H.Npsi[ii].get_state_configurations(threshold=1e-2, sector_indices=sector_indices)
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
