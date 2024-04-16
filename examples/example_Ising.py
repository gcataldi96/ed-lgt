# %%
import numpy as np
from math import prod
from ed_lgt.modeling import abelian_sector_indices
from ed_lgt.operators import get_Pauli_operators
from ed_lgt.modeling import (
    LocalTerm,
    TwoBodyTerm,
    QMB_hamiltonian,
    NBodyTerm,
    truncation,
    get_loc_states_from_qmb_state,
)
from time import time
from scipy.sparse import csr_matrix
import logging

logger = logging.getLogger(__name__)

# N eigenvalues
n_eigs = 2
# LATTICE GEOMETRY
lvals = [10]
dim = len(lvals)
directions = "xyz"[:dim]
n_sites = prod(lvals)
has_obc = [False]
loc_dims = np.array([2 for i in range(n_sites)])
# ACQUIRE HAMILTONIAN COEFFICIENTS
coeffs = {"J": 1, "h": 1}
# ACQUIRE OPERATORS
ops = get_Pauli_operators()
# SYMMETRY SECTOR
sector = True
if sector:
    for op in ops.keys():
        ops[op] = ops[op].toarray()
    sector_indices, sector_basis = abelian_sector_indices(
        loc_dims, [ops["Sz"]], [1], sym_type="P"
    )
    logger.info(sector_indices.shape[0])
else:
    sector_indices = None
    sector_basis = None
    ranges = [range(dim) for dim in loc_dims]
    basis = np.transpose(np.meshgrid(*ranges, indexing="ij")).reshape(-1, len(loc_dims))
    basis_indices = np.ravel_multi_index(basis.T, loc_dims)
    basis = basis[np.argsort(basis_indices)]
    basis_indices = basis_indices[np.argsort(basis_indices)]
start = time()
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
        sector_basis=sector_basis,
        sector_indices=sector_indices,
    )
    H.Ham += h_terms[f"NN_{d}"].get_Hamiltonian(strength=-coeffs["J"])
# EXTERNAL MAGNETIC FIELD
op_name = "Sz"
h_terms[op_name] = LocalTerm(
    ops[op_name],
    op_name,
    lvals=lvals,
    has_obc=has_obc,
    sector_basis=sector_basis,
    sector_indices=sector_indices,
)
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
"""
[
    ["Sz", "Sz"],
    ["Sx", "Sm"],
    ["Sx", "Sp"],
    ["Sp", "Sx"],
    ["Sm", "Sx"],
    ["SpSm", "Sz"],
    ["Sz", "SpSm"],
]"""
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
# NBODY TERMS
nbody_obs = []
"""[
    ["Sm", "Sz", "Sz", "Sm"],
    ["Sz", "Sp", "Sm"],
    ["Sm", "Sp", "Sz"],
    ["Sx", "Sm", "Sz"],
    ["Sz", "Sp", "Sx"],
]"""
# LIST OF NBODY CORRELATORS
for op_names_list in nbody_obs:
    obs = "-".join(op_names_list)
    op_list = [ops[op] for op in op_names_list]
    h_terms[obs] = NBodyTerm(
        op_list=op_list,
        op_names_list=op_names_list,
        lvals=lvals,
        has_obc=has_obc,
        sector_indices=sector_indices,
        sector_basis=sector_basis,
    )
# ===========================================================================
for ii in range(n_eigs):
    logger.info("====================================================")
    logger.info(f"{ii} ENERGY: {format(res['energy'][ii], '.9f')}")
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
        logger.info("----------------------------------------------------")
        logger.info(f"{obs1}_{obs2}")
        logger.info("----------------------------------------------------")
        h_terms[f"{obs1}_{obs2}"].get_expval(H.Npsi[ii])
    # MEASURE NBODY OBSERVABLES:
    for op_names_list in nbody_obs:
        obs = "-".join(op_names_list)
        logger.info("----------------------------------------------------")
        logger.info(f"{obs}")
        logger.info("----------------------------------------------------")
        h_terms[f"{obs}"].get_expval(H.Npsi[ii])
end = time()
tot_time = end - start
logger.info("")
logger.info("TOT TIME {tot_time}")

# %%
"""C = np.outer(H.Npsi[1].psi, np.conjugate(H.Npsi[0].psi))
indices = np.array(np.where(np.abs(C) > 1e-3)).T
data = csr_matrix(truncation(C, threshold=1e-3)).data

if sector:
    true_indices_r = sector_indices[indices[:, 0]]
    true_indices_c = sector_indices[indices[:, 1]]
    true_indices = np.array([true_indices_r, true_indices_c]).T
    basis_r = sector_basis[indices[:, 0], :]
    basis_c = sector_basis[indices[:, 1], :]
else:
    basis_r = basis[indices[:, 0], :]
    basis_c = basis[indices[:, 1], :]
    true_indices = indices

# Order data and indices according to data in descending order
order = np.argsort(-np.abs(data))
true_indices = true_indices[order, :]
o_basis_r = basis_r[order, :]
o_basis_c = basis_c[order, :]
true_data = data[order]
for ii in range(data.shape[0]):
    logger.info(
        true_indices[ii],
        round(true_data[ii], 6),
        o_basis_r[ii],
        o_basis_c[ii],
        list(np.where(~np.equal(o_basis_r[ii], o_basis_c[ii]))[0]),
    )
"""
# %%
