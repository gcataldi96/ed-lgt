# %%
import numpy as np
from scipy.linalg import schur
from math import prod
from copy import deepcopy
from ed_lgt.operators import (
    Zn_rishon_operators,
    Zn_gauge_invariant_ops,
    Zn_corner_magnetic_basis,
    Zn_dressed_site_operators,
    Zn_gauge_invariant_states,
    QED_Hamiltonian_couplings,
    get_lambda_subspace,
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
from numpy.linalg import eigh
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import norm
from ed_lgt.tools import (
    check_hermitian,
    anti_commutator as anti_comm,
    check_commutator as check_comm,
)

# N eigenvalues
n_eigs = 1
# LATTICE DIMENSIONS
lvals = [2, 2]
dim = len(lvals)
directions = "xyz"[:dim]
# TOTAL NUMBER OF LATTICE SITES & particles
n_sites = prod(lvals)
# BOUNDARY CONDITIONS
has_obc = False
# DEFINE the truncation of the gauge field and the type of U
n = 3
# PURE or FULL THEORY
pure_theory = True
# GET g COUPLING
g = 0.1
if pure_theory:
    m = None
    staggered_basis = False
else:
    m = 0.1
    staggered_basis = True
# Obtain the gauge invariant operators
ops = Zn_gauge_invariant_ops(n, pure_theory, dim)
# ACQUIRE BASIS AND GAUGE INVARIANT STATES FOR ANY POSSIBLE TYPE OF LATTICE
M, states = Zn_gauge_invariant_states(n, pure_theory, lattice_dim=dim)
# ACQUIRE LOCAL DIMENSION OF EVERY SINGLE SITE
lattice_base, loc_dims = lattice_base_configs(
    M, lvals, has_obc, staggered=staggered_basis
)
loc_dims = loc_dims.transpose().reshape(n_sites)
lattice_base = lattice_base.transpose().reshape(n_sites)


def get_lambda_subspace(vals, vecs):
    subspaces_vals = []
    subspaces_vecs = []
    for ii, llambda in enumerate(vals):
        if ii == 0:
            subspaces_vals.append(llambda)
            subspaces_vecs.append([vecs[:, ii]])
        else:
            if np.any(np.isclose(llambda, subspaces_vals)):
                tmp = np.where(np.isclose(llambda, subspaces_vals))[0][0]
                subspaces_vecs[tmp].append(vecs[:, ii])
            else:
                subspaces_vals.append(llambda)
                subspaces_vecs.append([vecs[:, ii]])
        print(f"i={ii}", format(llambda, ".3f"), len(subspaces_vals))
    return subspaces_vals, subspaces_vecs


from numpy.linalg import eig


def diagonalize_corner(C):
    T, Z = eig(C.toarray())
    sorted_indices = np.lexsort((T.imag, T.real))
    T = T[sorted_indices]
    Z = Z[:, sorted_indices]
    return T, Z


# Obtain the gauge invariant operators
ops = Zn_gauge_invariant_ops(n, pure_theory, lattice_dim=2)
# ACQUIRE BASIS AND GAUGE INVARIANT STATES FOR ANY POSSIBLE TYPE OF LATTICE
M, _ = Zn_gauge_invariant_states(n, pure_theory, lattice_dim=2)
# DEFINE OBSERVABLES for MAGNETIC BASIS
dim_basis = M["site"].shape[1]
magnetic_basis = {
    "config": np.zeros((dim_basis, 4), dtype=complex),
    "basis": np.zeros((dim_basis, dim_basis), dtype=complex),
}
corner_names = ["C_mx,my", "C_my,px", "C_px,py", "C_py,mx"]
c_name_0 = corner_names[0]
c_name_1 = corner_names[1]
c_name_2 = corner_names[2]
c_name_3 = corner_names[3]

corners = {}
for ii, name in enumerate(corner_names):
    corners[name] = {
        "vals": [],
        "vecs": [],
        "s_vals": [],
        "s_vecs": [],
        "secs0": [],
        "secs1": [],
        "secs2": [],
    }

# %%
np.set_printoptions(precision=3, suppress=True)
# Start from diagonalizing the 1st corner operator
T0, Z0 = diagonalize_corner(ops[c_name_0])
corners[c_name_0]["vals"] = T0
corners[c_name_0]["vecs"] = Z0
# Register the first set of vals
magnetic_basis["config"][:, 0] = T0
# Look at the subeigenspaces of each eigenvalue
s_vals0, s_vecs0 = get_lambda_subspace(T0, Z0)
corners[c_name_0]["s_vals"] = deepcopy(s_vals0)
corners[c_name_0]["s_vecs"] = deepcopy(s_vecs0)
# %%
# Project the 2nd corner on each sector
for i0, s0 in enumerate(s_vecs0):
    print(f"{i0} =============================================")
    print(f"{s_vals0[i0]}")
    small_dim0 = len(s0)
    large_dim0 = dim_basis
    # Create the projector on the subspace
    P0 = csr_matrix(np.concatenate(s0).reshape((small_dim0, large_dim0)))
    # Project the other corners on this subspace
    for c_name in corner_names[1:]:
        c = P0 * ops[c_name_1] * P0.transpose().conjugate()
        corners[c_name]["secs0"].append(c)
    # ------------------------------------------------------------------
    # Focus on the 2nd corner and diagonalize it
    T1, Z1 = diagonalize_corner(corners[c_name_1]["secs0"][i0])
    corners[c_name_1]["vals"].append(T1)
    # For each diagonalization, obtain subsectors
    s_vals1, s_vecs1 = get_lambda_subspace(T1, Z1)
    # Register sector eigvals & eigvecs
    corners[c_name_1]["s_vals"].append(s_vals1)
    corners[c_name_1]["s_vecs"].append(s_vecs1)
    # ------------------------------------------------------------------
    # Focus on the 3rd corner and diagonalize it
    for i1, s1 in enumerate(s_vecs1):
        small_dim1 = len(s1)
        large_dim1 = small_dim0
        print("------------------------------")
        print(f"{s_vals0[i0]} {s_vals1[i1]}")
        # Create the projector on the subspace
        P1 = csr_matrix(np.concatenate(s1).reshape((small_dim1, large_dim1)))
        # Project the other corners on this subspace
        for c_name in corner_names[2:]:
            corners[c_name]["secs1"].append(
                P1 * corners[c_name]["secs0"][i0] * P1.transpose().conjugate()
            )
        # DIAGONALIZE THE 3RD corner
        T2, Z2 = diagonalize_corner(corners[c_name_2]["secs1"][i1])
        corners[c_name_2]["vals"].append(T2)
        # For each diagonalization, obtain subsectors
        s_vals2, s_vecs2 = get_lambda_subspace(T2, Z2)
        # Register sector eigvals & eigvecs
        corners[c_name_2]["s_vals"].append(s_vals2)
        corners[c_name_2]["s_vecs"].append(s_vecs2)
        # --------------------------------------------------------------
        # Focus on the 4th corner and diagonalize it
        for i2, s2 in enumerate(corners[c_name_2]["s_vecs"][-1]):
            small_dim2 = len(s2)
            large_dim2 = small_dim1
            print("#################")
            print(f"{c_name_2}={corners[c_name_2]['s_vals'][-1][i2]}")
            # Create the projector on the subspace
            P2 = csr_matrix(np.concatenate(s2).reshape((small_dim2, large_dim2)))
            # Project the other corners on this subspace
            corners[c_name_3]["secs2"].append(
                P2 * corners[c_name_3]["secs1"][-1] * P2.transpose().conjugate()
            )
            # DIAGONALIZE THE 3RD corner
            T3, Z3 = diagonalize_corner(corners[c_name_3]["secs2"][-1])
            corners[c_name_3]["vals"].append(T3)
            # For each diagonalization, obtain subsectors
            s_vals, s_vecs = get_lambda_subspace(T3, Z3)
            # Register sector eigvals & eigvecs
            corners[c_name_3]["s_vals"].append(s_vals)
            corners[c_name_3]["s_vecs"].append(s_vecs)
# %%
# Register the eigenvalues of the 2nd adn 3rd corner
corners[c_name_1]["vals"] = np.concatenate(corners[c_name_1]["vals"])
magnetic_basis["config"][:, 1] = corners[c_name_1]["vals"]
# %%
corners[c_name_2]["vals"] = np.concatenate(corners[c_name_2]["vals"])
magnetic_basis["config"][:, 2] = corners[c_name_2]["vals"]
# %%
corners[c_name_3]["vals"] = np.concatenate(corners[c_name_3]["vals"])
magnetic_basis["config"][:, 3] = corners[c_name_3]["vals"]


# %%
# ACQUIRE HAMILTONIAN COEFFICIENTS
coeffs = QED_Hamiltonian_couplings(pure_theory, g, m)
# CONSTRUCT THE HAMILTONIAN
H = 0
h_terms = {}
# -------------------------------------------------------------------------------
# LINK PENALTIES & Border penalties
for d in directions:
    op_name_list = [f"E_p{d}", f"E_m{d}"]
    op_list = [ops[op] for op in op_name_list]
    # Define the Hamiltonian term
    h_terms[f"W_{d}"] = TwoBodyTerm(
        axis=d,
        op_list=op_list,
        op_name_list=op_name_list,
        lvals=lvals,
        has_obc=has_obc,
        staggered_basis=staggered_basis,
        site_basis=M,
    )
    H += h_terms[f"W_{d}"].get_Hamiltonian(strength=2 * coeffs["eta"], add_dagger=False)
    # SINGLE SITE OPERATORS needed for the LINK SYMMETRY/OBC PENALTIES
    for s in "mp":
        op_name = f"E_square_{s}{d}"
        h_terms[op_name] = LocalTerm(
            ops[op_name],
            op_name,
            lvals=lvals,
            has_obc=has_obc,
            staggered_basis=staggered_basis,
            site_basis=M,
        )
        H += h_terms[op_name].get_Hamiltonian(strength=coeffs["eta"])
# -------------------------------------------------------------------------------
# ELECTRIC ENERGY
h_terms["E_square"] = LocalTerm(
    ops["E_square"],
    "E_square",
    lvals=lvals,
    has_obc=has_obc,
    staggered_basis=staggered_basis,
    site_basis=M,
)
H += h_terms["E_square"].get_Hamiltonian(strength=coeffs["E"])
# -------------------------------------------------------------------------------
# PLAQUETTE TERM: MAGNETIC INTERACTION
op_name_list = [
    "C_px,py",
    "C_py,mx",
    "C_my,px",
    "C_mx,my",
]
op_list = [ops[op] for op in op_name_list]
h_terms["plaq_xy"] = PlaquetteTerm(
    ["x", "y"],
    op_list,
    op_name_list,
    lvals=lvals,
    has_obc=has_obc,
    staggered_basis=staggered_basis,
    site_basis=M,
)
H += h_terms["plaq_xy"].get_Hamiltonian(strength=coeffs["B"], add_dagger=True)
# -------------------------------------------------------------------------------
if not pure_theory:
    # ---------------------------------------------------------------------------
    # STAGGERED MASS TERM
    for site in ["even", "odd"]:
        h_terms[f"N_{site}"] = LocalTerm(
            ops["N"],
            "N",
            lvals=lvals,
            has_obc=has_obc,
            staggered_basis=staggered_basis,
            site_basis=M,
        )
        H += h_terms[f"N_{site}"].get_Hamiltonian(
            coeffs[f"m_{site}"], staggered_mask(lvals, site)
        )
    # ---------------------------------------------------------------------------
    # HOPPING
    for d in directions:
        for site in ["even", "odd"]:
            # Define the list of the 2 non trivial operators
            op_name_list = [f"Q_p{d}_dag", f"Q_m{d}"]
            op_list = [ops[op] for op in op_name_list]
            # Define the Hamiltonian term
            h_terms[f"{d}_hop_{site}"] = TwoBodyTerm(
                d,
                op_list,
                op_name_list,
                lvals=lvals,
                has_obc=has_obc,
                staggered_basis=staggered_basis,
                site_basis=M,
            )
            H += h_terms[f"{d}_hop_{site}"].get_Hamiltonian(
                strength=coeffs[f"t{d}_{site}"],
                add_dagger=True,
                mask=staggered_mask(lvals, site),
            )
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
res["rho_vals"] = []

obs_list = [f"E_{s}{d}" for s in "mp" for d in directions] + ["E_square"]
if not pure_theory:
    obs_list.append("N")
for obs in obs_list:
    h_terms[obs] = LocalTerm(
        ops[obs], obs, lvals, has_obc, staggered_basis=staggered_basis, site_basis=M
    )
    res[obs] = []
if dim > 1:
    obs_list.append("plaq_xy")
    res["plaq_xy"] = []
if dim == 3:
    obs_list += ["plaq_xz", "plaq_yz"]
    res["plaq_xz"] = []
    res["plaq_yz"] = []
# ===========================================================================
for ii in range(n_eigs):
    print("====================================================")
    print(f"{ii} ENERGY: {format(res['energy'][ii], '.9f')}")
    # ENTROPY of a BIPARTITION
    # res["entropy"].append(
    #    entanglement_entropy(GS.Npsi[:, ii], loc_dims, lvals, lvals[0])
    # )
    # COMPUTE THE REDUCED DENSITY MATRIX
    rho = get_reduced_density_matrix(GS.Npsi[:, ii], loc_dims, lvals, 0)
    vals, _ = diagonalize_density_matrix(rho)
    res["rho_vals"].append(vals)
    if has_obc:
        # GET STATE CONFIGURATIONS
        get_state_configurations(truncation(GS.Npsi[:, ii], 1e-10), loc_dims, lvals)
    # ===========================================================================
    # MEASURE OBSERVABLES:
    # RISHON NUMBER OPERATORS, ELECTRIC ENERGY E^{2}, DENSITY OPERATOR N, B^{2}
    # ===========================================================================
    for obs in obs_list:
        h_terms[obs].get_expval(GS.Npsi[:, ii])
        res[obs].append(h_terms[obs].avg)
print(f"Energies {res['energy']}")
if not has_obc:
    print(f"DM vals {res['rho_vals']}")
if n_eigs > 1:
    res["DeltaE"] = np.abs(res["energy"][0] - res["energy"][1])

# %%
# Project the 2nd corner on each sector
for i0, s0 in enumerate(corners[c_name_0]["s_vecs"]):
    print("=============================================")
    print(f"{i0} {c_name_0}={corners[c_name_0]['s_vals'][i0]}")
    small_dim0 = len(s0)
    large_dim0 = dim_basis
    # Create the projector on the subspace
    P0 = csr_matrix(np.concatenate(s0).reshape((small_dim0, large_dim0)))
    # Project the other corners on this subspace
    for c_name in corner_names[1:]:
        c = P0 * ops[c_name_1] * P0.transpose().conjugate()
        corners[c_name]["secs0"].append(c)
    # ------------------------------------------------------------------
    # Focus on the 2nd corner and diagonalize it
    T1, Z1 = diagonalize_corner(corners[c_name_1]["secs0"][i0])
    corners[c_name_1]["vals"].append(T1)
    # For each diagonalization, obtain subsectors
    s_vals, s_vecs = get_lambda_subspace(T1, Z1)
    # Register sector eigvals & eigvecs
    corners[c_name_1]["s_vals"].append(s_vals)
    corners[c_name_1]["s_vecs"].append(s_vecs)
    # ------------------------------------------------------------------
    # Focus on the 3rd corner and diagonalize it
    for i1, s1 in enumerate(corners[c_name_1]["s_vecs"][-1]):
        small_dim1 = len(s1)
        large_dim1 = small_dim0
        print("------------------------------")
        print(f"{c_name_1}={corners[c_name_1]['s_vals'][-1][i1]}")
        # Create the projector on the subspace
        P1 = csr_matrix(np.concatenate(s1).reshape((small_dim1, large_dim1)))
        # Project the other corners on this subspace
        for c_name in corner_names[2:]:
            corners[c_name]["secs1"].append(
                P1 * corners[c_name]["secs0"][i0] * P1.transpose().conjugate()
            )
        # DIAGONALIZE THE 3RD corner
        T2, Z2 = diagonalize_corner(corners[c_name_2]["secs1"][-1])
        corners[c_name_2]["vals"].append(T2)
        # For each diagonalization, obtain subsectors
        s_vals, s_vecs = get_lambda_subspace(T2, Z2)
        # Register sector eigvals & eigvecs
        corners[c_name_2]["s_vals"].append(s_vals)
        corners[c_name_2]["s_vecs"].append(s_vecs)
        # --------------------------------------------------------------
        # Focus on the 4th corner and diagonalize it
        for i2, s2 in enumerate(corners[c_name_2]["s_vecs"][-1]):
            small_dim2 = len(s2)
            large_dim2 = small_dim1
            print("#################")
            print(f"{c_name_2}={corners[c_name_2]['s_vals'][-1][i2]}")
            # Create the projector on the subspace
            P2 = csr_matrix(np.concatenate(s2).reshape((small_dim2, large_dim2)))
            # Project the other corners on this subspace
            corners[c_name_3]["secs2"].append(
                P2 * corners[c_name_3]["secs1"][-1] * P2.transpose().conjugate()
            )
            # DIAGONALIZE THE 3RD corner
            T3, Z3 = diagonalize_corner(corners[c_name_3]["secs2"][-1])
            corners[c_name_3]["vals"].append(T3)
            # For each diagonalization, obtain subsectors
            s_vals, s_vecs = get_lambda_subspace(T3, Z3)
            # Register sector eigvals & eigvecs
            corners[c_name_3]["s_vals"].append(s_vals)
            corners[c_name_3]["s_vecs"].append(s_vecs)
# %%
# Register the eigenvalues of the 2nd adn 3rd corner
corners[c_name_1]["vals"] = np.concatenate(corners[c_name_1]["vals"])
magnetic_basis["config"][:, 1] = corners[c_name_1]["vals"]
# %%
corners[c_name_2]["vals"] = np.concatenate(corners[c_name_2]["vals"])
magnetic_basis["config"][:, 2] = corners[c_name_2]["vals"]
# %%
corners[c_name_3]["vals"] = np.concatenate(corners[c_name_3]["vals"])
magnetic_basis["config"][:, 3] = corners[c_name_3]["vals"]
