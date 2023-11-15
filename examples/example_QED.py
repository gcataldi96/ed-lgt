# %%
import numpy as np
from math import prod
from ed_lgt.operators import (
    QED_dressed_site_operators,
    QED_rishon_operators,
    QED_gauge_invariant_states,
    QED_Hamiltonian_couplings,
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
from ed_lgt.tools import check_hermitian, anti_commutator as anti_comm

# %%
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
spin = 1
U = "ladder"
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
# ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
in_ops = QED_rishon_operators(spin, pure_theory, U)
ops = QED_dressed_site_operators(spin, pure_theory, U, lattice_dim=dim)
# Acquire Basis and gauge invariant states
M, states = QED_gauge_invariant_states(spin, pure_theory, lattice_dim=dim)
# ACQUIRE LOCAL DIMENSION OF EVERY SINGLE SITE
lattice_base, loc_dims = lattice_base_configs(
    M, lvals, has_obc, staggered=staggered_basis
)
loc_dims = loc_dims.transpose().reshape(n_sites)
lattice_base = lattice_base.transpose().reshape(n_sites)
print("local dimensions:", loc_dims)
# ACQUIRE HAMILTONIAN COEFFICIENTS
coeffs = QED_Hamiltonian_couplings(pure_theory, g, m)
# %%
# %%
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
if dim > 1:
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
if dim == 3:
    # XZ Plane
    op_name_list = [
        "C_px,pz",
        "C_pz,mx",
        "C_mz,px",
        "C_mx,mz",
    ]
    op_list = [ops[op] for op in op_name_list]
    h_terms["plaq_xz"] = PlaquetteTerm(
        ["x", "z"],
        op_list,
        op_name_list,
        lvals=lvals,
        has_obc=has_obc,
        staggered_basis=staggered_basis,
        site_basis=M,
    )
    H += h_terms["plaq_xz"].get_Hamiltonian(strength=coeffs["B"], add_dagger=True)
    # YZ Plane
    op_name_list = [
        "C_py,pz",
        "C_pz,my",
        "C_mz,py",
        "C_my,mz",
    ]
    op_list = [ops[op] for op in op_name_list]
    h_terms["plaq_yz"] = PlaquetteTerm(
        ["y", "z"],
        op_list,
        op_name_list,
        lvals=lvals,
        has_obc=has_obc,
        staggered_basis=staggered_basis,
        site_basis=M,
    )
    H += h_terms["plaq_yz"].get_Hamiltonian(strength=coeffs["B"], add_dagger=True)
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
res["rho_eigvals"] = []

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
    eigvals, _ = diagonalize_density_matrix(rho)
    res["rho_eigvals"].append(eigvals)
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
    print(f"DM eigvals {res['rho_eigvals']}")
if n_eigs > 1:
    res["DeltaE"] = np.abs(res["energy"][0] - res["energy"][1])

# %%
