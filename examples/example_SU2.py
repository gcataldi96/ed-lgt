# %%
import numpy as np
from math import prod
from ed_lgt.operators import (
    SU2_Hamiltonian_couplings,
    SU2_dressed_site_operators,
    SU2_rishon_operators,
    SU2_gauge_invariant_states,
    SU2_check_gauss_law,
)
from ed_lgt.modeling import Ground_State, LocalTerm, TwoBodyTerm, PlaquetteTerm
from ed_lgt.modeling import (
    check_link_symmetry,
    entanglement_entropy,
    get_reduced_density_matrix,
    diagonalize_density_matrix,
    staggered_mask,
    get_state_configurations,
    truncation,
    lattice_base_configs,
)
from ed_lgt.tools import check_hermitian

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
j_max = 1 / 2
# PURE or FULL THEORY
pure_theory = True
# GET g COUPLING
g = 0.1
if pure_theory:
    m = None
else:
    m = 0.1
# ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
ops = SU2_dressed_site_operators(j_max, pure_theory, lattice_dim=dim)
# ACQUIRE SU2 BASIS and GAUGE INVARIANT STATES
M, states = SU2_gauge_invariant_states(j_max, pure_theory, lattice_dim=dim)
# ACQUIRE LOCAL DIMENSION OF EVERY SINGLE SITE
lattice_base, loc_dims = lattice_base_configs(M, lvals, has_obc, staggered=False)
loc_dims = loc_dims.transpose().reshape(n_sites)
lattice_base = lattice_base.transpose().reshape(n_sites)
print("local dimensions:", loc_dims)
# SU2_check_gauss_law(basis=M["site"], gauss_law_op=ops["S2_tot"])
# ACQUIRE HAMILTONIAN COEFFICIENTS
coeffs = SU2_Hamiltonian_couplings(pure_theory, g, m)
print("penalty", coeffs["eta"])
# ===============================================================================
# CONSTRUCT THE HAMILTONIAN
H = 0
h_terms = {}
# DEFINE THE OBSERVABLE LIST and RESULT DICTIONARY
obs_list = [f"T2_{s}{d}" for s in "mp" for d in directions] + ["E_square"]
for op_name in obs_list:
    h_terms[op_name] = LocalTerm(ops[op_name], op_name, lvals, has_obc, site_basis=M)
# -------------------------------------------------------------------------------
# LINK PENALTIES & Border penalties
for d in directions:
    op_name_list = [f"T2_p{d}", f"T2_m{d}"]
    op_list = [ops[op] for op in op_name_list]
    # Define the Hamiltonian term
    h_terms[f"W_{d}"] = TwoBodyTerm(
        axis=d,
        op_list=op_list,
        op_name_list=op_name_list,
        lvals=lvals,
        has_obc=has_obc,
        site_basis=M,
    )
    H += h_terms[f"W_{d}"].get_Hamiltonian(
        strength=-2 * coeffs["eta"], add_dagger=False
    )
    # SINGLE SITE OPERATORS needed for the LINK SYMMETRY/OBC PENALTIES
    for s in "mp":
        op_name = f"T4_{s}{d}"
        h_terms[op_name] = LocalTerm(
            ops[op_name],
            op_name,
            lvals=lvals,
            has_obc=has_obc,
            site_basis=M,
        )
        H += h_terms[op_name].get_Hamiltonian(strength=coeffs["eta"])
# -------------------------------------------------------------------------------
# ELECTRIC ENERGY
H += h_terms["E_square"].get_Hamiltonian(strength=coeffs["E"])
# -------------------------------------------------------------------------------
if not pure_theory:
    # -----------------------------------------------------------------------
    # STAGGERED MASS TERM
    for site in ["even", "odd"]:
        h_terms[f"N_{site}"] = LocalTerm(
            ops["N_tot"],
            "N_tot",
            lvals=lvals,
            has_obc=has_obc,
            site_basis=M,
        )
        H += h_terms[f"N_{site}"].get_Hamiltonian(
            coeffs[f"m_{site}"], staggered_mask(lvals, site)
        )
    # ADD NUMBER OPERATORS TO THE LIST OF OBSERVABLES
    for label in ["r", "g", "tot", "single", "pair"]:
        op_name = f"N_{label}"
        obs_list.append(op_name)
        h_terms[op_name] = LocalTerm(
            ops[op_name], op_name, lvals, has_obc, site_basis=M
        )
    # ------------------------------------------------------------------------
    # HOPPING
    for d in directions:
        for site in ["even", "odd"]:
            hopping_terms = [[f"Q1_p{d}_dag", f"Q2_m{d}"], [f"Q2_p{d}_dag", f"Q1_m{d}"]]
            for op_name_list in hopping_terms:
                op_list = [ops[op] for op in op_name_list]
                # Define the Hamiltonian term
                h_terms[f"{d}_hop_{site}"] = TwoBodyTerm(
                    d,
                    op_list,
                    op_name_list,
                    lvals=lvals,
                    has_obc=has_obc,
                    site_basis=M,
                )
                H += h_terms[f"{d}_hop_{site}"].get_Hamiltonian(
                    strength=coeffs[f"t{d}_{site}"],
                    add_dagger=True,
                    mask=staggered_mask(lvals, site),
                )
# -------------------------------------------------------------------------------
# PLAQUETTE TERM: MAGNETIC INTERACTION
plaq_list = []
plaquette_directions = ["xy", "xz", "yz"]
plaquette_set = [
    ["AB", "AB", "AB", "AB"],
    ["AA", "AB", "BB", "AB"],
    ["AB", "AB", "AA", "BB"],
    ["AB", "AB", "AA", "BB"],
    ["AB", "BB", "AB", "AA"],
    ["AA", "BB", "BB", "AA"],
    ["AB", "BB", "AA", "BA"],
    ["AA", "BB", "BA", "BA"],
    ["BB", "AA", "AB", "AB"],
    ["BA", "AA", "BB", "AB"],
    ["BB", "AA", "AA", "BB"],
    ["BA", "AA", "BA", "BB"],
    ["BB", "BA", "AB", "AA"],
    ["BA", "BA", "AB", "AA"],
    ["BB", "BA", "AA", "BA"],
    ["BA", "BA", "BA", "BA"],
]
plaquette_signs = [-1, +1, +1, -1, +1, -1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1]
for ii, pdir in enumerate(plaquette_directions):
    if (dim == 2 and ii == 0) or dim == 3:
        for jj, p_set in enumerate(plaquette_set):
            # DEFINE THE LIST OF CORNER OPERATORS
            op_name_list = [
                f"C{p_set[0]}_p{pdir[0]},p{pdir[1]}",
                f"C{p_set[1]}_p{pdir[1]},m{pdir[0]}",
                f"C{p_set[2]}_m{pdir[1]},p{pdir[0]}",
                f"C{p_set[3]}_m{pdir[0]},m{pdir[1]}",
            ]
            # CORRESPONDING LIST OF OPERATORS
            op_list = [ops[op] for op in op_name_list]
            # DEFINE THE PLAQUETTE CLASS
            plaq_name = f"P_{pdir}_" + "".join(p_set)
            h_terms[plaq_name] = PlaquetteTerm(
                [pdir[0], pdir[1]],
                op_list,
                op_name_list,
                lvals=lvals,
                has_obc=has_obc,
                site_basis=M,
                print_plaq=False,
            )
            # ADD THE HAMILTONIAN TERM
            H += h_terms[plaq_name].get_Hamiltonian(
                strength=plaquette_signs[jj] * coeffs["B"], add_dagger=True
            )
            # ADD THE PLAQUETTE TO THE LIST OF OBSERVABLES
            plaq_list.append(plaq_name)
# ===========================================================================
# CHECK THAT THE HAMILTONIAN IS HERMITIAN
check_hermitian(H)
# DIAGONALIZE THE HAMILTONIAN
GS = Ground_State(H, n_eigs)
# ===========================================================================
for ii in range(n_eigs):
    print("====================================================")
    print(f"{ii} ENERGY: {format(GS.Nenergies[ii], '.9f')}")
    # ===========================================================================
    # ENTROPY of a BIPARTITION
    # res["entropy"].append(
    #    entanglement_entropy(GS.Npsi[:, ii], loc_dims, lvals, lvals[0])
    # )
    # ===========================================================================
    # GET STATE CONFIGURATIONS
    get_state_configurations(truncation(GS.Npsi[:, ii], 1e-10), loc_dims, lvals)
    if not has_obc:
        # COMPUTE THE REDUCED DENSITY MATRIX
        rho = get_reduced_density_matrix(GS.Npsi[:, ii], loc_dims, lvals, 0)
        eigvals, _ = diagonalize_density_matrix(rho)
        print(eigvals)
    # ===========================================================================
    # MEASURE LOCAL OBSERVABLES:
    for obs in obs_list:
        h_terms[obs].get_expval(GS.Npsi[:, ii])
    # CHECK LINK SYMMETRIES
    for ax in directions:
        check_link_symmetry(
            ax, h_terms[f"T2_p{ax}"], h_terms[f"T2_m{ax}"], value=0, sign=-1
        )
    # MEASURE PLAQUETTES
    plaq_obs = 0
    for obs in plaq_list:
        h_terms[obs].get_expval(GS.Npsi[:, ii])
        plaq_obs += h_terms[obs].avg
    print("AVERAGE PLAQUETTE VALUE")
    print(plaq_obs)
# %%
