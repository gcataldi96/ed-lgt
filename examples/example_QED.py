# %%
from math import prod
from ed_lgt.operators import (
    QED_dressed_site_operators,
    QED_gauge_invariant_states,
    QED_Hamiltonian_couplings,
)
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm, QMB_hamiltonian
from ed_lgt.modeling import (
    check_link_symmetry,
    diagonalize_density_matrix,
    staggered_mask,
    lattice_base_configs,
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
has_obc = True
# DEFINE the truncation of the gauge field and the type of U
spin = 1
U = "ladder"
# PURE or FULL THEORY
pure_theory = False
# GET g COUPLING
g = 0.1
if pure_theory:
    m = None
    staggered_basis = False
else:
    m = 0.1
    staggered_basis = True
# ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
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
# ===============================================================================
# CONSTRUCT THE HAMILTONIAN
H = QMB_hamiltonian(0, lvals, loc_dims)
h_terms = {}
# -------------------------------------------------------------------------------
# LINK PENALTIES & Border penalties
for d in directions:
    op_names_list = [f"E_p{d}", f"E_m{d}"]
    op_list = [ops[op] for op in op_names_list]
    # Define the Hamiltonian term
    h_terms[f"W_{d}"] = TwoBodyTerm(
        axis=d,
        op_list=op_list,
        op_names_list=op_names_list,
        lvals=lvals,
        has_obc=has_obc,
        staggered_basis=staggered_basis,
        site_basis=M,
    )
    H.Ham += h_terms[f"W_{d}"].get_Hamiltonian(strength=2 * coeffs["eta"])
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
        H.Ham += h_terms[op_name].get_Hamiltonian(strength=coeffs["eta"])
# -------------------------------------------------------------------------------
# ELECTRIC ENERGY
op_name = "E_square"
h_terms[op_name] = LocalTerm(
    ops[op_name],
    op_name,
    lvals=lvals,
    has_obc=has_obc,
    staggered_basis=staggered_basis,
    site_basis=M,
)
H.Ham += h_terms[op_name].get_Hamiltonian(strength=coeffs["E"])
# -------------------------------------------------------------------------------
# PLAQUETTE TERM: MAGNETIC INTERACTION
if dim > 1:
    op_names_list = ["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]
    op_list = [ops[op] for op in op_names_list]
    h_terms["plaq_xy"] = PlaquetteTerm(
        ["x", "y"],
        op_list,
        op_names_list,
        lvals=lvals,
        has_obc=has_obc,
        staggered_basis=staggered_basis,
        site_basis=M,
    )
    H.Ham += h_terms["plaq_xy"].get_Hamiltonian(strength=coeffs["B"], add_dagger=True)
if dim == 3:
    # XZ Plane
    op_names_list = ["C_px,pz", "C_pz,mx", "C_mz,px", "C_mx,mz"]
    op_list = [ops[op] for op in op_names_list]
    h_terms["plaq_xz"] = PlaquetteTerm(
        ["x", "z"],
        op_list,
        op_names_list,
        lvals=lvals,
        has_obc=has_obc,
        staggered_basis=staggered_basis,
        site_basis=M,
    )
    H.Ham += h_terms["plaq_xz"].get_Hamiltonian(strength=coeffs["B"], add_dagger=True)
    # YZ Plane
    op_names_list = ["C_py,pz", "C_pz,my", "C_mz,py", "C_my,mz"]
    op_list = [ops[op] for op in op_names_list]
    h_terms["plaq_yz"] = PlaquetteTerm(
        ["y", "z"],
        op_list,
        op_names_list,
        lvals=lvals,
        has_obc=has_obc,
        staggered_basis=staggered_basis,
        site_basis=M,
    )
    H.Ham += h_terms["plaq_yz"].get_Hamiltonian(strength=coeffs["B"], add_dagger=True)
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
        H.Ham += h_terms[f"N_{site}"].get_Hamiltonian(
            coeffs[f"m_{site}"], staggered_mask(lvals, site)
        )
    # ---------------------------------------------------------------------------
    # HOPPING
    for d in directions:
        for site in ["even", "odd"]:
            # Define the list of the 2 non trivial operators
            op_names_list = [f"Q_p{d}_dag", f"Q_m{d}"]
            op_list = [ops[op] for op in op_names_list]
            # Define the Hamiltonian term
            h_terms[f"{d}_hop_{site}"] = TwoBodyTerm(
                d,
                op_list,
                op_names_list,
                lvals=lvals,
                has_obc=has_obc,
                staggered_basis=staggered_basis,
                site_basis=M,
            )
            H.Ham += h_terms[f"{d}_hop_{site}"].get_Hamiltonian(
                strength=coeffs[f"t{d}_{site}"],
                add_dagger=True,
                mask=staggered_mask(lvals, site),
            )
# ===========================================================================
# DIAGONALIZE THE HAMILTONIAN
H.diagonalize(n_eigs)
# Dictionary for results
res = {}
res["energy"] = H.Nenergies
# ===========================================================================
# DEFINE THE OBSERVABLE LIST
res["entropy"] = []
res["rho_eigvals"] = []
# LIST OF LOCAL OBSERVABLES
local_obs = [f"E_{s}{d}" for s in "mp" for d in directions] + ["E_square"]
if not pure_theory:
    local_obs.append("N")
for obs in local_obs:
    h_terms[obs] = LocalTerm(
        ops[obs],
        obs,
        lvals=lvals,
        has_obc=has_obc,
        staggered_basis=staggered_basis,
        site_basis=M,
    )
    res[obs] = []
# LIST OF PLAQUETTE OBSERVABLES
plaquette_obs = []
if dim > 1:
    plaquette_obs.append("plaq_xy")
if dim == 3:
    plaquette_obs += ["plaq_xz", "plaq_yz"]
for plaq_name in plaquette_obs:
    res[plaq_name] = []
# ===========================================================================
for ii in range(n_eigs):
    print("====================================================")
    print(f"{ii} ENERGY: {format(res['energy'][ii], '.9f')}")
    if dim < 3:
        # ENTROPY of a BIPARTITION
        res["entropy"].append(H.Npsi[ii].entanglement_entropy(int(prod(lvals) / 2)))
    # GET STATE CONFIGURATIONS
    H.Npsi[ii].get_state_configurations(threshold=1e-3)
    if not has_obc:
        # COMPUTE THE REDUCED DENSITY MATRIX
        rho = H.Npsi[ii].reduced_density_matrix(0)
        eigvals, _ = diagonalize_density_matrix(rho)
        res["rho_eigvals"].append(eigvals)
    # =======================================================================
    # MEASURE LOCAL OBSERVABLES:
    for obs in local_obs:
        h_terms[obs].get_expval(H.Npsi[ii])
        res[obs].append(h_terms[obs].avg)
    # CHECK LINK SYMMETRIES
    for ax in directions:
        check_link_symmetry(ax, h_terms[f"E_p{ax}"], h_terms[f"E_m{ax}"])
    # MEASURE PLAQUETTE OBSERVABLES:
    for plaq_name in plaquette_obs:
        print("----------------------------------------------------")
        print(plaq_name)
        print("----------------------------------------------------")
        h_terms[plaq_name].get_expval(H.Npsi[ii])
        res[plaq_name].append(h_terms[plaq_name].avg)

# %%
