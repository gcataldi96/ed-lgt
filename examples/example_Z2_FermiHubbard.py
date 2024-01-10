# %%
import numpy as np
from scipy.sparse import identity as ID
from math import prod
from ed_lgt.modeling import Ground_State, LocalTerm, TwoBodyTerm, PlaquetteTerm
from ed_lgt.modeling import (
    check_link_symmetry,
    get_reduced_density_matrix,
    diagonalize_density_matrix,
    get_state_configurations,
    truncation,
    lattice_base_configs,
)
from ed_lgt.tools import check_hermitian
from ed_lgt.operators import (
    Z2_FermiHubbard_dressed_site_operators,
    Z2_FermiHubbard_gauge_invariant_states,
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
# ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
ops = Z2_FermiHubbard_dressed_site_operators(lattice_dim=dim)
# ACQUIRE SU2 BASIS and GAUGE INVARIANT STATES
M, states = Z2_FermiHubbard_gauge_invariant_states(lattice_dim=dim)
# ACQUIRE LOCAL DIMENSION OF EVERY SINGLE SITE
lattice_base, loc_dims = lattice_base_configs(M, lvals, has_obc, staggered=False)
loc_dims = loc_dims.transpose().reshape(n_sites)
lattice_base = lattice_base.transpose().reshape(n_sites)
tot_dim = prod(loc_dims)
print("local dimensions:", loc_dims)
# OBTAIN OPERATORS FOR TTN SIMULATIONS
TTN_ops = {}
for op in ops.keys():
    TTN_ops[op] = M["site"].transpose() * ops[op] * M["site"]
# Hamiltonian Couplings
coeffs = {"t": -1, "V": 10, "eta": 100}
# Symmetry sector (# of particles)
sector = None
# CONSTRUCT THE HAMILTONIAN
H = 0
h_terms = {}
# -------------------------------------------------------------------------------
# LINK PENALTIES & Border penalties
for d in directions:
    op_name_list = [f"n_p{d}", f"n_m{d}"]
    op_list = [ops[op] for op in op_name_list]
    # Define the Hamiltonian term
    h_terms[f"W{d}"] = TwoBodyTerm(
        axis=d,
        op_list=op_list,
        op_name_list=op_name_list,
        lvals=lvals,
        has_obc=has_obc,
        site_basis=M,
    )
    H += h_terms[f"W{d}"].get_Hamiltonian(strength=-2 * coeffs["eta"], add_dagger=False)
    # SINGLE SITE OPERATORS needed for the LINK SYMMETRY/OBC PENALTIES
    for s in "mp":
        op_name = f"n_{s}{d}"
        h_terms[op_name] = LocalTerm(
            ops[op_name],
            op_name,
            lvals=lvals,
            has_obc=has_obc,
            site_basis=M,
        )
        H += h_terms[op_name].get_Hamiltonian(strength=coeffs["eta"])
# -------------------------------------------------------------------------------
# COULOMB POTENTIAL
h_terms["V"] = LocalTerm(
    ops["N_pair_half"],
    "N_pair_half",
    lvals=lvals,
    has_obc=has_obc,
    site_basis=M,
)
H += h_terms["V"].get_Hamiltonian(strength=coeffs["V"])
# -------------------------------------------------------------------------------
# HOPPING
for d in directions:
    for s in ["up", "down"]:
        # Define the list of the 2 non trivial operators
        op_name_list = [f"Q{s}_p{d}_dag", f"Q{s}_m{d}"]
        op_list = [ops[op] for op in op_name_list]
        # Define the Hamiltonian term
        h_terms[f"{d}_hop_{s}"] = TwoBodyTerm(
            d,
            op_list,
            op_name_list,
            lvals=lvals,
            has_obc=has_obc,
            site_basis=M,
        )
        H += h_terms[f"{d}_hop_{s}"].get_Hamiltonian(
            strength=coeffs["t"], add_dagger=True
        )
# ===========================================================================
# SYMMETRY SECTOR
if sector is not None:
    op_name = f"N_tot"
    h_terms[op_name] = LocalTerm(
        ops[op_name],
        op_name,
        lvals=lvals,
        has_obc=has_obc,
        site_basis=M,
    )
    H += (
        0.5
        * coeffs["eta"]
        * (h_terms[op_name].get_Hamiltonian(strength=1) - sector * ID(tot_dim)) ** 2
    )
# ===========================================================================
# CHECK THAT THE HAMILTONIAN IS HERMITIAN
check_hermitian(H)
# DIAGONALIZE THE HAMILTONIAN
GS = Ground_State(H, n_eigs)
# Dictionary for results
res = {}
res["energy"] = GS.Nenergies
res["rho_eigvals"] = []
# ===========================================================================
# LOCAL OBSERVABLE LIST
local_obs = [f"n_{s}{d}" for d in directions for s in "mp"]
local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
local_obs += ["X_Cross"]
for obs in local_obs:
    h_terms[obs] = LocalTerm(ops[obs], obs, lvals, has_obc, site_basis=M)
    res[obs] = []
# TWO BODY OBSERVABLE LIST
twobody_obs = [["P_px", "P_mx"], ["P_py", "P_my"]]
for obs1, obs2 in twobody_obs:
    op_list = [ops[obs1], ops[obs2]]
    h_terms[f"{obs1}_{obs2}"] = TwoBodyTerm(
        axis="x",
        op_list=op_list,
        op_name_list=[obs1, obs2],
        lvals=lvals,
        has_obc=has_obc,
        site_basis=M,
    )
# PLAQUETTE OBSERVABLE
plaq_name_list = ["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]
op_list = [ops[op] for op in plaq_name_list]
h_terms["Plaq_Sx"] = PlaquetteTerm(
    axes=["x", "y"],
    op_list=op_list,
    op_name_list=plaq_name_list,
    lvals=lvals,
    has_obc=has_obc,
    site_basis=M,
)
# ===========================================================================
for ii in range(n_eigs):
    print("====================================================")
    print(f"{ii} ENERGY: {format(res['energy'][ii], '.9f')}")
    # COMPUTE THE REDUCED DENSITY MATRIX
    rho = get_reduced_density_matrix(GS.Npsi[:, ii], loc_dims, lvals, 0)
    eigvals, _ = diagonalize_density_matrix(rho)
    res["rho_eigvals"].append(eigvals)
    print(f"DM eigvals {res['rho_eigvals']}")
    if has_obc:
        # GET STATE CONFIGURATIONS
        get_state_configurations(truncation(GS.Npsi[:, ii], 1e-10), loc_dims, lvals)
    # ===========================================================================
    # LOCAL OBSERVABLES:
    # ===========================================================================
    for obs in local_obs:
        h_terms[obs].get_expval(GS.Npsi[:, ii])
        res[obs].append(h_terms[obs].avg)
    # CHECK LINK SYMMETRIES
    for ax in directions:
        check_link_symmetry(
            ax, h_terms[f"n_p{ax}"], h_terms[f"n_m{ax}"], value=0, sign=-1
        )
    # ===========================================================================
    # TWOBODY OBSERVABLES:
    # ===========================================================================
    for obs1, obs2 in twobody_obs:
        print("----------------------------------------------------")
        print(f"{obs1}_{obs2}")
        print("----------------------------------------------------")
        h_terms[f"{obs1}_{obs2}"].get_expval(GS.Npsi[:, ii])
        print(h_terms[f"{obs1}_{obs2}"].corr)
    # ===========================================================================
    # PLAQUETTE OBSERVABLES:
    # ===========================================================================
    h_terms["Plaq_Sx"].get_expval(GS.Npsi[:, ii])

if n_eigs > 1:
    print(f"Energies {res['energy']}")
    res["DeltaE"] = np.abs(res["energy"][0] - res["energy"][1])

# %%
