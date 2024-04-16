# %%
import numpy as np
from math import prod
from numba import typed
import logging
from ed_lgt.symmetries import (
    get_symmetry_sector_generators,
    global_abelian_sector,
    link_abelian_sector,
)
from ed_lgt.modeling import QMB_hamiltonian, LocalTerm, TwoBodyTerm, PlaquetteTerm
from ed_lgt.modeling import (
    lattice_base_configs,
    zig_zag,
    get_neighbor_sites,
    check_link_symmetry,
)
from ed_lgt.operators import (
    Z2_FermiHubbard_dressed_site_operators,
    Z2_FermiHubbard_gauge_invariant_states,
)

logger = logging.getLogger(__name__)
# N eigenvalues
n_eigs = 2
# LATTICE DIMENSIONS
lvals = [3, 2]
dim = len(lvals)
directions = "xyz"[:dim]
# TOTAL NUMBER OF LATTICE SITES & particles
n_sites = prod(lvals)
# BOUNDARY CONDITIONS
has_obc = [True, True]
# ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
ops = Z2_FermiHubbard_dressed_site_operators(lattice_dim=dim)
# ACQUIRE SU2 BASIS and GAUGE INVARIANT STATES
M, states = Z2_FermiHubbard_gauge_invariant_states(lattice_dim=dim)
def_params = {
    "lvals": lvals,
    "has_obc": has_obc,
    "gauge_basis": M,
    "staggered_basis": False,
}
# ACQUIRE LOCAL DIMENSION OF EVERY SINGLE SITE
lattice_labels, loc_dims = lattice_base_configs(**def_params)
loc_dims = loc_dims.transpose().reshape(n_sites)
lattice_labels = lattice_labels.transpose().reshape(n_sites)
tot_dim = prod(loc_dims)
logger.info(f"local dimensions: {loc_dims}")
# GET operators for global symmetry sector
op_list = [ops["N_tot"], ops["N_up"]]
# op_diagonals = np.array([np.diag(op) for op in op_list], dtype=np.int64)
op_diagonals = get_symmetry_sector_generators(
    op_list, loc_dims, action="global", gauge_basis=M, lattice_labels=lattice_labels
)
op_sectors_list = np.array([n_sites, int(n_sites / 2)], dtype=np.int64)
# ================================================================================
# Acquire the twobody link symmetry operators
pair_list = typed.List()
for d in directions:
    dir_list = []
    for ii in range(prod(lvals)):
        # Compute the corresponding coords
        coords = zig_zag(lvals, ii)
        # Check if it admits a twobody term according to the lattice geometry
        _, sites_list = get_neighbor_sites(coords, lvals, d, has_obc)
        if sites_list is not None:
            dir_list.append(sites_list)
    pair_list.append(np.array(dir_list, dtype=np.uint8))

link_ops = [
    [ops["P_px"], ops["P_mx"]],
    [ops["P_py"], ops["P_my"]],
]


link_ops = get_symmetry_sector_generators(
    link_ops, loc_dims, action="link", gauge_basis=M, lattice_labels=lattice_labels
)
link_sectors = np.array([1, 1], dtype=np.int64)
# ===========================================================
# Compute the global symmetry sector
sector_ind, sector_config = global_abelian_sector(
    loc_dims, sym_op_diags=op_diagonals, sym_sectors=op_sectors_list, sym_type="U"
)
sector_ind, sector_configs = link_abelian_sector(
    loc_dims,
    sym_op_diags=link_ops,
    sym_sectors=link_sectors,
    pair_list=pair_list,
    configs=sector_config,
)
# Hamiltonian Couplings
coeffs = {"t": -1, "U": 0.1, "eta": 100}
# CONSTRUCT THE HAMILTONIAN
H = QMB_hamiltonian(0, lvals, loc_dims)
h_terms = {}
# -------------------------------------------------------------------------------
# COULOMB POTENTIAL
h_terms["U"] = LocalTerm(
    ops["N_pair_half"],
    "N_pair_half",
    lvals=lvals,
    has_obc=has_obc,
    gauge_basis=M,
    sector_configs=sector_configs,
)
H.Ham += h_terms["U"].get_Hamiltonian(strength=coeffs["U"])
# -------------------------------------------------------------------------------
# HOPPING
for d in directions:
    for s in ["up", "down"]:
        # Define the list of the 2 non trivial operators
        op_names_list = [f"Q{s}_p{d}_dag", f"Q{s}_m{d}"]
        op_list = [ops[op] for op in op_names_list]
        # Define the Hamiltonian term
        h_terms[f"{d}_hop_{s}"] = TwoBodyTerm(
            d,
            op_list,
            op_names_list,
            lvals=lvals,
            has_obc=has_obc,
            gauge_basis=M,
            sector_configs=sector_configs,
        )
        H.Ham += h_terms[f"{d}_hop_{s}"].get_Hamiltonian(
            strength=coeffs["t"], add_dagger=True
        )
# ===========================================================================
# DIAGONALIZE THE HAMILTONIAN
H.diagonalize(n_eigs)
# Dictionary for results
res = {}
res["energy"] = H.Nenergies
# ===========================================================================
# LOCAL OBSERVABLE LIST
local_obs = [f"n_{s}{d}" for d in directions for s in "mp"]
local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
local_obs += ["X_Cross", "S2"]
for obs in local_obs:
    h_terms[obs] = LocalTerm(
        ops[obs],
        obs,
        lvals=lvals,
        has_obc=has_obc,
        gauge_basis=M,
        sector_configs=sector_configs,
    )
# TWO BODY OBSERVABLES
twobody_obs = [["P_px", "P_mx"], ["P_py", "P_my"]]
for obs1, obs2 in twobody_obs:
    op_list = [ops[obs1], ops[obs2]]
    h_terms[f"{obs1}_{obs2}"] = TwoBodyTerm(
        axis="x",
        op_list=op_list,
        op_names_list=[obs1, obs2],
        lvals=lvals,
        has_obc=has_obc,
        gauge_basis=M,
        sector_configs=sector_configs,
    )
# PLAQUETTE OBSERVABLES
plaq_name_list = ["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]
op_list = [ops[op] for op in plaq_name_list]
h_terms["Plaq_Sz"] = PlaquetteTerm(
    axes=["x", "y"],
    op_list=op_list,
    op_names_list=plaq_name_list,
    lvals=lvals,
    has_obc=has_obc,
    gauge_basis=M,
    sector_configs=sector_configs,
)
# ===========================================================================
for ii in range(n_eigs):
    logger.info("====================================================")
    logger.info(f"{ii} ENERGY: {round(res['energy'][ii], 4)}")
    # GET STATE CONFIGURATIONS
    H.Npsi[ii].get_state_configurations(threshold=1e-1, sector_indices=sector_ind)
    """
    # COMPUTE THE REDUCED DENSITY MATRIX
    if not has_obc:
        rho = H.Npsi[ii].reduced_density_matrix(0)
        eigvals, _ = diagonalize_density_matrix(rho)
        logger.info(eigvals)
    """
    # ===========================================================================
    # LOCAL OBSERVABLES:
    # ===========================================================================
    for obs in local_obs:
        h_terms[obs].get_expval(H.Npsi[ii])
    # CHECK LINK SYMMETRIES
    for ax in directions:
        check_link_symmetry(
            ax, h_terms[f"n_p{ax}"], h_terms[f"n_m{ax}"], value=0, sign=-1
        )
    # ===========================================================================
    # TWOBODY OBSERVABLES:
    # ===========================================================================
    for obs1, obs2 in twobody_obs:
        logger.info("----------------------------------------------------")
        logger.info(f"{obs1}_{obs2}")
        logger.info("----------------------------------------------------")
        h_terms[f"{obs1}_{obs2}"].get_expval(H.Npsi[ii])
        print(h_terms[f"{obs1}_{obs2}"].corr)
    # ===========================================================================
    # PLAQUETTE OBSERVABLES:
    # ===========================================================================
    h_terms["Plaq_Sz"].get_expval(H.Npsi[ii])

# %%
