# %%
import numpy as np
from math import prod
import logging
from ed_lgt.operators import (
    SU2_Hamiltonian_couplings,
    SU2_dressed_site_operators,
    SU2_gauge_invariant_states,
    SU2_check_gauss_law,
    SU2_generators,
    SU2_rishon_operators,
)
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm, QMB_hamiltonian
from ed_lgt.modeling import (
    check_link_symmetry,
    diagonalize_density_matrix,
    staggered_mask,
    lattice_base_configs,
)
from ed_lgt.symmetries import (
    get_symmetry_sector_generators,
    global_abelian_sector,
    link_abelian_sector,
)
from numba import typed
from ed_lgt.modeling import zig_zag, get_neighbor_sites

logger = logging.getLogger(__name__)
# SYMMETRIES
symmetries = True
# N eigenvalues
n_eigs = 1
# LATTICE DIMENSIONS
lvals = [4]
dim = len(lvals)
directions = "xyz"[:dim]
n_sites = prod(lvals)
# BOUNDARY CONDITIONS
has_obc = [False]
# DEFINE the maximal truncation of the gauge link
j_max = 1 / 2
# PURE or FULL THEORY
pure_theory = False
# GET g COUPLING
g = 1
if pure_theory:
    m = None
else:
    m = 10
# ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
ops = SU2_dressed_site_operators(j_max, pure_theory, lattice_dim=dim)
# ACQUIRE SU2 BASIS and GAUGE INVARIANT STATES
gauge_basis, gauge_states = SU2_gauge_invariant_states(
    j_max, pure_theory, lattice_dim=dim
)
SU2_check_gauss_law(gauge_basis["site"])
def_params = {
    "lvals": lvals,
    "has_obc": has_obc,
    "gauge_basis": gauge_basis,
    "staggered_basis": False,
}
# ACQUIRE LOCAL DIMENSION OF EVERY SINGLE SITE
lattice_labels, loc_dims = lattice_base_configs(**def_params)
loc_dims = loc_dims.transpose().reshape(n_sites)
lattice_labels = lattice_labels.transpose().reshape(n_sites)
tot_dim = prod(loc_dims)
logger.info(f"local dimensions: {loc_dims}")
# SU2_check_gauss_law(basis=M["site"], gauss_law_op=ops["S2_tot"])
TTN_ops = {}
for op in ops.keys():
    TTN_ops[op] = gauge_basis["site"].transpose() * ops[op] * gauge_basis["site"]
# ACQUIRE HAMILTONIAN COEFFICIENTS
coeffs = SU2_Hamiltonian_couplings(dim, pure_theory, g, m)
coeffs["eta"] = 400
if symmetries:
    # GET operators for global symmetry sector
    op_list = [ops["N_tot"]]
    # op_diagonals = np.array([np.diag(op) for op in op_list], dtype=np.int64)
    op_diagonals = get_symmetry_sector_generators(
        op_list,
        loc_dims,
        action="global",
        gauge_basis=gauge_basis,
        lattice_labels=lattice_labels,
    )
    op_sectors_list = np.array([n_sites], dtype=np.int64)
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
        [ops["T2_px"], -ops["T2_mx"]],
    ]
    link_ops = get_symmetry_sector_generators(
        link_ops,
        loc_dims,
        action="link",
        gauge_basis=gauge_basis,
        lattice_labels=lattice_labels,
    )
    link_sectors = np.array([0], dtype=np.int64)
    # ===========================================================
    # Compute the global symmetry sector
    sector_indices, sector_config = global_abelian_sector(
        loc_dims, sym_op_diags=op_diagonals, sym_sectors=op_sectors_list, sym_type="U"
    )
    sector_indices, sector_configs = link_abelian_sector(
        loc_dims,
        sym_op_diags=link_ops,
        sym_sectors=link_sectors,
        pair_list=pair_list,
        configs=sector_config,
    )
else:
    sector_indices = None
    sector_configs = None

# ===============================================================================
# CONSTRUCT THE HAMILTONIAN
H = QMB_hamiltonian(0, lvals, loc_dims)
h_terms = {}
if not symmetries:
    # -------------------------------------------------------------------------------
    # LINK PENALTIES & Border penalties
    for d in directions:
        op_names_list = [f"T2_p{d}", f"T2_m{d}"]
        op_list = [ops[op] for op in op_names_list]
        # Define the Hamiltonian term
        h_terms[f"W_{d}"] = TwoBodyTerm(
            axis=d,
            op_list=op_list,
            op_names_list=op_names_list,
            lvals=lvals,
            has_obc=has_obc,
            gauge_basis=gauge_basis,
        )
        H.Ham += h_terms[f"W_{d}"].get_Hamiltonian(strength=-2 * coeffs["eta"])
        # SINGLE SITE OPERATORS needed for the LINK SYMMETRY/OBC PENALTIES
        for s in "mp":
            op_name = f"T4_{s}{d}"
            h_terms[op_name] = LocalTerm(
                ops[op_name],
                op_name,
                lvals=lvals,
                has_obc=has_obc,
                gauge_basis=gauge_basis,
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
    gauge_basis=gauge_basis,
    sector_configs=sector_configs,
)
H.Ham += h_terms[op_name].get_Hamiltonian(strength=coeffs["E"])

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
            gauge_basis=gauge_basis,
            sector_configs=sector_configs,
        )
        H.Ham += h_terms[f"N_{site}"].get_Hamiltonian(
            coeffs[f"m_{site}"], staggered_mask(lvals, site)
        )
    # ------------------------------------------------------------------------
    # HOPPING
    for d in directions:
        for site in ["even", "odd"]:
            hopping_terms = [[f"Qp{d}_dag", f"Qm{d}"]]
            for op_names_list in hopping_terms:
                op_list = [ops[op] for op in op_names_list]
                # Define the Hamiltonian term
                h_terms[f"{d}_hop_{site}"] = TwoBodyTerm(
                    d,
                    op_list,
                    op_names_list,
                    lvals=lvals,
                    has_obc=has_obc,
                    gauge_basis=gauge_basis,
                    sector_configs=sector_configs,
                )
                H.Ham += h_terms[f"{d}_hop_{site}"].get_Hamiltonian(
                    strength=coeffs[f"tx_{site}"],
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
            op_names_list = [
                f"C{p_set[0]}_p{pdir[0]},p{pdir[1]}",
                f"C{p_set[1]}_p{pdir[1]},m{pdir[0]}",
                f"C{p_set[2]}_m{pdir[1]},p{pdir[0]}",
                f"C{p_set[3]}_m{pdir[0]},m{pdir[1]}",
            ]
            # CORRESPONDING LIST OF OPERATORS
            op_list = [ops[op] for op in op_names_list]
            # DEFINE THE PLAQUETTE CLASS
            plaq_name = f"P_{pdir}_" + "".join(p_set)
            h_terms[plaq_name] = PlaquetteTerm(
                [pdir[0], pdir[1]],
                op_list,
                op_names_list,
                lvals=lvals,
                has_obc=has_obc,
                gauge_basis=gauge_basis,
                print_plaq=False,
            )
            # ADD THE HAMILTONIAN TERM
            H.Ham += h_terms[plaq_name].get_Hamiltonian(
                strength=plaquette_signs[jj] * coeffs["B"], add_dagger=True
            )
            # ADD THE PLAQUETTE TO THE LIST OF OBSERVABLES
            plaq_list.append(plaq_name)
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
local_obs = [f"T2_{s}{d}" for s in "mp" for d in directions] + ["E_square"]
if not pure_theory:
    for label in ["r", "g", "tot", "single", "pair"]:
        local_obs.append(f"N_{label}")
for op_name in local_obs:
    h_terms[op_name] = LocalTerm(
        ops[op_name],
        op_name,
        lvals=lvals,
        has_obc=has_obc,
        gauge_basis=gauge_basis,
        sector_configs=sector_configs,
    )
    res[op_name] = []
# ===========================================================================
for ii in range(n_eigs):
    logger.info("====================================================")
    logger.info(f"{ii} ENERGY: {format(res['energy'][ii], '.9f')}")
    # if dim < 3:
    # ENTROPY of a BIPARTITION
    # res["entropy"].append(H.Npsi[ii].entanglement_entropy(int(prod(lvals) / 2)))
    # GET STATE CONFIGURATIONS
    H.Npsi[ii].get_state_configurations(threshold=1e-5, sector_configs=sector_configs)
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
        check_link_symmetry(ax, h_terms[f"T2_p{ax}"], h_terms[f"T2_m{ax}"], sign=-1)
    # MEASURE PLAQUETTE OBSERVABLES
    plaq_obs = 0
    for plaq_name in plaq_list:
        h_terms[plaq_name].get_expval(H.Npsi[ii])
        plaq_obs += h_terms[plaq_name].avg
    logger.info("AVERAGE PLAQUETTE VALUE")
    logger.info(plaq_obs)
# %%
