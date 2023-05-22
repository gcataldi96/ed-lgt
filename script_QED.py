import numpy as np
from operators import (
    QED_dressed_site_operators,
    gauge_invariant_states,
    get_QED_Hamiltonian_couplings,
)
from modeling import Ground_State, LocalTerm2D, TwoBodyTerm2D, PlaquetteTerm2D
from modeling import (
    entanglement_entropy,
    get_reduced_density_matrix,
    diagonalize_density_matrix,
    staggered_mask,
    get_state_configurations,
    truncation,
    lattice_base_configs,
)
from tools import check_hermitian
from simsio import logger, run_sim

# ===================================================================================
with run_sim() as sim:
    # LATTICE DIMENSIONS
    lvals = sim.par["lvals"]
    dim = len(lvals)
    directions = "xyz"[:dim]
    # TOTAL NUMBER OF LATTICE SITES & particles
    n_sites = lvals[0] * lvals[1]
    # BOUNDARY CONDITIONS
    has_obc = sim.par["has_obc"]
    # GET g & m COUPLING
    g = sim.par["g"]
    m = sim.par["m"]
    # DEFINE THE GAUGE INVARIANT STATES OF THE BASIS
    spin = sim.par["spin"]
    # ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
    ops = QED_dressed_site_operators(spin, U=sim.par["U"])
    M, _ = gauge_invariant_states(spin)
    # ACQUIRE LOCAL DIMENSION OF EVERY SINGLE SITE
    lattice_base, loc_dims = lattice_base_configs(M, lvals, has_obc, staggered=True)
    loc_dims = loc_dims.transpose().reshape(n_sites)
    lattice_base = lattice_base.transpose().reshape(n_sites)
    logger.info(loc_dims)
    # ACQUIRE HAMILTONIAN COEFFICIENTS
    coeffs = get_QED_Hamiltonian_couplings(g, m)
    # CONSTRUCT THE HAMILTONIAN
    H = 0
    h_terms = {}
    # -------------------------------------------------------------------------------
    # LINK PENALTIES & Border penalties
    for d in directions:
        op_name_list = [f"E0_p{d}", f"E0_m{d}"]
        op_list = [ops[op] for op in op_name_list]
        # Define the Hamiltonian term
        h_terms[f"W_{d}"] = TwoBodyTerm2D(
            d, op_list, op_name_list, staggered_basis=True, site_basis=M
        )
        H += h_terms[f"W_{d}"].get_Hamiltonian(
            lvals,
            strength=2 * coeffs["eta"],
            has_obc=has_obc,
            add_dagger=False,
        )
        # SINGLE SITE OPERATORS needed for the LINK SYMMETRY/OBC PENALTIES
        for s in "mp":
            op_name = f"E0_square_{s}{d}"
            h_terms[op_name] = LocalTerm2D(
                ops[op_name], op_name, staggered_basis=True, site_basis=M
            )
            H += h_terms[op_name].get_Hamiltonian(
                lvals=lvals,
                has_obc=has_obc,
                strength=coeffs["eta"],
            )
    # -------------------------------------------------------------------------------
    # ELECTRIC ENERGY
    h_terms["E_square"] = LocalTerm2D(
        ops["E_square"], "E_square", staggered_basis=True, site_basis=M
    )
    H += h_terms["E_square"].get_Hamiltonian(lvals, has_obc, coeffs["E"])
    for site in ["even", "odd"]:
        # STAGGERED MASS TERM
        h_terms[f"N_{site}"] = LocalTerm2D(
            ops["N"], "N", staggered_basis=True, site_basis=M
        )
        H += h_terms[f"N_{site}"].get_Hamiltonian(
            lvals, has_obc, coeffs[f"m_{site}"], staggered_mask(lvals, site)
        )
    # -------------------------------------------------------------------------------
    # HOPPING
    for d in directions:
        for site in ["even", "odd"]:
            # Define the list of the 2 non trivial operators
            op_name_list = [f"Q_p{d}_dag", f"Q_m{d}"]
            op_list = [ops[op] for op in op_name_list]
            # Define the Hamiltonian term
            h_terms[f"{d}_hop_{site}"] = TwoBodyTerm2D(
                d, op_list, op_name_list, staggered_basis=True, site_basis=M
            )
            H += h_terms[f"{d}_hop_{site}"].get_Hamiltonian(
                lvals,
                strength=coeffs[f"t{d}_{site}"],
                has_obc=has_obc,
                add_dagger=True,
                mask=staggered_mask(lvals, site),
            )
    # -------------------------------------------------------------------------------
    # PLAQUETTE TERM: MAGNETIC INTERACTION
    op_name_list = [
        "C_px,py",
        "C_py,mx",
        "C_my,px",
        "C_mx,my",
    ]
    op_list = [ops[op] for op in op_name_list]
    h_terms["plaq"] = PlaquetteTerm2D(
        op_list, op_name_list, staggered_basis=True, site_basis=M
    )
    H += h_terms["plaq"].get_Hamiltonian(
        lvals,
        strength=coeffs["B"],
        has_obc=has_obc,
        add_dagger=True,
    )
    # ===========================================================================
    # CHECK THAT THE HAMILTONIAN IS HERMITIAN
    check_hermitian(H)
    # DIAGONALIZE THE HAMILTONIAN
    n_eigs = sim.par["n_eigs"]
    GS = Ground_State(H, n_eigs)
    sim.res["energy"] = GS.Nenergies
    # ===========================================================================
    # DEFINE THE OBSERVABLE LIST
    sim.res["entropy"] = []
    if not has_obc:
        sim.res["rho_eigvals"] = []
    obs_list = [f"n_{s}{d}" for s in "mp" for d in directions] + ["E_square", "N"]
    for obs in obs_list:
        h_terms[obs] = LocalTerm2D(ops[obs], obs, staggered_basis=True, site_basis=M)
        sim.res[obs] = []
    obs_list.append("plaq")
    sim.res["plaq"] = []
    # ===========================================================================
    for ii in range(n_eigs):
        logger.info("====================================================")
        logger.info(f"{ii} ENERGY: {format(sim.res['energy'][ii], '.9f')}")
        # ENTROPY of a BIPARTITION
        sim.res["entropy"].append(
            entanglement_entropy(
                GS.Npsi[:, ii], loc_dims, n_sites, partition_size=lvals[0]
            )
        )
        if has_obc:
            # GET STATE CONFIGURATIONS
            get_state_configurations(
                truncation(GS.Npsi[:, ii], 1e-10), loc_dims, n_sites
            )
        else:
            # COMPUTE THE REDUCED DENSITY MATRIX
            rho = get_reduced_density_matrix(GS.Npsi[:, ii], loc_dims, lvals, 0)
            eigvals, _ = diagonalize_density_matrix(rho)
            sim.res["rho_eigvals"].append(eigvals)
        # ===========================================================================
        # MEASURE OBSERVABLES:
        # RISHON NUMBER OPERATORS, ELECTRIC ENERGY E^{2}, DENSITY OPERATOR N, B^{2}
        # ===========================================================================
        for obs in obs_list:
            h_terms[obs].get_expval(GS.Npsi[:, ii], lvals, has_obc)
            sim.res[obs].append(h_terms[obs].avg)
    logger.info(f"Energies {sim.res['energy']}")
    if not has_obc:
        logger.info(f"DM eigvals {sim.res['rho_eigvals']}")
    if n_eigs > 1:
        sim.res["DeltaE"] = np.abs(sim.res["energy"][0] - sim.res["energy"][1])
