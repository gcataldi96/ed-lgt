import numpy as np
from scipy.sparse import identity as ID
from operators import (
    SU2_Hamiltonian_couplings,
    SU2_dressed_site_operators,
    SU2_gauge_basis,
)
from modeling import Ground_State, LocalTerm2D, TwoBodyTerm2D, PlaquetteTerm2D
from modeling import (
    lattice_base_configs,
    staggered_mask,
    entanglement_entropy,
    get_reduced_density_matrix,
    diagonalize_density_matrix,
    get_state_configurations,
    truncation,
)
from tools import get_energy_density, check_hermitian
from simsio import logger, run_sim

# ===================================================================================
with run_sim() as sim:
    # LATTICE DIMENSIONS
    lvals = sim.par["lvals"]
    dim = len(lvals)
    directions = "xyz"[:dim]
    # TOTAL NUMBER OF LATTICE SITES
    n_sites = lvals[0] * lvals[1]
    # BOUNDARY CONDITIONS
    has_obc = sim.par["has_obc"]
    # Get the spin s representation
    spin = sim.par["spin"]
    # PURE or FULL THEORY
    pure_theory = sim.par["pure"]
    # GET g COUPLING
    g = sim.par["g"]
    if pure_theory:
        loc_dim = 9
        m = None
    else:
        loc_dim = 30
        DeltaN = sim.par["DeltaN"]
        m = sim.par["m"]
    # ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
    ops = SU2_dressed_site_operators(spin, pure_theory)
    # ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
    M, _ = SU2_gauge_basis(spin, pure_theory)
    # ACQUIRE LOCAL DIMENSION OF EVERY SINGLE SITE
    lattice_base, loc_dims = lattice_base_configs(M, lvals, has_obc)
    loc_dims = loc_dims.transpose().reshape(n_sites)
    lattice_base = lattice_base.transpose().reshape(n_sites)
    logger.info(loc_dims)
    # ACQUIRE HAMILTONIAN COEFFICIENTS
    coeffs = SU2_Hamiltonian_couplings(pure_theory, g, m)
    logger.info(f"PENALTY {coeffs['eta']}")
    # CONSTRUCT THE HAMILTONIAN
    H = 0
    h_terms = {}
    # ELECTRIC ENERGY
    h_terms["E_square"] = LocalTerm2D(ops["E_square"], "E_square", site_basis=M)
    H += h_terms["E_square"].get_Hamiltonian(lvals, has_obc, coeffs["E"])
    # -------------------------------------------------------------------------------
    # LINK PENALTIES & Border penalties
    for d in directions:
        op_name_list = [f"T2_p{d}", f"T2_m{d}"]
        op_list = [ops[op] for op in op_name_list]
        # Define the Hamiltonian term
        h_terms[f"W_{d}"] = TwoBodyTerm2D(d, op_list, op_name_list, site_basis=M)
        H += h_terms[f"W_{d}"].get_Hamiltonian(
            lvals, strength=2 * coeffs["eta"], has_obc=has_obc
        )
        # SINGLE SITE OPERATORS needed for the LINK SYMMETRY/OBC PENALTIES
        for s in "mp":
            op_name = f"T4_{s}{d}"
            h_terms[op_name] = LocalTerm2D(ops[op_name], op_name, site_basis=M)
            H += h_terms[op_name].get_Hamiltonian(
                lvals=lvals, has_obc=has_obc, strength=coeffs["eta"]
            )
    if not pure_theory:
        # STAGGERED MASS TERM
        h_terms["mass"] = LocalTerm2D(ops["N_tot"], "N_tot", site_basis=M)
        for site in ["even", "odd"]:
            H += h_terms["mass"].get_Hamiltonian(
                lvals,
                has_obc,
                strength=coeffs[f"m_{site}"],
                mask=staggered_mask(lvals, site),
            )
        if DeltaN != 0:
            # SELECT THE SYMMETRY SECTOR with N PARTICLES
            tot_hilb_space = H.shape[0]
            h_terms["fix_N"] = LocalTerm2D(ops["N_tot"], "N_tot", site_basis=M)
            H += (
                -coeffs["eta"]
                * (
                    h_terms["fix_N"].get_Hamiltonian(lvals, has_obc, strength=1)
                    - (DeltaN + n_sites) * ID(tot_hilb_space)
                )
                ** 2
            )
    """
    # MAGNETIC ENERGY
    op_name_list = ["C_py_px", "C_py_mx", "C_my_px", "C_my_mx"]
    op_list = [ops[op] for op in op_name_list]
    h_terms["plaq"] = PlaquetteTerm2D(op_list, op_name_list)
    H += h_terms["plaq"].get_Hamiltonian(
        lvals, strength=coeffs["B"], has_obc=has_obc, add_dagger=True
    )
    # -----------------------------------------------------------------------------
    if not pure_theory:
        # HOPPING ACTIVITY
        for d in directions:
            for site in ["even", "odd"]:
                op_name_list = [f"Q_p{d}_dag", f"Q_m{d}"]
                op_list = [ops[op] for op in op_name_list]
                h_terms[f"{d}_hop_{site}"] = TwoBodyTerm2D(d, op_list, op_name_list)
                H += h_terms[f"{d}_hop_{site}"].get_Hamiltonian(
                    lvals,
                    strength=coeffs[f"t{d}_{site}"],
                    has_obc=has_obc,
                    add_dagger=True,
                    mask=staggered_mask(lvals, site),
                )
    """
    # CHECK THAT THE HAMILTONIAN IS HERMITIAN
    check_hermitian(H)
    # DIAGONALIZE THE HAMILTONIAN
    n_eigs = sim.par["n_eigs"]
    GS = Ground_State(H, n_eigs)
    sim.res["energy"] = GS.Nenergies
    sim.res["energy_density"] = []
    # ===========================================================================
    # DEFINE THE OBSERVABLE LIST
    sim.res["entropy"] = []
    if not has_obc:
        sim.res["rho_eigvals"] = []
    # ===========================================================================
    # LIST OF OBSERVABLES
    obs_list = [f"T2_{s}{d}" for s in "mp" for d in directions] + ["E_square"]
    for obs in obs_list:
        h_terms[obs] = LocalTerm2D(ops[obs], obs, site_basis=M)
        sim.res[obs] = []
    if not pure_theory:
        matter_obs = ["N_up", "N_down", "N_tot", "N_single", "N_pair"]
        for obs in matter_obs:
            h_terms[obs] = LocalTerm2D(ops[obs], obs, site_basis=M)
            sim.res[obs] = []
            obs_list.append(obs)
    # ===========================================================================
    # COMPUTE THE OBSERVABLES FOR EACH EIGENSTATES
    for ii in range(n_eigs):
        logger.info("====================================================")
        logger.info(f"{ii} ENERGY: {format(sim.res['energy'][ii], '.9f')}")
        # GET AND RESCALE SINGLE SITE ENERGIES
        sim.res["energy_density"].append(
            get_energy_density(
                GS.Nenergies[ii],
                lvals,
                penalty=coeffs["eta"],
                border_penalty=False,
                link_penalty=True,
                plaquette_penalty=False,
                has_obc=has_obc,
            )
        )
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
            # PRINT THE EIGENVALUES
            for eig in eigvals[::-1]:
                logger.info(eig)
        # ===========================================================================
        # MEASURE GAUGE OBSERVABLES: RISHON CASIMIR OPERATORS, ELECTRIC ENERGY E^{2}
        # ===========================================================================
        for obs in obs_list:
            h_terms[obs].get_expval(GS.Npsi[:, ii], lvals, has_obc)
            sim.res[obs].append(h_terms[obs].avg)
        # ===========================================================================
        # COMPUTE MATTER OBSERVABLES (STAGGERED)
        # ===========================================================================
        if not pure_theory:
            for obs in matter_obs:
                h_terms[obs].get_expval(GS.Npsi[:, ii], lvals, has_obc)
        logger.info("====================================================")

    if n_eigs > 1:
        sim.res["DeltaE"] = np.abs(sim.res["energy"][0] - sim.res["energy"][1])
