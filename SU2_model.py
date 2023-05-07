import numpy as np
from scipy.sparse import identity
from operators import get_su2_operators, get_SU2_Hamiltonian_couplings
from modeling import Ground_State, LocalTerm2D, TwoBodyTerm2D, PlaquetteTerm2D
from modeling import (
    entanglement_entropy,
    border_mask,
    staggered_mask,
    truncation,
    get_state_configurations,
    get_SU2_topological_invariant,
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
    ops = get_su2_operators(pure_theory)
    # ACQUIRE HAMILTONIAN COEFFICIENTS
    coeffs = get_SU2_Hamiltonian_couplings(pure_theory, g, m)
    logger.info(f"PENALTY {coeffs['eta']}")
    # CONSTRUCT THE HAMILTONIAN
    H = 0
    h_terms = {}
    # LINK PENALTIES
    for i, d in enumerate(directions):
        op_name_list = [f"W_{s}{d}" for s in "pm"]
        op_list = [ops[op] for op in op_name_list]
        # Define the Hamiltonian term
        h_terms[f"W_{d}"] = TwoBodyTerm2D(
            d, op_list, op_name_list, staggered_basis=False, site_basis=None
        )
        H += h_terms[f"W_{d}"].get_Hamiltonian(
            lvals,
            has_obc=has_obc,
            strength=coeffs["eta"],
            add_dagger=False,
        )
    # BORDER PENALTIES
    if has_obc:
        for d in directions:
            for s in "mp":
                op_name = f"P_{s}{d}"
                h_terms[op_name] = LocalTerm2D(
                    ops[op_name], op_name, staggered_basis=False, site_basis=None
                )
                H += h_terms[op_name].get_Hamiltonian(
                    lvals,
                    has_obc=has_obc,
                    strength=coeffs["eta"],
                    mask=border_mask(lvals, f"{s}{d}"),
                )
    # ELECTRIC ENERGY
    h_terms["gamma"] = LocalTerm2D(ops["gamma"], "gamma")
    H += h_terms["gamma"].get_Hamiltonian(lvals, has_obc=has_obc, strength=coeffs["E"])
    # MAGNETIC ENERGY
    op_name_list = ["C_py_px", "C_py_mx", "C_my_px", "C_my_mx"]
    op_list = [ops[op] for op in op_name_list]
    h_terms["plaq"] = PlaquetteTerm2D(op_list, op_name_list)
    H += h_terms["plaq"].get_Hamiltonian(
        lvals, strength=coeffs["B"], has_obc=has_obc, add_dagger=True
    )
    # -----------------------------------------------------------------------------
    if not pure_theory:
        # STAGGERED MASS TERM
        h_terms["mass_op"] = LocalTerm2D(ops["mass_op"], "mass_op")
        for site in ["even", "odd"]:
            H += h_terms["mass_op"].get_Hamiltonian(
                lvals,
                has_obc,
                strength=coeffs[f"m_{site}"],
                mask=staggered_mask(lvals, site),
            )
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
        if DeltaN != 0:
            # SELECT THE SYMMETRY SECTOR with N PARTICLES
            tot_hilb_space = H.shape[0]
            h_terms["fix_N"] = LocalTerm2D(ops["n_tot"], "n_tot")
            H += (
                -coeffs["eta"]
                * (
                    h_terms["fix_N"].get_Hamiltonian(lvals, has_obc, strength=1)
                    - (DeltaN + n_sites) * identity(tot_hilb_space)
                )
                ** 2
            )
    # CHECK THAT THE HAMILTONIAN IS HERMITIAN
    check_hermitian(H)
    # DIAGONALIZE THE HAMILTONIAN
    n_eigs = sim.par["n_eigs"]
    GS = Ground_State(H, n_eigs)
    # ===========================================================================
    # ENERGIES, STATE CONFIGURATIONS, AND TOPOLOGICAL SECTORS
    # ===========================================================================
    list_obs = ["energy", "entropy", "px_sector", "py_sector", "gamma", "plaq"]
    if not pure_theory:
        for obs in ["n_single", "n_pair", "n_tot"]:
            for site in ["even", "odd"]:
                list_obs.append(f"{obs}_{site}")
    for obs in list_obs:
        sim.res[obs] = []
    for ii in range(n_eigs):
        # GET AND RESCALE SINGLE SITE ENERGIES
        sim.res["energy"].append(
            get_energy_density(
                GS.Nenergies[ii],
                lvals,
                penalty=coeffs["eta"],
                border_penalty=True,
                link_penalty=True,
                plaquette_penalty=False,
                has_obc=has_obc,
            )
        )
        logger.info("====================================================")
        logger.info(f"{ii} ENERGY: {format(sim.res['energy'][ii], '.9f')}")
        # GET STATE CONFIGURATIONS
        get_state_configurations(truncation(GS.Npsi[:, ii], 1e-10), loc_dim, n_sites)
        # ENTROPY of a BIPARTITION
        sim.res["entropy"].append(
            entanglement_entropy(
                GS.Npsi[:, ii], loc_dim, n_sites, partition_size=int(n_sites / 2)
            )
        )
        # ===========================================================================
        # GROUND STATE SINGLE SITE OBSERVABLES
        # ===========================================================================
        # CHECK LINK PENALTIES
        for d in directions:
            h_terms[f"W_{d}"].get_expval(GS.Npsi[:, ii], lvals, has_obc)
            h_terms[f"W_{d}"].check_link_symm(value=1, has_obc=has_obc)
        # CHECK BORDER PENALTIES
        if has_obc:
            for d in directions:
                for s in "mp":
                    op_name = f"P_{s}{d}"
                    h_terms[op_name].get_expval(GS.Npsi[:, ii], lvals, has_obc)
                    h_terms[op_name].check_on_borders(border=f"{s}{d}", value=1)
        # COMPUTE GAUGE OBSERVABLES
        sim.res["gamma"].append(
            h_terms["gamma"].get_expval(GS.Npsi[:, ii], lvals, has_obc)
        )
        # sim.res["delta_gamma"] = h_terms["gamma"].get_fluctuations(GS.psi, lvals)
        sim.res["plaq"].append(
            h_terms["plaq"].get_expval(GS.Npsi[:, ii], lvals, has_obc)
        )
        if not pure_theory:
            # COMPUTE MATTER OBSERVABLES STAGGERED
            local_obs = ["n_single", "n_pair", "n_tot"]
            for obs in local_obs:
                # Generate the Operator
                h_terms[obs] = LocalTerm2D(ops[obs], obs)
                # Run over even and odd sites
                for site in ["odd", "even"]:
                    sim.res[f"{obs}_{site}"].append(
                        h_terms[obs].get_expval(GS.Npsi[:, ii], lvals, has_obc, site)
                    )
                    """                    
                    sim.res[f"delta_{obs}_{site}"] = h_terms[obs].get_fluctuations(
                        GS.psi, lvals, site
                    )
                    """
        # MEASURE TOPOLOGICAL SECTORS
        if not has_obc:
            for d1, d2 in [("y", "x"), ("x", "y")]:
                # select the link parity operator
                op = ops[f"p{d1}_link_P"]
                # measure the topological sector
                sim.res[f"p{d2}_sector"].append(
                    get_SU2_topological_invariant(op, lvals, GS.Npsi[:, ii], d2)
                )
