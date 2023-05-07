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
    logger.info(f"Penalty {coeffs['eta']}")
    # CONSTRUCT THE HAMILTONIAN
    ham_terms = {}
    H = 0
    # BORDER PENALTIES
    if has_obc:
        for d in directions:
            for s in "mp":
                ham_terms[f"P_{s}{d}"] = LocalTerm2D(ops[f"P_{s}{d}"], f"P_{s}{d}")
                H += ham_terms[f"P_{s}{d}"].get_Hamiltonian(
                    lvals, strength=coeffs["eta"], mask=border_mask(lvals, f"{s}{d}")
                )
    # LINK PENALTIES
    axes = ["x", "y"]
    for i, d in enumerate(directions):
        op_list = [ops[f"W_{s}{d}"] for s in "pm"]
        op_name_list = [f"W_{s}{d}" for s in "pm"]
        ham_terms[f"W_{axes[i]}_link"] = TwoBodyTerm2D(axes[i], op_list, op_name_list)
        H += ham_terms[f"W_{axes[i]}_link"].get_Hamiltonian(
            lvals, strength=coeffs["eta"], has_obc=has_obc, add_dagger=False
        )
    # ELECTRIC ENERGY
    ham_terms["gamma"] = LocalTerm2D(ops["gamma"], "gamma")
    H += ham_terms["gamma"].get_Hamiltonian(lvals, strength=coeffs["E"])
    # MAGNETIC ENERGY
    op_name_list = ["C_py_px", "C_py_mx", "C_my_px", "C_my_mx"]
    op_list = [ops[op] for op in op_name_list]
    ham_terms["plaq"] = PlaquetteTerm2D(op_list, op_name_list)
    H += ham_terms["plaq"].get_Hamiltonian(
        lvals, strength=coeffs["B"], has_obc=has_obc, add_dagger=True
    )
    if not pure_theory:
        # STAGGERED MASS TERM
        ham_terms["mass_op"] = LocalTerm2D(ops["mass_op"], "mass_op")
        for site in ["even", "odd"]:
            H += ham_terms["mass_op"].get_Hamiltonian(
                lvals, strength=coeffs[f"m_{site}"], mask=staggered_mask(lvals, site)
            )
        # HOPPING ACTIVITY along x AXIS
        op_name_list = ["Q_px_dag", "Q_mx"]
        op_list = [ops[op] for op in op_name_list]
        ham_terms["x_hopping"] = TwoBodyTerm2D("x", op_list, op_name_list)
        H += ham_terms["x_hopping"].get_Hamiltonian(
            lvals, strength=coeffs["tx"], has_obc=has_obc, add_dagger=True
        )
        # HOPPING ACTIVITY along y AXIS
        op_name_list = ["Q_py_dag", "Q_my"]
        op_list = [ops[op] for op in op_name_list]
        ham_terms["y_hopping"] = TwoBodyTerm2D("y", op_list, op_name_list)
        for site in ["even", "odd"]:
            H += ham_terms["y_hopping"].get_Hamiltonian(
                lvals,
                strength=coeffs[f"ty_{site}"],
                has_obc=has_obc,
                add_dagger=True,
                mask=staggered_mask(lvals, site),
            )
        if DeltaN != 0:
            # SELECT THE SYMMETRY SECTOR with N PARTICLES
            tot_hilb_space = loc_dim ** (lvals[0] * lvals[1])
            ham_terms["fix_N"] = LocalTerm2D(ops["n_tot"], "n_tot")
            H += (
                -coeffs["eta"]
                * (
                    ham_terms["fix_N"].get_Hamiltonian(lvals, strength=1)
                    - (DeltaN + n_sites) * identity(tot_hilb_space)
                )
                ** 2
            )
    # CHECK THAT THE HAMILTONIAN IS HERMITIAN
    check_hermitian(H)
    # DIAGONALIZE THE HAMILTONIAN
    n_eigs = sim.par["n_eigs"]
    GS = Ground_State(H, n_eigs)
    GS.normalize()
    # ===========================================================================
    # ENERGIES, STATE CONFIGURATIONS, AND TOPOLOGICAL SECTORS
    # ===========================================================================
    sim.res["px_sector"] = np.zeros(n_eigs)
    sim.res["py_sector"] = np.zeros(n_eigs)
    sim.res["energy"] = np.zeros(n_eigs)
    for ii in range(n_eigs):
        # GET AND RESCALE SINGLE SITE ENERGIES
        sim.res["energy"][ii] = get_energy_density(
            GS.Nenergies[ii],
            lvals,
            penalty=coeffs["eta"],
            border_penalty=True,
            link_penalty=True,
            plaquette_penalty=False,
            has_obc=has_obc,
        )
        logger.info(f" {ii} ENERGY VALUE: {sim.res['energy'][ii]}")
        # GET STATE CONFIGURATIONS
        get_state_configurations(truncation(GS.Npsi[:, ii], 1e-10), loc_dim, n_sites)
        # MEASURE TOPOLOGICAL SECTORS
        for jj, ax in enumerate(axes):
            # select the link parity operator
            op = ops[f"p{axes[::-1][jj]}_link_P"]
            # measure the topological sector
            sim.res[f"p{ax}_sector"][ii] = get_SU2_topological_invariant(
                op, lvals, GS.Npsi[:, ii], ax
            )
    if n_eigs == 1:
        sim.res["energy"] = sim.res["energy"][0]
        sim.res["px_sector"] = sim.res["px_sector"][0]
        sim.res["py_sector"] = sim.res["py_sector"][0]
    # ===========================================================================
    # GROUND STATE SINGLE SITE OBSERVABLES
    # ===========================================================================
    # CHECK BORDER PENALTIES
    if has_obc:
        for d in directions:
            for s in "mp":
                ham_terms[f"P_{s}{d}"].get_loc_expval(GS.psi, lvals)
                ham_terms[f"P_{s}{d}"].check_on_borders(border=f"{s}{d}", value=1)
    # CHECK LINK PENALTIES
    axes = ["x", "y"]
    for i, d in enumerate(directions):
        op_list = [ops[f"W_{s}{d}"] for s in "pm"]
        op_name_list = [f"W_{s}{d}" for s in "pm"]
        ham_terms[f"W_{axes[i]}_link"].get_expval(GS.psi, lvals, has_obc=has_obc)
        ham_terms[f"W_{axes[i]}_link"].check_link_symm(value=1, has_obc=has_obc)

    # COMPUTE GAUGE OBSERVABLES
    sim.res["gamma"] = ham_terms["gamma"].get_loc_expval(GS.psi, lvals)
    sim.res["delta_gamma"] = ham_terms["gamma"].get_fluctuations(GS.psi, lvals)
    sim.res["plaq"], sim.res["delta_plaq"] = ham_terms["plaq"].get_plaq_expval(
        GS.psi, lvals, has_obc=has_obc, get_imag=False
    )
    if not pure_theory:
        # COMPUTE MATTER OBSERVABLES STAGGERED
        local_obs = ["n_single", "n_pair", "n_tot"]
        for obs in local_obs:
            # Generate the Operator
            ham_terms[obs] = LocalTerm2D(ops[obs], obs)
            # Run over even and odd sites
            for site in ["odd", "even"]:
                sim.res[f"{obs}_{site}"] = ham_terms[obs].get_loc_expval(
                    GS.psi, lvals, site
                )
                sim.res[f"delta_{obs}_{site}"] = ham_terms[obs].get_fluctuations(
                    GS.psi, lvals, site
                )
    # COMPUTE ENTROPY of a BIPARTITION
    sim.res["entropy"] = entanglement_entropy(
        psi=GS.psi, loc_dim=loc_dim, n_sites=n_sites, partition_size=int(n_sites / 2)
    )
    # SUMMARIZE OBSERVABLES
    logger.info("----------------------------------------------------")
    logger.info(f" ENTROPY:  {sim.res['entropy']}")
    logger.info(f" ELECTRIC: {sim.res['gamma']} +- {sim.res['delta_gamma']}")
    logger.info(f" MAGNETIC: {sim.res['plaq']} +- {sim.res['delta_plaq']}")
    if not pure_theory:
        for obs in local_obs:
            logger.info(
                f" {obs}_EVEN: {sim.res[f'{obs}_even']} +- {sim.res[f'delta_{obs}_even']}"
            )
            logger.info(
                f" {obs}_ODD: {sim.res[f'{obs}_odd']} +- {sim.res[f'delta_{obs}_odd']}"
            )
