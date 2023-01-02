import numpy as np
from scipy.sparse import identity
from operators import get_su2_operators, get_Hamiltonian_couplings
from modeling import Pure_State, LocalTerm2D, TwoBodyTerm2D, PlaquetteTerm2D
from modeling import (
    entanglement_entropy,
    border_mask,
    staggered_mask,
    truncation,
    normalize,
    get_state_configurations,
    two_body_op,
    get_SU2_topological_invariant,
)
from tools import get_energy_density, check_hermitian
from simsio import logger, run_sim

# ===================================================================================

with run_sim() as sim:
    sim.link("psi")
    # LATTICE DIMENSIONS
    GS_only = sim.par["GS_only"]
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
    coeffs = get_Hamiltonian_couplings(pure_theory, g, m)
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
    psi = Pure_State()
    if GS_only:
        psi.ground_state(H)
        # CHECK THE STATE TO BE NORMALIZED:
        psi.GSpsi = normalize(psi.GSpsi)
        # RESCALE ENERGY
        sim.res["energy"] = get_energy_density(
            psi.GSenergy,
            lvals,
            penalty=coeffs["eta"],
            border_penalty=True,
            link_penalty=True,
            plaquette_penalty=False,
            has_obc=has_obc,
        )
        # GET STATE CONFIGURATIONS
        get_state_configurations(truncation(psi.GSpsi, 1e-10), loc_dim, n_sites)
        # TOPOLOGICAL SECTOR
        for ii, ax in enumerate(axes):
            # select the link parity operator
            op = ops[f"p{axes[::-1][ii]}_link_P"]
            # measure the topological sector
            sim.res[f"p{ax}_sector"] = get_SU2_topological_invariant(
                op, lvals, psi.GSpsi, ax
            )
    else:
        psi.get_first_n_eigs(H, n_eigs=8)
        sim.res["px_sector"] = np.zeros(psi.Npsi.shape[1])
        sim.res["py_sector"] = np.zeros(psi.Npsi.shape[1])
        sim.res["energies"] = np.zeros(psi.Npsi.shape[1])
        for ii in range(psi.Npsi.shape[1]):
            logger.info(f" {ii} ENERGY VALUE")
            sim.res["energies"][ii] = get_energy_density(
                psi.Nenergies[ii],
                lvals,
                penalty=coeffs["eta"],
                border_penalty=True,
                link_penalty=True,
                plaquette_penalty=False,
                has_obc=has_obc,
            )
            logger.info(f" ENERGY: {sim.res['energies'][ii]}")
            # GET STATE CONFIGURATIONS
            get_state_configurations(
                truncation(psi.Npsi[:, ii], 1e-10), loc_dim, n_sites
            )
            # TOPOLOGICAL SECTOR
            for jj, ax in enumerate(axes):
                # select the link parity operator
                op = ops[f"p{axes[::-1][jj]}_link_P"]
                # measure the topological sector
                sim.res[f"p{ax}_sector"][ii] = get_SU2_topological_invariant(
                    op, lvals, psi.Npsi[:, ii], ax
                )
    # ===========================================================================
    # OBSERVABLES for the GROUND STATE
    # ===========================================================================
    # CHECK BORDER PENALTIES
    if has_obc:
        for d in directions:
            for s in "mp":
                ham_terms[f"P_{s}{d}"].get_loc_expval(psi.GSpsi, lvals)
                ham_terms[f"P_{s}{d}"].check_on_borders(border=f"{s}{d}", value=1)
    # CHECK LINK PENALTIES
    axes = ["x", "y"]
    for i, d in enumerate(directions):
        op_list = [ops[f"W_{s}{d}"] for s in "pm"]
        op_name_list = [f"W_{s}{d}" for s in "pm"]
        ham_terms[f"W_{axes[i]}_link"].get_expval(psi.GSpsi, lvals, has_obc=has_obc)
        ham_terms[f"W_{axes[i]}_link"].check_link_symm(value=1, has_obc=has_obc)

    # COMPUTE GAUGE OBSERVABLES
    sim.res["gamma"] = ham_terms["gamma"].get_loc_expval(psi.GSpsi, lvals)
    sim.res["delta_gamma"] = ham_terms["gamma"].get_fluctuations(psi.GSpsi, lvals)
    sim.res["plaq"], sim.res["delta_plaq"] = ham_terms["plaq"].get_plaq_expval(
        psi.GSpsi, lvals, has_obc=has_obc, get_imag=False
    )
    if not pure_theory:
        # COMPUTE MATTER OBSERVABLES
        local_obs = ["n_single", "n_pair", "n_tot"]
        for obs in local_obs:
            ham_terms[obs] = LocalTerm2D(ops[obs], obs)
            sim.res[f"{obs}_even"], sim.res[f"{obs}_odd"] = ham_terms[
                obs
            ].get_loc_expval(psi.GSpsi, lvals, staggered=True)
            sim.res[f"delta_{obs}_even"], sim.res[f"delta_{obs}_odd"] = ham_terms[
                obs
            ].get_fluctuations(psi.GSpsi, lvals, staggered=True)
    # COMPUTE ENTROPY of a BIPARTITION
    sim.res["entropy"] = entanglement_entropy(
        psi=psi.GSpsi, loc_dim=loc_dim, n_sites=n_sites, partition_size=int(n_sites / 2)
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
