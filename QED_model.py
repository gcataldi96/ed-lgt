from operators import (
    get_QED_operators,
    get_QED_Hamiltonian_couplings,
)
from modeling import Ground_State, LocalTerm2D, TwoBodyTerm2D, PlaquetteTerm2D
from modeling import (
    entanglement_entropy,
    staggered_mask,
    get_state_configurations,
    truncation,
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
    n_rishons = sim.par["rishons_number"]
    # ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
    ops = get_QED_operators(n_rishons)
    # GET LOCAL DIMENSION
    loc_dim = ops["E_square_odd"].shape[0]
    # ACQUIRE HAMILTONIAN COEFFICIENTS
    coeffs = get_QED_Hamiltonian_couplings(g, m)
    # CONSTRUCT THE HAMILTONIAN
    h_terms = {}
    H = 0
    # -------------------------------------------------------------------------------
    # LINK PENALTIES & Border penalties
    for d in directions:
        for site, anti_site in [("even", "odd"), ("odd", "even")]:
            op_name_list = [f"E0_p{d}_{site}", f"E0_m{d}_{anti_site}"]
            op_list = [ops[op] for op in op_name_list]
            # Define the Hamiltonian term
            h_terms[f"WW{d}_{site}"] = TwoBodyTerm2D(d, op_list, op_name_list)
            H += h_terms[f"WW{d}_{site}"].get_Hamiltonian(
                lvals,
                strength=2 * coeffs["eta"],
                has_obc=has_obc,
                add_dagger=False,
                mask=staggered_mask(lvals, site),
            )
        # SINGLE SITE OPERATORS needed for the LINK SYMMETRY/OBC PENALTIES
        for s in "mp":
            for site in ["even", "odd"]:
                op_name = f"E0_square_{s}{d}_{site}"
                h_terms[op_name] = LocalTerm2D(ops[op_name], op_name)
                H += h_terms[op_name].get_Hamiltonian(
                    lvals=lvals,
                    strength=coeffs["eta"],
                    mask=staggered_mask(lvals, site),
                )
    # -------------------------------------------------------------------------------
    for site in ["even", "odd"]:
        # ELECTRIC ENERGY
        h_terms[f"E_square_{site}"] = LocalTerm2D(
            ops[f"E_square_{site}"], f"E_square_{site}"
        )
        H += h_terms[f"E_square_{site}"].get_Hamiltonian(
            lvals, coeffs["E"], staggered_mask(lvals, site)
        )
        # STAGGERED MASS TERM
        h_terms[f"N_{site}"] = LocalTerm2D(ops[f"N_{site}"], f"N_{site}")
        H += h_terms[f"N_{site}"].get_Hamiltonian(
            lvals, coeffs[f"m_{site}"], staggered_mask(lvals, site)
        )
    # HOPPING
    for d in directions:
        for site, anti_site in [("even", "odd"), ("odd", "even")]:
            # Define the list of the 2 non trivial operators
            op_name_list = [f"Q_p{d}_dag_{site}", f"Q_m{d}_{anti_site}"]
            op_list = [ops[op] for op in op_name_list]
            # Define the Hamiltonian term
            h_terms[f"{d}_hop_{site}"] = TwoBodyTerm2D(d, op_list, op_name_list)
            H += h_terms[f"{d}_hop_{site}"].get_Hamiltonian(
                lvals,
                strength=coeffs[f"t{d}_{site}"],
                has_obc=has_obc,
                add_dagger=True,
                mask=staggered_mask(lvals, site),
            )
    # -------------------------------------------------------------------------------
    # PLAQUETTE TERM: MAGNETIC INTERACTION
    if has_obc:
        site_list = [("even", "odd")]
    else:
        site_list = [("even", "odd"), ("odd", "even")]
    for site, anti_site in site_list:
        op_name_list = [
            f"C_px,py_{site}",
            f"C_py,mx_{anti_site}",
            f"C_my,px_{anti_site}",
            f"C_mx,my_{site}",
        ]
        op_list = [ops[op] for op in op_name_list]
        h_terms[f"plaq_{site}"] = PlaquetteTerm2D(op_list, op_name_list)
        H += h_terms[f"plaq_{site}"].get_Hamiltonian(
            lvals,
            strength=0.0 * coeffs["B"],
            has_obc=has_obc,
            add_dagger=True,
            mask=staggered_mask(lvals, site),
        )
    # ===========================================================================
    # CHECK THAT THE HAMILTONIAN IS HERMITIAN
    check_hermitian(H)
    # DIAGONALIZE THE HAMILTONIAN
    n_eigs = sim.par["n_eigs"]
    GS = Ground_State(H, n_eigs)
    sim.res["energy"] = GS.Nenergies
    for ii in range(n_eigs):
        logger.info("====================================================")
        logger.info(f"{ii} ENERGY: {format(sim.res['energy'][ii], '.9f')}")
        # GET STATE CONFIGURATIONS
        if not has_obc:
            get_state_configurations(
                truncation(GS.Npsi[:, ii], 1e-10), loc_dim, n_sites
            )
        # ENTROPY of a BIPARTITION
        sim.res["entropy"] = entanglement_entropy(
            GS.Npsi[:, ii], loc_dim, n_sites, partition_size=int(n_sites / 2)
        )
        # ===========================================================================
        # OBSERVABLES: RISHON NUMBER OPERATORS
        for d in directions:
            for s in "mp":
                for site in ["even", "odd"]:
                    op_name = f"n_{s}{d}_{site}"
                    h_terms[op_name] = LocalTerm2D(ops[op_name], op_name)
                    h_terms[op_name].get_loc_expval(GS.Npsi[:, ii], lvals, site)
        # OBSERVABLES: ElECTRIC ENERGY E^{2} and DENSITY OPERATOR N
        for obs in ["E_square", "N"]:
            for site in ["odd", "even"]:
                obs_name = f"{obs}_{site}"
                h_terms[obs_name].get_loc_expval(GS.Npsi[:, ii], lvals, site)
        # OBSERVABLES: PLAQUETTE ENERGY
        for site in ["even", "odd"]:
            if h_terms.get(f"plaq_{site}") is not None:
                sim.res["plaq"], _ = h_terms[f"plaq_{site}"].get_plaq_expval(
                    GS.Npsi[:, ii], lvals, has_obc, get_imag=False, site=site
                )
    if n_eigs == 1:
        sim.res["energy"] = sim.res["energy"][0]
