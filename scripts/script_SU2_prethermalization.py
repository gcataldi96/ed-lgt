import numpy as np
from ed_lgt.models import SU2_Model
from scipy.sparse import eye
from simsio import run_sim
from time import perf_counter
import logging

logger = logging.getLogger(__name__)
with run_sim() as sim:
    start_time = perf_counter()
    # -------------------------------------------------------------------------------
    # MODEL HAMILTONIAN
    model = SU2_Model(**sim.par["model"])
    m = sim.par["m"] if not model.pure_theory else None
    llambda = sim.par.get("lambda", 0.0)
    # Select Momentum Sector
    if sim.par["momentum"]["get_momentum_basis"]:
        unit_cell_size = sim.par["momentum"]["unit_cell_size"]
        k_vals = sim.par["momentum_k_vals"]
        TC_symmetry = sim.par["momentum"]["TC_symmetry"]
        model.set_momentum_sector(unit_cell_size, k_vals, TC_symmetry)
    # Save parameters
    model.default_params()
    # Build Hamiltonian
    if model.spin > 0.5:
        model.build_gen_Hamiltonian(sim.par["g"], m)
    else:
        model.build_Hamiltonian(sim.par["g"], m, lambda_noise=llambda)
    logger.info(f"loc dim {model.loc_dims}")
    # -------------------------------------------------------------------------------
    # DYNAMICS PARAMETERS
    name = sim.par["dynamics"]["state"]
    start = sim.par["dynamics"]["start"]
    stop = sim.par["dynamics"]["stop"]
    delta_t = sim.par["dynamics"]["delta_n"]
    time_line = np.arange(start, stop + delta_t, delta_t)
    sim.res["time_steps"] = time_line
    n_steps = len(sim.res["time_steps"])
    # -------------------------------------------------------------------------------
    # INITIAL STATE PREPARATION
    config = model.overlap_QMB_state(name)
    in_state = model.get_qmb_state_from_configs([config])
    # np.array([6, 12, 6, 12, 6, 12], dtype=int)
    # -------------------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = [f"T2_p{d}" for d in model.directions]
    local_obs += ["bg"]
    if not model.pure_theory:
        local_obs += [f"N_{label}" for label in ["tot", "single", "pair", "zero"]]
    # DEFINE OBSERVABLES
    if sim.par["observables"]["measure_obs"]:
        model.get_observables(local_obs)
        sim.res["E2"] = np.zeros(n_steps, dtype=float)
        for obs in local_obs:
            sim.res[obs] = np.zeros(n_steps, dtype=float)
    # -------------------------------------------------------------------------------
    # DEFINE THE PARTITION FOR THE ENTANGLEMENT ENTROPY
    partition_indices = sim.par["observables"]["entropy_partition"]
    if sim.par["observables"]["get_entropy"] or sim.par["observables"]["get_RDM"]:
        model._get_partition(partition_indices)
        sim.res["entropy"] = np.zeros(n_steps, dtype=float)
    if sim.par["observables"]["get_overlap"]:
        sim.res["overlap"] = np.zeros(n_steps, dtype=float)
    # -------------------------------------------------------------------------------
    # ENSEMBLE BEHAVIORS
    get_micro_avg = sim.par["ensemble"]["microcanonical"]["average"]
    # -------------------------------------------------------------------------------
    # MICROCANONICAL ENSEMBLE (it requires a large part of the Hamiltonian spectrum)
    if get_micro_avg:
        # DIAGONALIZE THE HAMILTONIAN
        model.diagonalize_Hamiltonian("full", "dense", print_results=True)
        sim.res["energy"] = model.H.Nenergies
        stag_avgs = {"N_single": None, "N_pair": "even", "N_zero": "odd"}
        norms = {"N_single": None, "N_pair": None, "N_zero": None}
        _, ME = model.microcanonical_avg(
            ["N_single", "N_pair", "N_zero"],
            in_state,
            staggered_avgs=stag_avgs,
            special_norms=norms,
        )
        sim.res |= ME
    # -------------------------------------------------------------------------------
    # CANONICAL ENSEMBLE (it does not need the full spectrum)
    if sim.par["ensemble"]["canonical"]["average"]:
        threshold = sim.par["ensemble"]["canonical"]["threshold"]
        if sim.par["ensemble"]["canonical"]["state"] != "micro":
            config = model.overlap_QMB_state(sim.par["ensemble"]["canonical"]["state"])
            ref_state = model.get_qmb_state_from_configs([config])
        beta = model.get_thermal_beta(ref_state, threshold)
        sim.res["canonical_avg"] = model.canonical_avg(obs, beta)
    # -------------------------------------------------------------------------------
    # DIAGONAL ENSEMBLE (it requires the full spectrum of the Hamiltonian)
    if sim.par["ensemble"]["diagonal"]["average"]:
        if not get_micro_avg:
            # DIAGONALIZE THE HAMILTONIAN
            model.diagonalize_Hamiltonian("full", "dense")
            sim.res["energy"] = model.H.Nenergies
        # MEASURE DIAGONAL ENSEMBLE of some OBSERVABLES
        sim.res |= model.diagonal_avg(
            ["N_single", "N_pair", "N_zero"],
            in_state,
            staggered_avgs=stag_avgs,
            special_norms=norms,
        )
    # -------------------------------------------------------------------------------
    # TIME EVOLUTION
    if sim.par["dynamics"]["time_evolution"]:
        model.time_evolution_Hamiltonian(in_state, time_line)
        # -------------------------------------------------------------------------------
        for ii, tstep in enumerate(time_line):
            msg = f"TIME {round(tstep, 2)}"
            logger.info(f"================== {msg} ==========================")
            if not model.momentum_basis:
                # -----------------------------------------------------------------------
                # ENTROPY
                if sim.par["observables"]["get_entropy"]:
                    sim.res["entropy"][ii] = model.H.psi_time[ii].entanglement_entropy(
                        partition_indices, model._partition_cache
                    )
                # -----------------------------------------------------------------------
                # STATE CONFIGURATIONS
                if sim.par["observables"]["get_state_configs"]:
                    model.H.psi_time[ii].get_state_configurations(
                        1e-2, model.sector_configs
                    )
            # -----------------------------------------------------------------------
            # MEASURE OBSERVABLES
            if sim.par["observables"]["measure_obs"]:
                model.measure_observables(ii, dynamics=True)
                sim.res["E2"][ii] = model.link_avg(obs_name="T2")
                sim.res["bg"][ii] = model.stag_avg(model.res["bg"])
                if not model.pure_theory:
                    sim.res["N_single"][ii] = model.stag_avg(model.res["N_single"])
                    sim.res["N_pair"][ii] += 0.5 * model.stag_avg(
                        model.res["N_pair"], "even"
                    )
                    sim.res["N_pair"][ii] += 0.5 * model.stag_avg(
                        model.res["N_zero"], "odd"
                    )
                    sim.res["N_zero"][ii] += 0.5 * model.stag_avg(
                        model.res["N_zero"], "even"
                    )
                    sim.res["N_zero"][ii] += 0.5 * model.stag_avg(
                        model.res["N_pair"], "odd"
                    )
                    sim.res["N_tot"][ii] = (
                        sim.res["N_single"][ii] + 2 * sim.res["N_pair"][ii]
                    )
                logger.info("----------------------------------------------------")
                logger.info(f"N_single {sim.res['N_single'][ii]}")
                logger.info(f"N_pair {sim.res['N_pair'][ii]}")
                logger.info(f"N_zero {sim.res['N_zero'][ii]}")
            # ---------------------------------------------------------------------------
            # OVERLAPS with the INITIAL STATE
            if sim.par["observables"]["get_overlap"]:
                sim.res["overlap"][ii] = model.measure_fidelity(
                    in_state, ii, True, True
                )
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
