import numpy as np
from ed_lgt.models import SU2_Model
from ed_lgt.operators import SU2_Hamiltonian_couplings
from simsio import run_sim
from time import perf_counter
import logging

logger = logging.getLogger(__name__)
with run_sim() as sim:
    start_time = perf_counter()
    model = SU2_Model(**sim.par["model"])
    # -------------------------------------------------------------------------------
    # BUILD THE HAMILTONIAN
    coeffs = SU2_Hamiltonian_couplings(
        model.dim, model.pure_theory, sim.par["g"], sim.par["m"]
    )
    model.build_Hamiltonian(coeffs)
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
    if sim.par["hamiltonian"]["diagonalize"]:
        model.diagonalize_Hamiltonian(
            n_eigs=sim.par["hamiltonian"]["n_eigs"],
            format=sim.par["hamiltonian"]["format"],
        )
        sim.res["energy"] = model.H.Nenergies
    # -------------------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = [f"T2_{s}{d}" for d in model.directions for s in "mp"]
    local_obs += ["E_square"]
    if not model.pure_theory:
        local_obs = [f"N_{label}" for label in ["tot", "single", "pair"]]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = [[f"P_p{d}", f"P_m{d}"] for d in model.directions]
    twobody_axes = [d for d in model.directions]
    # LIST OF PLAQUETTE OPERATORS
    plaquette_obs = []
    # DEFINE OBSERVABLES
    model.get_observables(local_obs)
    # -------------------------------------------------------------------------------
    # ENSEMBLE BEHAVIORS
    obs = sim.par["ensemble"]["local_obs"]
    # -------------------------------------------------------------------------------
    # MICROCANONICAL ENSEMBLE
    if sim.par["ensemble"]["microcanonical"]["average"]:
        config = model.overlap_QMB_state(sim.par["ensemble"]["microcanonical"]["state"])
        ref_state = model.get_qmb_state_from_config(config)
        micro_state, sim.res["microcan_avg"] = model.microcanonical_avg(obs, ref_state)
    # -------------------------------------------------------------------------------
    # CANONICAL ENSEMBLE
    if sim.par["ensemble"]["canonical"]["average"]:
        threshold = sim.par["ensemble"]["canonical"]["threshold"]
        if sim.par["ensemble"]["canonical"]["state"] != "micro":
            config = model.overlap_QMB_state(sim.par["ensemble"]["canonical"]["state"])
            ref_state = model.get_qmb_state_from_config(config)
        else:
            ref_state = micro_state
        beta = model.get_thermal_beta(ref_state, threshold)
        sim.res["canonical_avg"] = model.canonical_avg(obs, beta)
    # -------------------------------------------------------------------------------
    # DIAGONAL ENSEMBLE
    if sim.par["ensemble"]["diagonal"]["average"]:
        if sim.par["ensemble"]["diagonal"]["state"] != "micro":
            config = model.overlap_QMB_state(sim.par["ensemble"]["diagonal"]["state"])
            ref_state = model.get_qmb_state_from_config(config)
        else:
            ref_state = micro_state
        sim.res["diagonal_avg"] = model.diagonal_avg(obs, ref_state)
    # -------------------------------------------------------------------------------
    # DYNAMICS
    if sim.par["dynamics"]["time_evolution"]:
        start = sim.par["dynamics"]["start"]
        stop = sim.par["dynamics"]["stop"]
        delta_n = sim.par["dynamics"]["delta_n"]
        n_steps = int((stop - start) / delta_n)
        # INITIAL STATE PREPARATION
        name = sim.par["dynamics"]["state"]
        if name != "micro":
            config = model.overlap_QMB_state(name)
            in_state = model.get_qmb_state_from_config(config)
        else:
            in_state = micro_state
        # TIME EVOLUTION
        model.time_evolution_Hamiltonian(in_state, start, stop, n_steps)
    # -------------------------------------------------------------------------------
    # DEFINE THE STEP (FROM DYNAMICS OR DIAGONALIZATION)
    N = n_steps if sim.par["dynamics"]["time_evolution"] else model.n_eigs
    # -------------------------------------------------------------------------------
    sim.res["entropy"] = np.zeros(N, dtype=float)
    for overlap_state in sim.par["overlap_list"]:
        sim.res[f"overlap_{overlap_state}"] = np.zeros(N, dtype=float)
    for obs in local_obs:
        sim.res[obs] = np.zeros((N, model.n_sites), dtype=float)
    # -------------------------------------------------------------------------------
    for ii in range(N):
        logger.info(f"================== {ii} ===================")
        if not model.momentum_basis:
            # -----------------------------------------------------------------------
            # ENTROPY
            sim.res["entropy"][ii] = model.H.psi_time[ii].entanglement_entropy(
                list(np.arange(0, int(model.lvals[0] / 2), 1)),
                sector_configs=model.sector_configs,
            )
            # -----------------------------------------------------------------------
            # STATE CONFIGURATIONS
            model.H.psi_time[ii].get_state_configurations(
                threshold=1e-1, sector_configs=model.sector_configs
            )
        # -----------------------------------------------------------------------
        # MEASURE OBSERVABLES
        model.measure_observables(ii, sim.par["dynamics"]["time_evolution"])
        for obs in local_obs:
            sim.res[obs][ii, :] = model.res[obs]
        # ---------------------------------------------------------------------------
        # OVERLAPS with the INITIAL STATE
        for overlap_state in sim.par["overlap_list"]:
            sim.res[f"overlap_{overlap_state}"][ii] = model.measure_fidelity(
                in_state, ii, sim.par["dynamics"]["time_evolution"], True
            )
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
