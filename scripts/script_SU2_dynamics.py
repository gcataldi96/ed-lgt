import numpy as np
from ed_lgt.models import SU2_Model, SU2_Model_Gen
from ed_lgt.operators import SU2_Hamiltonian_couplings, SU2_gen_Hamiltonian_couplings
from simsio import run_sim
from time import perf_counter
import logging

logger = logging.getLogger(__name__)
with run_sim() as sim:
    start_time = perf_counter()
    # -------------------------------------------------------------------------------
    if sim.par["model"]["spin"] > 2:
        model = SU2_Model(**sim.par["model"])
        m = sim.par["m"] if not model.pure_theory else None
        coeffs = SU2_Hamiltonian_couplings(
            model.dim, model.pure_theory, sim.par["g"], m
        )
    else:
        model = SU2_Model_Gen(**sim.par["model"])
        m = sim.par["m"] if not model.pure_theory else None
        coeffs = SU2_gen_Hamiltonian_couplings(
            model.dim, model.pure_theory, sim.par["g"], m
        )
    logical_stag_basis = 2
    num_blocks = model.n_sites // (2 * logical_stag_basis)
    stag_array = np.array(
        [-1] * logical_stag_basis + [1] * logical_stag_basis, dtype=int
    )
    norm_scalar_product = np.tile(stag_array, num_blocks)
    norm_scalar_product = np.array([-1, 1, 1, -1, -1, -1, 1, 1, -1, -1])
    logger.info(f"norm scalar product {norm_scalar_product}")
    # -------------------------------------------------------------------------------
    # BUILD THE HAMILTONIAN
    model.build_Hamiltonian(coeffs)
    # -------------------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = [f"T2_{s}{d}" for d in model.directions for s in "mp"]
    local_obs = ["E_square"]
    if not model.pure_theory:
        local_obs = ["N_tot"]  # ["N_pair"]
        # [f"N_{label}" for label in ["r", "g", "tot", "single", "pair"]]
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
    get_micro_avg = sim.par["ensemble"]["microcanonical"]["average"]
    # -------------------------------------------------------------------------------
    # MICROCANONICAL ENSEMBLE (it requires a large part of the Hamiltonian spectrum)
    if get_micro_avg:
        # DIAGONALIZE THE HAMILTONIAN
        model.diagonalize_Hamiltonian("full", "dense")
        sim.res["energy"] = model.H.Nenergies
        config = model.overlap_QMB_state(sim.par["ensemble"]["microcanonical"]["state"])
        ref_state = model.get_qmb_state_from_configs([config])
        micro_state, sim.res["microcan_avg"] = model.microcanonical_avg(obs, ref_state)
    # -------------------------------------------------------------------------------
    # CANONICAL ENSEMBLE (it does not need the full spectrum)
    if sim.par["ensemble"]["canonical"]["average"]:
        threshold = sim.par["ensemble"]["canonical"]["threshold"]
        if sim.par["ensemble"]["canonical"]["state"] != "micro":
            config = model.overlap_QMB_state(sim.par["ensemble"]["canonical"]["state"])
            ref_state = model.get_qmb_state_from_configs([config])
        else:
            ref_state = micro_state
        beta = model.get_thermal_beta(ref_state, threshold)
        sim.res["canonical_avg"] = model.canonical_avg(obs, beta)
    # -------------------------------------------------------------------------------
    # DIAGONAL ENSEMBLE (it requires the full spectrum of the Hamiltonian)
    if sim.par["ensemble"]["diagonal"]["average"]:
        if sim.par["ensemble"]["diagonal"]["state"] != "micro":
            config = model.overlap_QMB_state(sim.par["ensemble"]["diagonal"]["state"])
            ref_state = model.get_qmb_state_from_configs([config])
            if not get_micro_avg:
                # DIAGONALIZE THE HAMILTONIAN
                model.diagonalize_Hamiltonian("full", "dense")
                sim.res["energy"] = model.H.Nenergies
        else:
            # Here we assume the microcanonical state has been already computed
            # and so the full Hamiltonian spectrum
            ref_state = micro_state
        sim.res["diagonal_avg"] = model.diagonal_avg(obs, ref_state)
    # -------------------------------------------------------------------------------
    # DYNAMICS
    start = sim.par["dynamics"]["start"]
    stop = sim.par["dynamics"]["stop"]
    delta_n = sim.par["dynamics"]["delta_n"]
    n_steps = int((stop - start) / delta_n)
    # INITIAL STATE PREPARATION
    name = sim.par["dynamics"]["state"]
    if name != "micro":
        config = model.overlap_QMB_state(name)
        logger.info(f"config {config}")
        in_state = model.get_qmb_state_from_configs([config])
    else:
        in_state = micro_state
    # TIME EVOLUTION
    model.time_evolution_Hamiltonian(in_state, start, stop, n_steps)
    # -------------------------------------------------------------------------------
    # ALLOCATE OBSERVABLES
    if len(model.has_obc) == 1:
        partition_indices = list(np.arange(0, int(model.lvals[0] / 2), 1))
    else:
        partition_indices = list(np.arange(0, int(model.lvals[0] / 2), 1)) + list(
            np.arange(model.lvals[0], model.lvals[0] + int(model.lvals[0] / 2), 1)
        )
    sim.res["entropy"] = np.zeros(n_steps, dtype=float)
    sim.res["overlap"] = np.zeros(n_steps, dtype=float)
    sim.res["delta"] = np.zeros(n_steps, dtype=float)
    for obs in local_obs:
        sim.res[obs] = np.zeros(n_steps, dtype=float)
    # -------------------------------------------------------------------------------
    for ii in range(n_steps):
        logger.info(
            f"================== TIME {format(delta_n*ii, '.2f')} ======================="
        )
        if not model.momentum_basis:
            # -----------------------------------------------------------------------
            # ENTROPY
            # sim.res["entropy"][ii] = model.H.psi_time[ii].entanglement_entropy(
            #    partition_indices, sector_configs=model.sector_configs
            # )
            # -----------------------------------------------------------------------
            # STATE CONFIGURATIONS
            # model.H.psi_time[ii].get_state_configurations(1e-1, model.sector_configs)
            # -----------------------------------------------------------------------
            # MEASURE OBSERVABLES
            model.measure_observables(ii, dynamics=True)
            for obs in local_obs:
                sim.res[obs][ii] = np.mean(model.res[obs])
            sim.res["delta"][ii] = (
                np.dot(model.res["N_tot"], norm_scalar_product) / model.n_sites
            )
        # ---------------------------------------------------------------------------
        # OVERLAPS with the INITIAL STATE
        sim.res["overlap"][ii] = model.measure_fidelity(in_state, ii, True, True)
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
