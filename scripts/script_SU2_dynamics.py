import numpy as np
import os
from numba import set_num_threads
from ed_lgt.tools import stag_avg
from ed_lgt.modeling import get_entropy_partition
from ed_lgt.models import SU2_Model
from simsio import run_sim
from time import perf_counter
import logging

logger = logging.getLogger(__name__)
with run_sim() as sim:
    # Set the number of threads per simulation
    set_num_threads(int(os.environ.get("NUMBA_NUM_THREADS", sim.par["n_threads"])))
    start_time = perf_counter()
    # -------------------------------------------------------------------------------
    # MODEL HAMILTONIAN
    model = SU2_Model(**sim.par["model"])
    m = sim.par["m"] if not model.pure_theory else None
    model.build_Hamiltonian(sim.par["g"], m)
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
    if name != "micro":
        config = model.overlap_QMB_state(name)
        logger.info(f"config {config}")
        in_state = model.get_qmb_state_from_configs([config])
    # -------------------------------------------------------------------------------
    # STAGGERED BASIS
    logical_stag_basis = sim.par["dynamics"]["logical_stag_basis"]
    num_blocks = model.n_sites // (2 * logical_stag_basis)
    stag_array = np.array(
        [-1] * logical_stag_basis + [1] * logical_stag_basis, dtype=int
    )
    norm_scalar_product = np.tile(stag_array, num_blocks)
    logger.info(f"norm scalar product {norm_scalar_product}")
    # -------------------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = [f"T2_{s}{d}" for d in model.directions for s in "mp"]
    local_obs = ["E_square"]
    if not model.pure_theory:
        local_obs = [f"N_{label}" for label in ["tot", "single", "pair", "zero"]]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = [[f"P_p{d}", f"P_m{d}"] for d in model.directions]
    twobody_axes = [d for d in model.directions]
    # LIST OF PLAQUETTE OPERATORS
    plaquette_obs = []
    # DEFINE OBSERVABLES
    model.get_observables(local_obs)
    # ALLOCATE OBSERVABLES
    partition_indices = get_entropy_partition(model.lvals)
    sim.res["entropy"] = np.zeros(n_steps, dtype=float)
    sim.res["overlap"] = np.zeros(n_steps, dtype=float)
    sim.res["delta"] = np.zeros(n_steps, dtype=float)
    for obs in local_obs:
        sim.res[obs] = np.zeros(n_steps, dtype=float)
    # -------------------------------------------------------------------------------
    # ENSEMBLE BEHAVIORS
    get_micro_avg = sim.par["ensemble"]["microcanonical"]["average"]
    # -------------------------------------------------------------------------------
    # MICROCANONICAL ENSEMBLE (it requires a large part of the Hamiltonian spectrum)
    if get_micro_avg:
        # DIAGONALIZE THE HAMILTONIAN
        model.diagonalize_Hamiltonian("full", "dense")
        sim.res["energy"] = model.H.Nenergies
        micro_state, sim.res["micro_delta"] = model.microcanonical_avg1(
            "N_tot", in_state, norm_scalar_product
        )
        _, sim.res["micro_N_single"] = model.microcanonical_avg1("N_single", in_state)
        _, sim.res["micro_N_pair"] = model.microcanonical_avg1(
            "N_pair", in_state, staggered_avg="even"
        )
        _, sim.res["micro_N_zero"] = model.microcanonical_avg1(
            "N_zero", in_state, staggered_avg="odd"
        )
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
        if not get_micro_avg:
            # DIAGONALIZE THE HAMILTONIAN
            model.diagonalize_Hamiltonian("full", "dense")
            sim.res["energy"] = model.H.Nenergies
        # MEASURE DIAGONAL ENSEMBLE of some OBSERVABLES
        sim.res["DE_delta"] = model.diagonal_avg1(
            "N_tot", in_state, norm_scalar_product
        )
        sim.res["DE_N_single"] = model.diagonal_avg1("N_single", in_state)
        sim.res["DE_N_pair"] = model.diagonal_avg1(
            "N_pair", in_state, staggered_avg="even"
        )
        sim.res["DE_N_zero"] = model.diagonal_avg1(
            "N_zero", in_state, staggered_avg="odd"
        )
    # ---------------------------------------------------------------------------
    # OVERLAPS and entropy of the eigenstates with the INITIAL STATE
    sim.res["eig_overlap"] = np.zeros(model.H.shape[0], dtype=float)
    sim.res["eig_entropy"] = np.zeros(model.H.shape[0], dtype=float)
    for ii in range(model.H.shape[0]):
        sim.res["eig_overlap"][ii] = model.measure_fidelity(in_state, ii)
        sim.res["eig_entropy"][ii] = model.H.Npsi[ii].entanglement_entropy(
            partition_indices, sector_configs=model.sector_configs
        )
    # -------------------------------------------------------------------------------
    # TIME EVOLUTION
    model.time_evolution_Hamiltonian(in_state, time_line)
    if hasattr(model.H, "Deff"):
        sim.res["Deff"] = model.H.Deff
        sim.res["Hspace_size"] = model.H.shape[0]
        logger.info(f"D {-np.log(model.H.Deff)/np.log(model.H.shape[0])}")
    # -------------------------------------------------------------------------------
    for ii, tstep in enumerate(time_line):
        msg = f"TIME {round(tstep, 2)}"
        logger.info(f"================== {msg} ========================")
        if not model.momentum_basis:
            # -----------------------------------------------------------------------
            # ENTROPY
            if sim.par["get_entropy"]:
                sim.res["entropy"][ii] = model.H.psi_time[ii].entanglement_entropy(
                    partition_indices, sector_configs=model.sector_configs
                )
            # -----------------------------------------------------------------------
            # STATE CONFIGURATIONS
            if sim.par["get_state_configs"]:
                model.H.psi_time[ii].get_state_configurations(
                    1e-1, model.sector_configs
                )
        # -----------------------------------------------------------------------
        # MEASURE OBSERVABLES
        model.measure_observables(ii, dynamics=True)
        sim.res["N_single"][ii] = stag_avg(model.res["N_single"])
        sim.res["N_pair"][ii] = (
            stag_avg(model.res["N_pair"], "even") + stag_avg(model.res["N_zero"], "odd")
        ) / 2
        sim.res["N_zero"][ii] = (
            stag_avg(model.res["N_zero"], "even") + stag_avg(model.res["N_pair"], "odd")
        ) / 2
        sim.res["delta"][ii] = (
            np.dot(model.res["N_tot"], norm_scalar_product) / model.n_sites
        )
        # ---------------------------------------------------------------------------
        # OVERLAPS with the INITIAL STATE
        # sim.res["overlap"][ii] = model.measure_fidelity(in_state, ii, True, True)
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
