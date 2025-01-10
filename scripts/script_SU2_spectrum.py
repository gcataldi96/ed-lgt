import numpy as np
import os
from numba import set_num_threads
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
    model.build_Hamiltonian1(sim.par["g"], m)
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
    n_eigs = sim.par["hamiltonian"]["n_eigs"]
    format = sim.par["hamiltonian"]["format"]
    model.diagonalize_Hamiltonian(n_eigs, format)
    sim.res["energy"] = model.H.Nenergies
    # -------------------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = [f"T2_{s}{d}" for d in model.directions for s in "mp"]
    local_obs = []  # ["E_square"]
    if not model.pure_theory:
        local_obs = ["N_single"]
        # [f"N_{label}" for label in ["tot", "single", "pair"]]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = []  # [[f"P_p{d}", f"P_m{d}"] for d in model.directions]
    twobody_axes = []  # [d for d in model.directions]
    # LIST OF PLAQUETTE OPERATORS
    plaquette_obs = []
    # DEFINE OBSERVABLES
    model.get_observables(local_obs)
    # QUENCH STATE FOR OVERLAP
    name = sim.par["hamiltonian"]["state"]
    config = model.overlap_QMB_state(name)
    logger.info(f"config {config}")
    in_state = model.get_qmb_state_from_configs([config])
    # -------------------------------------------------------------------------------
    # ALLOCATE OBSERVABLES
    partition_indices = get_entropy_partition(model.lvals)
    sim.res["entropy"] = np.zeros(model.H.n_eigs, dtype=float)
    sim.res["overlap"] = np.zeros(model.H.n_eigs, dtype=float)
    for obs in local_obs:
        sim.res[obs] = np.zeros(model.H.n_eigs, dtype=float)
    # -------------------------------------------------------------------------------
    for ii in range(model.H.n_eigs):
        model.H.print_energy(ii)
        if not model.momentum_basis:
            # -----------------------------------------------------------------------
            # ENTROPY
            if sim.par["get_entropy"]:
                sim.res["entropy"][ii] = model.H.Npsi[ii].entanglement_entropy(
                    partition_indices, model.sector_configs
                )
            # -----------------------------------------------------------------------
            # STATE CONFIGURATIONS
            if sim.par["get_state_configs"]:
                model.H.Npsi[ii].get_state_configurations(1e-1, model.sector_configs)
        # -----------------------------------------------------------------------
        # MEASURE OBSERVABLES
        model.measure_observables(ii)
        for obs in local_obs:
            sim.res[obs][ii] = np.mean(model.res[obs])
        # ---------------------------------------------------------------------------
        # OVERLAPS with the INITIAL STATE
        sim.res["overlap"][ii] = model.measure_fidelity(in_state, ii, print_value=True)
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
