import numpy as np
import os
from numba import set_num_threads
from ed_lgt.modeling import get_entropy_partition
from ed_lgt.models import QCD_Model
from simsio import run_sim
from time import perf_counter
import logging

logger = logging.getLogger(__name__)
with run_sim() as sim:
    # Set the number of threads per simulation
    set_num_threads(int(os.environ.get("NUMBA_NUM_THREADS", sim.par["n_threads"])))
    # -------------------------------------------------------------------------------
    start_time = perf_counter()
    # -------------------------------------------------------------------------------
    # MODEL
    h_format = sim.par["hamiltonian"]["format"]
    model = QCD_Model(**sim.par["model"])
    model.build_Hamiltonian(sim.par["g"], sim.par["mu"], sim.par["md"], h_format)
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
    n_eigs = sim.par["hamiltonian"]["n_eigs"]
    model.diagonalize_Hamiltonian(n_eigs, h_format)
    sim.res["energy"] = model.H.Nenergies
    # -------------------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = ["uu", "dd"]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = []
    twobody_axes = []
    # DEFINE OBSERVABLES
    model.get_observables(local_obs)
    # -------------------------------------------------------------------------------
    # ALLOCATE OBSERVABLES
    partition_indices = get_entropy_partition(model.lvals)
    sim.res["entropy"] = np.zeros(model.H.n_eigs, dtype=float)
    for obs in local_obs:
        sim.res[obs] = np.zeros(model.H.n_eigs, dtype=float)
    # -------------------------------------------------------------------------------
    for ii in range(model.H.n_eigs):
        logger.info(f"======================== {ii} =========================")
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
        # ---------------------------------------------------------------------------
        # MEASURE OBSERVABLES
        model.measure_observables(ii)
        for obs in local_obs:
            sim.res[obs][ii] = np.mean(model.res[obs])
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
