import os
import sys

# Ensure NUMBA_NUM_THREADS is set properly before importing anything else
B = int(sys.argv[-1])
# Read the B parameter from command-line arguments
os.environ["NUMBA_NUM_THREADS"] = str(B)

import numpy as np
from ed_lgt.models import SU2_Model
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
    if model.spin > 0.5:
        model.build_gen_Hamiltonian(sim.par["g"], m)
    else:
        model.build_Hamiltonian(sim.par["g"], m)
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
    n_eigs = sim.par["hamiltonian"]["n_eigs"]
    model.diagonalize_Hamiltonian(n_eigs, model.ham_format)
    sim.res["energy"] = model.H.Nenergies
    # -------------------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = [f"T2_{s}{d}" for d in model.directions for s in "mp"]
    local_obs += ["E_square"]
    if not model.pure_theory:
        local_obs += [f"N_{label}" for label in ["tot", "single", "pair"]]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = []
    twobody_axes = []
    # LIST OF PLAQUETTE OPERATORS
    plaquette_obs = []
    # DEFINE OBSERVABLES
    model.get_observables(local_obs)
    for obs in local_obs:
        sim.res[obs] = np.zeros(model.H.n_eigs, dtype=float)
    # QUENCH STATE FOR OVERLAP
    if sim.par["observables"]["get_overlap"]:
        name = sim.par["hamiltonian"]["state"]
        config = model.overlap_QMB_state(name)
        logger.info(f"config {config}")
        in_state = model.get_qmb_state_from_configs([config])
        sim.res["overlap"] = np.zeros(model.H.n_eigs, dtype=float)
    # -------------------------------------------------------------------------------
    # ENTROPY
    # DEFINE THE PARTITION FOR THE ENTANGLEMENT ENTROPY
    partition_indices = [0, 1, 2, 3]
    # Build the list of environment and subsystem sites configurations
    model.get_subsystem_environment_configs(keep_indices=partition_indices)
    sim.res["entropy"] = np.zeros(model.H.n_eigs, dtype=float)
    # -------------------------------------------------------------------------------
    for ii in range(model.H.n_eigs):
        model.H.print_energy(ii)
        if not model.momentum_basis:
            # -----------------------------------------------------------------------
            # ENTROPY
            if sim.par["observables"]["get_entropy"]:
                sim.res["entropy"][ii] = model.H.Npsi[ii].entanglement_entropy(
                    partition_indices,
                    model.subsystem_configs,
                    model.env_configs,
                    model.unique_subsys_configs,
                    model.unique_env_configs,
                )
            # -----------------------------------------------------------------------
            # STATE CONFIGURATIONS
            if sim.par["observables"]["get_state_configs"]:
                model.H.Npsi[ii].get_state_configurations(1e-2, model.sector_configs)
        # -----------------------------------------------------------------------
        # MEASURE OBSERVABLES
        if sim.par["observables"]["measure_obs"]:
            model.measure_observables(ii)
            for obs in local_obs:
                sim.res[obs][ii] = np.mean(model.res[obs])
        # ---------------------------------------------------------------------------
        # OVERLAPS with the INITIAL STATE
        if sim.par["observables"]["get_overlap"]:
            sim.res["overlap"][ii] = model.measure_fidelity(
                in_state, ii, print_value=True
            )
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
