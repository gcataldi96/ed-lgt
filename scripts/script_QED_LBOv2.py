import os
import sys

# Ensure NUMBA_NUM_THREADS is set properly before importing anything else
B = int(sys.argv[-1])
# Read the B parameter from command-line arguments
os.environ["NUMBA_NUM_THREADS"] = str(B)

import numpy as np
from ed_lgt.models import QED_Model
from simsio import run_sim
from time import perf_counter
import logging

logger = logging.getLogger(__name__)
with run_sim() as sim:
    start_time = perf_counter()
    spin_list = np.arange(1, 10, 1, dtype=int)
    n_spins = len(spin_list)
    sim.res["irrep_basis"] = np.zeros(n_spins, dtype=int)
    # -------------------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = ["E_square"]
    if not sim.par["model"]["pure_theory"]:
        local_obs += ["N"]
    n_eigs = sim.par["hamiltonian"]["n_eigs"]
    for obs in local_obs:
        sim.res[obs] = np.zeros((n_spins, n_eigs), dtype=float)
    sim.res["energy"] = np.zeros((n_spins, n_eigs), dtype=float)
    sim.res["entropy"] = np.zeros((n_spins, n_eigs), dtype=float)
    # -------------------------------------------------------------------------------
    for ss, spin in enumerate(spin_list):
        sim.par["model"]["spin"] = spin
        # ---------------------------------------------------------------------------
        # MODEL HAMILTONIAN
        model = QED_Model(**sim.par["model"])
        sim.res["irrep_basis"][ss] = model.loc_dims[0]
        m = sim.par["m"] if not model.pure_theory else None
        model.build_Hamiltonian(sim.par["g"], m)
        # ---------------------------------------------------------------------------
        # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
        model.diagonalize_Hamiltonian(n_eigs, model.ham_format)
        sim.res["energy"][ss] = model.H.Nenergies
        # DEFINE OBSERVABLES
        model.get_observables(local_obs)
        # ---------------------------------------------------------------------------
        # ENTROPY
        # DEFINE THE PARTITION FOR THE ENTANGLEMENT ENTROPY
        partition_indices = sim.par["observables"]["entropy_partition"]
        # ---------------------------------------------------------------------------
        for ii in range(model.H.n_eigs):
            model.H.print_energy(ii)
            if not model.momentum_basis:
                # -------------------------------------------------------------------
                # ENTROPY
                if sim.par["observables"]["get_entropy"]:
                    sim.res["entropy"][ss, ii] = model.H.Npsi[ii].entanglement_entropy(
                        partition_indices,
                        model.sector_configs,
                    )
                # -------------------------------------------------------------------
                # STATE CONFIGURATIONS
                if sim.par["observables"]["get_state_configs"]:
                    model.H.Npsi[ii].get_state_configurations(
                        1.5e-3, model.sector_configs
                    )
            # -----------------------------------------------------------------------
            # MEASURE OBSERVABLES
            if sim.par["observables"]["measure_obs"]:
                model.measure_observables(ii)
                sim.res["E_square"][ss, ii] = model.stag_avg(model.res["N_single"])
                if not model.pure_theory:
                    sim.res["N_single"][ss, ii] = model.stag_avg(model.res["N_single"])
                    sim.res["N_pair"][ss, ii] += 0.5 * model.stag_avg(
                        model.res["N_pair"], "even"
                    )
                    sim.res["N_pair"][ss, ii] += 0.5 * model.stag_avg(
                        model.res["N_zero"], "odd"
                    )
                    sim.res["N_zero"][ss, ii] += 0.5 * model.stag_avg(
                        model.res["N_zero"], "even"
                    )
                    sim.res["N_zero"][ss, ii] += 0.5 * model.stag_avg(
                        model.res["N_pair"], "odd"
                    )
                    sim.res["N_tot"][ss, ii] = (
                        sim.res["N_single"][ss, ii] + 2 * sim.res["N_pair"][ss, ii]
                    )
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
