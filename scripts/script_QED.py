import numpy as np
from ed_lgt.modeling import get_entropy_partition
from ed_lgt.models import QED_Model
from simsio import run_sim
from time import perf_counter
import logging

logger = logging.getLogger(__name__)
with run_sim() as sim:
    # Start measuring time
    start_time = perf_counter()
    # -------------------------------------------------------------------------------
    # MODEL HAMILTONIAN
    model = QED_Model(**sim.par["model"])
    m = sim.par["m"] if not model.pure_theory else None
    model.build_Hamiltonian(sim.par["g"], m)
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
    n_eigs = sim.par["hamiltonian"]["n_eigs"]
    model.diagonalize_Hamiltonian(n_eigs, model.ham_format)
    sim.res["energy"] = model.H.Nenergies
    # ---------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = ["E_square"]
    local_obs += [f"E_{s}{d}" for d in model.directions for s in "mp"]
    if not model.pure_theory:
        local_obs += ["N"]
    for obs in local_obs:
        sim.res[obs] = np.zeros(n_eigs, dtype=float)
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = [[f"P_p{d}", f"P_m{d}"] for d in model.directions]
    twobody_axes = [d for d in model.directions]
    # LIST OF PLAQUETTE OPERATORS
    if model.dim == 2:
        plaquette_obs = [["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]]
    elif model.dim == 3:
        plaquette_obs = [
            ["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"],
            ["C_px,pz", "C_pz,mx", "C_mz,px", "C_mx,mz"],
            ["C_py,pz", "C_pz,my", "C_mz,py", "C_my,mz"],
        ]
    else:
        plaquette_obs = None
    # DEFINE OBSERVABLES
    model.get_observables(
        local_obs, twobody_obs, plaquette_obs, twobody_axes=twobody_axes
    )
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
            if sim.par["get_entropy"]:
                sim.res["entropy"][ii] = model.H.Npsi[ii].entanglement_entropy(
                    partition_indices,
                    model.subsystem_configs,
                    model.env_configs,
                    model.unique_subsys_configs,
                    model.unique_env_configs,
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
