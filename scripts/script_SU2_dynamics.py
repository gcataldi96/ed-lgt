import numpy as np
from ed_lgt.models import SU2_Model
from ed_lgt.operators import SU2_Hamiltonian_couplings
from ed_lgt.symmetries import momentum_basis_k0
from simsio import run_sim
from time import perf_counter
import logging

logger = logging.getLogger(__name__)
with run_sim() as sim:
    start_time = perf_counter()
    model = SU2_Model(**sim.par["model"])
    # BUILD AND DIAGONALIZE HAMILTONIAN
    coeffs = SU2_Hamiltonian_couplings(
        model.dim, model.pure_theory, sim.par["g"], sim.par["m"]
    )
    model.build_Hamiltonian(coeffs)
    # -------------------------------------------------------------------------------
    # PROJECT ON THE MOMENUMT SECTOR k=0
    if model.momentum_basis:
        model.momentum_basis_projection(logical_unit_size=2)
        # Get the momentum basis
        B = momentum_basis_k0(model.sector_configs, 2)
    # -------------------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = [f"T2_{s}{d}" for d in model.directions for s in "mp"]
    local_obs += ["E_square"]
    if not model.pure_theory:
        local_obs = [f"N_{label}" for label in ["tot", "single", "pair"]]
        local_obs += ["E_square"]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = [[f"P_p{d}", f"P_m{d}"] for d in model.directions]
    twobody_axes = [d for d in model.directions]
    # LIST OF PLAQUETTE OPERATORS
    plaquette_obs = []
    # DEFINE OBSERVABLES
    model.get_observables(local_obs)
    # -------------------------------------------------------------------------------
    # TIME EVOLUTION
    start = 0
    stop = 10
    delta_n = 0.05
    n_steps = int((stop - start) / delta_n)
    # -------------------------------------------------------------------------------
    # OVERLAPS
    name = "M"
    ov_info = {"config": {}, "ind": {}, "in_state": {}}
    # Initialize a null state
    sim.res[f"overlap_{name}"] = np.zeros(n_steps, dtype=float)
    # Define the config_state associated to a specific axis
    config = model.overlap_QMB_state(name)
    ov_info["config"][name] = config
    # Get the corresponding QMB index
    ov_info["ind"][name] = np.where((model.sector_configs == config).all(axis=1))[0]
    # INITIAL STATE
    ov_info["in_state"][name] = np.zeros(len(model.sector_configs), dtype=float)
    ov_info["in_state"][name][ov_info["ind"][name]] = 1
    if model.momentum_basis:
        # Project the state in the momentum sector
        ov_info["in_state"][name] = B.transpose().dot(ov_info["in_state"][name])
    # -------------------------------------------------------------------------------
    model.time_evolution_Hamiltonian(ov_info["in_state"][name], start, stop, n_steps)
    # -------------------------------------------------------------------------------
    sim.res["entropy"] = np.zeros(n_steps, dtype=float)
    sim.res["fidelity"] = np.zeros(n_steps, dtype=float)
    for obs in local_obs:
        sim.res[obs] = np.zeros((n_steps, model.n_sites), dtype=float)
    for ii in range(n_steps):
        logger.info(f"================== STEP {ii} ===================")
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
            model.measure_observables(ii, dynamics=True)
            for obs in local_obs:
                sim.res[obs][ii, :] = model.res[obs]
        # ---------------------------------------------------------------------------
        # OVERLAPS with the INITIAL STATE
        if model.momentum_basis:
            sim.res[f"overlap_{name}"][ii] = (
                np.abs(ov_info["state"][name].conj().dot(model.H.psi_time[ii].psi)) ** 2
            )
        else:
            sim.res[f"overlap_{name}"][ii] = (
                np.abs(model.H.psi_time[ii].psi[ov_info["ind"][name]]) ** 2
            )
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
