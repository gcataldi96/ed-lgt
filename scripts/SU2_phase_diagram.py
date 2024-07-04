import numpy as np
from ed_lgt.models import SU2_Model_Gen
from ed_lgt.operators import SU2_Hamiltonian_couplings
from simsio import run_sim
from time import perf_counter
import logging

logger = logging.getLogger(__name__)
with run_sim() as sim:
    start_time = perf_counter()
    model = SU2_Model_Gen(**sim.par["model"])
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
    local_obs = ["E_square", "S2_tot"]
    if not model.pure_theory:
        local_obs += ["S2_matter"]
        local_obs += [f"N_{label}" for label in ["tot", "single", "pair"]]
    # DEFINE OBSERVABLES
    model.get_observables(local_obs)
    # -------------------------------------------------------------------------------
    sim.res["entropy"] = np.zeros(model.n_eigs, dtype=float)
    for obs in local_obs:
        sim.res[obs] = np.zeros((model.n_eigs, model.n_sites), dtype=float)
    # -------------------------------------------------------------------------------
    for ii in range(model.n_eigs):
        logger.info(f"================== {ii} ===================")
        if not model.momentum_basis:
            # -----------------------------------------------------------------------
            # ENTROPY
            sim.res["entropy"][ii] = model.H.Npsi[ii].entanglement_entropy(
                list(np.arange(0, int(model.lvals[0] / 2), 1)),
                sector_configs=model.sector_configs,
            )
            # -----------------------------------------------------------------------
            # STATE CONFIGURATIONS
            model.H.Npsi[ii].get_state_configurations(
                threshold=1e-1, sector_configs=model.sector_configs
            )
        # -----------------------------------------------------------------------
        # MEASURE OBSERVABLES
        model.measure_observables(ii)
        for obs in local_obs:
            sim.res[obs][ii, :] = model.res[obs]
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
