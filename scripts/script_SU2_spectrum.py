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
    # -------------------------------------------------------------------------------
    # BUILD THE HAMILTONIAN
    model.build_Hamiltonian(coeffs)
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
        local_obs = []  # [f"N_{label}" for label in ["tot", "single", "pair"]]
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
    if len(model.has_obc) == 1:
        partition_indices = list(np.arange(0, int(model.lvals[0] / 2), 1))
    else:
        partition_indices = list(np.arange(0, int(model.lvals[0] / 2), 1)) + list(
            np.arange(model.lvals[0], model.lvals[0] + int(model.lvals[0] / 2), 1)
        )
    sim.res["entropy"] = np.zeros(model.H.n_eigs, dtype=float)
    sim.res["overlap"] = np.zeros(model.H.n_eigs, dtype=float)
    for obs in local_obs:
        sim.res[obs] = np.zeros(model.H.n_eigs, dtype=float)
    # -------------------------------------------------------------------------------
    for ii in range(model.H.n_eigs):
        logger.info(f"================== {ii} ===================")
        if not model.momentum_basis:
            # -----------------------------------------------------------------------
            # ENTROPY
            sim.res["entropy"][ii] = model.H.Npsi[ii].entanglement_entropy(
                partition_indices, model.sector_configs
            )
            # -----------------------------------------------------------------------
            # STATE CONFIGURATIONS
            # model.H.Npsi[ii].get_state_configurations(1e-1, model.sector_configs)
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
