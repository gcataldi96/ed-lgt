import numpy as np
from ed_lgt.models import Z2_FermiHubbard_Model
from simsio import run_sim
from time import perf_counter
import logging

logger = logging.getLogger(__name__)

with run_sim() as sim:
    start_time = perf_counter()
    model = Z2_FermiHubbard_Model(**sim.par["model"])
    # -------------------------------------------------------------------------------
    # BUILD THE HAMILTONIAN
    coeffs = {
        "U": sim.par["U"],
        "t": sim.par["t"],
        "h": sim.par["h"],
        "J": sim.par["J"],
    }
    model.build_Hamiltonian(coeffs)
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
    model.diagonalize_Hamiltonian(
        n_eigs=sim.par["hamiltonian"]["n_eigs"],
        format=sim.par["hamiltonian"]["format"],
    )
    sim.res["energy"] = model.H.Nenergies
    # -------------------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = [f"n_{s}{d}" for d in model.directions for s in "mp"]
    local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
    local_obs += ["X_Cross", "S2_psi", "E"]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = [["Sz_psi", "Sz_psi"]]
    twobody_axes = None
    # LIST OF PLAQUETTE OPERATORS
    plaquette_obs = [["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]]
    # STRING OPERATOR
    if not model.has_obc[0]:
        nbody_obs = [["Sz_mx,px" for _ in range(model.lvals[0])]]
        nbody_dist = [[[ii, 0] for ii in range(1, model.lvals[0], 1)]]
    # DEFINE OBSERVABLES
    model.get_observables(
        local_obs,
        twobody_obs,
        plaquette_obs,
        nbody_obs=nbody_obs,
        nbody_dist=nbody_dist,
    )
    # ENTROPY
    sim.res["entropy"] = np.zeros(model.n_eigs, dtype=float)
    partition_indices = list(np.arange(0, int(model.n_sites / 2), 1))
    # MEASUREMENTS
    sim.res["plaq"] = np.zeros(model.n_eigs, dtype=float)
    for obs in local_obs:
        sim.res[obs] = np.zeros(model.n_eigs, dtype=float)
    # -------------------------------------------------------------------------------
    for ii in range(model.n_eigs):
        logger.info(f"================== {ii} ===================")
        # PRINT ENERGY
        model.H.print_energy(ii)
        # -----------------------------------------------------------------------
        # ENTROPY
        sim.res["entropy"][ii] = model.H.Npsi[ii].entanglement_entropy(
            partition_indices, model.sector_configs
        )
        # -----------------------------------------------------------------------
        # STATE CONFIGURATIONS
        model.H.Npsi[ii].get_state_configurations(1e-1, model.sector_configs)
        # ---------------------------------------------------------------------------
        # MEASURE OBSERVABLES
        model.measure_observables(ii)
        sim.res["plaq"][ii] = model.res["_".join(plaquette_obs[0])]
        for obs in local_obs:
            sim.res[obs][ii] = np.mean(model.res[obs])
        # CHECK LINK SYMMETRIES
        model.check_symmetries()
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
