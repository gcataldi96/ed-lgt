import numpy as np
from ed_lgt.models import Z2_FermiHubbard_Model
from simsio import run_sim
from ed_lgt.tools import analyze_correlator
from time import perf_counter
import logging

logger = logging.getLogger(__name__)

with run_sim() as sim:
    start_time = perf_counter()
    model = Z2_FermiHubbard_Model(**sim.par["model"])
    # -------------------------------------------------------------------------------
    # BUILD THE HAMILTONIAN
    coeffs = {"U": sim.par["U"], "t": sim.par["t"], "h": sim.par["h"]}
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
    local_obs = [f"n_{s}{d}" for d in model.directions for s in "mp"]
    local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
    local_obs += ["X_Cross", "S2_psi", "E"]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = [["Sz_psi", "Sz_psi"]]
    twobody_axes = None
    # LIST OF PLAQUETTE OPERATORS
    plaquette_obs = [["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]]
    # DEFINE OBSERVABLES
    model.get_observables(local_obs, twobody_obs, plaquette_obs)
    # ENTROPY
    sim.res["entropy"] = np.zeros(model.n_eigs, dtype=float)
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
            list(np.arange(0, int(model.lvals[0] / 2), 1)),
            sector_configs=model.sector_configs,
        )
        # -----------------------------------------------------------------------
        # STATE CONFIGURATIONS
        model.H.Npsi[ii].get_state_configurations(
            threshold=1e-1, sector_configs=model.sector_configs
        )
        # ---------------------------------------------------------------------------
        # MEASURE OBSERVABLES
        model.measure_observables(ii)
        sim.res["plaq"][ii] = model.res["_".join(plaquette_obs[0])]
        for obs in local_obs:
            sim.res[obs][ii] = np.mean(model.res[obs])
        # sim.res["Sz_Sz"] = analyze_correlator(model.res["Sz_psi_Sz_psi"])
        # CHECK LINK SYMMETRIES
        model.check_symmetries()
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
