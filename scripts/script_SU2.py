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
    # Project onto the momentum sector k=0
    if model.momentum_basis:
        model.momentum_basis_projection(logical_unit_size=2)
        # Get the momentum basis
        B = momentum_basis_k0(model.sector_configs, 2)
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
    model.diagonalize_Hamiltonian(n_eigs=sim.par["hamiltonian"]["n_eigs"])
    sim.res["energy"] = model.H.Nenergies
    # -------------------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = [f"T2_{s}{d}" for d in model.directions for s in "mp"]
    local_obs += ["E_square"]
    if not model.pure_theory:
        local_obs += [f"N_{label}" for label in ["r", "g", "tot", "single", "pair"]]
        local_obs += ["E_square"]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = [[f"P_p{d}", f"P_m{d}"] for d in model.directions]
    twobody_axes = [d for d in model.directions]
    # LIST OF PLAQUETTE OPERATORS
    plaquette_obs = []
    # DEFINE OBSERVABLES
    model.get_observables(
        local_obs, twobody_obs, plaquette_obs, twobody_axes=twobody_axes
    )
    # -------------------------------------------------------------------------------
    # ENTROPY
    sim.res["entropy"] = np.zeros(model.n_eigs, dtype=float)
    # -------------------------------------------------------------------------------
    # OVERLAPS
    ov_info = {"config": {}, "ind": {}, "state": {}}
    for name in ["V", "PV", "M"]:
        # Initialize a null state
        sim.res[f"overlap_{name}"] = np.zeros(model.n_eigs, dtype=float)
        # Define the config_state associated to a specific axis
        config = model.overlap_QMB_state(name)
        ov_info["config"][name] = config
        # Get the corresponding QMB index
        ov_info["ind"][name] = np.where((model.sector_configs == config).all(axis=1))[0]
        # Initialize the state
        ov_info["state"][name] = np.zeros(len(model.sector_configs), dtype=float)
        ov_info["state"][name][ov_info["ind"][name]] = 1
        if model.momentum_basis:
            # Project the vacuum in the momentum sector
            ov_info["state"][name] = B.transpose().dot(ov_info["state"][name])
    # -------------------------------------------------------------------------------
    # THERMAL AVERAGE with CANONICAL ENSEMBLE
    name = "M"
    logger.info(name)
    beta = model.get_thermal_beta(state=ov_info["state"][name], threshold=1e-8)
    val = model.thermal_average("N_single", beta)
    logger.info(f"Thermal average {name} {val}")
    val = model.thermal_average("E_square", beta)
    logger.info(f"Thermal average {name} {val}")
    # -------------------------------------------------------------------------------
    for ii in range(model.n_eigs):
        # PRINT ENERGY
        model.H.print_energy(ii)
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
            # CHECK LINK SYMMETRIES
            model.check_symmetries()
            """
            if ii == 0:
                # SAVE RESULTS
                for measure in model.res.keys():
                    sim.res[measure] = model.res[measure]
            """
        # ---------------------------------------------------------------------------
        # OVERLAPS
        for name in ["V", "PV"]:
            if model.momentum_basis:
                sim.res[f"overlap_{name}"][ii] = (
                    np.abs(ov_info["state"][name].conj().dot(model.H.Npsi[ii].psi)) ** 2
                )
            else:
                sim.res[f"overlap_{name}"][ii] = (
                    np.abs(model.H.Npsi[ii].psi[ov_info["ind"][name]]) ** 2
                )
        # ---------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
