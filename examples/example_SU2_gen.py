# %%
import numpy as np
from ed_lgt.models import SU2_Model_Gen
from ed_lgt.operators import SU2_gen_Hamiltonian_couplings
from time import perf_counter
import logging

logger = logging.getLogger(__name__)

par = {
    "model": {
        "lvals": [4],
        "has_obc": [False],
        "spin": 2,
        "pure_theory": False,
        "momentum_basis": False,
        "logical_unit_size": 2,
    },
    "hamiltonian": {"diagonalize": True, "n_eigs": 1, "format": "sparse"},
    "dynamics": {
        "time_evolution": False,
        "start": 0,
        "stop": 1,
        "delta_n": 0.5,
        "state": "PV",
    },
    "ensemble": {
        "local_obs": "N_single",
        "microcanonical": {
            "average": False,
            "state": "PV",
            "delta": 0.1,
        },
        "canonical": {
            "average": False,
            "state": "PV",
            "threshold": 1e-8,
        },
        "diagonal": {
            "average": False,
            "state": "PV",
        },
    },
    "overlap_list": [],
    "g": 1,
    "m": 5,
}
res = {}
start_time = perf_counter()
model = SU2_Model_Gen(**par["model"])
# -------------------------------------------------------------------------------
end_time = perf_counter()
logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
# %%
# -------------------------------------------------------------------------------
# BUILD THE HAMILTONIAN
coeffs = SU2_gen_Hamiltonian_couplings(model.dim, model.pure_theory, par["g"], par["m"])
model.build_Hamiltonian(coeffs)
# -------------------------------------------------------------------------------
# DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
if par["hamiltonian"]["diagonalize"]:
    model.diagonalize_Hamiltonian(
        n_eigs=par["hamiltonian"]["n_eigs"],
        format=par["hamiltonian"]["format"],
    )
    res["energy"] = model.H.Nenergies
# -------------------------------------------------------------------------------
# LIST OF LOCAL OBSERVABLES
local_obs = [f"T2_{s}{d}" for d in model.directions for s in "mp"]
local_obs += ["E_square"]
if not model.pure_theory:
    local_obs = [f"N_{label}" for label in ["tot", "single", "pair"]]
# LIST OF TWOBODY CORRELATORS
twobody_obs = [[f"P_p{d}", f"P_m{d}"] for d in model.directions]
twobody_axes = [d for d in model.directions]
# LIST OF PLAQUETTE OPERATORS
plaquette_obs = []
# DEFINE OBSERVABLES
model.get_observables(local_obs)
# -------------------------------------------------------------------------------
# ENSEMBLE BEHAVIORS
obs = par["ensemble"]["local_obs"]
# -------------------------------------------------------------------------------
# MICROCANONICAL ENSEMBLE
if par["ensemble"]["microcanonical"]["average"]:
    config = model.overlap_QMB_state(par["ensemble"]["microcanonical"]["state"])
    ref_state = model.get_qmb_state_from_config(config)
    micro_state, res["microcan_avg"] = model.microcanonical_avg(obs, ref_state)
# -------------------------------------------------------------------------------
# CANONICAL ENSEMBLE
if par["ensemble"]["canonical"]["average"]:
    threshold = par["ensemble"]["canonical"]["threshold"]
    if par["ensemble"]["canonical"]["state"] != "micro":
        config = model.overlap_QMB_state(par["ensemble"]["canonical"]["state"])
        ref_state = model.get_qmb_state_from_config(config)
    else:
        ref_state = micro_state
    beta = model.get_thermal_beta(ref_state, threshold)
    res["canonical_avg"] = model.canonical_avg(obs, beta)
# -------------------------------------------------------------------------------
# DIAGONAL ENSEMBLE
if par["ensemble"]["diagonal"]["average"]:
    if par["ensemble"]["diagonal"]["state"] != "micro":
        config = model.overlap_QMB_state(par["ensemble"]["diagonal"]["state"])
        ref_state = model.get_qmb_state_from_config(config)
    else:
        ref_state = micro_state
    res["diagonal_avg"] = model.diagonal_avg(obs, ref_state)
# -------------------------------------------------------------------------------
# DYNAMICS
if par["dynamics"]["time_evolution"]:
    start = par["dynamics"]["start"]
    stop = par["dynamics"]["stop"]
    delta_n = par["dynamics"]["delta_n"]
    n_steps = int((stop - start) / delta_n)
    # INITIAL STATE PREPARATION
    name = par["dynamics"]["state"]
    if name != "micro":
        config = model.overlap_QMB_state(name)
        in_state = model.get_qmb_state_from_config(config)
    else:
        in_state = micro_state
    # TIME EVOLUTION
    model.time_evolution_Hamiltonian(in_state, start, stop, n_steps)
# -------------------------------------------------------------------------------
# DEFINE THE STEP (FROM DYNAMICS OR DIAGONALIZATION)
N = n_steps if par["dynamics"]["time_evolution"] else model.n_eigs
# -------------------------------------------------------------------------------
res["entropy"] = np.zeros(N, dtype=float)
ov_states = {}
for overlap_state in par["overlap_list"]:
    res[f"overlap_{overlap_state}"] = np.zeros(N, dtype=float)
    config = model.overlap_QMB_state(overlap_state)
    ov_states[overlap_state] = model.get_qmb_state_from_config(config)
for obs in local_obs:
    res[obs] = np.zeros((N, model.n_sites), dtype=float)
# -------------------------------------------------------------------------------
if par["dynamics"]["time_evolution"]:
    for ii in range(N):
        logger.info(f"================== {ii} ===================")
        if not model.momentum_basis:
            # -----------------------------------------------------------------------
            # ENTROPY
            res["entropy"][ii] = model.H.psi_time[ii].entanglement_entropy(
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
        model.measure_observables(ii, par["dynamics"]["time_evolution"])
        for obs in local_obs:
            res[obs][ii, :] = model.res[obs]
        # ---------------------------------------------------------------------------
        # OVERLAPS with the INITIAL STATE
        for overlap_state in par["overlap_list"]:
            res[f"overlap_{overlap_state}"][ii] = model.measure_fidelity(
                in_state, ii, par["dynamics"]["time_evolution"], True
            )
elif par["hamiltonian"]["diagonalize"]:
    for ii in range(N):
        logger.info(f"================== {ii} ===================")
        if not model.momentum_basis:
            # -----------------------------------------------------------------------
            # ENTROPY
            res["entropy"][ii] = model.H.Npsi[ii].entanglement_entropy(
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
        model.measure_observables(ii, par["dynamics"]["time_evolution"])
        for obs in local_obs:
            res[obs][ii, :] = model.res[obs]
        # ---------------------------------------------------------------------------
        # OVERLAPS with the INITIAL STATE
        for overlap_state in par["overlap_list"]:
            res[f"overlap_{overlap_state}"][ii] = model.measure_fidelity(
                ov_states[overlap_state], ii, False, True
            )
# -------------------------------------------------------------------------------
end_time = perf_counter()
logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")

# %%
