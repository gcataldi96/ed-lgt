# %%
import numpy as np
from ed_lgt.modeling import get_entropy_partition
from ed_lgt.models import QED_Model
from time import perf_counter
import logging

logger = logging.getLogger(__name__)

par = {
    "model": {
        # LATTICE DIMENSIONS
        "lvals": [8],
        # BOUNDARY CONDITIONS
        "has_obc": [True],
        # GAUGE TRUNCATION
        "spin": 1,
        # PURE or FULL THEORY
        "pure_theory": False,
        "ham_format": "sparse",
        # "sectors": [4]
    },
    "dynamics": {
        "time_evolution": True,
        "start": 0,
        "stop": 10,
        "delta_n": 0.02,
        "state": "V",
        "logical_stag_basis": 2,
    },
    "observables": {
        "measure_obs": True,
        "get_entropy": False,
        "entropy_partition": [0, 1, 2, 3],
        "get_state_configs": False,
        "get_overlap": True,
    },
    # g COUPLING
    "g": 1,
    # m COUPLING
    "m": 7,
}

res = {}

logger = logging.getLogger(__name__)

start_time = perf_counter()
# -------------------------------------------------------------------------------
# MODEL HAMILTONIAN
model = QED_Model(**par["model"])
m = par["m"] if not model.pure_theory else None
model.build_Hamiltonian(par["g"], m)
# -------------------------------------------------------------------------------
# DYNAMICS PARAMETERS
name = par["dynamics"]["state"]
start = par["dynamics"]["start"]
stop = par["dynamics"]["stop"]
delta_t = par["dynamics"]["delta_n"]
time_line = np.arange(start, stop + delta_t, delta_t)
res["time_steps"] = time_line
n_steps = len(res["time_steps"])
# -------------------------------------------------------------------------------
# PREPARE THE INITIAL STATE
in_state = model.get_qmb_state_from_configs([[1, 3, 1, 3, 1, 3, 1, 3]])
# -------------------------------------------------------------------------------
# LIST OF LOCAL OBSERVABLES
local_obs = ["N", "E_square"]
# DEFINE OBSERVABLES
model.get_observables(local_obs)
for obs in local_obs:
    res[obs] = np.zeros(n_steps, dtype=float)
# -------------------------------------------------------------------------------
# DEFINE THE PARTITION FOR THE ENTANGLEMENT ENTROPY
partition_indices = get_entropy_partition(model.lvals)
res["entropy"] = np.zeros(n_steps, dtype=float)
if par["observables"]["get_overlap"]:
    res["overlap"] = np.zeros(n_steps, dtype=float)
# -------------------------------------------------------------------------------
# TIME EVOLUTION
if par["dynamics"]["time_evolution"]:
    model.time_evolution_Hamiltonian(in_state, time_line)
    # -------------------------------------------------------------------------------
    for ii, tstep in enumerate(time_line):
        msg = f"TIME {round(tstep, 2)}"
        logger.info(f"================== {msg} ==========================")
        if not model.momentum_basis:
            # -----------------------------------------------------------------------
            # ENTROPY
            # if par["observables"]["get_entropy"]:
            #   res["entropy"][ii] = model.H.psi_time[ii].entanglement_entropy(
            #       partition_indices,
            #       model.sector_configs,
            #   )
            # -----------------------------------------------------------------------
            # STATE CONFIGURATIONS
            if par["observables"]["get_state_configs"]:
                model.H.psi_time[ii].get_state_configurations(
                    1e-2, model.sector_configs
                )
        # -----------------------------------------------------------------------
        # MEASURE OBSERVABLES
        if par["observables"]["measure_obs"]:
            model.measure_observables(ii, dynamics=True)
            res["E_square"][ii] = np.mean(model.res["E_square"])
            res["N"][ii] = model.stag_avg(model.res["N"], "even")
            logger.info(f"N {res['N'][ii]}")
            logger.info(f"E_square {res['E_square'][ii]}")
        # ---------------------------------------------------------------------------
        # OVERLAPS with the INITIAL STATE
        if par["observables"]["get_overlap"]:
            res["overlap"][ii] = model.measure_fidelity(in_state, ii, True, True)
# -------------------------------------------------------------------------------
end_time = perf_counter()
logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")

# %%
