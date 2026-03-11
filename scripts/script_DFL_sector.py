import os
import sys

# Ensure NUMBA_NUM_THREADS is set properly before importing anything else
B = int(sys.argv[-1])
# Read the B parameter from command-line arguments
os.environ["NUMBA_NUM_THREADS"] = str(B)

import logging
from simsio import run_sim
from time import perf_counter
import numpy as np
from edlgt.workflows import (
    normalize_DFL_simsio_params,
    run_DFL_dynamics_sector_by_sector,
)

logger = logging.getLogger(__name__)


def _save_dfl_results(sim, res: dict):
    sim.res["time_steps"] = res["time"]
    sim.res["g_values"] = res["g_values"]
    for key, value in res.items():
        if key == "time":
            continue
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == 1:
            sim.res[key] = value[0]
        else:
            sim.res[key] = value


with run_sim() as sim:
    start_time = perf_counter()
    params = normalize_DFL_simsio_params(sim.par)
    res = run_DFL_dynamics_sector_by_sector(params)
    _save_dfl_results(sim, res)
    sim.res["total_time"] = perf_counter() - start_time
    logger.info(f"TIME SIMS {sim.res['total_time']:.5f}")
