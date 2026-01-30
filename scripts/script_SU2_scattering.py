import os
import sys

# Ensure NUMBA_NUM_THREADS is set properly before importing anything else
B = int(sys.argv[-1])
# Read the B parameter from command-line arguments
os.environ["NUMBA_NUM_THREADS"] = str(B)

from ed_lgt.models import SU2_Model
from ed_lgt.workflows import (
    su2_get_momentum_params,
    su2_get_convolution_gs_energy,
    su2_get_convolution_matrix,
)
from simsio import *
from time import perf_counter
import logging

logger = logging.getLogger(__name__)
with run_sim() as sim:
    start_time = perf_counter()
    # Simulation band name
    sim_band_name = sim.par["sim_band_name"]
    band_number = sim.par.get("band_number", 0)
    # -------------------------------------------------------------------------------
    # SU(2) MODEL in (1+1)D with MOMENTUM SYMMETRY
    model = SU2_Model(**sim.par["model"])
    zero_density = False if sim.par["model"]["sectors"][0] != model.n_sites else True
    logger.info(f"zero density {zero_density}")
    m = sim.par["m"]
    g = sim.par["g"]
    # Choose if TC symmetry is enabled
    TC_symmetry = sim.par.get("TC_symmetry", False)
    momentum_params = su2_get_momentum_params(TC_symmetry, model.n_sites)
    logger.info(f"Momentum params {momentum_params}")
    band_params = {
        "sim_band_name": sim_band_name,
        "zero_density": zero_density,
        "band_number": band_number,
        "m": m,
        "g": g,
    }
    # -------------------------------------------------------------------------------
    # GET THE GROUND STATE ENERGY density at momentum 0
    gsdensity = su2_get_convolution_gs_energy(model, momentum_params, band_params)
    # -------------------------------------------------------------------------------
    # CONVOLUTION MATRIX
    if TC_symmetry:
        band_params["R0"] = 0
        sim.res["k1k2matrix"] = su2_get_convolution_matrix(
            model, momentum_params, band_params
        )
        for ii in range(momentum_params["n_momenta"]):
            logger.info("==================")
            for jj in range(momentum_params["n_momenta"]):
                logger.info(f"{ii} {jj} {sim.res["k1k2matrix"][ii, jj]}")
    else:
        band_params["R0"] = 0
        sim.res["k1k2matrix_even"] = su2_get_convolution_matrix(
            model, momentum_params, band_params
        )
        band_params["R0"] = 1
        sim.res["k1k2matrix_odd"] = su2_get_convolution_matrix(
            model, momentum_params, band_params
        )
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
