# %%
from time import perf_counter
from ed_lgt.models import SU2_Model
from ed_lgt.workflows import (
    su2_get_momentum_params,
    su2_get_convolution_gs_energy,
    su2_get_convolution_matrix,
)
import logging

logger = logging.getLogger(__name__)


def run_SU2_convolution(params):
    sim_res = {}
    start_time = perf_counter()
    # -------------------------------------------------------------------------------
    # SU(2) MODEL in (1+1)D with MOMENTUM SYMMETRY
    model = SU2_Model(**params["model"])
    # -------------------------------------------------------------------------------
    TC_symmetry = params.get("TC_symmetry", False)
    momentum_params = su2_get_momentum_params(TC_symmetry, model.n_sites)
    band_params = {
        "sim_band_name": params["sim_band_name"],
        "band_number": params.get("band_number", 0),
        "m": params["m"],
        "g": params["g"],
        "R0": 0,
    }
    # -------------------------------------------------------------------------------
    # GET THE GROUND STATE ENERGY density at momentum 0
    sim_res["gsdensity"] = su2_get_convolution_gs_energy(
        model, momentum_params, band_params
    )
    # -------------------------------------------------------------------------------
    # CONVOLUTION MATRIX
    if TC_symmetry:
        band_params["R0"] = 0
        sim_res["k1k2matrix"] = su2_get_convolution_matrix(
            model, momentum_params, band_params
        )
        for ii in range(momentum_params["n_momenta"]):
            logger.info("==================")
            for jj in range(momentum_params["n_momenta"]):
                logger.info(f"{ii} {jj} {sim_res['k1k2matrix'][ii, jj]}")
    else:
        band_params["R0"] = 0
        sim_res["k1k2matrix_even"] = su2_get_convolution_matrix(
            model, momentum_params, band_params
        )
        band_params["R0"] = 1
        sim_res["k1k2matrix_odd"] = su2_get_convolution_matrix(
            model, momentum_params, band_params
        )
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")


# %%
params = {
    "model": {
        "lvals": [14],
        "sectors": [14],
        "has_obc": [False],
        "spin": 0.5,
        "pure_theory": False,
        "background": 0,
        "ham_format": "sparse",
    },
    "hamiltonian": {
        "n_eigs": 1,
        "save_psi": False,
    },
    "momentum": {
        "get_momentum_basis": False,
        "unit_cell_size": [2],
        "TC_symmetry": False,
    },
    "observables": {
        "measure_obs": True,
        "get_entropy": False,
        "entropy_partition": [0],
        "get_state_configs": True,
        "get_overlap": False,
    },
    "TC_symmetry": True,
    "sim_band_name": "scattering/band1_N0",
    "g": 1,
    "m": 3,
}

run_SU2_convolution(params)

# %%
