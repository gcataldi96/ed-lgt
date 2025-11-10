import os
import sys

# Ensure NUMBA_NUM_THREADS is set properly before importing anything else
B = int(sys.argv[-1])
# Read the B parameter from command-line arguments
os.environ["NUMBA_NUM_THREADS"] = str(B)

import numpy as np
from itertools import product
from ed_lgt.models import SU2_Model
from ed_lgt.modeling import exp_val_data2
from simsio import *
from time import perf_counter
import logging


def get_data_from_sim(sim_filename, obs_name, kindex):
    config_filename = f"scattering/{sim_filename}"
    match = SimsQuery(group_glob=config_filename)
    ugrid, _ = uids_grid(match.uids, ["momentum_k_vals"])
    return get_sim(ugrid[kindex]).res[obs_name]


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
    k_unit_cell_size = [1] if TC_symmetry else [2]
    n_momenta = model.n_sites if TC_symmetry else model.n_sites // 2
    k_indices = np.arange(0, n_momenta, 1)
    sim.res["k_indices"] = k_indices
    if TC_symmetry:
        k_phys = 2 * np.pi * k_indices / model.n_sites
    else:
        k_phys = 4 * np.pi * k_indices / model.n_sites
    # -------------------------------------------------------------------------------
    # GET THE GROUND STATE ENERGY density at momentum 0
    if zero_density:
        GS = get_data_from_sim(sim_band_name, "psi0", 0)
        # Check Translational Hamiltonian
        model.set_momentum_pair([0], [0], k_unit_cell_size, TC_symmetry)
        model.default_params()
        # Check the momentum bases
        model.check_momentum_pair()
        # Build the local hamiltonian
        model.build_local_Hamiltonian(g, m, 0, TC_symmetry)
        eg_single_block = exp_val_data2(
            GS,
            GS,
            model.Hlocal.row_list,
            model.Hlocal.col_list,
            model.Hlocal.value_list,
        )
        logger.info(f"E0 single block size {k_unit_cell_size}: {eg_single_block}")
        logger.info(f"E0 {eg_single_block * n_momenta}")
        sim.res["gs_energy"] = eg_single_block
    # -------------------------------------------------------------------------------
    # CONVOLUTIONAL expectation values
    R0 = 0
    # Save the convolution matrix
    shape = (len(k_indices), len(k_indices))
    sim.res["k1k2matrix"] = np.zeros(shape, dtype=np.complex128)
    for k1, k2 in product(k_indices, k_indices):
        # Set the momentum pair
        model.set_momentum_pair([k1], [k2], k_unit_cell_size, TC_symmetry)
        model.default_params()
        # Check the momentum bases
        model.check_momentum_pair()
        # Build the local hamiltonian
        model.build_local_Hamiltonian(g, m, R0, TC_symmetry)
        # Acquire the state vectors
        state_idx_k1 = 1 if (zero_density and k1 == 0) else 0
        state_idx_k2 = 1 if (zero_density and k2 == 0) else 0
        state_idx_k1 += band_number
        state_idx_k2 += band_number
        psik1 = get_data_from_sim(sim_band_name, f"psi{state_idx_k1}", k1)
        psik2 = get_data_from_sim(sim_band_name, f"psi{state_idx_k2}", k2)
        # Measure the overlap with k1 & k2
        sim.res["k1k2matrix"][k1, k2] = exp_val_data2(
            psik1,
            psik2,
            model.Hlocal.row_list,
            model.Hlocal.col_list,
            model.Hlocal.value_list,
        )
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
