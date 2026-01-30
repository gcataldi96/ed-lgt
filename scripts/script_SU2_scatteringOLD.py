import os
import sys

# Ensure NUMBA_NUM_THREADS is set properly before importing anything else
B = int(sys.argv[-1])
# Read the B parameter from command-line arguments
os.environ["NUMBA_NUM_THREADS"] = str(B)

import numpy as np
from itertools import product
from ed_lgt.models import SU2_Model
from ed_lgt.modeling import mixed_exp_val_data
from simsio import *
from time import perf_counter
import logging

logger = logging.getLogger(__name__)
with run_sim() as sim:
    start_time = perf_counter()
    # -------------------------------------------------------------------------------
    # SU(2) MODEL in (1+1)D with MOMENTUM SYMMETRY
    model = SU2_Model(**sim.par["model"])
    zero_density = False if sim.par["model"]["sectors"][0] != model.n_sites else True
    logger.info(f"zero density {zero_density}")
    m = sim.par["m"] if not model.pure_theory else None
    # Build momentum grid
    kdict = {}
    # Choose if TC symmetry is enabled
    TC_symmetry = sim.par.get("TC_symmetry", False)
    k_unit_cell_size = [1] if TC_symmetry else [2]
    n_momenta = model.n_sites if TC_symmetry else model.n_sites // 2
    k_vals = np.arange(0, n_momenta, 1)
    sim.res["kvals"] = k_vals
    if TC_symmetry:
        k_phys = 2 * np.pi * k_vals / model.n_sites
    else:
        k_phys = 4 * np.pi * k_vals / model.n_sites
    # -------------------------------------------------------------------------------
    for kidx in k_vals:
        kdict[f"{kidx}"] = {}
        # Set the momentum basis on the model
        model.set_momentum_sector(k_unit_cell_size, [kidx], TC_symmetry)
        model.default_params()
        # Generate HAMILTONIAN
        if model.spin > 0.5:
            model.build_gen_Hamiltonian(sim.par["g"], m)
        else:
            model.build_Hamiltonian(sim.par["g"], m)
        # ----------------------------------------------------------------------------
        # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
        n_eigs = 2 if (kidx == 0 and zero_density) else 1
        model.diagonalize_Hamiltonian(n_eigs, model.ham_format)
        # Save the states of the energy band
        idx = 1 if (kidx == 0 and zero_density) else 0
        kdict[f"{kidx}"]["psi"] = model.H.Npsi[idx].psi
        if kidx == 0 and zero_density:
            kdict[f"{kidx}"]["gs"] = model.H.Npsi[0].psi
        for ii in range(model.H.n_eigs):
            model.H.print_energy(ii)
    # -------------------------------------------------------------------------------
    if zero_density:
        GS = kdict["0"]["gs"]
        # Check Translational Hamiltonian
        model.set_momentum_pair([0], [0], k_unit_cell_size, TC_symmetry)
        model.default_params()
        for ii in range(n_momenta):
            # Build the local hamiltonian
            model.build_local_Hamiltonian(sim.par["g"], m, ii, TC_symmetry)
            eg_single_block = mixed_exp_val_data(
                GS,
                GS,
                model.Hlocal.row_list,
                model.Hlocal.col_list,
                model.Hlocal.value_list,
            )
            logger.info(f"E0 single block size {k_unit_cell_size}: {eg_single_block}")
            logger.info(f"E0 {eg_single_block * n_momenta}")
            if ii == 0:
                sim.res["gs_energy"] = eg_single_block
    # -------------------------------------------------------------------------------
    # CONVOLUTIONAL expectation values
    R0 = 0
    kdict["overlaps"] = np.zeros((len(k_vals), len(k_vals)), dtype=np.complex128)
    for k1, k2 in product(k_vals, k_vals):
        model.set_momentum_pair([k1], [k2], k_unit_cell_size, TC_symmetry)
        model.default_params()
        model.check_momentum_pair()
        # Build the local hamiltonian
        model.build_local_Hamiltonian(sim.par["g"], m, R0, TC_symmetry)
        # Measure the overlap with k1 & k2
        kdict["overlaps"][k1, k2] = mixed_exp_val_data(
            kdict[f"{k1}"]["psi"],
            kdict[f"{k2}"]["psi"],
            model.Hlocal.row_list,
            model.Hlocal.col_list,
            model.Hlocal.value_list,
        )
    # Save the convolution matrix
    sim.res["k1k2matrix"] = kdict["overlaps"]
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
