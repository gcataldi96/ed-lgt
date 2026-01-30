import os
import sys

# Ensure NUMBA_NUM_THREADS is set properly before importing anything else
B = int(sys.argv[-1])
os.environ["NUMBA_NUM_THREADS"] = str(B)

import numpy as np
from itertools import product
from ed_lgt.models import SU2_Model
from ed_lgt.modeling import mixed_exp_val_data
from simsio import *
from time import perf_counter
import logging

logger = logging.getLogger(__name__)


def get_data_from_sim(sim_filename, obs_name, kindex):
    config_filename = f"new_scattering/{sim_filename}"
    match = SimsQuery(group_glob=config_filename)
    ugrid, _ = uids_grid(match.uids, ["momentum_k_vals"])
    return get_sim(ugrid[kindex]).res[obs_name]


def k_opposite_index(kindx, n_momenta):
    """
    Return the index corresponding to -k in your discrete grid.
    """
    return (-kindx) % n_momenta


def k_canonical(kindx, n_momenta):
    k_op = k_opposite_index(kindx, n_momenta)
    return kindx if kindx <= k_op else k_op


with run_sim() as sim:
    start_time = perf_counter()
    # ----------------------------------------------------------------
    # MODEL INITIALIZATION
    model = SU2_Model(**sim.par["model"])
    zero_density = sim.par["model"]["sectors"][0] == model.n_sites
    logger.info(f"zero density {zero_density}")
    m = sim.par["m"]
    g = sim.par["g"]
    # ----------------------------------------------------------------
    # MOMENTUM SYMMETRY
    TC_symmetry = sim.par.get("TC_symmetry", False)
    k_unit_cell_size = [1] if TC_symmetry else [2]
    n_momenta = model.n_sites if TC_symmetry else model.n_sites // 2
    k_indices = np.arange(0, n_momenta, 1)
    sim.res["k_indices"] = k_indices
    if TC_symmetry:
        k_phys = 2 * np.pi * k_indices / model.n_sites
    else:
        k_phys = 4 * np.pi * k_indices / model.n_sites
    sim.res["k_phys"] = k_phys
    sim_band_name = sim.par["sim_band_name"]
    band_number = sim.par.get("band_number", 0)
    # ----------------------------------------------------------------
    # Ground state energy density at k=0 (unchanged)
    if zero_density:
        GS = get_data_from_sim(sim_band_name, "psi0", 0)
        model.set_momentum_sector(k_unit_cell_size, [0], TC_symmetry)
        model.default_params()
        model.build_local_Hamiltonian(g, m, 0, TC_symmetry)
        eg_single_block = mixed_exp_val_data(
            GS,
            GS,
            model.Hlocal.row_list,
            model.Hlocal.col_list,
            model.Hlocal.value_list,
        )
        logger.info(f"E0 single block size {k_unit_cell_size}: {eg_single_block}")
        logger.info(f"E0 {eg_single_block * n_momenta}")
        sim.res["gs_energy"] = eg_single_block
    else:
        sim.res["gs_energy"] = 2 * (-4.580269235030599 - 1.251e-18j)
    # ----------------------------------------------------------------
    # PARITY OPERATOR (INVERSION SYMMETRY)
    use_parity = sim.par.get("use_parity", True)
    wrt_site = sim.par.get("wrt_site", True)  # True=site, False=bond
    logger.info(f"wrt_site {wrt_site}")
    if use_parity:
        # the local Hamiltonian lives in real space as parity
        model.momentum_basis = None
        model.default_params()
        model.build_local_Hamiltonian(g, m, 0, TC_symmetry)
        # Make it in sparse format
        model.Hlocal.build(format="sparse")
        H0 = model.Hlocal.Ham.copy()
        # Get the parity operator
        model.get_parity_inversion_operator(wrt_site)
    # ----------------------------------------------------------------
    # Convolutional expectation values
    shape = (len(k_indices), len(k_indices))
    sim.res["k1k2matrix"] = np.zeros(shape, dtype=np.complex128)
    for k1, k2 in product(k_indices, k_indices):
        logger.info(f"*************************************************************")
        logger.info(f"K1 {k1} K2 {k2}")
        state_idx_k1 = 1 if (zero_density and k1 == 0) else 0
        state_idx_k1 += band_number
        state_idx_k2 = 1 if (zero_density and k2 == 0) else 0
        state_idx_k2 += band_number
        k1c = k_canonical(k1, n_momenta)
        k2c = k_canonical(k2, n_momenta)
        neg1 = k1 != k1c
        neg2 = k2 != k2c
        # Two approaches: one with parity, one without
        if use_parity and (neg1 or neg2):
            # 1) get canonical states (only use sim data at k1c and k2c)
            psi_k1c = get_data_from_sim(sim_band_name, f"psi{state_idx_k1}", k1c)
            psi_k2c = get_data_from_sim(sim_band_name, f"psi{state_idx_k2}", k2c)
            # 2) build projectors to real space for canonical k's
            # (call set_momentum_pair / _basis_Pk_as_csr only for canonical k's)
            model.set_momentum_sector(k_unit_cell_size, [k1c], TC_symmetry)
            Bk1c = model._basis_Pk_as_csr()
            model.set_momentum_sector(k_unit_cell_size, [k2c], TC_symmetry)
            Bk2c = model._basis_Pk_as_csr()
            psi1_r = Bk1c @ psi_k1c
            psi2_r = Bk2c @ psi_k2c
            # --------------------------------------------------------
            # Parity-aware case for opposite momenta:
            # <k1|P^† H0|k1> = <psi_k1^r|P H0|psi_k1^r>
            # --------------------------------------------------------
            if neg1 and not neg2:
                # <k1c|P H0 |k2c>
                elem = np.vdot(model.parityOP @ psi1_r, H0 @ psi2_r)
            elif not neg1 and neg2:
                # <k1c| H0 P |k2c>
                elem = np.vdot(psi1_r, H0 @ model.parityOP @ psi2_r)
            else:
                # <k1c| P H0 P |k2c>
                elem = np.vdot(model.parityOP @ psi1_r, H0 @ model.parityOP @ psi2_r)
            logger.info(f"ELEM {elem}")
            sim.res["k1k2matrix"][k1, k2] = elem
        else:
            # --------------------------------------------------------
            # Generic case (no parity, or not opposite momenta):
            # use your original momentum-pair local Hamiltonian
            #   <k1|H0|k2> in the (k1,k2)-pair basis.
            # --------------------------------------------------------
            # load states from energy band
            psi_k1 = get_data_from_sim(sim_band_name, f"psi{state_idx_k1}", k1)
            psi_k2 = get_data_from_sim(sim_band_name, f"psi{state_idx_k2}", k2)
            model.set_momentum_pair([k1], [k2], k_unit_cell_size, TC_symmetry)
            model.default_params()
            model.check_momentum_pair()
            model.build_local_Hamiltonian(g, m, 0, TC_symmetry)
            sim.res["k1k2matrix"][k1, k2] = mixed_exp_val_data(
                psi_k1,
                psi_k2,
                model.Hlocal.row_list,
                model.Hlocal.col_list,
                model.Hlocal.value_list,
            )
            logger.info(f"ELEM {sim.res['k1k2matrix'][k1, k2]}")
    # ----------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
