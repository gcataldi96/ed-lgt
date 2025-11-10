import os
import sys

# Ensure NUMBA_NUM_THREADS is set properly before importing anything else
B = int(sys.argv[-1])
# Read the B parameter from command-line arguments
os.environ["NUMBA_NUM_THREADS"] = str(B)

import numpy as np
from scipy.sparse import csr_array
from ed_lgt.models import SU2_Model
from ed_lgt.modeling import QMB_state
from ed_lgt.tools import (
    get_data_from_sim,
    get_Wannier_support,
    localize_Wannier,
    operator_to_mpo_via_mps,
)
from ed_lgt.symmetries import build_sector_expansion_projector
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
    m = sim.par["m"]
    g = sim.par["g"]
    # -------------------------------------------------------------------------------
    # Save parameters
    model.default_params()
    # Build Hamiltonian
    if model.spin > 0.5:
        model.build_gen_Hamiltonian(g, m)
    else:
        model.build_Hamiltonian(g, m)
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN to get the GLOBAL GROUND STATE
    model.diagonalize_Hamiltonian(1, model.ham_format)
    GS = model.H.Npsi[0]
    # -------------------------------------------------------------------------------
    # Acquire the optimal theta phases that localize the Wannier
    Eprofile, _, theta_phases = localize_Wannier("convolution1_N0", center_mode=1)
    # Get the partition to the model according to the optimal support of the Wannier
    w_supports = get_Wannier_support(Eprofile, epsilons=(1e-3, 1e-4))
    support_indices = w_supports["supports"][0.001]
    model._get_partition(support_indices)
    # Initialize the Wannier State
    psi_wannier = np.zeros(model.sector_configs.shape[0], dtype=np.complex128)
    # Simulation band name where to extract the momentum states
    sim_band_name = sim.par["sim_band_name"]
    band_number = sim.par.get("band_number", 0)
    state_idx = 1 + band_number
    # -------------------------------------------------------------------------------
    # Define the momentum grid
    TC_symmetry = sim.par.get("TC_symmetry", False)
    k_unit_cell_size = [1] if TC_symmetry else [2]
    n_momenta = model.n_sites if TC_symmetry else model.n_sites // 2
    k_indices = np.arange(0, n_momenta, 1)
    # -------------------------------------------------------------------------------
    for kidx in k_indices:
        # Load the momentum state forming the energy band
        psik = get_data_from_sim(sim_band_name, f"psi{state_idx}", kidx)
        # Set the corresponding momentum sector
        model.get_momentum_sector(k_unit_cell_size, [kidx], TC_symmetry)
        model.default_params()
        # Build the projector from the momentum sector to the global one
        Pk = model._basis_Pk_as_csr()
        # Project the State from the momentum sector to the coordinate one
        psik_exp = Pk @ psik
        # Add it to the Wannier state with the corresponding theta phase
        psi_wannier += np.exp(1j * theta_phases[kidx]) * psik_exp / np.sqrt(n_momenta)
    # Promote the Wannier state as an item of the QMB state class
    Wannier = QMB_state(psi=psi_wannier, lvals=model.lvals, loc_dims=model.loc_dims)
    W_psimatrix = Wannier._get_psi_matrix(support_indices, model._partition_cache)
    # Promote the Ground State as an item of the QMB state class
    GS_psimatrix = GS._get_psi_matrix(support_indices, model._partition_cache)
    # ------------------------------------------------------------------
    # Build the cross operator on the support Tr_{Env}|Wannier><GS|
    qp_operator = W_psimatrix.conj().T @ GS_psimatrix
    logger.info(f"quasi-particle creation operator {qp_operator.shape}")
    # ------------------------------------------------------------------
    # Build the operator that promotes the symmetry sector to the global space
    logger.info("Build the projector from symmetry-sector to full space")
    P = build_sector_expansion_projector(
        model._partition_cache[tuple(sorted(support_indices))]["unique_subsys_configs"],
        model.loc_dims[support_indices],
    )
    # Get the MPO
    support_loc_dims = list(model.loc_dims[support_indices])
    MPO = operator_to_mpo_via_mps(
        operator=qp_operator,
        projector=P,
        loc_dims=support_loc_dims,
        op_svd_rel_tol=1e-4,  # controls rank R of operator
        max_rank=10,  # hard safety cap
        mps_chi_max=16,  # controls entanglement per vector
        mps_svd_rel_tol=1e-6,  # per-bond truncation inside MPS
    )
    for ii, site in enumerate(support_loc_dims):
        sim.res[f"MPO[{ii}]"] = MPO[ii]
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
