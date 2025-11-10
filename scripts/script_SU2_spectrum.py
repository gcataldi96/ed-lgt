import os
import sys

# Ensure NUMBA_NUM_THREADS is set properly before importing anything else
B = int(sys.argv[-1])
# Read the B parameter from command-line arguments
os.environ["NUMBA_NUM_THREADS"] = str(B)

import numpy as np
from ed_lgt.models import SU2_Model
from ed_lgt.modeling import diagonalize_density_matrix
from simsio import run_sim
from time import perf_counter
import logging

logger = logging.getLogger(__name__)
with run_sim() as sim:
    start_time = perf_counter()
    # -------------------------------------------------------------------------------
    # Initialize the model
    model = SU2_Model(**sim.par["model"])
    m = sim.par["m"] if not model.pure_theory else None
    # Select Momentum Sector
    if sim.par["momentum"]["get_momentum_basis"]:
        unit_cell_size = sim.par["momentum"]["unit_cell_size"]
        k_vals = sim.par["momentum_k_vals"]
        TC_symmetry = sim.par["momentum"]["TC_symmetry"]
        model.get_momentum_sector(unit_cell_size, k_vals, TC_symmetry)
    # Save parameters
    model.default_params()
    # Build Hamiltonian
    if model.spin > 0.5:
        model.build_gen_Hamiltonian(sim.par["g"], m)
    else:
        model.build_Hamiltonian(sim.par["g"], m)
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
    n_eigs = sim.par["hamiltonian"]["n_eigs"]
    model.diagonalize_Hamiltonian(n_eigs, model.ham_format)
    sim.res["energy"] = model.H.Nenergies
    if sim.par["hamiltonian"]["save_psi"]:
        for ii in range(n_eigs):
            sim.res[f"psi{ii}"] = model.H.Npsi[ii].psi
    # -------------------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = [f"T2_p{d}" for d in model.directions]
    if not model.pure_theory:
        local_obs += [f"N_{label}" for label in ["tot", "single", "pair", "zero"]]
    sim.res["E2"] = np.zeros(n_eigs, dtype=float)
    for obs in local_obs:
        sim.res[obs] = np.zeros(n_eigs, dtype=float)
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = []
    twobody_axes = []
    # LIST OF PLAQUETTE OPERATORS
    if model.spin < 1:
        if model.dim == 2:
            plaquette_obs = [["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]]
        elif model.dim == 3:
            plaquette_obs = [
                ["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"],
                ["C_px,pz", "C_pz,mx", "C_mz,px", "C_mx,mz"],
                ["C_py,pz", "C_pz,my", "C_mz,py", "C_my,mz"],
            ]
        else:
            plaquette_obs = []
    else:
        plaquette_obs = []
    for obs_names_list in plaquette_obs:
        obs = "_".join(obs_names_list)
        sim.res[obs] = np.zeros(n_eigs, dtype=float)
    # DEFINE OBSERVABLES
    model.get_observables(
        local_obs, twobody_obs, plaquette_obs, twobody_axes=twobody_axes
    )
    # QUENCH STATE FOR OVERLAP
    if sim.par["observables"]["get_overlap"]:
        name = sim.par["hamiltonian"]["state"]
        config = model.overlap_QMB_state(name)
        logger.info(f"config {config}")
        in_state = model.get_qmb_state_from_configs([config])
        sim.res["overlap"] = np.zeros(model.H.n_eigs, dtype=float)
    # -------------------------------------------------------------------------------
    # ENTROPY
    # DEFINE THE PARTITION FOR THE ENTANGLEMENT ENTROPY
    partition_indices = sim.par["observables"]["entropy_partition"]
    sim.res["entropy"] = np.zeros(model.H.n_eigs, dtype=float)
    # -------------------------------------------------------------------------------
    for ii in range(model.H.n_eigs):
        model.H.print_energy(ii)
        if model.momentum_basis is None:
            # -------------------------------------------------------------------------------
            # REDUCED DENSITY MATRIX
            if sim.par["observables"]["get_RDM"]:
                # Get the reduced density matrix of a partition in the ground state
                RDM = model.H.Npsi[ii].reduced_density_matrix(
                    partition_indices,
                    model.sector_configs,
                )
                logger.info(f"RDM shape {RDM.shape}")
                rho_eigvals, rho_eigvecs = diagonalize_density_matrix(RDM)
                # Sort eigenvalues and eigenvectors in descending order.
                # Note: np.argsort sorts in ascending order; we reverse to get descending order.
                sorted_indices = np.argsort(rho_eigvals)[::-1]
                rho_eigvals = rho_eigvals[sorted_indices]
                logger.info(f"eigvals {rho_eigvals}")
                sim.res["eigvals"] = rho_eigvals
                rho_eigvecs = rho_eigvecs[:, sorted_indices]
            # -----------------------------------------------------------------------
            # ENTROPY
            if sim.par["observables"]["get_entropy"]:
                sim.res["entropy"][ii] = model.H.Npsi[ii].entanglement_entropy(
                    partition_indices,
                    model.sector_configs,
                )
            # -----------------------------------------------------------------------
            # STATE CONFIGURATIONS
            if sim.par["observables"]["get_state_configs"]:
                model.H.Npsi[ii].get_state_configurations(1e-3, model.sector_configs)
        # -----------------------------------------------------------------------
        # MEASURE OBSERVABLES
        if sim.par["observables"]["measure_obs"]:
            model.measure_observables(ii)
            sim.res["E2"][ii] = model.link_avg(obs_name="T2")
            if not model.pure_theory:
                sim.res["N_single"][ii] = model.stag_avg(model.res["N_single"])
                sim.res["N_pair"][ii] += 0.5 * model.stag_avg(
                    model.res["N_pair"], "even"
                )
                sim.res["N_pair"][ii] += 0.5 * model.stag_avg(
                    model.res["N_zero"], "odd"
                )
                sim.res["N_zero"][ii] += 0.5 * model.stag_avg(
                    model.res["N_zero"], "even"
                )
                sim.res["N_zero"][ii] += 0.5 * model.stag_avg(
                    model.res["N_pair"], "odd"
                )
                sim.res["N_tot"][ii] = (
                    sim.res["N_single"][ii] + 2 * sim.res["N_pair"][ii]
                )
        # ---------------------------------------------------------------------------
        # OVERLAPS with the INITIAL STATE
        if sim.par["observables"]["get_overlap"]:
            sim.res["overlap"][ii] = model.measure_fidelity(
                in_state, ii, print_value=True
            )
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
