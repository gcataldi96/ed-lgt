import os
import sys

# Ensure NUMBA_NUM_THREADS is set properly before importing anything else
B = int(sys.argv[-1])
# Read the B parameter from command-line arguments
os.environ["NUMBA_NUM_THREADS"] = str(B)

import numpy as np
from ed_lgt.models import QED_Model
from simsio import run_sim
from time import perf_counter
from ed_lgt.modeling import diagonalize_density_matrix
import logging

logger = logging.getLogger(__name__)
with run_sim() as sim:
    # Start measuring time
    start_time = perf_counter()
    # -------------------------------------------------------------------------------
    # MODEL HAMILTONIAN
    model = QED_Model(**sim.par["model"])
    m = sim.par["m"] if not model.pure_theory else None
    model.build_Hamiltonian(sim.par["g"], m, theta=sim.par["theta"])
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
    n_eigs = sim.par["hamiltonian"]["n_eigs"]
    model.diagonalize_Hamiltonian(n_eigs, model.ham_format)
    sim.res["energy"] = model.H.Nenergies
    # -------------------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = ["E_square"]
    local_obs += [f"E_{s}{d}" for d in model.directions for s in "mp"]
    if not model.pure_theory:
        local_obs += ["N"]
    for obs in local_obs:
        sim.res[obs] = np.zeros(n_eigs, dtype=float)
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = []
    twobody_axes = []
    # LIST OF PLAQUETTE OPERATORS
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
    for obs_names_list in plaquette_obs:
        obs = "_".join(obs_names_list)
        sim.res[obs] = np.zeros(n_eigs, dtype=float)
    # DEFINE OBSERVABLES
    model.get_observables(
        local_obs, twobody_obs, plaquette_obs, twobody_axes=twobody_axes
    )
    # -------------------------------------------------------------------------------
    # ENTROPY
    # DEFINE THE PARTITION FOR THE ENTANGLEMENT ENTROPY
    partition_indices = sim.par["observables"]["entropy_partition"]
    sim.res["entropy"] = np.zeros(model.H.n_eigs, dtype=float)
    # -------------------------------------------------------------------------------
    for ii in range(model.H.n_eigs):
        model.H.print_energy(ii)
        if not model.momentum_basis:
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
                model.H.Npsi[ii].get_state_configurations(1e-2, model.sector_configs)
        # ---------------------------------------------------------------------------
        # MEASURE OBSERVABLES
        model.measure_observables(ii)
        for obs in local_obs:
            sim.res[obs][ii] = np.mean(model.res[obs])
        for obs_names_list in plaquette_obs:
            obs = "_".join(obs_names_list)
            sim.res[obs][ii] = model.res[obs]
    # -------------------------------------------------------------------------------
    if sim.par["observables"]["get_RDM"]:
        # Get the reduced density matrix of a partition in the ground state
        RDM = model.H.Npsi[0].reduced_density_matrix(
            partition_indices,
            model.sector_configs,
        )
        logger.info(f"RDM shape {RDM.shape}")
        rho_eigvals, rho_eigvecs = diagonalize_density_matrix(RDM)
        # Sort eigenvalues and eigenvectors in descending order.
        # Note: np.argsort sorts in ascending order; we reverse to get descending order.
        sorted_indices = np.argsort(rho_eigvals)[::-1]
        rho_eigvals = rho_eigvals[sorted_indices]
        sim.res["eigvals"] = rho_eigvals
        rho_eigvecs = rho_eigvecs[:, sorted_indices]
        # Set a list of truncation values to reduce the RDM
        truncation_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        for tt, threshold in enumerate(truncation_values):
            # Determine how many eigenvectors have eigenvalues > the threshold.
            # If too few are significant, relax the threshold until at least 2 are selected.
            P_columns = np.sum(rho_eigvals > threshold)
            while P_columns < 2:
                threshold /= 10
                P_columns = np.sum(rho_eigvals > threshold)
            logger.info(f"SIGNIFICANT EIGENVALUES {P_columns} > {threshold}")
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
