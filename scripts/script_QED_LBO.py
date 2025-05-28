import numpy as np
from ed_lgt.models import QED_Model
from simsio import run_sim
from time import perf_counter
from ed_lgt.modeling import (
    get_projector_for_efficient_density_matrix as project_RDM,
    diagonalize_density_matrix,
)
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
    # ---------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = ["E_square"]
    local_obs += [f"E_{s}{d}" for d in model.directions for s in "mp"]
    if not model.pure_theory:
        local_obs += ["N"]
    for obs in local_obs:
        sim.res[obs] = np.zeros(n_eigs, dtype=float)
        sim.res[f"eff_{obs}"] = np.zeros(n_eigs, dtype=float)
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
        sim.res[f"eff_{obs}"] = np.zeros(n_eigs, dtype=float)
    # DEFINE OBSERVABLES
    model.get_observables(local_obs=local_obs, plaquette_obs=plaquette_obs)
    # ENTROPY
    # DEFINE THE PARTITION FOR THE ENTANGLEMENT ENTROPY
    partition_indices = sim.par["observables"]["entropy_partition"]
    # Build the list of environment and subsystem sites configurations
    model.get_subsystem_environment_configs(keep_indices=partition_indices)
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
                    model.subsystem_configs,
                    model.env_configs,
                    model.unique_subsys_configs,
                    model.unique_env_configs,
                )
            # -----------------------------------------------------------------------
            # STATE CONFIGURATIONS
            if sim.par["observables"]["get_state_configs"]:
                model.H.Npsi[ii].get_state_configurations(1e-1, model.sector_configs)
        # ---------------------------------------------------------------------------
        # MEASURE OBSERVABLES
        model.measure_observables(ii)
        for obs in local_obs:
            sim.res[obs][ii] = np.mean(model.res[obs])
        for obs_names_list in plaquette_obs:
            obs = "_".join(obs_names_list)
            sim.res[obs][ii] = model.res[obs]
    # -------------------------------------------------------------------------------
    # EFFECTIVE MODEL OUT OF THE LOCAL BASIS OPTIMIZATION
    # -------------------------------------------------------------------------------
    # Get the reduced density matrix of a single site in the ground state
    RDM = model.H.Npsi[0].reduced_density_matrix(
        partition_indices,
        model.subsystem_configs,
        model.env_configs,
        model.unique_subsys_configs,
        model.unique_env_configs,
    )
    rho_eigvals, rho_eigvecs = diagonalize_density_matrix(RDM)
    # -------------------------------------------------------------------------------
    # For each truncation value, we will build the effective model
    truncation_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    n_trunc = len(truncation_values)
    sim.res["eff_basis"] = np.zeros(n_trunc, dtype=float)
    for obs in local_obs + ["entropy", "energy"]:
        sim.res[f"eff_{obs}"] = np.zeros((n_trunc, n_eigs), dtype=float)
    for obs_names_list in plaquette_obs:
        obs = "_".join(obs_names_list)
        sim.res[f"eff_{obs}"] = np.zeros((n_trunc, n_eigs), dtype=float)
    # -------------------------------------------------------------------------------
    for tt, truncation in enumerate(truncation_values):
        # Get the reduced and optimized operators of a single site in the ground state
        proj = project_RDM(rho_eigvals, rho_eigvecs, truncation)
        if proj.shape[1] < max(model.loc_dims):
            sim.res["eff_basis"][tt] = proj.shape[1]
            # build the effective model Hamiltonian
            eff_model = QED_Model(**sim.par["model"], basis_projector=proj)
            eff_model.build_Hamiltonian(sim.par["g"], m, theta=sim.par["theta"])
            # diagonalize the effective Hamiltonian and save energy eigvals
            eff_model.diagonalize_Hamiltonian(n_eigs, eff_model.ham_format)
            sim.res[f"eff_energy"][tt] = eff_model.H.Nenergies
            eff_model.get_observables(local_obs=local_obs, plaquette_obs=plaquette_obs)
            # Build the list of environment and subsystem sites configurations
            eff_model.get_subsystem_environment_configs(keep_indices=partition_indices)
            # -------------------------------------------------------------------------------
            for ii in range(eff_model.H.n_eigs):
                eff_model.H.print_energy(ii)
                if not eff_model.momentum_basis:
                    # -----------------------------------------------------------------------
                    # ENTROPY
                    if sim.par["observables"]["get_entropy"]:
                        sim.res["eff_entropy"][tt, ii] = eff_model.H.Npsi[
                            ii
                        ].entanglement_entropy(
                            partition_indices,
                            eff_model.subsystem_configs,
                            eff_model.env_configs,
                            eff_model.unique_subsys_configs,
                            eff_model.unique_env_configs,
                        )
                    # -----------------------------------------------------------------------
                    # STATE CONFIGURATIONS
                    if sim.par["observables"]["get_state_configs"]:
                        eff_model.H.Npsi[ii].get_state_configurations(
                            1e-1, eff_model.sector_configs
                        )
                # ----------------------------------------------------------------------------+
                # MEASURE OBSERVABLES
                eff_model.measure_observables(ii)
                for obs in local_obs:
                    sim.res[f"eff_{obs}"][tt, ii] = np.mean(eff_model.res[obs])
                for obs_names_list in plaquette_obs:
                    obs = "_".join(obs_names_list)
                    sim.res[f"eff_{obs}"][tt, ii] = eff_model.res[obs]
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
