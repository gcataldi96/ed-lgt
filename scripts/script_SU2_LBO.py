import os
import sys

# Ensure NUMBA_NUM_THREADS is set properly before importing anything else
B = int(sys.argv[-1])
# Read the B parameter from command-line arguments
os.environ["NUMBA_NUM_THREADS"] = str(B)

import numpy as np
from ed_lgt.models import SU2_Model
from simsio import run_sim
from ed_lgt.modeling import (
    get_projector_for_efficient_density_matrix as project_RDM,
    diagonalize_density_matrix,
)
from time import perf_counter
import logging

logger = logging.getLogger(__name__)
with run_sim() as sim:
    start_time = perf_counter()
    # -------------------------------------------------------------------------------
    # MODEL HAMILTONIAN
    model = SU2_Model(**sim.par["model"])
    m = sim.par["m"] if not model.pure_theory else None
    if model.spin > 0.5:
        model.build_gen_Hamiltonian(sim.par["g"], m)
    else:
        model.build_Hamiltonian(sim.par["g"], m)
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
    n_eigs = sim.par["hamiltonian"]["n_eigs"]
    model.diagonalize_Hamiltonian(n_eigs, model.ham_format)
    sim.res["energy"] = model.H.Nenergies
    # -------------------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = [f"T2_p{d}" for d in model.directions]
    local_obs += ["E_square"]
    if not model.pure_theory:
        local_obs += [f"N_{label}" for label in ["tot", "single", "pair", "zero"]]
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
                model.H.Npsi[ii].get_state_configurations(1.5e-3, model.sector_configs)
        # -----------------------------------------------------------------------
        # MEASURE OBSERVABLES
        if sim.par["observables"]["measure_obs"]:
            model.measure_observables(ii)
            sim.res["E_square"][ii] = model.stag_avg(model.res["N_single"])
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
    # EFFECTIVE MODEL OUT OF THE LOCAL BASIS OPTIMIZATION
    # -------------------------------------------------------------------------------
    if sim.par["observables"]["basis_reduction"]:
        # Get the reduced density matrix of a single site in the ground state
        RDM = model.H.Npsi[0].reduced_density_matrix(
            partition_indices,
            model.sector_configs,
        )
        rho_eigvals, rho_eigvecs = diagonalize_density_matrix(RDM)
        # -------------------------------------------------------------------------------
        # For each truncation value, we will build the effective model
        truncation_values = [
            1e-2,
            1e-3,
            1e-4,
            1e-5,
            1e-6,
            1e-7,
            1e-8,
            1e-9,
            1e-10,
            1e-11,
            1e-12,
            1e-13,
            1e-14,
        ]
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
            if 3 < proj.shape[1] < max(model.loc_dims):
                sim.res["eff_basis"][tt] = proj.shape[1]
                # build the effective model Hamiltonian
                eff_model = SU2_Model(**sim.par["model"], basis_projector=proj)
                eff_model.build_gen_Hamiltonian(sim.par["g"], m)
                # diagonalize the effective Hamiltonian and save energy eigvals
                eff_model.diagonalize_Hamiltonian(n_eigs, eff_model.ham_format)
                sim.res["eff_energy"][tt] = eff_model.H.Nenergies
                eff_model.get_observables(
                    local_obs=local_obs, plaquette_obs=plaquette_obs
                )
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
                                eff_model.sector_configs,
                            )
                        # -----------------------------------------------------------------------
                        # STATE CONFIGURATIONS
                        if sim.par["observables"]["get_state_configs"]:
                            eff_model.H.Npsi[ii].get_state_configurations(
                                1.5e-3, eff_model.sector_configs
                            )
                    # ----------------------------------------------------------------------------+
                    # MEASURE OBSERVABLES
                    eff_model.measure_observables(ii)
                    sim.res["eff_E_square"][tt, ii] = eff_model.stag_avg(
                        eff_model.res["E_square"]
                    )
                    if not model.pure_theory:
                        sim.res["eff_N_single"][tt, ii] = eff_model.stag_avg(
                            eff_model.res["N_single"]
                        )
                        sim.res["eff_N_pair"][tt, ii] += 0.5 * eff_model.stag_avg(
                            eff_model.res["N_pair"], "even"
                        )
                        sim.res["eff_N_pair"][tt, ii] += 0.5 * eff_model.stag_avg(
                            eff_model.res["N_zero"], "odd"
                        )
                        sim.res["eff_N_zero"][tt, ii] += 0.5 * eff_model.stag_avg(
                            eff_model.res["N_zero"], "even"
                        )
                        sim.res["eff_N_zero"][tt, ii] += 0.5 * eff_model.stag_avg(
                            eff_model.res["N_pair"], "odd"
                        )
                        sim.res["eff_N_tot"][tt, ii] = (
                            sim.res["eff_N_single"][tt, ii]
                            + 2 * sim.res["eff_N_pair"][tt, ii]
                        )
                    for obs_names_list in plaquette_obs:
                        obs = "_".join(obs_names_list)
                        sim.res[f"eff_{obs}"][tt, ii] = eff_model.res[obs]
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
