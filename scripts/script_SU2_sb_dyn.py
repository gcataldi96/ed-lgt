import os
import sys

# Ensure NUMBA_NUM_THREADS is set properly before importing anything else
B = int(sys.argv[-1])
# Read the B parameter from command-line arguments
os.environ["NUMBA_NUM_THREADS"] = str(B)

import numpy as np
from ed_lgt.models import DFL_Model
from ed_lgt.modeling import get_lattice_link_site_pairs
from ed_lgt.symmetries import get_symmetry_sector_generators, symmetry_sector_configs
from simsio import run_sim
from time import perf_counter
import logging


logger = logging.getLogger(__name__)

with run_sim() as sim:
    start_time = perf_counter()
    # ==============================================================================
    # MODEL
    model = DFL_Model(**sim.par["model"])
    if model.background < 1:
        bg = 0.75
    else:
        bg = 2
    # ==============================================================================
    # GLOBAL SYMMETRIES
    global_ops = [model.ops["N_tot"]]
    global_sectors = [sim.par["sector"]]
    # GLOBAL OPERATORS
    global_ops = get_symmetry_sector_generators(global_ops, action="global")
    # ==============================================================================
    # ABELIAN LINK SYMMETRIES
    link_ops = [
        [model.ops[f"T2_p{d}"], -model.ops[f"T2_m{d}"]] for d in model.directions
    ]
    link_sectors = [0 for _ in model.directions]
    # LINK OPERATORS
    link_ops = get_symmetry_sector_generators(link_ops, action="link")
    pair_list = get_lattice_link_site_pairs(model.lvals, model.has_obc)
    # ==============================================================================
    # SELECT THE BACKGROUND SYMMETRY SECTOR CONFIGURATION
    if model.lvals == [5, 2]:
        bg_sector = [bg, 0, 0, 0, 0, 0, 0, 0, 0, bg]
    elif model.lvals == [4, 3]:
        bg_sector = [bg, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, bg]
    elif model.lvals == [3, 2]:
        bg_sector = [bg, 0, 0, 0, 0, bg]
    logger.info(f"bg sector configs {bg_sector}")
    # BACKGROUND OPERATOR
    bg_global_ops = get_symmetry_sector_generators([model.ops["bg"]], action="global")
    # ==============================================================================
    # SELECT THE U(1) GLOBAL and LINK SYMMETRY & BACKGROUND STRING SECTOR
    # ==============================================================================
    model.sector_configs = symmetry_sector_configs(
        loc_dims=model.loc_dims,
        glob_op_diags=global_ops,
        glob_sectors=np.array(global_sectors, dtype=float),
        sym_type_flag="U",
        link_op_diags=link_ops,
        link_sectors=link_sectors,
        pair_list=pair_list,
        string_op_diags=bg_global_ops,
        string_sectors=np.array([bg_sector], dtype=float),
    )
    # DEFINE SETTINGS, OBSERVABLES, and BUILD HAMILTONIAN
    model.default_params()
    if sim.par["model"]["spin"] < 1:
        model.build_Hamiltonian(sim.par["g"], sim.par["m"])
    else:
        model.build_gen_Hamiltonian(sim.par["g"], sim.par["m"])
    # ===========================================================================
    # DYNAMICS PARAMETERS
    start = sim.par["dynamics"]["start"]
    stop = sim.par["dynamics"]["stop"]
    delta_t = sim.par["dynamics"]["delta_n"]
    time_line = np.arange(start, stop + delta_t, delta_t)
    sim.res["time_steps"] = time_line
    n_steps = len(sim.res["time_steps"])
    # ===========================================================================
    # OBSERVABLES
    matter_obs = [f"N_{label}" for label in ["tot", "single", "pair", "zero"]]
    extra_obs = ["bg", "T2_px", "T2_py"]
    local_obs = matter_obs + extra_obs
    sim.res["E2"] = np.zeros(n_steps, dtype=float)
    for obs in local_obs:
        sim.res[obs] = np.zeros(n_steps, dtype=float)
    model.get_observables(local_obs)
    # ENTROPY
    partition_indices = sim.par["observables"]["entropy_partition"]
    sim.res["entropy"] = np.zeros(n_steps, dtype=float)
    # ---------------------------------------------------------------------------
    # DYNAMICS: INITIAL STATE PREPARATION
    # ---------------------------------------------------------------------------
    finite_density = int(sim.par["sector"] - model.n_sites)
    model.get_string_breaking_configs(finite_density)
    states_dic = {}
    if sim.par["dynamics"]["initial_state"] == "min0":
        str_type = "min"
        n_strings = model.n_min_strings
    else:
        str_type = "max"
        n_strings = model.n_max_strings
    logger.info(f"{str_type} string configs")
    sim.res[f"tot_ov_{str_type}"] = np.zeros(n_steps, dtype=float)
    for ii in range(n_strings):
        states_dic[f"{str_type}{ii}"] = model.get_qmb_state_from_configs(
            [model.string_cfgs[f"{str_type}{ii}"]]
        )
        sim.res[f"ov_{str_type}{ii}"] = np.zeros(n_steps, dtype=float)
    if sim.par["dynamics"]["time_evolution"]:
        model.time_evolution_Hamiltonian(states_dic[f"{str_type}0"], time_line)
        # -----------------------------------------------------------------------
        for ii, tstep in enumerate(time_line):
            msg_tstep = f"TIME {round(tstep, 4)}"
            msg = f"================= {msg_tstep} ========================="
            logger.info(msg)
            if not model.momentum_basis:
                # ------------------------------------------------------------------
                # ENTROPY
                if sim.par["observables"]["get_entropy"]:
                    sim.res["entropy"][ii] = model.H.psi_time[ii].entanglement_entropy(
                        partition_indices,
                        model.sector_configs,
                    )
                # -------------------------------------------------------------------
                # STATE CONFIGURATIONS
                if sim.par["observables"]["get_state_configs"]:
                    model.H.psi_time[ii].get_state_configurations(
                        1e-2, model.sector_configs
                    )
            # -----------------------------------------------------------------------
            # MEASURE OBSERVABLES
            model.measure_observables(ii, dynamics=True)
            sim.res["E2"][ii] = model.link_avg(model.res["T2_px"], model.res["T2_py"])
            sim.res["N_single"][ii] = model.stag_avg(model.res["N_single"])
            sim.res["N_pair"][ii] += 0.5 * model.stag_avg(model.res["N_pair"], "even")
            sim.res["N_pair"][ii] += 0.5 * model.stag_avg(model.res["N_zero"], "odd")
            sim.res["N_zero"][ii] += 0.5 * model.stag_avg(model.res["N_zero"], "even")
            sim.res["N_zero"][ii] += 0.5 * model.stag_avg(model.res["N_pair"], "odd")
            sim.res["N_tot"][ii] = sim.res["N_single"][ii] + 2 * sim.res["N_pair"][ii]
            logger.info(f"Nsingle {sim.res['N_single'][ii]}")
            logger.info(f"Npair {sim.res['N_pair'][ii]}")
            logger.info(f"Ntot {sim.res['N_tot'][ii]}")
            logger.info(f"E2 {sim.res['E2'][ii]}")
            # -----------------------------------------------------------------------
            # OVERLAPS with the INITIAL STATE & OTHER CONFIGURATIONS
            for kk in range(n_strings):
                sim.res[f"ov_{str_type}{kk}"][ii] = model.measure_fidelity(
                    states_dic[f"{str_type}{kk}"],
                    ii,
                    dynamics=True,
                    print_value=True,
                )
                sim.res[f"tot_ov_{str_type}"][ii] += sim.res[f"ov_{str_type}{kk}"][ii]
            logger.info(f"Tot fidelity {sim.res[f'tot_ov_{str_type}'][ii]}")
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
