import os
import sys

# Ensure NUMBA_NUM_THREADS is set properly before importing anything else
B = int(sys.argv[-1])
# Read the B parameter from command-line arguments
os.environ["NUMBA_NUM_THREADS"] = str(B)

import numpy as np
from ed_lgt.models import DFL_Model
from ed_lgt.tools import stag_avg
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
    # ==============================================================================
    # GLOBAL SYMMETRIES
    global_ops = [model.ops["N_tot"]]
    global_sectors = [model.n_sites]
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
    bg_sector = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
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
    model.build_Hamiltonian(sim.par["g"], sim.par["m"])
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
    extra_obs = ["bg", "E_square", "T2_px", "T2_py"]
    local_obs = matter_obs + extra_obs
    for obs in local_obs:
        sim.res[obs] = np.zeros(n_steps, dtype=float)
    model.get_observables(local_obs)
    # ENTROPY
    partition_indices = sim.par["observables"]["entropy_partition"]
    sim.res["entropy"] = np.zeros(n_steps, dtype=float)
    # ---------------------------------------------------------------------------
    # DYNAMICS: INITIAL STATE PREPARATION
    # ---------------------------------------------------------------------------
    logger.info("----------------------------------------------------")
    logger.info(f"Minimal string configs")
    strings_dic = {
        "cfg_snake": np.array([6, 10, 2, 10, 1, 5, 3, 10, 3, 11], dtype=int),
        "cfg0": np.array([7, 12, 3, 12, 1, 4, 0, 9, 0, 11], dtype=int),
        "cfg1": np.array([7, 12, 3, 11, 0, 4, 0, 9, 1, 12], dtype=int),
        "cfg2": np.array([7, 12, 2, 9, 0, 4, 0, 10, 2, 12], dtype=int),
        "cfg3": np.array([7, 11, 0, 9, 0, 4, 1, 11, 2, 12], dtype=int),
        "cfg4": np.array([6, 9, 0, 9, 0, 5, 2, 11, 2, 12], dtype=int),
    }
    for ii in range(5):
        strings_dic[f"state{ii}"] = model.get_qmb_state_from_configs(
            [strings_dic[f"cfg{ii}"]]
        )
        sim.res[f"overlap{ii}"] = np.zeros(n_steps, dtype=float)
    logger.info(f"Maximal string config")
    strings_dic[f"state_snake"] = model.get_qmb_state_from_configs(
        [strings_dic["cfg_snake"]]
    )
    sim.res[f"overlap_snake"] = np.zeros(n_steps, dtype=float)
    if sim.par["dynamics"]["time_evolution"]:
        model.time_evolution_Hamiltonian(strings_dic["state_snake"], time_line)
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
            sim.res["E_square"][ii] = model.link_avg(
                model.res["T2_px"], model.res["T2_py"]
            )
            sim.res["N_single"][ii] = stag_avg(model.res["N_single"])
            sim.res["N_pair"][ii] += 0.5 * stag_avg(model.res["N_pair"], "even")
            sim.res["N_pair"][ii] += 0.5 * stag_avg(model.res["N_zero"], "odd")
            sim.res["N_zero"][ii] += 0.5 * stag_avg(model.res["N_zero"], "even")
            sim.res["N_zero"][ii] += 0.5 * stag_avg(model.res["N_pair"], "odd")
            sim.res["N_tot"][ii] = sim.res["N_single"][ii] + 2 * sim.res["N_pair"][ii]
            # -----------------------------------------------------------------------
            # OVERLAPS with the INITIAL STATE & OTHER CONFIGURATIONS
            sim.res[f"overlap_snake"][ii] = model.measure_fidelity(
                strings_dic[f"state_snake"], ii, dynamics=True, print_value=True
            )
            for kk in range(5):
                sim.res[f"overlap{kk}"][ii] = model.measure_fidelity(
                    strings_dic[f"state{kk}"], ii, dynamics=True, print_value=True
                )
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
