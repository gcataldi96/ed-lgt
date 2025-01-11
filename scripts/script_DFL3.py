import numpy as np
import os
from numba import set_num_threads
from ed_lgt.models import DFL_Model
from ed_lgt.modeling import get_lattice_link_site_pairs, get_entropy_partition
from ed_lgt.symmetries import get_symmetry_sector_generators, symmetry_sector_configs
from simsio import run_sim
from time import perf_counter
import logging


logger = logging.getLogger(__name__)

with run_sim() as sim:
    # Set the number of threads per simulation
    set_num_threads(int(os.environ.get("NUMBA_NUM_THREADS", sim.par["n_threads"])))
    start_time = perf_counter()
    # ==============================================================================
    # DYNAMICS PARAMETERS
    start = sim.par["dynamics"]["start"]
    stop = sim.par["dynamics"]["stop"]
    delta_t = sim.par["dynamics"]["delta_n"]
    time_line = np.arange(start, stop + delta_t, delta_t)
    sim.res["time_steps"] = time_line
    n_steps = len(sim.res["time_steps"])
    # ==============================================================================
    # MODEL PROPERTIES
    model = DFL_Model(**sim.par["model"])
    m = sim.par["m"] if not model.pure_theory else None
    # ===============================================================================
    # OBSERVABLES
    if not model.pure_theory:
        local_obs = [f"N_{label}" for label in ["tot", "single"]] + ["S2_matter"]
        for measure in local_obs:
            sim.res[measure] = np.zeros(n_steps, dtype=float)
    # Store the observables
    partition_indices = get_entropy_partition(model.lvals)
    for measure in ["delta", "entropy", "overlap"][:1]:
        sim.res[measure] = np.zeros(n_steps, dtype=float)
    sim.res["microcan_avg"] = 0.0
    sim.res["micro_Nsingle"] = 0.0
    sim.res["micro_S2_matter"] = 0.0
    # ==============================================================================
    # GLOBAL SYMMETRIES
    global_ops = [model.ops["N_tot"]]
    global_sectors = [model.n_sites]
    # GLOBAL OPERATORS
    global_ops = get_symmetry_sector_generators(
        global_ops,
        loc_dims=model.loc_dims,
        action="global",
        gauge_basis=model.gauge_basis,
        lattice_labels=model.lattice_labels,
    )
    # ==============================================================================
    # ABELIAN Z2 SYMMETRIES
    link_ops = [
        [model.ops[f"T2_p{d}"], -model.ops[f"T2_m{d}"]] for d in model.directions
    ]
    link_sectors = [0 for _ in model.directions]
    # LINK OPERATORS
    link_ops = get_symmetry_sector_generators(
        link_ops,
        loc_dims=model.loc_dims,
        action="link",
        gauge_basis=model.gauge_basis,
        lattice_labels=model.lattice_labels,
    )
    pair_list = get_lattice_link_site_pairs(model.lvals, model.has_obc)
    # ==============================================================================
    # ENUMERATE ALL THE BACKGROUND SYMMETRY SECTORS
    logical_stag_basis = sim.par["dynamics"]["logical_stag_basis"]
    bg_configs, bg_sectors = model.get_background_charges_configs(logical_stag_basis)
    logger.info(f"charge sector configs {bg_sectors.shape[0]}")
    # Calculate the norm_scalar product with the matter config of the initial state
    if model.n_sites % (2 * logical_stag_basis) != 0:
        raise ValueError(f"the staggered basis is not compatible with n_sites")
    # -------------------------------------------------------------------------------
    num_blocks = model.n_sites // (2 * logical_stag_basis)
    stag_array = np.array(
        [-1] * logical_stag_basis + [1] * logical_stag_basis, dtype=int
    )
    norm_scalar_product = np.tile(stag_array, num_blocks)
    logger.info(f"norm scalar product {norm_scalar_product}")
    # ==============================================================================
    # STRING BACKGROUND SYMMETRY OPERATORS
    bg_global_ops = get_symmetry_sector_generators(
        [model.ops["bg"]],
        loc_dims=model.loc_dims,
        action="global",
        gauge_basis=model.gauge_basis,
        lattice_labels=model.lattice_labels,
    )
    # Step 1: Find unique rows and their indices
    unique_bg_sectors, indices = np.unique(bg_sectors, axis=0, return_inverse=True)
    num_bg_sectors = len(unique_bg_sectors)
    # Step 2: Define the array where bg configs are stored according to their associated sector
    bg_configs_per_sector = np.zeros((num_bg_sectors, 2, model.n_sites), dtype=int)
    counts = np.zeros(num_bg_sectors, dtype=int)
    # Step 3: Group rows from the second array based on the indices
    for idx, group_id in enumerate(indices):
        bg_configs_per_sector[group_id, counts[group_id]] = bg_configs[idx]
        counts[group_id] += 1  # Increment the count for the current group
    # Measures of the effective Hilbert space of each sector
    sim.res["Deff"] = np.zeros(num_bg_sectors, dtype=float)
    if sim.par["get_entropy"]:
        sim.res["eigen_entropy"] = np.zeros(
            (num_bg_sectors, int(model.n_sites / 2)), dtype=float
        )
    # -------------------------------------------------------------------------------
    # RUN OVER THE POSSIBLE BACKGROUND SECTORS
    for bg_num, bg_sector in enumerate(unique_bg_sectors):
        bg_config_list = [
            list(bg_configs_per_sector[bg_num, 0, :]),
            list(bg_configs_per_sector[bg_num, 1, :]),
        ]
        logger.info("----------------------------------------------")
        logger.info(f"BG SECTOR {bg_sector}")
        for bg_config in bg_config_list:
            logger.info(f"BG CONFIG {bg_config}")
        # ==============================================================================
        # SELECT THE U(1) GLOBAL and LINK SYMMETRY & BACKGROUND STRING SECTOR
        # ==============================================================================
        model.sector_indices, model.sector_configs = symmetry_sector_configs(
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
        model.get_observables(local_obs)
        model.build_Hamiltonian(sim.par["g"], m)
        # ---------------------------------------------------------------------------
        # DYNAMICS: INITIAL STATE PREPARATION
        # ---------------------------------------------------------------------------
        name = sim.par["dynamics"]["state"]
        if name == "background" and sim.par["model"]["background"]:
            in_state = model.get_qmb_state_from_configs(bg_config_list)
        else:
            raise ValueError("initial state expected to be background")
        # -------------------------------------------------------------------------------
        # MICROCANONICAL ENSEMBLE (it requires a large part of the Hamiltonian spectrum)
        if sim.par["ensemble"]["microcanonical"]["average"]:
            # OBSERVABLE
            obs = sim.par["ensemble"]["local_obs"]
            # DIAGONALIZE THE HAMILTONIAN
            model.diagonalize_Hamiltonian("full", "dense")
            _, microavg = model.microcanonical_avg1(obs, in_state, norm_scalar_product)
            _, microNsingle = model.microcanonical_avg1("N_single", in_state)
            _, microS2casimir = model.microcanonical_avg1("S2_matter", in_state)
            sim.res["microcan_avg"] += microavg / num_bg_sectors
            sim.res["micro_Nsingle"] += microNsingle / num_bg_sectors
            sim.res["micro_S2_matter"] += microS2casimir / num_bg_sectors
        # TIME EVOLUTION
        model.time_evolution_Hamiltonian(in_state, time_line)
        # model.H.get_r_value()
        if hasattr(model.H, "Deff"):
            # Save the Effective Hilbert space of each superselection sector
            sim.res["Deff"][bg_num] = model.H.Deff
            if sim.par["get_entropy"]:
                # Save the eigenstate entropy as a function of the partition
                for L in range(int(model.n_sites / 2)):
                    partition = list(np.arange(0, L + 1, 1))
                    for kk in range(model.H.shape[0]):
                        sim.res["eigen_entropy"][bg_num, L] += (
                            model.H.Npsi[kk].entanglement_entropy(
                                partition,
                                model.sector_configs,
                            )
                            / model.H.shape[0]
                        )
        # -----------------------------------------------------------------------
        for ii, tstep in enumerate(time_line):
            msg_tstep = f"TIME {round(tstep, 2)}"
            msg = f"====== {bg_num} ============ {msg_tstep} ===================="
            logger.info(msg)
            if not model.momentum_basis:
                # ---------------------------------------------------------------
                # ENTROPY
                if sim.par["get_entropy"]:
                    entropy = model.H.psi_time[ii].entanglement_entropy(
                        partition_indices,
                        model.sector_configs,
                    )
                    # Save the entropy
                    sim.res["entropy"][ii] += (
                        entropy + np.log2(num_bg_sectors)
                    ) / num_bg_sectors
                # STATE CONFIGURATIONS
                if sim.par["get_state_configs"]:
                    model.H.psi_time[ii].get_state_configurations(
                        1e-1, model.sector_configs
                    )
            # -------------------------------------------------------------------
            # MEASURE OBSERVABLES
            model.measure_observables(ii, dynamics=True)
            for obs in local_obs:
                sim.res[obs][ii] += np.mean(model.res[obs]) / num_bg_sectors
            # TAKE THE SPECIAL AVERAGE TO LOOK AT THE IMBALANCE
            delta = np.dot(model.res["N_tot"], norm_scalar_product) / model.n_sites
            sim.res["delta"][ii] += delta / num_bg_sectors
            # OVERLAPS with the INITIAL STATE
            # overlap = model.measure_fidelity(in_state, ii, True, True)
            # sim.res["overlap"][ii] += overlap / num_bg_sectors
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
