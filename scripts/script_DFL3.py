import numpy as np
from ed_lgt.models import DFL_Model
from ed_lgt.tools import stag_avg
from ed_lgt.modeling import get_lattice_link_site_pairs, get_entropy_partition
from ed_lgt.symmetries import get_symmetry_sector_generators, symmetry_sector_configs
from simsio import run_sim
from time import perf_counter
import logging


logger = logging.getLogger(__name__)

with run_sim() as sim:
    start_time = perf_counter()
    # ==============================================================================
    # MODEL PROPERTIES
    model = DFL_Model(**sim.par["model"])
    m = sim.par["m"] if not model.pure_theory else None
    # ==============================================================================
    # DYNAMICS PARAMETERS
    start = sim.par["dynamics"]["start"]
    stop = sim.par["dynamics"]["stop"]
    delta_t = sim.par["dynamics"]["delta_n"]
    time_line = np.arange(start, stop + delta_t, delta_t)
    sim.res["time_steps"] = time_line
    n_steps = len(sim.res["time_steps"])
    # ===============================================================================
    # OBSERVABLES
    local_obs = [f"N_{label}" for label in ["tot", "single", "pair", "zero"]]
    for obs in local_obs:
        sim.res[obs] = np.zeros(n_steps, dtype=float)
        sim.res[f"ME_{obs}"] = 0.0
        sim.res[f"DE_{obs}"] = 0.0
    # Usefull quantities for the staggered averages
    stag_avgs = {"N_tot": None, "N_single": None, "N_pair": "even", "N_zero": "odd"}
    # Store the observables
    partition_indices = get_entropy_partition(model.lvals)
    for measure in ["delta", "entropy", "overlap"][:1]:
        sim.res[measure] = np.zeros(n_steps, dtype=float)
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
    # Set the norm only for the total number of particles
    norms = {
        "N_tot": norm_scalar_product,
        "N_single": None,
        "N_pair": None,
        "N_zero": None,
    }
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
    n_bg_sectors = len(unique_bg_sectors)
    # Step 2: Define the array where bg configs are stored according to their associated sector
    bg_configs_per_sector = np.zeros((n_bg_sectors, 2, model.n_sites), dtype=int)
    counts = np.zeros(n_bg_sectors, dtype=int)
    # Step 3: Group rows from the second array based on the indices
    for idx, group_id in enumerate(indices):
        bg_configs_per_sector[group_id, counts[group_id]] = bg_configs[idx]
        counts[group_id] += 1  # Increment the count for the current group
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
            # DIAGONALIZE THE HAMILTONIAN
            model.diagonalize_Hamiltonian("full", "dense")
            sim.res["energy"] = model.H.Nenergies
            _, ME = model.microcanonical_avg1(
                local_obs,
                in_state,
                staggered_avg=stag_avgs,
                special_norm=norms,
            )
            for obs in local_obs:
                sim.res[f"ME_{obs}"] += ME[f"ME_{obs}"] / n_bg_sectors
        # -------------------------------------------------------------------------------
        # DIAGONAL ENSEMBLE (it requires the full spectrum of the Hamiltonian)
        if sim.par["ensemble"]["diagonal"]["average"]:
            if not sim.par["ensemble"]["microcanonical"]["average"]:
                # DIAGONALIZE THE HAMILTONIAN
                model.diagonalize_Hamiltonian("full", "dense")
                sim.res["energy"] = model.H.Nenergies
            # MEASURE DIAGONAL ENSEMBLE of some OBSERVABLES
            DE = model.diagonal_avg1(
                ["N_tot", "N_single", "N_pair", "N_zero"],
                in_state,
                staggered_avg=stag_avgs,
                special_norms=norms,
            )
            for obs in local_obs:
                sim.res[f"DE_{obs}"] += DE[f"DE_{obs}"] / n_bg_sectors
        # TIME EVOLUTION
        if sim.par["dynamics"]["time_evolution"]:
            model.time_evolution_Hamiltonian(in_state, time_line)
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
                        sim.res["entropy"][ii] += entropy / n_bg_sectors
                    # STATE CONFIGURATIONS
                    if sim.par["get_state_configs"]:
                        model.H.psi_time[ii].get_state_configurations(
                            1e-1, model.sector_configs
                        )
                # -------------------------------------------------------------------
                # MEASURE OBSERVABLES
                model.measure_observables(ii, dynamics=True)
                sim.res["N_single"][ii] += (
                    stag_avg(model.res["N_single"]) / n_bg_sectors
                )
                sim.res["N_pair"][ii] += (
                    stag_avg(model.res["N_pair"], "even")
                    + stag_avg(model.res["N_zero"], "odd")
                ) / (2 * n_bg_sectors)
                sim.res["N_zero"][ii] += (
                    stag_avg(model.res["N_zero"], "even")
                    + stag_avg(model.res["N_pair"], "odd")
                ) / (2 * n_bg_sectors)
                # TAKE THE SPECIAL AVERAGE TO LOOK AT THE IMBALANCE
                delta = np.dot(model.res["N_tot"], norm_scalar_product) / model.n_sites
                sim.res["delta"][ii] += delta / n_bg_sectors
                # OVERLAPS with the INITIAL STATE
                # overlap = model.measure_fidelity(in_state, ii, True, True)
                # sim.res["overlap"][ii] += overlap / n_bg_sectors
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
