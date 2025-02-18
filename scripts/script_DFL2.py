import numpy as np
from numba import set_num_threads
from ed_lgt.models import DFL_Model
from ed_lgt.modeling import get_lattice_link_site_pairs, get_entropy_partition
from ed_lgt.symmetries import get_symmetry_sector_generators, symmetry_sector_configs
from simsio import run_sim
from time import perf_counter
import logging

logger = logging.getLogger(__name__)

with run_sim() as sim:
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
    local_obs = []  # [f"N_{label}" for label in ["tot", "single", "pair", "zero"]]
    for measure in local_obs:
        sim.res[measure] = np.zeros(n_steps, dtype=float)
    # Store the observables
    partition_indices = get_entropy_partition(model.lvals)
    for measure in ["entropy", "delta", "overlap"][:1]:
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
    # SELECT THE U(1) GLOBAL and LINK SYMMETRY SECTOR
    # ==============================================================================
    model.sector_indices, model.sector_configs = symmetry_sector_configs(
        loc_dims=model.loc_dims,
        glob_op_diags=global_ops,
        glob_sectors=np.array(global_sectors, dtype=float),
        sym_type_flag="U",
        link_op_diags=link_ops,
        link_sectors=link_sectors,
        pair_list=pair_list,
    )
    # DEFINE SETTINGS, OBSERVABLES, and BUILD HAMILTONIAN
    model.default_params()
    model.get_observables(local_obs)
    model.build_Hamiltonian(sim.par["g"], m)
    # ==============================================================================
    # ENUMERATE ALL THE BACKGROUND SYMMETRY SECTORS
    logical_stag_basis = sim.par["dynamics"]["logical_stag_basis"]
    bg_configs, bg_sectors = model.get_background_charges_configs(logical_stag_basis)
    n_charge_sectors = bg_sectors.shape[0]
    logger.info(f"charge sector configs {n_charge_sectors}")
    for ii in range(len(bg_configs)):
        logger.info(f"{bg_configs[ii]}")
    # Calculate the norm_scalar product with the matter config of the initial state
    if model.n_sites % (2 * logical_stag_basis) != 0:
        raise ValueError(f"the staggered basis is not compatible with n_sites")
    # ---------------------------------------------------------------------------
    num_blocks = model.n_sites // (2 * logical_stag_basis)
    stag_array = np.array(
        [-1] * logical_stag_basis + [1] * logical_stag_basis, dtype=int
    )
    norm_scalar_product = np.tile(stag_array, num_blocks)
    logger.info(f"norm scalar product {norm_scalar_product}")
    # ---------------------------------------------------------------------------
    # DYNAMICS: INITIAL STATE PREPARATION
    # ---------------------------------------------------------------------------
    name = sim.par["dynamics"]["state"]
    if name == "background" and sim.par["model"]["background"]:
        in_state = model.get_qmb_state_from_configs(bg_configs)
    else:
        raise ValueError("initial state expected to be background")
    # TIME EVOLUTION
    model.time_evolution_Hamiltonian(in_state, time_line)
    # -----------------------------------------------------------------------
    for ii, tstep in enumerate(time_line):
        msg_tstep = f"TIME {round(tstep, 2)}"
        msg = f"==================== {msg_tstep} ===================="
        logger.info(msg)
        if not model.momentum_basis:
            # ---------------------------------------------------------------
            # ENTROPY
            if sim.par["get_entropy"]:
                sim.res["entropy"][ii] = model.H.psi_time[ii].entanglement_entropy(
                    partition_indices,
                    model.sector_configs,
                )
            # STATE CONFIGURATIONS
            if sim.par["get_state_configs"]:
                model.H.psi_time[ii].get_state_configurations(
                    1e-1, model.sector_configs
                )
        # -------------------------------------------------------------------
        # MEASURE OBSERVABLES
        # model.measure_observables(ii, dynamics=True)
        # TAKE THE SPECIAL AVERAGE TO LOOK AT THE IMBALANCE
        # delta = np.dot(model.res["N_tot"], norm_scalar_product) / model.n_sites
        # sim.res["delta"][ii] = delta
        # OVERLAPS with the INITIAL STATE
        # sim.res["overlap"][ii] = model.measure_fidelity(in_state, ii, True, True)
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
