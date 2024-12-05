import numpy as np
from ed_lgt.models import DFL_Model
from ed_lgt.modeling import get_lattice_link_site_pairs, QMB_hamiltonian
from ed_lgt.operators import SU2_Hamiltonian_couplings
from ed_lgt.symmetries import (
    get_symmetry_sector_generators,
    global_abelian_sector,
    link_abelian_sector,
    symmetry_sector_configs,
)
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
    delta_n = sim.par["dynamics"]["delta_n"]
    n_steps = int((stop - start) / delta_n)
    sim.res["time_steps"] = np.arange(n_steps) * delta_n
    # ==============================================================================
    # MODEL PROPERTIES
    model = DFL_Model(**sim.par["model"])
    m = sim.par["m"] if not model.pure_theory else None
    coeffs = SU2_Hamiltonian_couplings(model.dim, model.pure_theory, sim.par["g"], m)
    # ===============================================================================
    # OBSERVABLES
    if not model.pure_theory:
        local_obs = [f"N_{label}" for label in ["tot"]]
    # Store the observables
    partition_indices = list(np.arange(0, int(np.prod(model.lvals) / 2), 1))
    for measure in ["delta", "entropy", "overlap"][:2]:
        sim.res[measure] = np.zeros(n_steps, dtype=float)
    # ==============================================================================
    # GLOBAL SYMMETRIES
    if model.pure_theory:
        global_ops = None
        global_sectors = None
    else:
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
    if global_ops is not None and link_ops is not None:
        sector_indices, sector_configs = symmetry_sector_configs(
            loc_dims=model.loc_dims,
            glob_op_diags=global_ops,
            glob_sectors=np.array(global_sectors, dtype=float),
            sym_type_flag="U",
            link_op_diags=link_ops,
            link_sectors=link_sectors,
            pair_list=pair_list,
        )
    elif link_ops is not None:
        sector_indices, sector_configs = link_abelian_sector(
            loc_dims=model.loc_dims,
            sym_op_diags=link_ops,
            sym_sectors=link_sectors,
            pair_list=pair_list,
        )
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
    # -------------------------------------------------------------------------------
    # DEFINE THE GLOBAL OPERATOR for the BACKGROUND CHARGE
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
    # -------------------------------------------------------------------------------
    # RUN OVER THE POSSIBLE BACKGROUND SECTORS
    for bg_num, bg_sector in enumerate(unique_bg_sectors):
        bg_config_list = [
            list(bg_configs_per_sector[bg_num, 0, :]),
            list(bg_configs_per_sector[bg_num, 1, :]),
        ]
        logger.info("----------------------------------------------")
        logger.info(f"BG SECTOR {bg_sector}  CONFIGS {bg_config_list}")
        # ---------------------------------------------------------------------------
        # SELECT THE SYMMETRY SECTOR OF THE BACKGROUNG CHARGE
        model.sector_indices, model.sector_configs = global_abelian_sector(
            loc_dims=model.loc_dims,
            sym_op_diags=bg_global_ops,
            sym_sectors=np.array([bg_sector], dtype=float),
            sym_type="string",
            configs=sector_configs,
        )
        model.default_params()
        # DEFINE OBSERVABLES
        model.get_observables(local_obs)
        # ---------------------------------------------------------------------------
        # BUILD THE HAMILTONIAN
        model.H = QMB_hamiltonian(0, model.lvals)
        model.build_Hamiltonian(coeffs)
        # ---------------------------------------------------------------------------
        # DYNAMICS: INITIAL STATE PREPARATION
        # ---------------------------------------------------------------------------
        name = sim.par["dynamics"]["state"]
        if name == "background" and sim.par["model"]["background"]:
            in_state = model.get_qmb_state_from_configs(bg_config_list)
        else:
            raise ValueError("initial state expected to be background")
        # TIME EVOLUTION
        model.time_evolution_Hamiltonian(in_state, start, stop, n_steps)
        # -----------------------------------------------------------------------
        for ii in range(n_steps):
            t_step = format(delta_n * ii, ".2f")
            msg = f"====== {bg_num} ============ TIME {t_step} ===================="
            logger.info(msg)
            """
            if not model.momentum_basis:
                # ---------------------------------------------------------------
                # ENTROPY
                entropy = model.H.psi_time[ii].entanglement_entropy(
                    partition_indices,
                    model.sector_configs,
                )
                # Save the entropy
                sim.res["entropy"][ii] += (
                    entropy + np.log2(num_bg_sectors)
                ) / num_bg_sectors
                # STATE CONFIGURATIONS
                # model.H.psi_time[ii].get_state_configurations(
                #    1e-1, model.sector_configs
                # )
            """
            # -------------------------------------------------------------------
            # MEASURE OBSERVABLES
            model.measure_observables(ii, dynamics=True)
            # TAKE THE SPECIAL AVERAGE TO LOOK AT THE IMBALANCE
            delta = np.dot(model.res["N_tot"], norm_scalar_product) / model.n_sites
            sim.res["delta"][ii] += delta / num_bg_sectors
            # OVERLAPS with the INITIAL STATE
            # overlap = model.measure_fidelity(in_state, ii, True, True)
            # sim.res["overlap"][ii] += overlap / num_bg_sectors
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
