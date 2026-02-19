# %%
import numpy as np
import logging
from time import perf_counter
from ed_lgt.modeling import diagonalize_density_matrix, get_lattice_link_site_pairs
from ed_lgt.models import SU2_Model
from ed_lgt.models import DFL_Model
from ed_lgt.symmetries import get_symmetry_sector_generators, symmetry_sector_configs

logger = logging.getLogger(__name__)


def _get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def run_SU2_spectrum(par: dict) -> dict:
    """
    Run the SU2 ED workflow using a plain parameter dictionary.

    Parameters
    ----------
    par : dict
        Dictionary of parameters (same structure as par).
    Returns
    -------
    res : dict
        Results dictionary (same role as res).
    """
    res = {}
    start_time = perf_counter()
    # -------------------------------------------------------------------------------
    # Initialize model
    model = SU2_Model(**par["model"])
    # -------------------------------------------------------------------------------
    # Momentum sector (optional)
    if _get(par, ["momentum", "get_momentum_basis"], False):
        unit_cell_size = _get(par, ["momentum", "unit_cell_size"], None)
        k_vals = _get(par, ["momentum", "momentum_k_vals"], None)
        TC_symmetry = _get(par, ["momentum", "TC_symmetry"], False)
        model.set_momentum_sector(unit_cell_size, k_vals, TC_symmetry)
    # -------------------------------------------------------------------------------
    model.default_params()
    # -------------------------------------------------------------------------------
    # Build Hamiltonian
    g = par["g"]
    m = par.get("m", None) if not model.pure_theory else None
    theta = par.get("theta", 0.0) if model.pure_theory else 0
    model.build_Hamiltonian(g, m, theta)
    # -------------------------------------------------------------------------------
    # Diagonalize
    n_eigs = _get(par, ["hamiltonian", "n_eigs"], "full")
    save_psi = _get(par, ["hamiltonian", "save_psi"], False)
    model.diagonalize_Hamiltonian(n_eigs, model.ham_format, print_results=True)
    n_eigs_eff = len(model.sector_configs) if n_eigs == "full" else int(n_eigs)
    res["energy"] = model.H.Nenergies
    if save_psi:
        for ii in range(n_eigs_eff):
            res[f"psi{ii}"] = model.H.Npsi[ii].psi
    # -------------------------------------------------------------------------------
    # Observables definition
    local_obs = [f"T2_p{d}" for d in model.directions]
    if not model.pure_theory:
        local_obs += [f"N_{label}" for label in ["tot", "single", "pair", "zero"]]
    res["E2"] = np.zeros(n_eigs_eff, dtype=float)
    for obs in local_obs:
        res[obs] = np.zeros(n_eigs_eff, dtype=float)
    twobody_obs = []
    twobody_axes = []
    # Plaquettes
    if np.all([model.spin < 1, model.dim in (2, 3), not model.use_generic_model]):
        if model.dim == 2:
            plaquette_obs = [["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]]
        else:  # dim == 3
            plaquette_obs = [
                ["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"],
                ["C_px,pz", "C_pz,mx", "C_mz,px", "C_mx,mz"],
                ["C_py,pz", "C_pz,my", "C_mz,py", "C_my,mz"],
            ]
            if np.abs(theta) > 1e-10:
                plaquette_obs += [
                    ["EzC_px,py", "C_py,mx", "C_my,px", "C_mx,my"],
                    ["EyC_px,pz", "C_pz,mx", "C_mz,px", "C_mx,mz"],
                    ["ExC_py,pz", "C_pz,my", "C_mz,py", "C_my,mz"],
                ]
    else:
        plaquette_obs = []
    for obs_names_list in plaquette_obs:
        obs = "_".join(obs_names_list)
        res[obs] = np.zeros(n_eigs_eff, dtype=float)
    model.get_observables(
        local_obs, twobody_obs, plaquette_obs, twobody_axes=twobody_axes
    )
    measure_obs = _get(par, ["observables", "measure_obs"], False)
    # -------------------------------------------------------------------------------
    # Overlap state (optional)
    get_overlap = _get(par, ["observables", "get_overlap"], False)
    if get_overlap:
        name = _get(par, ["hamiltonian", "state"], None)
        config = model.overlap_QMB_state(name)
        logger.info(f"config {config}")
        in_state = model.get_qmb_state_from_configs([config])
        res["overlap"] = np.zeros(model.H.n_eigs, dtype=float)
    # -------------------------------------------------------------------------------
    # Entropy / RDM partition (optional)
    partition_indices = _get(par, ["observables", "entropy_partition"], [])
    get_entropy = _get(par, ["observables", "get_entropy"], False)
    get_rdm = _get(par, ["observables", "get_RDM"], False)
    if get_entropy or get_rdm:
        model._get_partition(partition_indices)
    res["entropy"] = np.zeros(model.H.n_eigs, dtype=float)
    get_state_configs = _get(par, ["observables", "get_state_configs"], False)
    get_PE = _get(par, ["observables", "get_PE"], False)
    if get_PE:
        res["PE"] = np.zeros(model.H.n_eigs, dtype=float)
    get_SRE = _get(par, ["observables", "get_SRE"], False)
    if get_SRE:
        res["SRE"] = np.zeros(model.H.n_eigs, dtype=float)
    # -------------------------------------------------------------------------------
    # Parity (optional)
    apply_parity = False
    wrt_site = False
    if isinstance(par.get("inversion", None), dict):
        apply_parity = _get(par, ["inversion", "get_inversion_sym"], False)
        wrt_site = _get(par, ["inversion", "wrt_site"], False)
    if apply_parity:
        model.get_parity_inversion_operator(wrt_site)
    # -------------------------------------------------------------------------------
    # Main loop over eigenstates
    for ii in range(model.H.n_eigs):
        model.H.print_energy(ii)
        # Parity expectation (debug/info)
        if apply_parity:
            if model.momentum_basis is None:
                psi = model.H.Npsi[ii].psi
            else:
                Pk = model._basis_Pk_as_csr()
                psi = Pk @ model.H.Npsi[ii].psi
            psiP = model.parityOP @ psi
            logger.info(f"<psi{ii}|P|psi{ii}> {np.real(np.vdot(psi, psiP))}")
        # ---------------------------------------------------------------------------
        # ONLY IN THE COORDINATE BASIS
        if model.momentum_basis is None:
            # ---------------------------------------------------------------------------
            # REDUCED DENSITY MATRIX
            if get_rdm:
                RDM = model.H.Npsi[ii].reduced_density_matrix(
                    partition_indices, model._partition_cache
                )
                logger.info(f"RDM shape {RDM.shape}")
                rho_eigvals, rho_eigvecs = diagonalize_density_matrix(RDM)
                sorted_indices = np.argsort(rho_eigvals)[::-1]
                rho_eigvals = rho_eigvals[sorted_indices]
                logger.info(f"eigvals {rho_eigvals}")
                res["eigvals"] = rho_eigvals
                rho_eigvecs = rho_eigvecs[:, sorted_indices]
            # ---------------------------------------------------------------------------
            # ENTANGLEMENT ENTROPY
            if get_entropy:
                res["entropy"][ii] = model.H.Npsi[ii].entanglement_entropy(
                    partition_indices, model._partition_cache
                )
            # ---------------------------------------------------------------------------
            # STATE CONFIGURATIONS
            if get_state_configs:
                model.H.Npsi[ii].get_state_configurations(1e-3, model.sector_configs)
            if get_PE:
                res["PE"][ii] = model.H.Npsi[ii].participation_renyi_entropy()
            if get_SRE:
                res["SRE"][ii] = model.H.Npsi[ii].stabilizer_renyi_entropy(
                    model.sector_configs, prob_threshold=1e-3
                )
                res["SRE"][ii] = model.H.Npsi[ii].stabilizer_renyi_entropy(
                    model.sector_configs, prob_threshold=1e-4
                )
                res["SRE"][ii] = model.H.Npsi[ii].stabilizer_renyi_entropy(
                    model.sector_configs, prob_threshold=1e-5
                )
                res["SRE"][ii] = model.H.Npsi[ii].stabilizer_renyi_entropy(
                    model.sector_configs, prob_threshold=1e-6
                )
                res["SRE"][ii] = model.H.Npsi[ii].stabilizer_renyi_entropy(
                    model.sector_configs, prob_threshold=1e-7
                )
        # -------------------------------------------------------------------------------
        # LOCAL OBSERVABLES
        if measure_obs:
            model.measure_observables(ii)
            res["E2"][ii] = model.link_avg(obs_name="T2")
            if not model.pure_theory:
                res["N_single"][ii] = model.stag_avg(model.res["N_single"])
                res["N_pair"][ii] += 0.5 * model.stag_avg(model.res["N_pair"], "even")
                res["N_pair"][ii] += 0.5 * model.stag_avg(model.res["N_zero"], "odd")
                res["N_zero"][ii] += 0.5 * model.stag_avg(model.res["N_zero"], "even")
                res["N_zero"][ii] += 0.5 * model.stag_avg(model.res["N_pair"], "odd")
                res["N_tot"][ii] = res["N_single"][ii] + 2.0 * res["N_pair"][ii]
            for obs_names_list in plaquette_obs:
                obs = "_".join(obs_names_list)
                res[obs][ii] = model.res[obs]
        # -------------------------------------------------------------------------------
        # Overlaps
        if get_overlap:
            res["overlap"][ii] = model.measure_fidelity(in_state, ii, print_value=True)
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time - start_time, 5)}")
    return res


def run_SU2_dynamics(par: dict) -> dict:
    """
    Run the SU2 ED workflow using a plain parameter dictionary.

    Parameters
    ----------
    par : dict
        Dictionary of parameters (same structure as par).
    Returns
    -------
    res : dict
        Results dictionary (same role as res).
    """
    res = {}
    start_time = perf_counter()
    # -------------------------------------------------------------------------------
    # Initialize model
    model = SU2_Model(**par["model"])
    # -------------------------------------------------------------------------------
    # Momentum sector (optional)
    if _get(par, ["momentum", "get_momentum_basis"], False):
        unit_cell_size = _get(par, ["momentum", "unit_cell_size"], None)
        k_vals = _get(par, ["momentum", "momentum_k_vals"], None)
        TC_symmetry = _get(par, ["momentum", "TC_symmetry"], False)
        model.set_momentum_sector(unit_cell_size, k_vals, TC_symmetry)
    # -------------------------------------------------------------------------------
    model.default_params()
    # -------------------------------------------------------------------------------
    # Build Hamiltonian
    g = par["g"]
    m = par.get("m", None) if not model.pure_theory else None
    theta = par.get("theta", 0.0) if model.pure_theory else 0
    model.build_Hamiltonian(g, m, theta)
    # -------------------------------------------------------------------------------
    # Time evolution setup
    name = par["dynamics"]["state"]
    start = par["dynamics"]["start"]
    stop = par["dynamics"]["stop"]
    delta_t = par["dynamics"]["delta_n"]
    time_line = np.arange(start, stop + delta_t, delta_t)
    res["time_steps"] = time_line
    n_steps = len(time_line)
    # -------------------------------------------------------------------------------
    # INITIAL STATE PREPARATION
    config = model.overlap_QMB_state(name)
    in_state = model.get_qmb_state_from_configs([config])
    # Overlap state (optional)
    get_overlap = _get(par, ["observables", "get_overlap"], False)
    if get_overlap:
        res["overlap"] = np.zeros(n_steps, dtype=float)
    # -------------------------------------------------------------------------------
    # Observables definition
    local_obs = [f"T2_p{d}" for d in model.directions]
    if not model.pure_theory:
        local_obs += [f"N_{label}" for label in ["tot", "single", "pair", "zero"]]
    res["E2"] = np.zeros(n_steps, dtype=float)
    for obs in local_obs:
        res[obs] = np.zeros(n_steps, dtype=float)
    twobody_obs = []
    twobody_axes = []
    # Plaquettes
    if np.all([model.spin < 1, model.dim in (2, 3), not model.use_generic_model]):
        if model.dim == 2:
            plaquette_obs = [["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]]
        else:  # dim == 3
            plaquette_obs = [
                ["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"],
                ["C_px,pz", "C_pz,mx", "C_mz,px", "C_mx,mz"],
                ["C_py,pz", "C_pz,my", "C_mz,py", "C_my,mz"],
            ]
            if np.abs(theta) > 1e-10:
                plaquette_obs += [
                    ["EzC_px,py", "C_py,mx", "C_my,px", "C_mx,my"],
                    ["EyC_px,pz", "C_pz,mx", "C_mz,px", "C_mx,mz"],
                    ["ExC_py,pz", "C_pz,my", "C_mz,py", "C_my,mz"],
                ]
    else:
        plaquette_obs = []
    for obs_names_list in plaquette_obs:
        obs = "_".join(obs_names_list)
        res[obs] = np.zeros(n_steps, dtype=float)
    model.get_observables(
        local_obs, twobody_obs, plaquette_obs, twobody_axes=twobody_axes
    )
    measure_obs = _get(par, ["observables", "measure_obs"], False)
    # -------------------------------------------------------------------------------
    # Entropy / RDM partition (optional)
    partition_indices = _get(par, ["observables", "entropy_partition"], [])
    get_entropy = _get(par, ["observables", "get_entropy"], False)
    get_rdm = _get(par, ["observables", "get_RDM"], False)
    if get_entropy or get_rdm:
        model._get_partition(partition_indices)
        res["entropy"] = np.zeros(n_steps, dtype=float)
    get_state_configs = _get(par, ["observables", "get_state_configs"], False)
    get_PE = _get(par, ["observables", "get_PE"], False)
    if get_PE:
        res["PE"] = np.zeros(n_steps, dtype=float)
    get_SRE = _get(par, ["observables", "get_SRE"], False)
    if get_SRE:
        res["SRE1"] = np.zeros(n_steps, dtype=float)
        res["SRE2"] = np.zeros(n_steps, dtype=float)
        res["SRE3"] = np.zeros(n_steps, dtype=float)
    # -------------------------------------------------------------------------------
    # TIME EVOLUTION
    if par["dynamics"]["time_evolution"]:
        model.time_evolution_Hamiltonian(in_state, time_line)
    # Main loop over time steps
    for ii in range(n_steps):
        msg = f"TIME {time_line[ii]}"
        logger.info(f"================== {msg} ==========================")
        # ---------------------------------------------------------------------------
        # ONLY IN THE COORDINATE BASIS
        if model.momentum_basis is None:
            # -----------------------------------------------------------------------
            # REDUCED DENSITY MATRIX
            if get_rdm:
                RDM = model.H.psi_time[ii].reduced_density_matrix(
                    partition_indices, model._partition_cache
                )
                logger.info(f"RDM shape {RDM.shape}")
                rho_eigvals, rho_eigvecs = diagonalize_density_matrix(RDM)
                sorted_indices = np.argsort(rho_eigvals)[::-1]
                rho_eigvals = rho_eigvals[sorted_indices]
                logger.info(f"eigvals {rho_eigvals}")
                res["eigvals"] = rho_eigvals
                rho_eigvecs = rho_eigvecs[:, sorted_indices]
            # -----------------------------------------------------------------------
            # ENTANGLEMENT ENTROPY
            if get_entropy:
                res["entropy"][ii] = model.H.psi_time[ii].entanglement_entropy(
                    partition_indices, model._partition_cache
                )
            if get_PE:
                res["PE"][ii] = model.H.psi_time[ii].participation_renyi_entropy()
            if get_SRE:
                res["SRE1"][ii] = model.H.psi_time[ii].stabilizer_renyi_entropy(
                    model.sector_configs, prob_threshold=1e-3
                )
                res["SRE2"][ii] = model.H.psi_time[ii].stabilizer_renyi_entropy(
                    model.sector_configs, prob_threshold=1e-4
                )
                res["SRE3"][ii] = model.H.psi_time[ii].stabilizer_renyi_entropy(
                    model.sector_configs, prob_threshold=1e-5
                )
            # -----------------------------------------------------------------------
            # STATE CONFIGURATIONS
            if get_state_configs:
                model.H.psi_time[ii].get_state_configurations(
                    1e-3, model.sector_configs
                )
        # ---------------------------------------------------------------------------
        # LOCAL OBSERVABLES
        if measure_obs:
            model.measure_observables(ii, dynamics=True)
            res["E2"][ii] = model.link_avg(obs_name="T2")
            if not model.pure_theory:
                res["N_single"][ii] = model.stag_avg(model.res["N_single"])
                res["N_pair"][ii] += 0.5 * model.stag_avg(model.res["N_pair"], "even")
                res["N_pair"][ii] += 0.5 * model.stag_avg(model.res["N_zero"], "odd")
                res["N_zero"][ii] += 0.5 * model.stag_avg(model.res["N_zero"], "even")
                res["N_zero"][ii] += 0.5 * model.stag_avg(model.res["N_pair"], "odd")
                res["N_tot"][ii] = res["N_single"][ii] + 2.0 * res["N_pair"][ii]
            for obs_names_list in plaquette_obs:
                obs = "_".join(obs_names_list)
                res[obs][ii] = model.res[obs]
        # ---------------------------------------------------------------------------
        # Overlaps
        if get_overlap:
            res["overlap"][ii] = model.measure_fidelity(in_state, ii, True, True)
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time - start_time, 5)}")
    return res


def run_SU2_bg_groundstate(par: dict) -> dict:
    start_time = perf_counter()
    res = {}
    # ==============================================================================
    # MODEL
    model = DFL_Model(**par["model"])
    bg = 0.75 if model.background < 1 else 2
    # ==============================================================================
    # GLOBAL SYMMETRIES
    global_ops = [model.ops["N_tot"]]
    global_sectors = par["sectors"]
    # GLOBAL OPERATORS
    global_ops = get_symmetry_sector_generators(global_ops, action="global")
    # ==============================================================================
    # ABELIAN LINK SYMMETRIES
    dirs = model.directions
    link_ops = [[model.ops[f"T2_p{d}"], -model.ops[f"T2_m{d}"]] for d in dirs]
    link_sectors = [0 for _ in dirs]
    # LINK OPERATORS
    link_ops = get_symmetry_sector_generators(link_ops, action="link")
    pair_list = get_lattice_link_site_pairs(model.lvals, model.has_obc)
    # ==============================================================================
    # SELECT THE BACKGROUND SYMMETRY SECTOR CONFIGURATION
    if model.lvals == [5, 2]:
        bg_sector = [bg, 0, 0, 0, 0, 0, 0, 0, 0, bg]
    elif model.lvals == [4, 3] or model.lvals == [6, 2]:
        bg_sector = [bg, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, bg]
    elif model.lvals == [3, 2]:
        bg_sector = [bg, 0, 0, 0, 0, bg]
    elif model.lvals == [2, 2]:
        bg_sector = [bg, 0, 0, bg]
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
    if par["model"]["spin"] < 1:
        model.build_Hamiltonian(par["g"], par["m"])
    else:
        model.build_gen_Hamiltonian(par["g"], par["m"])
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
    n_eigs = par["hamiltonian"]["n_eigs"]
    model.diagonalize_Hamiltonian(n_eigs, model.ham_format)
    res["energy"] = model.H.Nenergies
    # ===========================================================================
    # OBSERVABLES
    matter_obs = [f"N_{label}" for label in ["tot", "single", "pair", "zero"]]
    extra_obs = ["bg", "T2_px", "T2_py"]
    local_obs = matter_obs + extra_obs
    for obs in local_obs:
        res[obs] = np.zeros(n_eigs, dtype=float)
    res["E2"] = np.zeros(n_eigs, dtype=float)
    model.get_observables(local_obs)
    measure_obs = _get(par, ["observables", "measure_obs"], False)
    # GET STATE CONFIGURATIONS
    get_state_configs = _get(par, ["observables", "get_state_configs"], False)
    # ENTROPY
    partition_indices = _get(par, ["observables", "entropy_partition"], [])
    get_entropy = _get(par, ["observables", "get_entropy"], False)
    if get_entropy:
        model._get_partition(partition_indices)
        res["entropy"] = np.zeros(n_eigs, dtype=float)
    # -------------------------------------------------------------------------------
    for ii in range(model.H.n_eigs):
        model.H.print_energy(ii)
        if not model.momentum_basis:
            # -----------------------------------------------------------------------
            # ENTROPY
            if get_entropy:
                res["entropy"][ii] = model.H.Npsi[ii].entanglement_entropy(
                    partition_indices, model._partition_cache
                )
            # -----------------------------------------------------------------------
            # STATE CONFIGURATIONS
            if get_state_configs:
                model.H.Npsi[ii].get_state_configurations(1e-2, model.sector_configs)
        # -----------------------------------------------------------------------
        # MEASURE OBSERVABLES
        if measure_obs:
            model.measure_observables(ii, dynamics=False)
            res["E2"][ii] = model.link_avg(obs_name="T2")
            res["N_single"][ii] = model.stag_avg(model.res["N_single"])
            res["N_pair"][ii] += 0.5 * model.stag_avg(model.res["N_pair"], "even")
            res["N_pair"][ii] += 0.5 * model.stag_avg(model.res["N_zero"], "odd")
            res["N_zero"][ii] += 0.5 * model.stag_avg(model.res["N_zero"], "even")
            res["N_zero"][ii] += 0.5 * model.stag_avg(model.res["N_pair"], "odd")
            res["N_tot"][ii] = res["N_single"][ii] + 2 * res["N_pair"][ii]
            logger.info(f"Nsingle {res['N_single'][ii]}")
            logger.info(f"Npair {res['N_pair'][ii]}")
            logger.info(f"Ntot {res['N_tot'][ii]}")
            logger.info(f"E2 {res['E2'][ii]}")
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
    return res


# %%
par = {
    "model": {
        "lvals": [10],
        "sectors": [10],
        "has_obc": [False],
        "spin": 0.5,
        "pure_theory": False,
        "background": 0,
        "ham_format": "sparse",
    },
    "hamiltonian": {
        "n_eigs": 1,
        "save_psi": False,
    },
    "momentum": {
        "get_momentum_basis": False,
        "unit_cell_size": [2],
        "TC_symmetry": False,
    },
    "observables": {
        "measure_obs": True,
        "get_entropy": True,
        "entropy_partition": [0, 1, 2],
        "get_state_configs": True,
        "get_overlap": False,
        "get_PE": True,
        "get_SRE": True,
    },
    "g": 5,
    "m": 1,
}
run_SU2_spectrum(par)
# %%
par = {
    "model": {
        "lvals": [10],
        "sectors": [10],
        "has_obc": [False],
        "spin": 0.5,
        "pure_theory": False,
        "background": 0,
        "ham_format": "sparse",
    },
    "dynamics": {
        "time_evolution": True,
        "start": 0,
        "stop": 5,
        "delta_n": 0.05,
        "state": "PV",
        "logical_stag_basis": 2,
    },
    "momentum": {
        "get_momentum_basis": False,
        "unit_cell_size": [2],
        "TC_symmetry": False,
    },
    "observables": {
        "measure_obs": True,
        "get_entropy": False,
        "get_PE": True,
        "get_SRE": True,
        "entropy_partition": [0, 1, 2, 3, 4],
        "get_state_configs": False,
        "get_overlap": True,
    },
    "ensemble": {
        "microcanonical": {"average": False},
        "diagonal": {"average": False},
        "canonical": {"average": False},
    },
    "g": 5,
    "m": 1,
}
res = run_SU2_dynamics(par)
# %%
import numpy as np
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # right y-axis sharing the same x-axis

xticks = np.arange(0, 5.1, 0.5)

# left axis
(l1,) = ax1.plot(res["time_steps"], res["overlap"], label="Fidelity", c="#1f77b4")
ax1.set_ylabel("Fidelity")  # left ylabel

# right axis
(l2,) = ax2.plot(res["time_steps"], res["PE"], label="PE2", c="#ff7f0e")
(l3,) = ax2.plot(res["time_steps"], res["SRE1"], label="PE2", c="green")
(l4,) = ax2.plot(res["time_steps"], res["SRE2"], label="PE2", c="green")
(l5,) = ax2.plot(res["time_steps"], res["SRE3"], label="PE2", c="green")
ax2.set_ylabel("Participation Renyi-2 Entropy PE2")  # right ylabel
ax1.set(xticks=xticks, xlabel="time t")

# optional: set y-limits independently (edit as needed)
# ax1.set_ylim(0, 1)
# ax2.set_ylim(res["PE"].min(), res["PE"].max())
# grid (draw it from the left axis)
ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

fig.tight_layout()
fig.savefig("PE2_Scar_SU2_PV.pdf")

# %%
par = {
    "model": {
        "lvals": [2, 2, 2],
        "has_obc": [True, True, True],
        "spin": 0.5,
        "pure_theory": False,
        "background": 0,
        "ham_format": "linear",
        "sectors": [6],
    },
    "hamiltonian": {
        "n_eigs": 1,
        "save_psi": False,
    },
    "momentum": {
        "get_momentum_basis": False,
        "unit_cell_size": [1, 1, 1],
        "TC_symmetry": False,
        "momentum_k_vals": [0, 0, 0],
    },
    "observables": {
        "measure_obs": True,
        "get_entropy": False,
        "entropy_partition": [0, 1],
        "get_state_configs": True,
        "get_overlap": False,
    },
    "g": 1,
    "m": 1,
    "theta": 0,
}
res = run_SU2_spectrum(par)

# %%
par = {
    "model": {
        "lvals": [2, 2],
        "has_obc": [True, True],
        "spin": 0.5,
        "pure_theory": False,
        "background": 0.5,
        "ham_format": "sparse",
    },
    "sectors": [4],
    "hamiltonian": {
        "n_eigs": 1,
        "save_psi": False,
    },
    "observables": {
        "measure_obs": True,
        "get_entropy": True,
        "entropy_partition": [0, 1],
        "get_state_configs": True,
        "get_overlap": False,
        "get_PE": True,
    },
    "g": 0.1,
    "m": 0.1,
}
res = run_SU2_bg_groundstate(par)

# %%
