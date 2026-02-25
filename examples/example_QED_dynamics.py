# %%
import numpy as np
import logging
from time import perf_counter

logger = logging.getLogger(__name__)
from edlgt.models import QED_Model
from edlgt.modeling import diagonalize_density_matrix


def _get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def run_QED_dynamics(par: dict) -> dict:
    """
    Run the QED ED workflow using a plain parameter dictionary.

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
    model = QED_Model(**par["model"])
    # -------------------------------------------------------------------------------
    # Momentum sector (optional)
    if _get(par, ["momentum", "get_momentum_basis"], False):
        unit_cell_size = _get(par, ["momentum", "unit_cell_size"], None)
        k_vals = _get(par, ["momentum", "momentum_k_vals"], None)
        model.set_momentum_sector(unit_cell_size, k_vals)
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
    res["time"] = time_line
    n_steps = len(time_line)
    # -------------------------------------------------------------------------------
    # INITIAL STATE PREPARATION
    config = model.overlap_QMB_state(name)
    in_state = model.get_qmb_state_from_configs([config])
    # Overlap state (optional)
    get_fidelity = _get(par, ["observables", "get_fidelity"], False)
    if get_fidelity:
        res["fidelity"] = np.zeros(n_steps, dtype=float)
    # List of other states to measure overlap with (optional)
    overlap_with = _get(par, ["observables", "overlap_with"], [])
    overlap_states = {}
    for state in overlap_with:
        overlap_config = model.overlap_QMB_state(state)
        overlap_states[state] = model.get_qmb_state_from_configs([overlap_config])
        res[f"overlap_{state}"] = np.zeros(n_steps, dtype=float)
    # -------------------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = ["E2"]
    local_obs += [f"E2_{s}{d}" for d in model.directions for s in "mp"]
    if not model.pure_theory:
        local_obs += ["N", "N_zero"]
    measure_obs = ["E2"]
    if not model.pure_theory:
        measure_obs += ["N"]
    for obs in measure_obs:
        res[obs] = np.zeros(n_steps, dtype=float)
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
    # DEFINE OBSERVABLES
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
        res["SRE"] = np.zeros(n_steps, dtype=float)
    # -------------------------------------------------------------------------------
    # TIME EVOLUTION
    if par["dynamics"]["time_evolution"]:
        model.time_evolution_Hamiltonian(in_state, time_line)
    # Main loop over time steps
    for ii in range(n_steps):
        msg = f"TIME {time_line[ii]:.3f}"
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
                res["SRE"][ii] = model.H.psi_time[ii].stabilizer_renyi_entropy(
                    model.sector_configs, prob_threshold=1e-3
                )
            # -----------------------------------------------------------------------
            # STATE CONFIGURATIONS
            if get_state_configs:
                cfgs, vals = model.H.psi_time[ii].get_state_configurations(
                    1e-2, model.sector_configs, return_configs=True
                )
                logger.info(f"State configurations at time {time_line[ii]:.3f}:")
                for cfg, val in zip(cfgs, vals):
                    model.print_state_config(cfg, amplitude=val)
        # ---------------------------------------------------------------------------
        # MEASURE OBSERVABLES
        if measure_obs:
            model.measure_observables(ii, dynamics=True)
            res["E2"][ii] = model.link_avg(obs_name="E2")
            if not model.pure_theory:
                res["N"][ii] += 0.5 * model.stag_avg(model.res["N"], "even")
                res["N"][ii] += 0.5 * model.stag_avg(model.res["N_zero"], "odd")
            for obs_names_list in plaquette_obs:
                obs = "_".join(obs_names_list)
                res[obs][ii] = model.res[obs]
        # ---------------------------------------------------------------------------
        # Overlaps
        if get_fidelity:
            res["fidelity"][ii] = model.measure_fidelity(in_state, ii, True, True)
            for state in overlap_with:
                res[f"overlap_{state}"][ii] = model.measure_fidelity(
                    overlap_states[state], ii, True, True
                )
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    tot_time = end_time - start_time
    logger.info(f"TIME SIMS {tot_time:.5f}")
    return res


# %%
# 3+1 QED STRING BREAKING DYNAMICS
par = {
    "model": {
        "lvals": [2, 2, 2],
        "has_obc": [True, True, True],
        "spin": 1,
        "pure_theory": False,
        "ham_format": "sparse",
        "bg_list": [-1, 0, 0, 0, 0, 0, 0, 1],
    },
    "dynamics": {
        "time_evolution": True,
        "start": 0,
        "stop": 10,
        "delta_n": 0.1,
        "state": "S1",
    },
    "momentum": {
        "get_momentum_basis": False,
        "unit_cell_size": [1, 1, 1],
        "momentum_k_vals": [0, 0, 0],
    },
    "observables": {
        "measure_obs": True,
        "get_entropy": True,
        "get_PE": False,
        "get_SRE": False,
        "entropy_partition": [0, 1, 2, 3],
        "get_state_configs": True,
        "get_fidelity": True,
        "overlap_with": ["S2", "S3", "S4", "S5", "S6"],
    },
    "g": 3,
    "m": 1,
}
res = run_QED_dynamics(par)
# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
# Fidelity plot
axs[0].plot(
    res["time"],
    res["fidelity"],
    "-",
    markersize=5,
    label=r"$\mathcal{F}_{S1}$",
)
for state in par["observables"]["overlap_with"]:
    axs[0].plot(
        res["time"],
        res[f"overlap_{state}"],
        "-",
        markersize=5,
        label=rf"$\mathcal{{F}}_{{{state}}}$",
    )
axs[0].set(
    ylabel=r"fidelities $\mathcal{F}(t)$",
    yticks=np.arange(0, 1.1, 0.2),
)
axs[0].grid(True, which="both", linestyle="-", linewidth=0.4)
axs[0].legend(
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.67, 0.92),
    frameon=True,  # <- box on
    fancybox=True,  # rounded corners (set False for sharp)
    framealpha=1.0,  # opaque box (e.g. 0.6 for semi-transparent)
    edgecolor="black",
    facecolor="white",
    borderpad=0.3,
)
# Local observables plot
axs[1].plot(
    res["time"],
    res["E2"],
    "-",
    color="darkblue",
    markersize=5,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=1,
    label=r"casimir $E^{2}$",
)
axs[1].plot(
    res["time"],
    res["N"],
    "-",
    color="darkred",
    markersize=5,
    markeredgecolor="darkred",
    markerfacecolor="white",
    markeredgewidth=1,
    label=r"particle density $N$",
)
axs[1].set(ylabel=r"observables $\langle\hat{O}(t)\rangle$")
axs[1].grid(True, which="both", linestyle="-", linewidth=0.4)
axs[1].legend(
    loc="upper center",
    ncol=1,
    bbox_to_anchor=(0.67, 0.87),
    frameon=True,  # <- box on
    fancybox=True,  # rounded corners (set False for sharp)
    framealpha=1.0,  # opaque box (e.g. 0.6 for semi-transparent)
    edgecolor="black",
    facecolor="white",
    borderpad=0.3,
)
# Entropy plot
axs[2].plot(
    res["time"],
    res["entropy"],
    "-",
    color="black",
    markersize=5,
    markeredgecolor="black",
    markerfacecolor="white",
    markeredgewidth=1,
)
axs[2].set(
    xlabel=r"time $t$",
    ylabel=r"entropy $S(t)$",
    xticks=np.arange(0, 11, 1),
)
axs[2].grid(True, which="both", linestyle="-", linewidth=0.4)
# %%
