# %%
import numpy as np
import logging
from time import perf_counter
from edlgt.modeling import diagonalize_density_matrix
from edlgt.models import SU2_Model

logger = logging.getLogger(__name__)


def _get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


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
        res["SRE"] = np.zeros(n_steps, dtype=float)
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
                res["SRE"][ii] = model.H.psi_time[ii].stabilizer_renyi_entropy(
                    model.sector_configs, prob_threshold=1e-2
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


# %%
par = {
    "model": {
        "lvals": [10],
        "sectors": [10],
        "has_obc": [False],
        "spin": 0.5,
        "pure_theory": False,
        "ham_format": "sparse",
    },
    "dynamics": {
        "time_evolution": True,
        "start": 0,
        "stop": 10,
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
        "get_PE": False,
        "get_SRE": False,
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
(l3,) = ax2.plot(res["time_steps"], res["SRE"], label="SRE", c="green")
ax2.set_ylabel("Participation Renyi-2 Entropy PE2")  # right ylabel
ax1.set(xticks=xticks, xlabel="time t")

# optional: set y-limits independently (edit as needed)
# ax1.set_ylim(0, 1)
# ax2.set_ylim(res["PE"].min(), res["PE"].max())
# grid (draw it from the left axis)
ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

fig.tight_layout()
fig.savefig("PE2_Scar_SU2_PV.pdf")
