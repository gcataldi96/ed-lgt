from time import perf_counter
import numpy as np
import logging
from ed_lgt.modeling import diagonalize_density_matrix
from ed_lgt.models import QED_Model

logger = logging.getLogger(__name__)

__all__ = ["run_QED_spectrum"]


def _get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def QED_build_model_and_hamiltonian(par: dict) -> QED_Model:
    model = QED_Model(**par["model"])
    # momentum sector
    if _get(par, ["momentum", "get_momentum_basis"], False):
        unit_cell_size = _get(par, ["momentum", "unit_cell_size"], None)
        k_vals = _get(par, ["momentum", "momentum_k_vals"], None)
        TC = _get(par, ["momentum", "TC_symmetry"], False)
        model.set_momentum_sector(unit_cell_size, k_vals, TC)
    model.default_params()
    # Hamiltonian
    g = par["g"]
    m = par.get("m", None) if not model.pure_theory else None
    theta = par.get("theta", 0)
    model.build_Hamiltonian(g, m, theta)
    return model


def QED_prepare_observables(model: QED_Model, par, n_points):
    # Local observables
    local_obs = [f"E2_p{d}" for d in model.directions]
    if not model.pure_theory:
        local_obs += ["N"]
    # Two-body observables
    twobody_obs, twobody_axes = [], []
    # plaquettes
    if model.spin < 1 and model.dim in (2, 3):
        if model.dim == 2:
            plaquette_obs = [["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]]
        else:
            plaquette_obs = [
                ["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"],
                ["C_px,pz", "C_pz,mx", "C_mz,px", "C_mx,mz"],
                ["C_py,pz", "C_pz,my", "C_mz,py", "C_my,mz"],
            ]
    else:
        plaquette_obs = []
    # register observables
    model.get_observables(
        local_obs, twobody_obs, plaquette_obs, twobody_axes=twobody_axes
    )
    # allocate results
    res = {}
    res["E2"] = np.zeros(n_points, dtype=float)
    for obs in local_obs:
        res[obs] = np.zeros(n_points, dtype=float)
    for names in plaquette_obs:
        res["_".join(names)] = np.zeros(n_points, dtype=float)
    # entropy/RDM config
    partition = _get(par, ["observables", "entropy_partition"], [])
    get_entropy = _get(par, ["observables", "get_entropy"], False)
    get_rdm = _get(par, ["observables", "get_RDM"], False)
    if get_entropy or get_rdm:
        model._get_partition(partition)
    res["entropy"] = np.zeros(n_points, dtype=float)
    # flags
    flags = dict(
        measure_obs=_get(par, ["observables", "measure_obs"], False),
        get_entropy=get_entropy,
        get_rdm=get_rdm,
        get_state_configs=_get(par, ["observables", "get_state_configs"], False),
        partition_indices=partition,
    )
    return res, flags


def QED_measure_on_states(
    model: QED_Model,
    par,
    res,
    flags,
    n_points,
    state_getter,
    overlap_state=None,
    dynamics=False,
):
    for ii in range(n_points):
        st = state_getter(ii)  # QMB_state-like
        if not dynamics:
            model.H.print_energy(ii)
        else:
            tstep = ii * par["dynamics"]["delta_n"]
            msg = f"TIME {tstep:.2f}"
            logger.info(f"================== {msg} =======================")
        if model.momentum_basis is None:
            if flags["get_rdm"]:
                RDM = st.reduced_density_matrix(
                    flags["partition_indices"], model._partition_cache
                )
                rho_eigvals, _ = diagonalize_density_matrix(RDM)
                rho_eigvals = rho_eigvals[np.argsort(rho_eigvals)[::-1]]
                res["eigvals"] = (
                    rho_eigvals  # last one wins, consistent with your script
                )
            if flags["get_entropy"]:
                res["entropy"][ii] = st.entanglement_entropy(
                    flags["partition_indices"], model._partition_cache
                )
            if flags["get_state_configs"]:
                st.get_state_configurations(1e-3, model.sector_configs)
        # observables
        if flags["measure_obs"]:
            model.measure_observables(ii, dynamics=dynamics)
            res["E2"][ii] = model.link_avg(obs_name="E2")
            if not model.pure_theory:
                res["N"][ii] += 0.5 * model.stag_avg(model.res["N_pair"], "even")
        # overlap
        if overlap_state is not None and _get(
            par, ["observables", "get_overlap"], False
        ):
            res["overlap"][ii] = model.measure_fidelity(
                overlap_state, ii, print_value=True, dynamics=dynamics
            )


def run_QED_spectrum(par):
    start_time = perf_counter()
    model = QED_build_model_and_hamiltonian(par)
    # diagonalize
    n_eigs = _get(par, ["hamiltonian", "n_eigs"], "full")
    save_psi = _get(par, ["hamiltonian", "save_psi"], False)
    model.diagonalize_Hamiltonian(n_eigs, model.ham_format, print_results=True)
    n_points = len(model.sector_configs) if n_eigs == "full" else int(n_eigs)
    # observables
    res, flags = QED_prepare_observables(model, par, n_points)
    res["energy"] = model.H.Nenergies
    if save_psi:
        for ii in range(n_points):
            res[f"psi{ii}"] = model.H.Npsi[ii].psi
    # overlap reference state (optional)
    overlap_state = None
    if _get(par, ["observables", "get_overlap"], False):
        name = _get(par, ["hamiltonian", "state"], None)
        config = model.overlap_QMB_state(name)
        overlap_state = model.get_qmb_state_from_configs([config])
        res["overlap"] = np.zeros(n_points, dtype=float)
    # measure observables
    QED_measure_on_states(
        model,
        par,
        res,
        flags,
        n_points,
        state_getter=lambda ii: model.H.Npsi[ii],
        overlap_state=overlap_state,
        dynamics=False,
    )
    end_time = perf_counter()
    tot_time = end_time - start_time
    res["total_time"] = tot_time
    logger.info(f"TIME SIMS {tot_time:.5f}")
    return res


def run_QED_dynamics(par):
    start_time = perf_counter()
    model = QED_build_model_and_hamiltonian(par)
    # timeline
    start = par["dynamics"]["start"]
    stop = par["dynamics"]["stop"]
    dt = par["dynamics"]["delta_n"]
    time_line = np.arange(start, stop + dt, dt)
    n_points = len(time_line)
    res, flags = QED_prepare_observables(model, par, n_points)
    res["time_steps"] = time_line
    # initial state
    name = par["dynamics"]["state"]
    config = model.overlap_QMB_state(name)
    in_state = model.get_qmb_state_from_configs([config])
    # overlap reference state (optional)
    overlap_state = (
        in_state if _get(par, ["observables", "get_overlap"], False) else None
    )
    if overlap_state is not None:
        res["overlap"] = np.zeros(n_points, dtype=float)
    if par["dynamics"]["time_evolution"]:
        model.time_evolution_Hamiltonian(in_state, time_line)
    # measure observables
    QED_measure_on_states(
        model,
        par,
        res,
        flags,
        n_points,
        state_getter=lambda ii: model.H.psi_time[ii],
        overlap_state=overlap_state,
        dynamics=True,
    )
    end_time = perf_counter()
    tot_time = end_time - start_time
    res["total_time"] = tot_time
    logger.info(f"TIME SIMS {tot_time:.5f}")
    return res
