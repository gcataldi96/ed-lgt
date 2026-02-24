from time import perf_counter
import numpy as np
import logging
from ed_lgt.modeling import diagonalize_density_matrix
from ed_lgt.models import SU2_Model

logger = logging.getLogger(__name__)

__all__ = [
    "_get",
    "run_SU2_spectrum",
    "run_SU2_dynamics",
    "su2_get_momentum_params",
    "compare_SU2_models",
]


def _get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def compare_SU2_models(lvals, pure_theory, has_obc, g=0.1, m=0.1, atol=1e-10, neigs=4):
    if not pure_theory:
        sectors = [np.prod(lvals)]
    else:
        sectors = None
        m = None
    # ------------------------------------------------------------------------
    model_base = SU2_Model(
        spin=0.5,
        pure_theory=pure_theory,
        sectors=sectors,
        use_generic_model=False,
        lvals=lvals,
        has_obc=has_obc,
        ham_format="sparse",
    )
    model_base.default_params()
    # ------------------------------------------------------------------------
    model_gen = SU2_Model(
        spin=0.5,
        pure_theory=pure_theory,
        sectors=sectors,
        use_generic_model=True,
        lvals=lvals,
        has_obc=has_obc,
        ham_format="sparse",
    )
    model_gen.default_params()
    # ------------------------------------------------------------------------
    # Build SU2 Hamiltonians
    model_base.build_base_Hamiltonian(g, m)
    model_gen.build_gen_Hamiltonian(g, m)
    # ------------------------------------------------------------------------
    # 1) Ensure the symmetry sector basis is identical
    # (If this fails, comparing Hamiltonians is meaningless.)
    sc_base = model_base.sector_configs
    sc_gen = model_gen.sector_configs
    assert sc_base.shape == sc_gen.shape
    if not np.array_equal(sc_base, sc_gen):
        raise ValueError("std sector_configs differs from generalized one")
    # ------------------------------------------------------------------------
    # 2) Compare Hamiltonian eigenvalues
    if neigs > model_base.sector_dim - 1:
        neigs = "full"
        model_base.ham_format = "dense"
        model_gen.ham_format = "dense"
    model_base.diagonalize_Hamiltonian(neigs, model_base.ham_format, True)
    model_gen.diagonalize_Hamiltonian(neigs, model_gen.ham_format, True)
    for ii in range(len(model_base.H.Nenergies)):
        e_base = model_base.H.Nenergies[ii]
        e_gen = model_gen.H.Nenergies[ii]
        if not np.isclose(e_base, e_gen, atol=atol):
            msg = f"Hamiltonian eigenvalue {ii} mismatch: {e_base} vs {e_gen}"
            raise ValueError(msg)


def su2_build_model_and_hamiltonian(par: dict) -> SU2_Model:
    model = SU2_Model(**par["model"])
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
    model.build_Hamiltonian(g, m)
    return model


def su2_prepare_observables(model: SU2_Model, par, n_points):
    # Topological term
    theta = par.get("theta", 0.0) if model.pure_theory else 0
    # Local observables
    local_obs = [f"T2_p{d}" for d in model.directions]
    if not model.pure_theory:
        local_obs += [f"N_{label}" for label in ["tot", "single", "pair", "zero"]]
    # Two-body observables
    twobody_obs, twobody_axes = [], []
    # plaquettes
    if np.all([model.spin < 1, model.dim in (2, 3), not model.use_generic_model]):
        if model.dim == 2:
            plaquette_obs = [["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]]
        else:
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
    get_overlap = _get(par, ["observables", "get_overlap"], False)
    measure_obs = _get(par, ["observables", "measure_obs"], False)
    get_state_configs = _get(par, ["observables", "get_state_configs"], False)
    # flags
    flags = dict(
        measure_obs=measure_obs,
        get_entropy=get_entropy,
        get_rdm=get_rdm,
        get_overlap=get_overlap,
        get_state_configs=get_state_configs,
        partition_indices=partition,
        local_obs=local_obs,
        twobody_obs=twobody_obs,
        plaquette_obs=plaquette_obs,
    )
    return res, flags


def su2_measure_on_states(
    model: SU2_Model,
    par,
    res,
    flags,
    n_points,
    state_getter,
    overlap_state=None,
    dynamics=False,
):
    logger.info(f"----------------------------------------------------")
    logger.info(f"MEASURE OBSERVABLES on STATES")
    logger.info(f"----------------------------------------------------")
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
            res["E2"][ii] = model.link_avg(obs_name="T2")
            if not model.pure_theory:
                res["N_single"][ii] = model.stag_avg(model.res["N_single"])
                res["N_pair"][ii] += 0.5 * model.stag_avg(model.res["N_pair"], "even")
                res["N_pair"][ii] += 0.5 * model.stag_avg(model.res["N_zero"], "odd")
                res["N_zero"][ii] += 0.5 * model.stag_avg(model.res["N_zero"], "even")
                res["N_zero"][ii] += 0.5 * model.stag_avg(model.res["N_pair"], "odd")
                res["N_tot"][ii] = res["N_single"][ii] + 2.0 * res["N_pair"][ii]
            for obs_names_list in flags["plaquette_obs"]:
                obs = "_".join(obs_names_list)
                res[obs][ii] = model.res[obs]
        # overlap
        if overlap_state is not None and flags["get_overlap"]:
            res["overlap"][ii] = model.measure_fidelity(
                overlap_state, ii, print_value=True, dynamics=dynamics
            )
    return res


def run_SU2_spectrum(par):
    logger.info(f"----------------------------------------------------")
    logger.info(f"RUN SU2 SPECTRUM")
    logger.info(f"----------------------------------------------------")
    start_time = perf_counter()
    model = su2_build_model_and_hamiltonian(par)
    # diagonalize
    n_eigs = _get(par, ["hamiltonian", "n_eigs"], "full")
    save_psi = _get(par, ["hamiltonian", "save_psi"], False)
    model.diagonalize_Hamiltonian(n_eigs, model.ham_format, print_results=True)
    n_points = len(model.sector_configs) if n_eigs == "full" else int(n_eigs)
    # observables
    res, flags = su2_prepare_observables(model, par, n_points)
    res["energy"] = model.H.Nenergies
    if save_psi:
        for ii in range(n_points):
            res[f"psi{ii}"] = model.H.Npsi[ii].psi
    # overlap reference state (optional)
    overlap_state = None
    get_overlap = _get(par, ["observables", "get_overlap"], False)
    if get_overlap:
        name = _get(par, ["hamiltonian", "state"], None)
        config = model.overlap_QMB_state(name)
        overlap_state = model.get_qmb_state_from_configs([config])
        res["overlap"] = np.zeros(n_points, dtype=float)
    # measure observables
    res = su2_measure_on_states(
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


def run_SU2_dynamics(par):
    logger.info(f"----------------------------------------------------")
    logger.info(f"RUN SU2 DYNAMICS")
    logger.info(f"----------------------------------------------------")
    start_time = perf_counter()
    model = su2_build_model_and_hamiltonian(par)
    # timeline
    start = par["dynamics"]["start"]
    stop = par["dynamics"]["stop"]
    dt = par["dynamics"]["delta_n"]
    time_line = np.arange(start, stop + dt, dt)
    n_points = len(time_line)
    res, flags = su2_prepare_observables(model, par, n_points)
    res["time_steps"] = time_line
    # initial state
    name = par["dynamics"]["state"]
    config = model.overlap_QMB_state(name)
    in_state = model.get_qmb_state_from_configs([config])
    # overlap reference state (optional)
    get_overlap = _get(par, ["observables", "get_overlap"], False)
    overlap_state = in_state if get_overlap else None
    if overlap_state is not None:
        res["overlap"] = np.zeros(n_points, dtype=float)
    if par["dynamics"]["time_evolution"]:
        model.time_evolution_Hamiltonian(in_state, time_line)
    # measure observables
    res = su2_measure_on_states(
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


def su2_get_momentum_params(TC_symmetry: bool, n_sites: int):
    logger.info(f"----------------------------------------------------")
    logger.info(f"GET MOMENTUM PARAMS")
    logger.info(f"----------------------------------------------------")
    if TC_symmetry:
        n_momenta = n_sites
        k_indices = np.arange(0, n_momenta, 1)
        k_physical = 2 * np.pi * k_indices / n_sites
        k_unit_cell_size = [1]
    else:
        n_momenta = n_sites // 2
        k_indices = np.arange(0, n_momenta, 1)
        k_physical = 4 * np.pi * k_indices / n_sites
        k_unit_cell_size = [2]
    momentum_params = {
        "n_momenta": n_momenta,
        "k_indices": k_indices,
        "k_physical": k_physical,
        "k_unit_cell_size": k_unit_cell_size,
        "TC_symmetry": TC_symmetry,
    }
    return momentum_params
