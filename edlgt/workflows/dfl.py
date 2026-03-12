from copy import deepcopy
from time import perf_counter
import logging

import numpy as np

from edlgt.modeling import get_lattice_link_site_pairs
from edlgt.models import DFL_Model
from edlgt.symmetries import get_symmetry_sector_generators, symmetry_sector_configs

logger = logging.getLogger(__name__)

__all__ = [
    "_get",
    "_get_gvals",
    "run_DFL_dynamics",
    "run_DFL_dynamics_sector_by_sector",
    "normalize_DFL_simsio_params",
]


def _get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _get_dtype_mode(par: dict):
    mode = _get(par, ["hamiltonian", "dtype_mode"], "auto")
    if mode == "auto":
        legacy_flag = _get(par, ["hamiltonian", "is_complex"], None)
        if legacy_flag is not None:
            mode = bool(legacy_flag)
    return mode


def _get_gvals(params: dict) -> np.ndarray:
    if "g_values" in params:
        return np.asarray(params["g_values"], dtype=float)
    if "g" in params:
        return np.atleast_1d(np.asarray(params["g"], dtype=float))
    return np.linspace(5, 10, 10)


def _get_time_line(params: dict) -> np.ndarray:
    start = params["dynamics"]["start"]
    stop = params["dynamics"]["stop"]
    delta_t = params["dynamics"]["delta_n"]
    return np.arange(start, stop + delta_t, delta_t)


def _get_dfl_symmetry_inputs(model: DFL_Model):
    global_ops = [model.ops["N_tot"]]
    global_sectors = np.array([model.n_sites], dtype=float)
    global_ops = get_symmetry_sector_generators(global_ops, action="global")
    dirs = model.directions
    link_ops = [[model.ops[f"T2_p{d}"], -model.ops[f"T2_m{d}"]] for d in dirs]
    link_sectors = [0 for _ in model.directions]
    link_ops = get_symmetry_sector_generators(link_ops, action="link")
    pair_list = get_lattice_link_site_pairs(model.lvals, model.has_obc)
    return global_ops, global_sectors, link_ops, link_sectors, pair_list


def _get_bgsector_groups(model: DFL_Model, logical_stag_basis: int):
    if model.n_sites % (2 * logical_stag_basis) != 0:
        raise ValueError("staggered matter config is not compatible with n_sites")
    bg_configs, bg_sectors = model.get_background_charges_configs(logical_stag_basis)
    unique_bg_sectors, indices = np.unique(bg_sectors, axis=0, return_inverse=True)
    grouped_configs = [
        bg_configs[indices == ii] for ii in range(len(unique_bg_sectors))
    ]
    weights = np.array([len(cfgs) for cfgs in grouped_configs], dtype=float)
    weights /= weights.sum()
    return bg_configs, bg_sectors, unique_bg_sectors, grouped_configs, weights


def _allocate_dynamics_results(
    params: dict, gvals: np.ndarray, time_line: np.ndarray
) -> tuple[dict, dict]:
    res = {"time": time_line, "g_values": gvals}
    flags = params["observables"]
    ngvals = len(gvals)
    n_steps = len(time_line)
    if flags["measure_obs"]:
        res["local_obs"] = ["T2_px"]
        matter_obs = [f"N_{lab}" for lab in ["tot", "single", "pair", "zero"]]
        res["local_obs"] += matter_obs
        for obs in res["local_obs"] + ["E2"]:
            res[obs] = np.zeros((ngvals, n_steps), dtype=float)
    res["partition_indices"] = _get(params, ["observables", "entropy_partition"], [])
    if flags["get_entropy"]:
        res["entropy"] = np.zeros((ngvals, n_steps), dtype=float)
    if flags["get_PE"]:
        res["PE"] = np.zeros((ngvals, n_steps), dtype=float)
    if flags["get_SRE"]:
        res["SRE"] = np.zeros((ngvals, n_steps), dtype=float)
    if flags["get_fidelity"]:
        res["fidelity"] = np.zeros((ngvals, n_steps), dtype=float)
    return res, flags


def _record_dynamics_step(
    model: DFL_Model, res: dict, gidx: int, tidx: int, in_state: np.ndarray, flags: dict
):
    if model.momentum_basis is None:
        if flags["get_entropy"]:
            res["entropy"][gidx, tidx] = model.H.psi_time[tidx].entanglement_entropy(
                res["partition_indices"], model._partition_cache
            )
        if flags["get_PE"]:
            res["PE"][gidx, tidx] = model.H.psi_time[tidx].participation_renyi_entropy()
        if flags["get_SRE"]:
            res["SRE"][gidx, tidx] = model.H.psi_time[tidx].stabilizer_renyi_entropy(
                model.sector_configs, prob_threshold=5e-2
            )
        if flags["get_state_configs"]:
            model.H.psi_time[tidx].get_state_configurations(1e-3, model.sector_configs)
    if flags["measure_obs"]:
        model.measure_observables(tidx, dynamics=True)
        res["E2"][gidx, tidx] = model.link_avg(obs_name="T2")
        if not model.pure_theory:
            res["N_single"][gidx, tidx] = model.stag_avg("N_single")
            res["N_pair"][gidx, tidx] = 0.5 * model.stag_avg("N_pair", "even")
            res["N_pair"][gidx, tidx] += 0.5 * model.stag_avg("N_zero", "odd")
            res["N_zero"][gidx, tidx] = 0.5 * model.stag_avg("N_zero", "even")
            res["N_zero"][gidx, tidx] += 0.5 * model.stag_avg("N_pair", "odd")
            res["N_tot"][gidx, tidx] += res["N_single"][gidx, tidx]
            res["N_tot"][gidx, tidx] += 2.0 * res["N_pair"][gidx, tidx]
    if flags["get_fidelity"]:
        res["fidelity"][gidx, tidx] = model.measure_fidelity(in_state, tidx, True, True)


def run_DFL_dynamics(params: dict) -> dict:
    start_time = perf_counter()
    gvals = _get_gvals(params)
    time_line = _get_time_line(params)
    n_steps = len(time_line)
    model = DFL_Model(**params["model"])
    global_ops, global_sectors, link_ops, link_sectors, pair_list = (
        _get_dfl_symmetry_inputs(model)
    )
    model.sector_configs = symmetry_sector_configs(
        loc_dims=model.loc_dims,
        glob_op_diags=global_ops,
        glob_sectors=global_sectors,
        sym_type_flag="U",
        link_op_diags=link_ops,
        link_sectors=link_sectors,
        pair_list=pair_list,
    )
    logical_stag_basis = params["dynamics"]["logical_stag_basis"]
    bg_configs, _, _, _, _ = _get_bgsector_groups(model, logical_stag_basis)
    in_state = model.get_qmb_state_from_configs(bg_configs)
    res, flags = _allocate_dynamics_results(params, gvals, time_line)
    if flags["get_entropy"]:
        model._partition_cache = {}
        model._get_partition(res["partition_indices"])
    dtype_mode = _get_dtype_mode(params)
    for gidx, g in enumerate(gvals):
        model.default_params()
        if flags["measure_obs"]:
            model.get_observables(res["local_obs"])
        model.build_Hamiltonian(g, params["m"], dtype_mode=dtype_mode)
        model.time_evolution_Hamiltonian(in_state, time_line)
        for tidx in range(n_steps):
            msg = f"TIME {time_line[tidx]:.3f}"
            logger.info(f"==================== {msg} ====================")
            _record_dynamics_step(model, res, gidx, tidx, in_state, flags)
    logger.info(f"TIME SIMS {round(perf_counter() - start_time, 5)}")
    return res


def run_DFL_dynamics_sector_by_sector(params: dict) -> dict:
    start_time = perf_counter()
    gvals = _get_gvals(params)
    time_line = _get_time_line(params)
    n_steps = len(time_line)
    model = DFL_Model(**params["model"])
    res, flags = _allocate_dynamics_results(params, gvals, time_line)
    if flags["get_PE"]:
        pe_ipr = np.zeros((len(gvals), n_steps), dtype=float)
    global_ops, global_sectors, link_ops, link_sectors, pair_list = (
        _get_dfl_symmetry_inputs(model)
    )
    bg_global_ops = get_symmetry_sector_generators([model.ops["bg"]], action="global")
    bg_sector_value = float(np.max(bg_global_ops))
    logical_stag_basis = params["dynamics"]["logical_stag_basis"]
    _, _, unique_bg_sectors, grouped_configs, weights = _get_bgsector_groups(
        model, logical_stag_basis
    )
    dtype_mode = _get_dtype_mode(params)
    for bg_num, (bg_sector, bg_config_group, weight) in enumerate(
        zip(unique_bg_sectors, grouped_configs, weights)
    ):
        sector_res, _ = _allocate_dynamics_results(params, gvals, time_line)
        logger.info("----------------------------------------------")
        logger.info(f"BG SECTOR {bg_sector}")
        for bg_config in bg_config_group:
            logger.info(f"BG CONFIG {bg_config}")
        model.sector_configs = symmetry_sector_configs(
            loc_dims=model.loc_dims,
            glob_op_diags=global_ops,
            glob_sectors=global_sectors,
            sym_type_flag="U",
            link_op_diags=link_ops,
            link_sectors=link_sectors,
            pair_list=pair_list,
            string_op_diags=bg_global_ops,
            string_sectors=bg_sector_value * np.atleast_2d(bg_sector).astype(float),
        )
        in_state = model.get_qmb_state_from_configs(bg_config_group)
        if flags["get_entropy"]:
            model._partition_cache = {}
            model._get_partition(sector_res["partition_indices"])
        for gidx, g in enumerate(gvals):
            model.default_params()
            if flags["measure_obs"]:
                model.get_observables(sector_res["local_obs"])
            model.build_Hamiltonian(g, params["m"], dtype_mode=dtype_mode)
            model.time_evolution_Hamiltonian(in_state, time_line)
            for tidx in range(n_steps):
                msg = f"SECTOR {bg_num} TIME {time_line[tidx]:.3f}"
                logger.info(f"=============== {msg} ===============")
                _record_dynamics_step(model, sector_res, gidx, tidx, in_state, flags)
        if flags["get_PE"]:
            pe_ipr += (weight**2) * np.exp(-sector_res["PE"])
        for key, value in sector_res.items():
            if key in ["time", "g_values", "local_obs", "partition_indices", "PE"]:
                continue
            conditions = [key in res, isinstance(value, np.ndarray)]
            conditions += [value.shape == (len(gvals), n_steps)]
            if np.all(conditions):
                res[key] += weight * value
    if flags["get_PE"]:
        res["PE"] = -np.log(pe_ipr + 1e-16)
    logger.info(f"TIME SIMS {round(perf_counter() - start_time, 5)}")
    return res


def normalize_DFL_simsio_params(par: dict) -> dict:
    params = deepcopy(par)
    observables = deepcopy(params.get("observables", {}))
    defaults = {
        "measure_obs": False,
        "get_entropy": False,
        "get_PE": False,
        "get_SRE": False,
        "get_state_configs": False,
        "get_fidelity": False,
        "entropy_partition": [],
    }
    legacy_keys = {
        "measure_obs": "measure_obs",
        "get_entropy": "get_entropy",
        "get_PE": "get_PE",
        "get_SRE": "get_SRE",
        "get_state_configs": "get_state_configs",
        "entropy_partition": "entropy_partition",
    }
    for new_key, old_key in legacy_keys.items():
        if new_key not in observables and old_key in params:
            observables[new_key] = params[old_key]
    if "get_fidelity" not in observables:
        if "get_fidelity" in params:
            observables["get_fidelity"] = params["get_fidelity"]
        else:
            observables["get_fidelity"] = params.get("get_overlap", False)
    for key, value in defaults.items():
        observables.setdefault(key, value)
    params["observables"] = observables
    params.setdefault(
        "ensemble",
        {
            "microcanonical": {"average": False},
            "diagonal": {"average": False},
            "canonical": {"average": False},
        },
    )
    state = _get(params, ["dynamics", "state"], None)
    if state not in (None, "background"):
        logger.warning(
            "Current DFL dynamics ignores dynamics.state=%r and uses the equal-weight "
            "superposition of background-charge configurations.",
            state,
        )
    for ensemble_name in ("microcanonical", "diagonal", "canonical"):
        if _get(params, ["ensemble", ensemble_name, "average"], False):
            logger.warning(
                "Current DFL dynamics wrapper ignores ensemble.%s.average.",
                ensemble_name,
            )
    if not _get(params, ["dynamics", "time_evolution"], True):
        logger.warning(
            "Current DFL simsio wrapper is dynamics-only; it will still evolve over "
            "the provided time grid."
        )
    return params
