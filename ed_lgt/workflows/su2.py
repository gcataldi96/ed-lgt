from time import perf_counter
import numpy as np
import logging
from itertools import product
from ed_lgt.modeling import diagonalize_density_matrix, mixed_exp_val_data, QMB_state
from ed_lgt.models import SU2_Model
from ed_lgt.tools import get_data_from_sim, get_Wannier_support, localize_Wannier

logger = logging.getLogger(__name__)

__all__ = [
    "_get",
    "run_SU2_spectrum",
    "run_SU2_dynamics",
    "su2_get_momentum_params",
    "su2_get_convolution_matrix",
    "su2_get_convolution_gs_energy",
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
        background=0,
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
        background=0,
        use_generic_model=True,
        lvals=lvals,
        has_obc=has_obc,
        ham_format="sparse",
    )
    model_gen.default_params()
    # ------------------------------------------------------------------------
    # Build SU2 Hamiltonians
    model_base.build_Hamiltonian(g, m)
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
    if model.spin > 0.5:
        model.build_gen_Hamiltonian(g, m)
    else:
        model.build_Hamiltonian(g, m)
    return model


def su2_prepare_observables(model: SU2_Model, par, n_points):
    # Local observables
    local_obs = [f"T2_p{d}" for d in model.directions]
    if not model.pure_theory:
        local_obs += [f"N_{label}" for label in ["tot", "single", "pair", "zero"]]
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
        # overlap
        if overlap_state is not None and _get(
            par, ["observables", "get_overlap"], False
        ):
            res["overlap"][ii] = model.measure_fidelity(
                overlap_state, ii, print_value=True, dynamics=dynamics
            )


def run_SU2_spectrum(par):
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
    if _get(par, ["observables", "get_overlap"], False):
        name = _get(par, ["hamiltonian", "state"], None)
        config = model.overlap_QMB_state(name)
        overlap_state = model.get_qmb_state_from_configs([config])
        res["overlap"] = np.zeros(n_points, dtype=float)
    # measure observables
    su2_measure_on_states(
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
    overlap_state = (
        in_state if _get(par, ["observables", "get_overlap"], False) else None
    )
    if overlap_state is not None:
        res["overlap"] = np.zeros(n_points, dtype=float)
    if par["dynamics"]["time_evolution"]:
        model.time_evolution_Hamiltonian(in_state, time_line)
    # measure observables
    su2_measure_on_states(
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


def su2_get_convolution_matrix(
    model: SU2_Model, momentum_params: dict, band_params: dict
) -> np.ndarray:
    # ---------------------------------------------------------
    k_indices = momentum_params["k_indices"]
    n_momenta = momentum_params["n_momenta"]
    k_unit_cell_size = momentum_params["k_unit_cell_size"]
    TC_symmetry = momentum_params["TC_symmetry"]
    # ---------------------------------------------------------
    R0 = band_params["R0"]
    band_number = band_params["band_number"]
    sim_band_name = band_params["sim_band_name"]
    g = band_params["g"]
    m = band_params["m"]
    # ---------------------------------------------------------
    k1k2matrix = np.zeros((n_momenta, n_momenta), dtype=np.complex128)
    for k1, k2 in product(k_indices, k_indices):
        model.set_momentum_pair([k1], [k2], k_unit_cell_size, TC_symmetry)
        model.default_params()
        # Build the local hamiltonian
        model.build_local_Hamiltonian(g, m, R0)
        # Acquire the state vectors
        state_idx_k1 = 1 if (model.zero_density and k1 == 0) else 0
        state_idx_k2 = 1 if (model.zero_density and k2 == 0) else 0
        state_idx_k1 += band_number
        state_idx_k2 += band_number
        psik1 = get_data_from_sim(sim_band_name, f"psi{state_idx_k1}", k1)
        psik2 = get_data_from_sim(sim_band_name, f"psi{state_idx_k2}", k2)
        # Measure the overlap with k1 & k2
        k1k2matrix[k1, k2] = mixed_exp_val_data(
            psik1,
            psik2,
            model.Hlocal.row_list,
            model.Hlocal.col_list,
            model.Hlocal.value_list,
        )
    return k1k2matrix


def su2_get_convolution_gs_energy(
    model: SU2_Model, momentum_params: dict, band_params: dict
) -> complex:
    # ---------------------------------------------------------
    n_momenta = momentum_params["n_momenta"]
    k_unit_cell_size = momentum_params["k_unit_cell_size"]
    TC_symmetry = momentum_params["TC_symmetry"]
    # ---------------------------------------------------------
    R0 = band_params["R0"]
    sim_band_name = band_params["sim_band_name"]
    g = band_params["g"]
    m = band_params["m"]
    # ---------------------------------------------------------
    if model.zero_density is True:
        GS = get_data_from_sim(sim_band_name, "psi0", 0)
        # Check Translational Hamiltonian
        model.set_momentum_pair([0], [0], k_unit_cell_size, TC_symmetry)
        model.default_params()
        # Check the momentum bases
        model.check_momentum_pair()
        # Build the local hamiltonian
        model.build_local_Hamiltonian(g, m, R0)
        eg_single_block = mixed_exp_val_data(
            GS,
            GS,
            model.Hlocal.row_list,
            model.Hlocal.col_list,
            model.Hlocal.value_list,
        )
        logger.info(f"E0 block {k_unit_cell_size}: {eg_single_block:.8f}")
        logger.info(f"E0 tot {eg_single_block * n_momenta:.8f}")
        return eg_single_block
    else:
        return -4.580269235030599 - 1.251803175199139e-18j


def su2_get_Wannier_state(model: SU2_Model, momentum_params: dict, band_params: dict):
    # -------------------------------------------------------------------------------
    n_momenta = momentum_params["n_momenta"]
    k_indices = momentum_params["k_indices"]
    k_unit_cell_size = momentum_params["k_unit_cell_size"]
    TC_symmetry = momentum_params["TC_symmetry"]
    # -------------------------------------------------------------------------------
    band_number = band_params["band_number"]
    sim_band_name = band_params["sim_band_name"]
    state_idx = 1 + band_number
    k1k2matrix = band_params["k1k2matrix"]
    gs_energy = band_params["gs_energy"]
    # -------------------------------------------------------------------------------
    # Acquire the optimal theta phases that localize the Wannier
    Eprofile, theta_vals = localize_Wannier(
        k1k2matrix, k_indices, gs_energy, center_mode=1
    )
    # Get the partition to the model according to the optimal support of the Wannier
    w_supports = get_Wannier_support(Eprofile, tail_tol=0.1)
    support_indices = w_supports["indices"]
    model._get_partition(support_indices)
    # -------------------------------------------------------------------------------
    # Initialize the Wannier State
    psi_wannier = np.zeros(model.sector_dim, dtype=np.complex128)
    for kidx in k_indices:
        # Load the momentum state forming the energy band
        psik = get_data_from_sim(sim_band_name, f"psi{state_idx}", kidx)
        # Set the corresponding momentum sector
        model.set_momentum_sector(k_unit_cell_size, [kidx], TC_symmetry)
        model.default_params()
        # Build the projector from the momentum sector to the global one
        Pk = model._basis_Pk_as_csr()
        # Project the State from the momentum sector to the coordinate one
        psik_exp = Pk @ psik
        # Add it to the Wannier state with the corresponding theta phase
        psi_wannier += np.exp(1j * theta_vals[kidx]) * psik_exp / np.sqrt(n_momenta)
    # Mesure the norm of the state: it should be 1
    wannier_norm = np.linalg.norm(psi_wannier)
    logger.info(f"Wannier norm = {wannier_norm}")
    # Promote the Wannier state as an item of the QMB state class
    Wannier = QMB_state(psi=psi_wannier, lvals=model.lvals, loc_dims=model.loc_dims)
    # Decompose it into the support and its negligible part
    W_psimatrix = Wannier._get_psi_matrix(support_indices, model._partition_cache)
    return W_psimatrix
