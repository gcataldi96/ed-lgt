# %%
import numpy as np
import logging
from time import perf_counter

logger = logging.getLogger(__name__)
from ed_lgt.models import QED_Model
from ed_lgt.modeling import diagonalize_density_matrix


def _get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


par = {
    "model": {
        "lvals": [2, 2, 2],
        "has_obc": [False, False, False],
        "spin": 1,
        "pure_theory": True,
        "ham_format": "sparse",
    },
    "hamiltonian": {
        "n_eigs": 1,
        "save_psi": False,
    },
    "momentum": {
        "get_momentum_basis": False,
        "unit_cell_size": [1, 1, 1],
        "momentum_k_vals": [0, 0, 0],
    },
    "observables": {
        "measure_obs": True,
        "get_entropy": False,
        "entropy_partition": [0, 1],
        "get_state_configs": True,
        "get_overlap": False,
    },
    "g": 2.3277777777777775,
    "theta": 10,
}


def run_QED_simulation(par: dict) -> dict:
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
        TC_symmetry = _get(par, ["momentum", "TC_symmetry"], False)
        model.set_momentum_sector(unit_cell_size, k_vals, TC_symmetry)
    # -------------------------------------------------------------------------------
    model.default_params()
    # -------------------------------------------------------------------------------
    # Build Hamiltonian
    m = par["m"] if not model.pure_theory else None
    model.build_Hamiltonian(par["g"], m, theta=par.get("theta", 0.0))
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
    n_eigs = _get(par, ["hamiltonian", "n_eigs"], "full")
    save_psi = _get(par, ["hamiltonian", "save_psi"], False)
    model.diagonalize_Hamiltonian(n_eigs, model.ham_format, print_results=True)
    n_eigs_eff = len(model.sector_configs) if n_eigs == "full" else int(n_eigs)
    res["energy"] = model.H.Nenergies
    if save_psi:
        for ii in range(n_eigs_eff):
            res[f"psi{ii}"] = model.H.Npsi[ii].psi
    # -------------------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = ["E2"]
    local_obs += [f"E_{s}{d}" for d in model.directions for s in "mp"]
    local_obs += [f"E2_{s}{d}" for d in model.directions for s in "mp"]
    if not model.pure_theory:
        local_obs += ["N"]
    for obs in local_obs:
        res[obs] = np.zeros(n_eigs, dtype=float)
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
    else:
        plaquette_obs = []
    for obs_names_list in plaquette_obs:
        obs = "_".join(obs_names_list)
        res[obs] = np.zeros(n_eigs, dtype=float)
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
    res["entropy"] = np.zeros(model.H.n_eigs, dtype=float)
    get_state_configs = _get(par, ["observables", "get_state_configs"], False)
    # -------------------------------------------------------------------------------
    # Main loop over eigenstates
    for ii in range(model.H.n_eigs):
        model.H.print_energy(ii)
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
                # Sort eigenvalues and eigenvectors in descending order.
                # Note: np.argsort sorts in ascending order;
                # we reverse to get descending order.
                sorted_indices = np.argsort(rho_eigvals)[::-1]
                rho_eigvals = rho_eigvals[sorted_indices]
                logger.info(f"eigvals {rho_eigvals}")
                res["eigvals"] = rho_eigvals
                rho_eigvecs = rho_eigvecs[:, sorted_indices]
                # Set a list of truncation values to reduce the RDM
                truncation_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
                for _, threshold in enumerate(truncation_values):
                    # Determine how many eigenvectors have eigenvalues > the threshold.
                    # If too few are significant, relax the threshold to get at least 2.
                    P_columns = np.sum(rho_eigvals > threshold)
                    while P_columns < 2:
                        threshold /= 10
                        P_columns = np.sum(rho_eigvals > threshold)
                    logger.info(f"SIGNIFICANT EIGENVALUES {P_columns} > {threshold}")
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
        # ---------------------------------------------------------------------------
        # MEASURE OBSERVABLES
        if measure_obs:
            model.measure_observables(ii)
            for obs in local_obs:
                res[obs][ii] = np.mean(model.res[obs])
            for obs_names_list in plaquette_obs:
                obs = "_".join(obs_names_list)
                res[obs][ii] = model.res[obs]
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
    return res


# %%
run_QED_simulation(par)
# %%
