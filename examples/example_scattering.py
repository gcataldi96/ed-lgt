# %%
import numpy as np
from itertools import product
from time import perf_counter
from simsio import SimsQuery, get_sim, uids_grid
from ed_lgt.workflows import su2_get_momentum_params
from ed_lgt.symmetries import build_sector_expansion_projector
from ed_lgt.modeling import mixed_exp_val_data, QMB_state
from ed_lgt.models import SU2_Model
from ed_lgt.tools import (
    get_Wannier_support,
    localize_Wannier,
    choose_rank_by_frobenius,
    dense_operator_to_mpo,
    mpo_to_dense_operator,
)
import logging

logger = logging.getLogger(__name__)


def get_data_from_sim(sim_filename, obs_name, kindex):
    match = SimsQuery(group_glob=sim_filename)
    ugrid, _ = uids_grid(match.uids, ["momentum_k_vals"])
    return get_sim(ugrid[kindex]).res[obs_name]


def run_SU2_convolution(params):
    start_time = perf_counter()
    # -------------------------------------------------------------------------------
    # SU(2) MODEL in (1+1)D with MOMENTUM SYMMETRY
    model = SU2_Model(**params["model"])
    # -------------------------------------------------------------------------------
    TC_symmetry = params.get("TC_symmetry", False)
    momentum_params = su2_get_momentum_params(TC_symmetry, model.n_sites)
    band_params = {
        "sim_band_name": params["sim_band_name"],
        "band_number": params.get("band_number", 0),
        "m": params["m"],
        "g": params["g"],
        "R0": 0,
    }
    # -------------------------------------------------------------------------------
    # GET THE GROUND STATE ENERGY density at momentum 0
    band_params["gs_energy"] = su2_get_convolution_gs_energy(
        model, momentum_params, band_params
    )
    # -------------------------------------------------------------------------------
    # CONVOLUTION MATRIX
    if TC_symmetry:
        band_params["R0"] = 0
        band_params["k1k2matrix"] = su2_get_convolution_matrix(
            model, momentum_params, band_params
        )
        for ii in range(momentum_params["n_momenta"]):
            logger.info("==================")
            for jj in range(momentum_params["n_momenta"]):
                logger.info(f"{ii} {jj} {band_params['k1k2matrix'][ii, jj]}")
    else:
        band_params["R0"] = 0
        band_params["k1k2matrix_even"] = su2_get_convolution_matrix(
            model, momentum_params, band_params
        )
        band_params["R0"] = 1
        band_params["k1k2matrix_odd"] = su2_get_convolution_matrix(
            model, momentum_params, band_params
        )
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
    return model, momentum_params, band_params


def su2_get_convolution_matrix(
    model: SU2_Model, momentum_params: dict, band_params: dict
) -> np.ndarray:
    logger.info(f"----------------------------------------------------")
    logger.info(f"GET CONVOLUTION MATRIX")
    logger.info(f"----------------------------------------------------")
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
    logger.info(f"----------------------------------------------------")
    logger.info(f"GET CONVOLUTION Ground State ENERGY")
    logger.info(f"----------------------------------------------------")
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


def su2_get_Wannier_state(model: SU2_Model, momentum_params: dict, band_params: dict):
    logger.info(f"----------------------------------------------------")
    logger.info(f"GET WANNIER STATE")
    logger.info(f"----------------------------------------------------")
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
    support_tail = band_params["support_tail"]
    # -------------------------------------------------------------------------------
    # Acquire the optimal theta phases that localize the Wannier
    Eprofile, theta_vals = localize_Wannier(
        k1k2matrix, k_indices, gs_energy, center_mode=1
    )
    # -------------------------------------------------------------------------------
    # Get the partition to the model according to the optimal support of the Wannier
    W_support_indices = get_Wannier_support(Eprofile, tail_tol=support_tail)
    model._get_partition(W_support_indices)
    # -------------------------------------------------------------------------------
    # Initialize the Wannier State
    logger.info(f"----------------------------------------------------")
    logger.info("Initialize the Wannier State")
    psi_wannier = np.zeros(model.sector_configs.shape[0], dtype=np.complex128)
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
    logger.info(f"----------------------------------------------------")
    wannier_norm = np.linalg.norm(psi_wannier)
    logger.info(f"Wannier norm = {wannier_norm}")
    # Promote the Wannier state as an item of the QMB state class
    Wannier = QMB_state(psi=psi_wannier, lvals=model.lvals, loc_dims=model.loc_dims)
    # Decompose it into the support and its negligible part
    Wannier_matrix = Wannier._get_psi_matrix(W_support_indices, model._partition_cache)
    return Wannier_matrix, W_support_indices


def su2_get_Wannier_MPO(params: dict, momentum_params: dict, band_params: dict):
    # SU(2) MODEL in (1+1)D with MOMENTUM SYMMETRY
    model = SU2_Model(**params["model"])
    model.default_params()
    # Build Hamiltonian
    model.build_Hamiltonian(params["g"], params["m"])
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN to get the GLOBAL GROUND STATE
    model.diagonalize_Hamiltonian(1, model.ham_format, print_results=True)
    GS = model.H.Npsi[0]
    # -------------------------------------------------------------------------------
    # ACQUIRE THE WANNIER STATE ALREADY PARTITIONED
    band_params["support_tail"] = params["support_tail"]
    Wannier_matrix, W_support_indices = su2_get_Wannier_state(
        model, momentum_params, band_params
    )
    # -------------------------------------------------------------------------------
    # Partition of the GS in the support of Wannier and the environment
    GS_matrix = GS._get_psi_matrix(W_support_indices, model._partition_cache)
    # -------------------------------------------------------------------------------
    # Compute the Wannier operator on the support based on
    # the orthogonal precustes problem
    U, _, Vh = np.linalg.svd(Wannier_matrix @ GS_matrix.conj().T)
    Wannier_op = U @ Vh
    # -------------------------------------------------------------------------------
    # Check the action of the Wannier operator on the Ground state
    Wannier_op_on_GS = Wannier_op @ GS_matrix
    Wannier_op_on_GS_norm = np.linalg.norm(Wannier_op_on_GS)
    logger.info(f"norm ||W_op|GS>|| {Wannier_op_on_GS_norm}")
    Wannier_op_on_GS /= Wannier_op_on_GS_norm
    overlap_WWop = np.vdot(Wannier_matrix.ravel(), Wannier_op_on_GS.ravel())
    fidelity_WWop = np.abs(overlap_WWop) ** 2
    logger.info(f"|<W|W_op|GS>| = {np.abs(overlap_WWop):.6f}")
    logger.info(f"|<W|W_op|GS>|^2 = {fidelity_WWop:.6f}")
    # -------------------------------------------------------------------------------
    # Check the SVD to see the required rank for the Wannier operator
    logger.info(f"----------------------------------------------------")
    logger.info(f"Wannier_op SVD singular values:")
    Wannier_U, Wannier_S, Wannier_Vh = np.linalg.svd(Wannier_op, full_matrices=False)
    rank = choose_rank_by_frobenius(Wannier_S, rel_tol=1e-6)
    U_R = Wannier_U[:, :rank]
    S_R = Wannier_S[:rank]
    Vh_R = Wannier_Vh[:rank, :]
    Wannier_op_trunc = (U_R * S_R) @ Vh_R
    # Check the action of the truncated Wannier operator on the Ground state
    Wannier_op_trunc_on_GS = Wannier_op_trunc @ GS_matrix
    overlap_trunc = np.vdot(Wannier_matrix.ravel(), Wannier_op_trunc_on_GS.ravel())
    overlap_trunc /= np.linalg.norm(Wannier_op_trunc_on_GS)
    fidelity = np.abs(overlap_trunc) ** 2
    logger.info(f"|<W|W_op_trunc|GS>| = {np.abs(overlap_trunc):.6f}")
    logger.info(f"|<W|W_op_trunc|GS>|^2 = {fidelity:.6f}")
    # -------------------------------------------------------------------------------
    # Build the operator that promotes the symmetry sector
    # of the support to the global space
    logger.info(f"----------------------------------------------------")
    logger.info(f"BUILD PROJECTOR FROM SYMMETRIC --> FULL SUPPORT")
    W_support_indices_key = tuple(sorted(W_support_indices))
    support_loc_dims = list(model.loc_dims[W_support_indices])
    P = build_sector_expansion_projector(
        model._partition_cache[W_support_indices_key]["unique_subsys_configs"],
        model.loc_dims[W_support_indices],
    )
    logger.info(f"----------------------------------------------------")
    logger.info(f"PROJECT the Wannier Operator on THE FULL SUPPORT")
    W_op_full = P @ Wannier_op @ P.conj().T
    # -------------------------------------------------------------------------------
    # BUILD the MPO from the full dense version of the Wannier operator
    Wannier_MPO = dense_operator_to_mpo(
        W_op_full,
        support_loc_dims,
        svd_relative_tolerance=1e-8,
        max_bond_dimension=300,
    )
    # -------------------------------------------------------------------------------
    # Reconstruct the Wannier dense operator from the MPO
    W_op_full_v2 = mpo_to_dense_operator(
        Wannier_MPO, np.array(support_loc_dims), projector=P, return_sparse=True
    )
    logger.info(f"----------------------------------------------------")
    logger.info(f"W_op_full from MPO {W_op_full_v2.shape}")
    W_MPO = W_op_full_v2 @ GS_matrix
    overlap_WMPO = np.vdot(Wannier_matrix.ravel(), W_MPO.ravel())
    overlap_WMPO /= np.linalg.norm(W_MPO)
    fidelity = np.abs(overlap_WMPO) ** 2
    logger.info(f"|<W|QP_MPO|GS>| = {np.abs(overlap_WMPO):.6f}")
    logger.info(f"FIDELITY = {fidelity:.6f}")
    # -------------------------------------------------------------------------------
    return Wannier_MPO


def run_SU2_convolution(params):
    start_time = perf_counter()
    # -------------------------------------------------------------------------------
    # SU(2) MODEL in (1+1)D with MOMENTUM SYMMETRY
    model = SU2_Model(**params["model"])
    # -------------------------------------------------------------------------------
    TC_symmetry = params.get("TC_symmetry", False)
    momentum_params = su2_get_momentum_params(TC_symmetry, model.n_sites)
    band_params = {
        "sim_band_name": params["sim_band_name"],
        "band_number": params.get("band_number", 0),
        "m": params["m"],
        "g": params["g"],
        "R0": 0,
    }
    # -------------------------------------------------------------------------------
    # GET THE GROUND STATE ENERGY density at momentum 0
    band_params["gs_energy"] = su2_get_convolution_gs_energy(
        model, momentum_params, band_params
    )
    # -------------------------------------------------------------------------------
    # CONVOLUTION MATRIX
    if TC_symmetry:
        band_params["R0"] = 0
        band_params["k1k2matrix"] = su2_get_convolution_matrix(
            model, momentum_params, band_params
        )
        for ii in range(momentum_params["n_momenta"]):
            logger.info("==================")
            for jj in range(momentum_params["n_momenta"]):
                logger.info(f"{ii} {jj} {band_params['k1k2matrix'][ii, jj]}")
    else:
        band_params["R0"] = 0
        band_params["k1k2matrix_even"] = su2_get_convolution_matrix(
            model, momentum_params, band_params
        )
        band_params["R0"] = 1
        band_params["k1k2matrix_odd"] = su2_get_convolution_matrix(
            model, momentum_params, band_params
        )
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
    return model, momentum_params, band_params


def plot_energy_bands(sim_name_list, textwidth_pt=510.0, columnwidth_pt=246.0):
    res = {}
    # Acquire data from simulations
    for sim_name in sim_name_list:
        res[sim_name] = {}
        config_filename = f"scattering/{sim_name}"
        match = SimsQuery(group_glob=config_filename)
        ugrid, vals = uids_grid(match.uids, ["momentum_k_vals"])
        # Universal parameters across the simulations
        g = get_sim(ugrid[0]).par["g"]
        m = get_sim(ugrid[0]).par["m"]
        n_eigs = get_sim(ugrid[0]).par["hamiltonian"]["n_eigs"]
        # Momentum parameters
        k_indices = vals["momentum_k_vals"]
        Nk = len(k_indices)
        # Sort k values centering 0 in the BZ
        k_values, order, tick_labels = bz_axis(Nk)
        # Specific parameters of the simulation
        res[sim_name] = {
            "energy": np.zeros((Nk, n_eigs)),
            "E2": np.zeros((Nk, n_eigs)),
            "N_single": np.zeros((Nk, n_eigs)),
            "N_pair": np.zeros((Nk, n_eigs)),
        }
        for ll in k_indices:
            sim_res = get_sim(ugrid[ll]).res
            res[sim_name]["energy"][ll] = sim_res["energy"]
            res[sim_name]["E2"][ll] = sim_res["E2"]
            res[sim_name]["N_single"][ll] = sim_res["N_single"]
            res[sim_name]["N_pair"][ll] = sim_res["N_pair"]
        # Sort the energy bands centering k=0 in the BZ
        res[sim_name]["bands"] = res[sim_name]["energy"][..., order, :]
    # -------------------------------------------------------------------------
    # BUILD THE FIGURE
    fig, ax = plt.subplots(
        1,
        1,
        figsize=set_size(2 * columnwidth_pt, subplots=(1, 1), height_factor=3),
        constrained_layout=True,
        sharex=True,
    )
    ax.set_xticks(k_values)
    ax.set_xticklabels(tick_labels)
    ax.set(ylabel=r"energy E")
    ax.set(xlabel=r"momentum $k$")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.annotate(
        rf"$g^{2}\!=\!{g}, m\!=\!{m}$",
        xy=(0.935, 0.1),
        xycoords="axes fraction",
        fontsize=10,
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox=dict(facecolor="white", edgecolor="black"),
    )
    colors = ["green", "blue", "red"]
    markersize_list = [7, 6, 5]
    # PLOT THE DATA
    for sim_idx, sim_name in enumerate(sim_name_list):
        for ss in range(n_eigs):
            ax.plot(
                k_values,
                res[sim_name]["bands"][:, ss],
                "o",
                color=colors[sim_idx],  # line & marker edge color
                markeredgecolor=colors[sim_idx],
                markerfacecolor=lighten_color(colors[sim_idx], 0.6),
                markeredgewidth=1.3,
                markersize=markersize_list[sim_idx],
            )
    # MAKE THE LEGEND
    sector_labels = [r"$N_{\rm bar}=0$", r"$N_{\rm bar}=+1$", r"$N_{\rm bar}=+2$"]
    handles = []
    for c, ms, lab in zip(colors, markersize_list, sector_labels):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                color=c,  # legend line color (not shown since linestyle=None)
                markeredgecolor=c,
                markerfacecolor=lighten_color(c, 0.6),
                markeredgewidth=1.3,
                markersize=ms,
                label=lab,
            )
        )
    # Put the legend where you like:
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.3, 0.1),  # tweak or remove to place inside
        frameon=True,
        ncol=1,
        handlelength=1.0,
        handletextpad=0.3,
        borderpad=0.3,
        labelspacing=0.15,
        fontsize=10,
        title=r"Baryon sectors",  # ← title here
        title_fontsize=10,
    )
    plt.savefig(f"bands_g{g}_m{m}.pdf", **default_params["save_plot"])
    return fig, ax, res


# %%
params = {
    "model": {
        "lvals": [14],
        "sectors": [14],
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
        "get_entropy": False,
        "entropy_partition": [0],
        "get_state_configs": True,
        "get_overlap": False,
    },
    "TC_symmetry": True,
    "sim_band_name": "scattering/band1_N0",
    "g": 1,
    "m": 3,
}

model, momentum_params, band_params = run_SU2_convolution(params)
# %%
MPO = su2_get_Wannier_MPO(model, momentum_params, band_params)
