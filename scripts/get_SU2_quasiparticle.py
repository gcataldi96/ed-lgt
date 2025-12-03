import os
import sys

# Ensure NUMBA_NUM_THREADS is set properly before importing anything else
B = int(sys.argv[-1])
# Read the B parameter from command-line arguments
os.environ["NUMBA_NUM_THREADS"] = str(B)

import numpy as np
from scipy.sparse import csr_matrix, kron
from ed_lgt.models import SU2_Model
from ed_lgt.modeling import QMB_state
from ed_lgt.tools import (
    get_data_from_sim,
    get_Wannier_support,
    localize_Wannier,
    operator_to_mpo_via_mps2,
    choose_rank_by_frobenius,
)
from ed_lgt.symmetries import build_sector_expansion_projector
from simsio import *
from time import perf_counter
import logging


def vector_to_mps(
    state_vector: np.ndarray,
    local_dims: list[int],
    chi_max: int | None = None,
    svd_rel_tol: float = 1e-6,
):
    """
    Decompose a state |psi> on a finite chain into an MPS with controlled bonds.
    Args
    ----
    state_vector:
        1D array of length prod(local_dims).
        State restricted to the support S in the product basis (matching local_dims).
    local_dims:
        [d1, d2, ..., dL] local Hilbert-space dimensions on each site in S,
        in the SAME order as used to flatten state_vector.
    chi_max:
        Optional maximum MPS bond dimension (global cap per bond).
    svd_rel_tol:
        Relative truncation tolerance at each bond:
        Discard singular values so that discarded Frobenius weight
        <= svd_rel_tol^2 of that bond's total.

    Returns
    -------
    mps:
        List of length L.
        mps[i] has shape (bond_left, d_i, bond_right).
        First tensor has bond_left = 1.
        Last tensor  has bond_right = 1.
    """
    L = len(local_dims)
    D = int(np.prod(local_dims))
    assert state_vector.size == D, "state_vector size incompatible with local_dims"
    logger.info("VECTOR TO MPS")
    # Reshape into tensor with one physical leg per site
    logger.info("Reshape")
    psi = state_vector.reshape(local_dims)
    mps: list[np.ndarray] = []
    left_bond_dim = 1
    # Sweep from left to right, stopping before the last site
    for site in range(L - 1):
        d_site = local_dims[site]
        logger.info("Reshape")
        # Group all left legs (current bond + this site) vs all right legs
        psi = psi.reshape(left_bond_dim * d_site, -1)  # (left * d_i) x rest
        # SVD on this bipartition and truncate
        # U shape: (left_bond_dim * d_site, keep)
        # S shape: (keep,)
        # Vh shape: (keep, rest)
        logger.info("SVD")
        U, S, Vh = svd_truncate(psi, svd_rel_tol, chi_max)
        keep = len(S)
        # Reshape U into the MPS tensor A_site
        #   (left_bond_dim, d_site, new_bond_dim)
        logger.info("reshape")
        A_site = U.reshape(left_bond_dim, d_site, keep)
        mps.append(A_site)
        # Propagate remaining entanglement to the right:
        # new "psi" has shape (keep, rest)
        psi = S[:, None] * Vh
        left_bond_dim = keep
    # Last site: whatever remains is (left_bond_dim * d_last) components
    d_last = local_dims[-1]
    psi = psi.reshape(left_bond_dim, d_last)
    # Final tensor with right bond dimension fixed to 1
    A_last = psi.reshape(left_bond_dim, d_last, 1)
    mps.append(A_last)
    # Sanity checks
    assert mps[0].shape[0] == 1, "First MPS tensor must have left bond = 1."
    assert mps[-1].shape[2] == 1, "Last MPS tensor must have right bond = 1."
    logger.info("----------------------------------------------------")
    logger.info("Final MPS tensors:")
    for i, W in enumerate(mps):
        logger.info(f"site {i}: shape {W.shape}")
    return mps


def svd_truncate(op: np.ndarray, svd_rel_tol: float = 1e-6, chi_max: int | None = None):
    """
    Perform SVD on matrix op and truncate singular values according to:
        discarded_weight <= svd_rel_tol^2 * total_weight
    and optionally enforce chi_max (maximal bond dimension).
    Returns
    -------
    U_trunc, S_trunc, Vh_trunc
    """
    U, S, Vh = np.linalg.svd(op, full_matrices=False)
    # Frobenius weights
    S2 = S * S
    total = S2.sum()
    cumsum = np.cumsum(S2)
    # relative tolerance rule
    target = (svd_rel_tol**2) * total
    keep = len(S)
    for r in range(len(S)):
        discarded = total - cumsum[r]
        if discarded <= target:
            keep = r + 1
            break
    # hard cap
    if chi_max is not None:
        keep = min(keep, chi_max)
    # truncate
    U = U[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]
    return U, S, Vh


def from_sym_operator_to_mpo(
    Osym: np.ndarray, P: np.ndarray, loc_dims, svd_rel_tol=1e-8, chi_max=None
):
    """
    Returns:
        list of L MPO tensors W[i] with shape (chiL_i, di, di, chiR_i).
    """
    # ------------------------------------------------------
    # SVD of operator in sector basis
    logger.info("SVD of QP operator (in sector basis)")
    U, S, Vh = np.linalg.svd(Osym, full_matrices=False)
    eff_dsym = choose_rank_by_frobenius(S, rel_tol=1e-6, max_rank=110)
    if eff_dsym == 0:
        # Completely negligible operator ⇒ raise error
        raise ValueError("Truncated QP operator to 0-rank; increase tol or max_rank.")
    # Truncate the SVD components up to eff_dsym
    U = U[:, :eff_dsym]
    S = S[:eff_dsym]
    Vh = Vh[:eff_dsym, :]
    logger.info(f"keep {eff_dsym} sing values")
    # ------------------------------------------------------
    # Absort np.sqrt(S) on to the Left and Right projectors P
    LeftP = P @ (U * np.sqrt(S)[None, :])
    RightP = P @ (Vh.conj().T * np.sqrt(S)[None, :])
    # ------------------------------------------------------
    # Below I assume the MPO is on a support of 4 sites
    # ------------------------------------------------------
    # Decompose these projectors to MPS
    # Internally, it reshape projectors in (d,d,d,d,eff_dsym)
    # LeftP = LeftP.reshape(loc_dims + [eff_dsym])
    # RightP = RightP.reshape(loc_dims + [eff_dsym])
    LP_mps = vector_to_mps(LeftP, loc_dims + [eff_dsym], chi_max, svd_rel_tol)
    # Merging all the tensors together
    # LP_mps has shape = (1, d, chi1, d, chi2, d, chi3, d, chi_sym, eff_dsym, 1)
    # where chi1 ... chi_sym are bounded by chi_max
    RP_mps = vector_to_mps(RightP, loc_dims + [eff_dsym], chi_max, svd_rel_tol)
    # RP_mps has shape = (1, d, chi1, d, chi2, d, chi3, d, chi_sym, eff_dsym, 1)
    mpo = []
    # ------------------------------------------------------
    # Build the MPO fusing the two mps and svd site by site
    # Contract LP_mps and RP_mps through the shared link of dimension eff_dsym
    TL = LP_mps[-1]  # shape (chi_sym, eff_dsym, 1)
    TR = RP_mps[-1].conj()  # shape (chi_sym, eff_dsym, 1)
    T = np.tensordot(TL, TR, axes=(1, 1))
    # After this, T has shape: (chi_sym, chi_sym, 1)
    logger.info(f"T shape: {T.shape}")
    # So contract on chi_sym on the LR and then on chi_sym on the TR
    TL = LP_mps[-2]  # shape (chi3, d, chi_sym)
    TR = RP_mps[-2].conj()  # shape (chi3, d, chi_sym)
    T = np.tensordot(TL, T, axes=(2, 0))  # new shape: (chi3, d, chi_sym, 1)
    logger.info(f"T shape: {T.shape}")
    T = np.tensordot(T, TR, axes=(2, 2))  # new shape: (chi3, chi3, d, d, 1)
    logger.info(f"T shape: {T.shape}")
    # (1, d, chi1, d, chi2, d, chi3, d, d, chi3, d, chi2, d, chi1, d, 1)
    # 2) Contract LP_mps and RP_mps on the next (first physical) site (on Left and right) via chi_sym
    # (1, d, chi1, d, chi2, d, chi3, d, d, chi3, d, chi2, d, chi1, d, 1)
    # Focus on this central tensor T of shape (chi3L, chi3R, d, d, 1). Reshape into (chi3L*chi3R, d*d)
    # SVD it and replace the tensor T with U, and reshape it as (chi_keep, d, d, 1)
    # where chi_keep depends on the previous svd. This is the rightmost tensor of the MPO
    # mpo.append(U)
    # Similarly, reshape (S@Vh) as (chi3L, chi3R, chi_keep)
    # Focus on the next tensors (chi2L, d, chi3L) (chi2R, d, chi3R) and contract them with S@Vh.
    # The new single tensor of shape (chi2L, chi2R, d, d, chi_keep, 1)undergoes the same SVD proceeding:
    # Reshape it into (chi2L*chi2R, d*d*chi_keep) SVD it and replace the tensor with U.
    # Reshape U into (new_chi_keep, d, d, chi_keep): this is the new tensor of the MPO.
    # mpo.append(U)
    # Go on with the S@ Vh part ...
    return mpo


def dense_operator_to_mpo(O: np.ndarray, loc_dims, svd_rel_tol=1e-8, chi_max=None):
    """
    Convert a dense operator on L sites into an MPO via sequential SVD.

    O has shape (d1,d2,...,dL, d1,d2,...,dL).

    Returns:
        list of L MPO tensors W[i] with shape (chiL_i, di, di, chiR_i).
    """
    if O.ndim != 2 or O.shape[0] != O.shape[1]:
        raise ValueError("operator must be a square 2D array")
    L = len(loc_dims)
    dims = loc_dims
    # reshape operator into a rank-2L tensor
    O_tensor = O.reshape(*dims, *dims)
    mpo = []
    R = O_tensor
    chi_left = 1
    for site in range(L - 1):
        d = dims[site]
        # group left indices: (chi_left, d, d)
        # and right indices: everything else
        R = R.reshape(chi_left * d * d, -1)
        U, S, Vh = np.linalg.svd(R, full_matrices=False)
        # truncation
        S2 = S * S
        total = S2.sum()
        cumsum = np.cumsum(S2)
        keep = len(S)
        target = svd_rel_tol**2 * total
        for r in range(len(S)):
            discarded = total - cumsum[r]
            if discarded <= target:
                keep = r + 1
                break
        if chi_max is not None:
            keep = min(keep, chi_max)
        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]
        # first MPO tensor
        W = U.reshape(chi_left, d, d, keep)
        mpo.append(W)
        # new remainder
        R = S[:, None] * Vh
        chi_left = keep
        # reshape back into tensor of remaining sites
        remaining_dims = dims[site + 1 :]
        R = R.reshape(chi_left, *remaining_dims, *remaining_dims)
    # last site
    d_last = dims[-1]
    R = R.reshape(chi_left, d_last, d_last, 1)
    mpo.append(R)
    logger.info("----------------------------------------------------")
    logger.info("Final MPO tensors:")
    for i, W in enumerate(mpo):
        logger.info(f"site {i}: shape {W.shape}")
    return mpo


def mpo_to_dense_operator(mpo: list[np.ndarray], loc_dims: list[int], P: np.ndarray):
    """
    Contract an MPO into a dense operator O of shape (D_full, D_full).

    mpo[i]: (chiL, d, d, chiR)
    loc_dims: [d1, d2, ..., dL]

    Returns
    -------
    O : np.ndarray of shape (D_full, D_full)
    """
    L = len(mpo)
    assert L == len(loc_dims)
    # Start from first tensor: shape (1, d1, d1, chi1)
    T = mpo[0]
    for i in range(1, L):
        W = mpo[i]  # (chiL_i, d_i, d_i, chiR_i)
        # Contract right bond of T with left bond of W
        # T: (..., chi_prev), W: (chi_prev, d_i, d_i, chi_next)
        T = np.tensordot(T, W, axes=(T.ndim - 1, 0))
        # After this, T has shape:
        # (1, d1, d1, ..., d_i, d_i, chi_next)
        logger.info("T shape after site {}: {}".format(i, T.shape))
    # At the end, T has shape (1, d1, d1, ..., dL, dL, 1)
    # Drop the trivial boundary bonds and reshape to (D_full, D_full)
    phys_dims = []
    for d in loc_dims:
        phys_dims.extend([d, d])  # in, out per site
    T = T.reshape(*phys_dims)  # (d1, d1, d2, d2, ..., dL, dL)
    D_full = int(np.prod(loc_dims))
    logger.info(f"Projection")
    return P.conj().T @ csr_matrix(T.reshape(D_full, D_full)) @ P


def build_optimal_qp_operator(
    W_psimatrix: np.ndarray, G_psimatrix: np.ndarray, rcond=1e-12
):
    """
    Compute the optimal local operator A_opt acting on the subsystem, such that
        A_opt @ G_psimatrix  ≈  W_psimatrix
    in least-squares sense.

    Both W_psimatrix and G_psimatrix must have shape (d_sub, d_env).

    The formula is always:
        A_opt = W G^+
    but this function computes G^+ using the *smaller* side
    (subsys or env) to avoid huge inversions.

    Returns
    -------
    A_opt : np.ndarray (d_sub, d_sub)
        Optimal subsystem operator.
    method : str
        Which branch was used ("subsys" or "env").
    """
    logger.info(f"----------------------------------------------------")
    G = G_psimatrix
    W = W_psimatrix
    d_sub, d_env = G.shape
    # --------------------------------------------------------------
    # CASE 1: Subsystem is the smaller space  (invert GG† of size d_sub)
    # --------------------------------------------------------------
    if d_sub <= d_env:
        # Left Gram matrix: GG†   (subsystem space)
        Gram_left = G @ G.conj().T  # (d_sub, d_sub)
        # Cross operator: W G†
        Cross = W @ G.conj().T  # (d_sub, d_sub)
        # Pseudo-inverse of GG† via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(Gram_left)
        inv_eigs = np.zeros_like(eigvals)
        mask = eigvals > rcond
        inv_eigs[mask] = 1.0 / eigvals[mask]
        Gram_left_pinv = (eigvecs * inv_eigs) @ eigvecs.conj().T
        A_opt = Cross @ Gram_left_pinv  # (d_sub, d_sub)
        method = "subsys"
    # --------------------------------------------------------------
    # CASE 2: Environment is smaller (invert G†G of size d_env)
    # --------------------------------------------------------------
    else:
        # Right Gram matrix: G†G   (environment space)
        Gram_right = G.conj().T @ G  # (d_env, d_env)
        # Pseudo-inverse of G†G via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(Gram_right)
        inv_eigs = np.zeros_like(eigvals)
        mask = eigvals > rcond
        inv_eigs[mask] = 1.0 / eigvals[mask]
        Gram_right_pinv = (eigvecs * inv_eigs) @ eigvecs.conj().T
        # G⁺ = (G†G)⁺ G†
        G_pinv = Gram_right_pinv @ G.conj().T  # (d_env, d_sub)
        # Optimal operator: A = W G⁺
        A_opt = W @ G_pinv  # (d_sub, d_sub)
        method = "env"
    logger.info(f"QP op via {method}-space inversion: shape {A_opt.shape}")
    return A_opt, method


logger = logging.getLogger(__name__)

with run_sim() as sim:
    start_time = perf_counter()
    # -------------------------------------------------------------------------------
    # SU(2) MODEL in (1+1)D with MOMENTUM SYMMETRY
    model = SU2_Model(**sim.par["model"])
    zero_density = False if sim.par["model"]["sectors"][0] != model.n_sites else True
    logger.info(f"zero density {zero_density}")
    m = sim.par["m"]
    g = sim.par["g"]
    # -------------------------------------------------------------------------------
    # Save parameters
    model.default_params()
    # Build Hamiltonian
    if model.spin > 0.5:
        model.build_gen_Hamiltonian(g, m)
    else:
        model.build_Hamiltonian(g, m)
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN to get the GLOBAL GROUND STATE
    model.diagonalize_Hamiltonian(1, model.ham_format)
    GS = model.H.Npsi[0]
    # -------------------------------------------------------------------------------
    # Acquire the optimal theta phases that localize the Wannier
    Eprofile, _, theta_phases = localize_Wannier("convolution1_N0", center_mode=1)
    # Get the partition to the model according to the optimal support of the Wannier
    w_supports = get_Wannier_support(Eprofile, epsilons=(0.05, 1e-2, 1e-3))
    support_indices = w_supports["supports"][0.05]
    model._get_partition(support_indices)
    # Initialize the Wannier State
    psi_wannier = np.zeros(model.sector_configs.shape[0], dtype=np.complex128)
    # Simulation band name where to extract the momentum states
    sim_band_name = sim.par["sim_band_name"]
    band_number = sim.par.get("band_number", 0)
    state_idx = 1 + band_number
    # -------------------------------------------------------------------------------
    # Define the momentum grid
    TC_symmetry = sim.par.get("TC_symmetry", False)
    k_unit_cell_size = [1] if TC_symmetry else [2]
    n_momenta = model.n_sites if TC_symmetry else model.n_sites // 2
    k_indices = np.arange(0, n_momenta, 1)
    # -------------------------------------------------------------------------------
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
        psi_wannier += np.exp(1j * theta_phases[kidx]) * psik_exp / np.sqrt(n_momenta)
    logger.info(f"wannier norm = {np.linalg.norm(psi_wannier)}")
    # Promote the Wannier state as an item of the QMB state class
    Wannier = QMB_state(psi=psi_wannier, lvals=model.lvals, loc_dims=model.loc_dims)
    W_psimatrix = Wannier._get_psi_matrix(support_indices, model._partition_cache)
    # Promote the Ground State as an item of the QMB state class
    GS_psimatrix = GS._get_psi_matrix(support_indices, model._partition_cache)
    # ------------------------------------------------------------------
    # Compute optimal operator using smallest inversion
    qp_opt, method_used = build_optimal_qp_operator(W_psimatrix, GS_psimatrix)
    op_norm = np.linalg.norm(qp_opt, "fro")
    qp_opt = qp_opt / op_norm
    # ------------------------------------------------------------------
    # Build the cross operator on the support Tr_{Env}|Wannier><GS|
    A_cross = W_psimatrix @ GS_psimatrix.conj().T
    W_rec_cross = A_cross @ GS_psimatrix
    overlap_cross = np.vdot(W_psimatrix.ravel(), W_rec_cross.ravel()) / (
        np.linalg.norm(W_psimatrix) * np.linalg.norm(W_rec_cross)
    )
    # ------------------------------------------------------------------
    # Optimal LS operator (via qp_opt)
    W_rec_opt = qp_opt @ GS_psimatrix
    overlap_opt = np.vdot(W_psimatrix.ravel(), W_rec_opt.ravel()) / (
        np.linalg.norm(W_psimatrix) * np.linalg.norm(W_rec_opt)
    )
    fidelity_cross = np.abs(overlap_cross) ** 2
    fidelity_opt = np.abs(overlap_opt) ** 2
    logger.info(
        f"|<W|QP|GS>| = {np.abs(overlap_cross):.6f} FIDELITY = {fidelity_cross:.6f}"
    )
    logger.info(
        f"|<W|QP_opt|GS>| = {np.abs(overlap_opt):.6f} FIDELITY = {fidelity_opt:.6f}"
    )
    # Check the SVD to see the required rank for the operator
    U, S, Vh = np.linalg.svd(qp_opt, full_matrices=False)
    logger.info(f"QP_opt operator SVD singular values:")
    R = choose_rank_by_frobenius(S, rel_tol=1e-6)
    U_R = U[:, :R]
    S_R = S[:R]
    Vh_R = Vh[:R, :]
    A_trunc = (U_R * S_R) @ Vh_R
    # Truncated action
    W_rec_trunc = A_trunc @ GS_psimatrix
    overlap_trunc = np.vdot(W_psimatrix.ravel(), W_rec_trunc.ravel()) / (
        np.linalg.norm(W_psimatrix) * np.linalg.norm(W_rec_trunc)
    )
    fidelity = np.abs(overlap_trunc) ** 2
    logger.info(
        f"|<W|QP_opt^{{R}}|GS>| = {np.abs(overlap_trunc):.6f} FIDELITY = {fidelity:.6f}"
    )
    # ------------------------------------------------------------------
    # Build the operator that promotes the symmetry sector to the global space
    support_indices_key = tuple(sorted(support_indices))
    support_loc_dims = list(model.loc_dims[support_indices])
    P = build_sector_expansion_projector(
        model._partition_cache[support_indices_key]["unique_subsys_configs"],
        model.loc_dims[support_indices],
    )
    O_full = P @ qp_opt @ P.conj().T
    MPO = dense_operator_to_mpo(
        O_full, support_loc_dims, svd_rel_tol=1.5e-4, chi_max=300
    )
    """# Try another solution
    mpo = from_sym_operator_to_mpo(
        qp_opt, P, support_loc_dims, svd_rel_tol=1e-6, chi_max=300
    )"""
    # ------------------------------------------------------------------
    # Transform the optimal operator to MPO form via MPS compression
    """MPO = operator_to_mpo_via_mps2(
        operator=qp_opt,
        projector=P,
        loc_dims=support_loc_dims,
        op_svd_rel_tol=1e-6,
        max_rank=None,
        mps_chi_max=40,
        mps_svd_rel_tol=1e-4,
        mpo_chi_max=300,
        mpo_svd_rel_tol=1e-4,
    )"""
    # ------------------------------------------------------------------
    # Save results
    for ii, site in enumerate(support_loc_dims):
        sim.res[f"MPO[{ii}]"] = MPO[ii]
    Op_full = mpo_to_dense_operator(MPO, np.array(support_loc_dims), P)
    logger.info(f"Op full {Op_full.shape}")
    W_MPO = Op_full @ GS_psimatrix
    overlap = np.vdot(W_psimatrix.ravel(), W_MPO.ravel()) / (
        np.linalg.norm(W_psimatrix) * np.linalg.norm(W_MPO)
    )
    fidelity = np.abs(overlap) ** 2
    logger.info(f"|<W|QP_MPO|GS>| = {abs(overlap):.6f}, FIDELITY = {fidelity:.6f}")
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
