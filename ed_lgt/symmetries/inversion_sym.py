import numpy as np
from numba import njit, prange
from .generate_configs import config_to_index_binarysearch

__all__ = [
    "apply_parity_to_state",
    "build_parity_operator",
]


@njit(cache=True)
def parity_perm_site(n_sites: int, j0: int) -> np.ndarray:
    """
    Site-centered inversion about site j0:
        j -> (2*j0 - j) mod n_sites

    Parameters
    ----------
    n_sites : int
        Number of lattice sites (L).
    j0 : int
        Site index about which we invert. Can be any integer; reduced mod L.

    Returns
    -------
    perm : (n_sites,) int32
        perm[j] = new site index of the DOF originally at site j.
    """
    L = n_sites
    j0_mod = j0 % L
    perm = np.empty(L, dtype=np.int32)
    center2 = 2 * j0_mod
    for j in range(L):
        perm[j] = (center2 - j) % L
    return perm


@njit(cache=True)
def parity_perm_bond(n_sites: int, j0: int) -> np.ndarray:
    """
    Bond-centered inversion about the bond (j0, j0+1),
    i.e. center at j0 + 0.5:

        j -> (2*(j0 + 0.5) - j) mod n_sites
          = (2*j0 + 1 - j) mod n_sites

    Parameters
    ----------
    n_sites : int
        Number of lattice sites (L).
    j0 : int
        LEFT site index of the bond (j0, j0+1). Can be any integer; reduced mod L.

    Returns
    -------
    perm : (n_sites,) int32
        perm[j] = new site index of the DOF originally at site j.
    """
    L = n_sites
    j0_mod = j0 % L
    perm = np.empty(L, dtype=np.int32)
    center2 = 2 * j0_mod + 1  # 2*(j0 + 0.5)
    for j in range(L):
        perm[j] = (center2 - j) % L
    return perm


@njit(cache=True)
def parity_image_config(
    config: np.ndarray,
    site_perm: np.ndarray,
    loc_perm: np.ndarray,
    loc_phase: np.ndarray,
):
    """
    Apply parity to a single configuration.

    Parameters
    ----------
    config : (n_sites,)
        Local basis indices of original configuration.
    site_perm : (n_sites,)
        site_perm[j] = new site index of DOF originally at j.
    loc_perm, loc_phase : Local parity maps

    Returns
    -------
    new_config : (n_sites,)
        Parity-transformed configuration.
    total_phase : int8
        Overall ±1 phase from local parity eigenvalues.
    """
    n_sites = len(config)
    new_config = np.empty(n_sites, dtype=np.int32)
    total_phase = 1
    for j in range(n_sites):
        s = config[j]
        s_p = loc_perm[s]
        total_phase = total_phase * loc_phase[s]
        new_j = site_perm[j]
        new_config[new_j] = s_p
    return new_config, total_phase


@njit(cache=True, parallel=True)
def build_parity_operator(
    sector_configs: np.ndarray,
    loc_perm: np.ndarray,
    loc_phase: np.ndarray,
    wrt_site: np.uint8 = 0,
):
    """
    Build site (or bond)-centered parity operator P in the sector basis of a generic model.
    Parameters
    ----------
    sector_configs : (n_configs, n_sites) int32
        Basis configurations of the symmetry sector (lex sorted).
    Returns
    -------
    row : (n_configs,) int32
    col : (n_configs,) int32
    data : (n_configs,) float64
    Triplet representation of P, with a single ±1 per column.
    """
    n_configs, n_sites = sector_configs.shape
    if wrt_site == 0:
        site_perm = parity_perm_site(n_sites, n_sites // 2 - 1)
    else:
        site_perm = parity_perm_bond(n_sites, n_sites // 2 - 1)
    row = np.empty(n_configs, dtype=np.int32)
    col = np.empty(n_configs, dtype=np.int32)
    data = np.empty(n_configs, dtype=np.float64)
    for cidx in prange(n_configs):
        cfg = sector_configs[cidx]
        new_cfg, phase = parity_image_config(cfg, site_perm, loc_perm, loc_phase)
        new_cfg_idx = config_to_index_binarysearch(new_cfg, sector_configs)
        # In a consistent sector construction, new_cfg_idx should never be -1.
        row[cidx] = new_cfg_idx
        col[cidx] = cidx
        data[cidx] = np.float64(phase)
    return row, col, data


@njit(cache=True, parallel=True)
def apply_parity_to_state(
    psi: np.ndarray,  # (n_configs,), complex128 expected
    row: np.ndarray,  # (n_configs,), int32
    col: np.ndarray,  # (n_configs,), int32
    data: np.ndarray,  # (n_configs,), float64 in {+1, -1}
) -> np.ndarray:
    """
    Apply psi_out = P psi using triplets (row, col, data).
    Assumes exactly one nonzero per column (true for a permutation-with-signs).
    """
    n_configs = psi.size
    psi_out = np.zeros(n_configs, dtype=psi.dtype)
    for cfg_idx in prange(n_configs):
        row_idx = row[cfg_idx]
        col_idx = col[cfg_idx]
        psi_out[row_idx] += psi[col_idx] * np.array(data[cfg_idx], dtype=psi.dtype)
    return psi_out


"""    if apply_parity:
        sector_dim = model.sector_configs.shape[0]
        r, c, d = model.get_parity_operator(wrt_site)
        P = csr_matrix((d, (r, c)), shape=(sector_dim, sector_dim))
        I = identity(sector_dim)
        norma = norm(P @ P - identity(sector_dim))
        logger.info(f"norm: (PP^{2}-1): {norma}")
        # nnz per row
        nnz_per_row = np.diff(P.indptr)
        # nnz per col
        nnz_per_col = np.diff(P.tocsc().indptr)
        logger.info(f"row nnz counts: {np.unique(nnz_per_row)}")
        logger.info(f"col nnz counts: {np.unique(nnz_per_col)}")
        # values
        unique_vals = np.unique(np.round(P.data, 8))
        logger.info(f"unique data values in P: {unique_vals}")
        # P^2 = I
        err_P2 = norm(P @ P - I)
        logger.info(f"||P^2 - I|| = {err_P2}")
        # Hermitian: P^† = P
        PH = P.getH()
        err_herm = norm(PH - P)
        logger.info(f"||P^† - P|| = {err_herm}")
        # Unitary: P^† P = I
        err_unit = norm(PH @ P - I)
        logger.info(f"||P^† P - I|| = {err_unit}")
        model.H.convert_hamiltonian("sparse")
        # Parity Hamiltonian
        err_PH = norm(P @ model.H.Ham - model.H.Ham @ P)
        logger.info(f"|PH-HP| = {err_PH}")
        psi = (np.random.randn(sector_dim) + 1j * np.random.randn(sector_dim)).astype(
            np.complex128
        )
        if model.momentum_basis is not None:
            # Build the projector from the momentum sector to the global one
            Pk = model._basis_Pk_as_csr()
            # Project the State from the momentum sector to the coordinate one
            psi0 = Pk @ model.H.Npsi[0].psi
            logger.info(f"{model.H.Npsi[0].psi.shape[0]}, {Pk.shape} {psi0.shape[0]}")
        else:
            psi0 = model.H.Npsi[0].psi
        psiP = apply_parity_to_state(psi0, r, c, d)
        psiPP = apply_parity_to_state(psiP, r, c, d)
        logger.info(f"Ppsi0 {np.vdot(psi0,psiP)}")
        logger.info(f"PPpsi0 {np.vdot(psi0,psiPP)}")
        psi0QS = QMB_state(psiP, model.lvals, model.loc_dims)
        psi0QS.get_state_configurations(1e-3, model.sector_configs)
        logger.info("PROVA")
        s1cfg = [2, 3, 0, 4, 0, 4, 0, 4, 2, 3]
        s1 = model.get_qmb_state_from_configs([s1cfg])
        Ss1 = QMB_state(s1, model.lvals, model.loc_dims)
        Ss1.get_state_configurations(1e-3, model.sector_configs)
        Ps1 = QMB_state(P @ s1, model.lvals, model.loc_dims)
        Ps1.get_state_configurations(1e-3, model.sector_configs)
        logger.info("PROVA 2")
        for ii, cfg in enumerate(model.sector_configs):
            s1 = model.get_qmb_state_from_configs([cfg])
            Ss1 = QMB_state(s1, model.lvals, model.loc_dims)
            # Ss1.get_state_configurations(1e-3, model.sector_configs)
            Ps1 = QMB_state(P @ s1, model.lvals, model.loc_dims)
            # Ps1.get_state_configurations(1e-3, model.sector_configs)
            exp_s1 = Ss1.expectation_value(model.H.Ham)
            exp_Ps1 = Ps1.expectation_value(model.H.Ham)
            Hpsi = model.H.Ham @ s1  # H|psi>
            Ppsi = P @ s1  # P|psi>
            HPpsi = model.H.Ham @ Ppsi  # H P |psi>
            PHpsi = P @ Hpsi  # P H |psi>
            diff = HPpsi - PHpsi
            absnorm = np.linalg.norm(diff)
            if np.abs(exp_Ps1 - exp_s1) > 1e-10:
                logger.info(f"{exp_s1}")
                logger.info(f"{exp_Ps1}")
                raise ValueError(f"config {ii} {cfg} not symmetrized")
            if np.abs(absnorm) > 1e-10:
                logger.info("==============================================")
                logger.info("State i")
                Ss1.get_state_configurations(1e-3, model.sector_configs)
                logger.info("State P|i>")
                Ps1.get_state_configurations(1e-3, model.sector_configs)
                A = QMB_state(HPpsi, model.lvals, model.loc_dims)
                logger.info("State HP|i>")
                A.get_state_configurations(1e-3, model.sector_configs)
                B = QMB_state(PHpsi, model.lvals, model.loc_dims)
                logger.info("State PH|i>")
                B.get_state_configurations(1e-3, model.sector_configs)
                logger.info(f"VDOT{np.vdot(PHpsi,HPpsi)}")
                logger.info(f"relative ||HP - PH|| ={absnorm}")
                raise ValueError(f"config {ii} {cfg} not symmetrized")"""
