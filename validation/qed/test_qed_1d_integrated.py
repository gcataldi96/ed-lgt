import logging
import numpy as np
from scipy.sparse import csc_matrix, identity

from edlgt.models import QED_Model
from edlgt.operators import QED_gauge_integrated_operators
from edlgt.modeling import local_op, two_body_op
from edlgt.symmetries.generate_configs import build_sector_expansion_projector

logger = logging.getLogger(__name__)


def _reference_integrated_hamiltonian(
    lvals: list[int], has_obc: list[bool], g: float, m: float, static_charges: np.ndarray
) -> csc_matrix:
    """Direct implementation of Eq. H_integrated (up to no simplification)."""
    ops = QED_gauge_integrated_operators()
    site_count = int(lvals[0])
    hilbert_dim = int(2**site_count)
    ident = identity(hilbert_dim, dtype=np.complex128, format="csc")
    ham_ref = csc_matrix((hilbert_dim, hilbert_dim), dtype=np.complex128)
    # Mass term: m * sum_i (-1)^i N_i (same convention used in the model)
    for ii in range(site_count):
        mass_coeff = m if ii % 2 == 0 else -m
        ham_ref += mass_coeff * local_op(
            operator=ops["N"],
            op_site=ii,
            lvals=lvals,
            has_obc=has_obc,
            staggered_basis=False,
            gauge_basis=None,
        )
    # Hopping term with staggered-fermion convention used in the model:
    # t * Sp_i Sm_{i+1} + t* * Sm_i Sp_{i+1}, with t = -i/2.
    hop_coeff = -0.5j
    for ii in range(site_count - 1):
        ham_ref += hop_coeff * two_body_op(
            op_list=[ops["Sp"], ops["Sm"]],
            op_sites_list=[ii, ii + 1],
            lvals=lvals,
            has_obc=has_obc,
            staggered_basis=False,
            gauge_basis=None,
        )
        ham_ref += np.conjugate(hop_coeff) * two_body_op(
            op_list=[ops["Sm"], ops["Sp"]],
            op_sites_list=[ii, ii + 1],
            lvals=lvals,
            has_obc=has_obc,
            staggered_basis=False,
            gauge_basis=None,
        )
    # Electric long-range term:
    # E * sum_{n=0}^{L-2} [sum_{k=0}^{n} (q_k + (Sz_k + (-1)^k)/2)]^2
    electric_coeff = g / 2.0
    for nn in range(site_count - 1):
        cumulative_op = csc_matrix((hilbert_dim, hilbert_dim), dtype=np.complex128)
        for kk in range(nn + 1):
            stag_sign = 1.0 if kk % 2 == 0 else -1.0
            cumulative_op += (static_charges[kk] + 0.5 * stag_sign) * ident
            cumulative_op += 0.5 * local_op(
                operator=ops["Sz"],
                op_site=kk,
                lvals=lvals,
                has_obc=has_obc,
                staggered_basis=False,
                gauge_basis=None,
            )
        ham_ref += electric_coeff * (cumulative_op @ cumulative_op)
    return ham_ref


def main():
    lvals = [4]
    has_obc = [True]
    g = 1.4
    m = 0.7
    static_charges = np.array([1.0, -1.0, 0.0, 0.0], dtype=float)

    model = QED_Model(
        spin="integrated",
        pure_theory=False,
        lvals=lvals,
        has_obc=has_obc,
        ham_format="dense",
        bg_list=static_charges.tolist(),
    )
    model.build_Hamiltonian(g=g, m=m)
    ham_model = np.asarray(model.H.Ham, dtype=np.complex128)

    ham_ref_full = _reference_integrated_hamiltonian(
        lvals=lvals,
        has_obc=has_obc,
        g=g,
        m=m,
        static_charges=static_charges,
    ).toarray()
    expansion = build_sector_expansion_projector(
        model.sector_configs,
        np.asarray(model.loc_dims, dtype=np.int32),
    ).astype(np.complex128)
    ham_ref = expansion.conj().T @ ham_ref_full @ expansion

    delta = ham_model - ham_ref
    max_residual = float(np.max(np.abs(delta)))
    atol = 1e-10
    if max_residual > atol:
        raise AssertionError(
            "QED 1D integrated test01: FAIL "
            f"(max residual = {max_residual:.3e}, atol={atol:.1e})"
        )
    # ---------------------------------------------------------------------------
    # Validate reconstructed link Casimir <E^2> from measured N.
    model.diagonalize_Hamiltonian(n_eigs=1, format=model.ham_format)
    model.get_observables(local_obs=["N"])
    model.measure_observables(0)
    e2_reconstructed = model.reconstruct_integrated_E2_from_N(state_index=0)
    # Direct reference: <E_n^2> from full-space operators and the same state.
    psi_sector = model.H.Npsi[0].psi
    psi_full = expansion @ psi_sector
    ops = QED_gauge_integrated_operators()
    hilbert_dim = int(2**lvals[0])
    ident = identity(hilbert_dim, dtype=np.complex128, format="csc")
    e2_reference = np.zeros(lvals[0] - 1, dtype=float)
    for nn in range(lvals[0] - 1):
        electric_op = csc_matrix((hilbert_dim, hilbert_dim), dtype=np.complex128)
        for kk in range(nn + 1):
            stag_offset = 0.0 if kk % 2 == 0 else -1.0
            electric_op += (static_charges[kk] + stag_offset) * ident
            electric_op += local_op(
                operator=ops["N"],
                op_site=kk,
                lvals=lvals,
                has_obc=has_obc,
                staggered_basis=False,
                gauge_basis=None,
            )
        psi_tmp = electric_op.dot(psi_full)
        e2_reference[nn] = float(np.real(np.vdot(psi_tmp, psi_tmp)))
    e2_residual = float(np.max(np.abs(e2_reconstructed - e2_reference)))
    if e2_residual > atol:
        raise AssertionError(
            "QED 1D integrated test02: FAIL "
            f"(max residual = {e2_residual:.3e}, atol={atol:.1e})"
        )
    logger.info("****************************************************")
    logger.info(
        "QED 1D integrated test01: PASS " f"(max residual = {max_residual:.3e})"
    )
    logger.info(
        "QED 1D integrated test02: PASS " f"(max residual = {e2_residual:.3e})"
    )
    logger.info("****************************************************")


if __name__ == "__main__":
    main()
