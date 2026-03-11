import logging
import numpy as np
from scipy.sparse import coo_matrix

from edlgt.symmetries.generate_configs import get_state_configs
from edlgt.symmetries.translational_sym import (
    get_momentum_basis,
    nbody_data_momentum,
    nbody_data_momentum_2sites,
    nbody_data_momentum_4sites,
)

logger = logging.getLogger(__name__)


def _sorted_full_sector_configs(num_sites: int, local_dim: int) -> np.ndarray:
    local_dims = np.full(num_sites, local_dim, dtype=np.uint8)
    sector_configs = get_state_configs(local_dims).astype(np.int32)
    sort_keys = []
    for site_idx in range(num_sites - 1, -1, -1):
        sort_keys.append(sector_configs[:, site_idx])
    sort_perm = np.lexsort(tuple(sort_keys))
    return np.ascontiguousarray(sector_configs[sort_perm], dtype=np.int32)


def _build_operator_list(op_sites_list: np.ndarray, num_sites: int) -> np.ndarray:
    num_ops = len(op_sites_list)
    op_list = np.zeros((num_ops, num_sites, 2, 2), dtype=np.complex128)
    for op_idx in range(num_ops):
        site_idx = int(op_sites_list[op_idx])
        coeff = float(op_idx + 1)
        op_matrix = np.array(
            [
                [0.7 * coeff, 0.2 + 0.05j * coeff],
                [-0.3 + 0.11j * coeff, -0.4 * coeff],
            ],
            dtype=np.complex128,
        )
        op_list[op_idx, site_idx] = op_matrix
    return op_list


def _triplets_to_csr(
    row_list: np.ndarray, col_list: np.ndarray, value_list: np.ndarray, dim: int
):
    return coo_matrix(
        (value_list, (row_list, col_list)), shape=(dim, dim), dtype=np.complex128
    ).tocsr()


def _assert_triplets_match(
    check_name: str,
    old_triplets: tuple[np.ndarray, np.ndarray, np.ndarray],
    generic_triplets: tuple[np.ndarray, np.ndarray, np.ndarray],
    proj_dim: int,
    atol: float = 1e-12,
) -> None:
    row_old, col_old, value_old = old_triplets
    row_gen, col_gen, value_gen = generic_triplets
    mat_old = _triplets_to_csr(row_old, col_old, value_old, proj_dim)
    mat_gen = _triplets_to_csr(row_gen, col_gen, value_gen, proj_dim)
    delta = mat_old - mat_gen
    max_abs_diff = float(np.max(np.abs(delta.data))) if delta.nnz else 0.0
    if max_abs_diff > atol:
        raise AssertionError(
            f"{check_name}: FAIL (max |delta| = {max_abs_diff:.3e}, atol={atol:.1e})"
        )
    logger.info(f"{check_name}: PASS (max |delta| = {max_abs_diff:.3e})")


def _run_case(
    case_name: str,
    lvals: list[int],
    unit_cell_size: list[int],
    k_vals: list[int],
    op_sites_2: list[int],
    op_sites_4: list[int],
) -> int:
    num_sites = int(np.prod(np.array(lvals, dtype=np.int32)))
    sector_configs = _sorted_full_sector_configs(num_sites=num_sites, local_dim=2)

    basis_data = get_momentum_basis(
        sector_configs=sector_configs,
        lvals=lvals,
        unit_cell_size=np.ascontiguousarray(unit_cell_size, dtype=np.int32),
        k_vals=np.ascontiguousarray(k_vals, dtype=np.int32),
        TC_symmetry=False,
    )
    (
        L_col_ptr,
        L_row_idx,
        L_data,
        R_row_ptr,
        R_col_idx,
        R_data,
    ) = basis_data
    proj_dim = int(L_col_ptr.shape[0] - 1)
    if proj_dim <= 0:
        raise AssertionError(f"{case_name}: momentum projected dimension is zero")

    op_sites_arr_2 = np.ascontiguousarray(op_sites_2, dtype=np.int32)
    op_list_2 = _build_operator_list(op_sites_arr_2, num_sites=num_sites)
    old_triplets_2 = nbody_data_momentum_2sites(
        op_list_2,
        op_sites_arr_2,
        sector_configs,
        L_col_ptr,
        L_row_idx,
        L_data,
        R_row_ptr,
        R_col_idx,
        R_data,
    )
    gen_triplets_2 = nbody_data_momentum(
        op_list_2,
        op_sites_arr_2,
        sector_configs,
        L_col_ptr,
        L_row_idx,
        L_data,
        R_row_ptr,
        R_col_idx,
        R_data,
    )
    _assert_triplets_match(
        check_name=f"{case_name} n_ops=2",
        old_triplets=old_triplets_2,
        generic_triplets=gen_triplets_2,
        proj_dim=proj_dim,
    )

    op_sites_arr_4 = np.ascontiguousarray(op_sites_4, dtype=np.int32)
    op_list_4 = _build_operator_list(op_sites_arr_4, num_sites=num_sites)
    old_triplets_4 = nbody_data_momentum_4sites(
        op_list_4,
        op_sites_arr_4,
        sector_configs,
        L_col_ptr,
        L_row_idx,
        L_data,
        R_row_ptr,
        R_col_idx,
        R_data,
    )
    gen_triplets_4 = nbody_data_momentum(
        op_list_4,
        op_sites_arr_4,
        sector_configs,
        L_col_ptr,
        L_row_idx,
        L_data,
        R_row_ptr,
        R_col_idx,
        R_data,
    )
    _assert_triplets_match(
        check_name=f"{case_name} n_ops=4",
        old_triplets=old_triplets_4,
        generic_triplets=gen_triplets_4,
        proj_dim=proj_dim,
    )
    return 2


def main():
    cases = [
        dict(
            case_name="Momentum sector k=0 (1D)",
            lvals=[6],
            unit_cell_size=[1],
            k_vals=[0],
            op_sites_2=[1, 4],
            op_sites_4=[0, 2, 3, 5],
        ),
        dict(
            case_name="Momentum sector k=(0,0,0) (3D)",
            lvals=[2, 1, 2],
            unit_cell_size=[1, 1, 1],
            k_vals=[0, 0, 0],
            op_sites_2=[0, 3],
            op_sites_4=[0, 1, 2, 3],
        ),
    ]
    n_checks = 0
    for case_params in cases:
        logger.info("----------------------------------------------------")
        logger.info(f"Running {case_params['case_name']}")
        logger.info("----------------------------------------------------")
        n_checks += _run_case(**case_params)
    logger.info("====================================================")
    logger.info(f"Momentum generic n-body validation: PASS ({n_checks} checks).")
    logger.info("====================================================")


if __name__ == "__main__":
    main()
