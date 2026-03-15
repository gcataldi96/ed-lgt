import logging
import numpy as np

from edlgt.symmetries.generate_configs import subenv_map_to_unique_indices
from edlgt.models.quantum_model import _unique_configs_with_inverse

logger = logging.getLogger(__name__)


def _assert_array_equal(name: str, got: np.ndarray, expected: np.ndarray) -> None:
    if not np.array_equal(got, expected):
        raise AssertionError(f"{name}: FAIL\nexpected={expected}\nobtained={got}")
    logger.info(f"{name}: PASS")


def _build_partition_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    sector_dim = 512
    subsystem_width = 4
    environment_width = 5
    subsystem_configs = rng.integers(
        0, 4, size=(sector_dim, subsystem_width), dtype=np.int64
    )
    environment_configs = rng.integers(
        0, 3, size=(sector_dim, environment_width), dtype=np.int64
    )
    unique_subsys_configs, subsys_inverse = np.unique(
        subsystem_configs, axis=0, return_inverse=True
    )
    unique_env_configs, env_inverse = np.unique(
        environment_configs, axis=0, return_inverse=True
    )
    return (
        subsystem_configs,
        environment_configs,
        unique_subsys_configs,
        unique_env_configs,
        subsys_inverse,
        env_inverse,
    )


def main():
    (
        subsystem_configs,
        environment_configs,
        unique_subsys_configs,
        unique_env_configs,
        subsys_inverse,
        env_inverse,
    ) = _build_partition_data()

    subsys_map, env_map = subenv_map_to_unique_indices(
        subsystem_configs=subsystem_configs,
        environment_configs=environment_configs,
        unique_subsys_configs=unique_subsys_configs,
        unique_env_configs=unique_env_configs,
    )
    _assert_array_equal("sorted unique subsystem map", subsys_map, subsys_inverse)
    _assert_array_equal("sorted unique environment map", env_map, env_inverse)

    subsys_perm = np.array(
        [2, 0, 3, 1] + list(range(4, unique_subsys_configs.shape[0]))
    )
    env_perm = np.array([1, 3, 0, 2] + list(range(4, unique_env_configs.shape[0])))
    unique_subsys_unsorted = unique_subsys_configs[subsys_perm]
    unique_env_unsorted = unique_env_configs[env_perm]
    expected_subsys_unsorted = np.empty_like(subsys_inverse)
    expected_env_unsorted = np.empty_like(env_inverse)
    for idx, original_idx in enumerate(subsys_perm):
        expected_subsys_unsorted[subsys_inverse == original_idx] = idx
    for idx, original_idx in enumerate(env_perm):
        expected_env_unsorted[env_inverse == original_idx] = idx

    subsys_map_unsorted, env_map_unsorted = subenv_map_to_unique_indices(
        subsystem_configs=subsystem_configs,
        environment_configs=environment_configs,
        unique_subsys_configs=unique_subsys_unsorted,
        unique_env_configs=unique_env_unsorted,
    )
    _assert_array_equal(
        "unsorted unique subsystem map", subsys_map_unsorted, expected_subsys_unsorted
    )
    _assert_array_equal(
        "unsorted unique environment map", env_map_unsorted, expected_env_unsorted
    )

    encodable_configs = np.array(
        [
            [0, 1, 2, 0],
            [1, 1, 0, 2],
            [0, 1, 2, 0],
            [2, 0, 1, 1],
        ],
        dtype=np.int64,
    )
    encodable_loc_dims = np.array([3, 3, 3, 3], dtype=np.int64)
    unique_configs_ref, inverse_ref = np.unique(
        encodable_configs, axis=0, return_inverse=True
    )
    unique_configs_fast, inverse_fast = _unique_configs_with_inverse(
        encodable_configs, encodable_loc_dims
    )
    _assert_array_equal(
        "packed unique configs", unique_configs_fast, unique_configs_ref
    )
    _assert_array_equal("packed inverse map", inverse_fast, inverse_ref)

    overflow_loc_dims = np.array([2**40, 2**30], dtype=np.int64)
    overflow_configs = np.array([[0, 1], [1, 0], [0, 1], [2, 2]], dtype=np.int64)
    unique_configs_safe, inverse_safe = _unique_configs_with_inverse(
        overflow_configs, overflow_loc_dims
    )
    unique_configs_safe_ref, inverse_safe_ref = np.unique(
        overflow_configs, axis=0, return_inverse=True
    )
    _assert_array_equal(
        "overflow fallback unique configs", unique_configs_safe, unique_configs_safe_ref
    )
    _assert_array_equal("overflow fallback inverse map", inverse_safe, inverse_safe_ref)

    logger.info("====================================================")
    logger.info("Subsystem/environment mapping validation: PASS")
    logger.info("====================================================")


if __name__ == "__main__":
    main()
