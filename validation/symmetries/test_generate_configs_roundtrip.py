import logging

import numpy as np

from edlgt.symmetries.generate_configs import (
    config_to_index,
    get_state_configs,
    index_to_config,
)

logger = logging.getLogger(__name__)


def _assert_roundtrip(loc_dims: np.ndarray) -> None:
    total_dim = int(np.prod(loc_dims))
    for idx in range(total_dim):
        cfg = index_to_config(idx, loc_dims)
        idx_back = config_to_index(cfg, loc_dims)
        if idx_back != idx:
            raise AssertionError(
                f"Roundtrip mismatch for loc_dims={loc_dims.tolist()}: "
                f"idx={idx} -> cfg={cfg.tolist()} -> idx_back={int(idx_back)}"
            )


def _assert_matches_state_enumeration(loc_dims: np.ndarray) -> None:
    configs = get_state_configs(loc_dims).astype(np.uint8)
    for idx in range(configs.shape[0]):
        cfg_from_index = index_to_config(idx, loc_dims)
        cfg_from_enum = configs[idx]
        if not np.array_equal(cfg_from_index, cfg_from_enum):
            raise AssertionError(
                f"Enumeration mismatch for loc_dims={loc_dims.tolist()}: "
                f"idx={idx}, index_to_config={cfg_from_index.tolist()}, "
                f"get_state_configs={cfg_from_enum.tolist()}"
            )
        idx_from_cfg = config_to_index(cfg_from_enum, loc_dims)
        if idx_from_cfg != idx:
            raise AssertionError(
                f"Inverse mismatch for loc_dims={loc_dims.tolist()}: "
                f"cfg={cfg_from_enum.tolist()} -> idx={int(idx_from_cfg)}, expected {idx}"
            )


def main() -> None:
    cases = [
        np.array([2, 3], dtype=np.int32),
        np.array([2, 3, 4], dtype=np.int32),
        np.array([2, 2, 2, 2], dtype=np.int32),
        np.array([3, 2, 5], dtype=np.int32),
    ]
    for loc_dims in cases:
        logger.info("----------------------------------------------------")
        logger.info(f"Checking loc_dims={loc_dims.tolist()}")
        _assert_roundtrip(loc_dims)
        _assert_matches_state_enumeration(loc_dims)
        logger.info("PASS")
    logger.info("====================================================")
    logger.info("generate_configs index/config roundtrip: PASS")
    logger.info("====================================================")


if __name__ == "__main__":
    main()
