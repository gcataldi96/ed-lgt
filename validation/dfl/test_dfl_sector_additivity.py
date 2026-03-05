import ast
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _load_example_dfl_functions():
    """Load the DFL example functions without executing the demo block."""
    path = Path(__file__).resolve().parents[2] / "examples" / "example_DFL.py"
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            keep.append(node)
    module = ast.Module(body=keep, type_ignores=[])
    ns = {"logger": logging.getLogger("validation.dfl")}
    exec(compile(module, str(path), "exec"), ns)
    return ns["run_DFL_dynamics"], ns["run_DFL_dynamics_sector_by_sector"]


def _assert_series_close(name, got, ref, atol):
    diff = np.abs(np.asarray(got) - np.asarray(ref))
    max_abs = float(np.max(diff))
    if max_abs > atol:
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        raise AssertionError(
            f"{name} mismatch: max_abs={max_abs:.3e} at {idx}, "
            f"got={got[idx]}, ref={ref[idx]}"
        )


def main():
    logger.info("****************************************************")
    logger.info("")
    logger.info("Running DFL sector additivity test")
    logger.info("")
    logger.info("****************************************************")

    run_dfl, run_dfl_sector = _load_example_dfl_functions()

    params = {
        "model": {
            "lvals": [8],
            "has_obc": [False],
            "spin": 0.5,
            "pure_theory": False,
            "ham_format": "sparse",
        },
        "dynamics": {
            "time_evolution": True,
            "start": 0.0,
            "stop": 0.4,
            "delta_n": 0.1,
            "logical_stag_basis": 2,
        },
        "momentum": {
            "get_momentum_basis": False,
            "unit_cell_size": [2],
            "TC_symmetry": False,
        },
        "observables": {
            "measure_obs": True,
            "get_entropy": False,
            "get_PE": True,
            "get_SRE": False,
            "entropy_partition": [0, 1, 2, 3],
            "get_state_configs": False,
            "get_fidelity": False,
        },
        "ensemble": {
            "microcanonical": {"average": False},
            "diagonal": {"average": False},
            "canonical": {"average": False},
        },
        "g_values": np.array([5.0]),
        "m": 1,
    }

    res_all = run_dfl(params)
    res_sector = run_dfl_sector(params)

    for obs in ["E2", "N_single", "N_pair", "N_zero", "N_tot"]:
        _assert_series_close(
            f"{obs}(t) [whole space vs sector sum]",
            res_sector[obs],
            res_all[obs],
            atol=1e-10,
        )
    _assert_series_close(
        "PE(t) [whole space vs sector reconstruction]",
        res_sector["PE"],
        res_all["PE"],
        atol=1e-10,
    )

    logger.info("****************************************************")
    logger.info("")
    logger.info("DFL sector additivity: PASS")
    logger.info("")
    logger.info("****************************************************")


if __name__ == "__main__":
    main()
