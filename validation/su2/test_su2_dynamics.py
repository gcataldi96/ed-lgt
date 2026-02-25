import os
import numpy as np
from edlgt.workflows.su2 import run_SU2_dynamics  # adjust to your module
import logging

logger = logging.getLogger(__name__)
GOLD_PATH = os.path.join(os.path.dirname(__file__), "gold_dynamics_case01_dense.npz")


def _series_metrics(x, y):
    """Return (max_abs, rmse)."""
    d = np.asarray(x) - np.asarray(y)
    max_abs = float(np.max(np.abs(d)))
    rmse = float(np.sqrt(np.mean(np.abs(d) ** 2)))
    return max_abs, rmse


def _assert_series_close(name, got, ref, max_abs_tol, rmse_tol):
    max_abs, rmse = _series_metrics(got, ref)
    if max_abs > max_abs_tol or rmse > rmse_tol:
        raise AssertionError(
            f"{name} mismatch: max_abs={max_abs:.3e} (tol {max_abs_tol:.3e}), "
            f"rmse={rmse:.3e} (tol {rmse_tol:.3e})"
        )


def main():
    # --- Small, quick, stable case
    base_par = {
        "model": {
            "lvals": [6],
            "sectors": [6],
            "has_obc": [False],
            "spin": 0.5,
            "pure_theory": False,
            "ham_format": "dense",
        },
        "dynamics": {
            "time_evolution": True,
            "start": 0.0,
            "stop": 0.5,
            "delta_n": 0.01,
            "state": "V",
        },
        "momentum": {
            "get_momentum_basis": False,
            "unit_cell_size": [2],
            "TC_symmetry": False,
        },
        "observables": {
            "measure_obs": True,
            "get_entropy": True,
            "entropy_partition": [0, 1, 2],
            "get_overlap": True,
            "get_state_configs": False,
            "get_RDM": False,
        },
        "g": 1.0,
        "m": 5.0,
    }
    logger.info("****************************************************")
    logger.info("")
    logger.info("Running SU2 dynamics test01")
    logger.info("")
    logger.info("****************************************************")
    # --- Load (or create) dense reference
    if not os.path.exists(GOLD_PATH):
        par_ref = dict(base_par)
        par_ref["model"] = dict(base_par["model"])
        par_ref["model"]["ham_format"] = "dense"
        res_ref = run_SU2_dynamics(par_ref)

        np.savez(
            GOLD_PATH,
            time_steps=res_ref["time_steps"],
            F=res_ref["overlap"],  # assumes overlap is fidelity series
            E2=res_ref["E2"],
            S=res_ref["entropy"],
        )
        logger.info(f"Created golden file: {GOLD_PATH}")
        logger.info("Re-run the test now that goldens exist.")
        return
    gold = np.load(GOLD_PATH)
    t_ref = gold["time_steps"]
    F_ref = gold["F"]
    E2_ref = gold["E2"]
    S_ref = gold["S"]
    # --- Run the other backends
    results = {}
    for fmt in ["sparse", "linear"]:
        par = dict(base_par)
        par["model"] = dict(base_par["model"])
        par["model"]["ham_format"] = fmt
        res = run_SU2_dynamics(par)
        assert np.allclose(res["time_steps"], t_ref, atol=0.0, rtol=0.0)
        results[fmt] = res
    # --- Tolerances for expm_multiply-style agreement
    # Use both max_abs and rmse to avoid brittle per-step assertions.
    # Start conservative; tighten later if you see it's always smaller.
    F_max_tol, F_rmse_tol = 1e-6, 1e-6
    O_max_tol, O_rmse_tol = 1e-6, 1e-6
    S_max_tol, S_rmse_tol = 1e-6, 1e-6
    for fmt, res in results.items():
        F = res["overlap"]
        E2 = res["E2"]
        S = res["entropy"]
        _assert_series_close(f"F(t) [{fmt} vs dense]", F, F_ref, F_max_tol, F_rmse_tol)
        _assert_series_close(
            f"E2(t) [{fmt} vs dense]", E2, E2_ref, O_max_tol, O_rmse_tol
        )
        _assert_series_close(f"S(t) [{fmt} vs dense]", S, S_ref, S_max_tol, S_rmse_tol)
    # Optional: also enforce sparse vs linear agreement (sometimes catches LinearOperator mistakes)
    _assert_series_close(
        "F(t) [linear vs sparse]",
        results["linear"]["overlap"],
        results["sparse"]["overlap"],
        1e-10,
        1e-11,
    )
    _assert_series_close(
        "E2(t) [linear vs sparse]",
        results["linear"]["E2"],
        results["sparse"]["E2"],
        1e-10,
        1e-11,
    )
    logger.info("****************************************************")
    logger.info("")
    logger.info("SU2 dynamics backend: PASS")
    logger.info("")
    logger.info("****************************************************")


if __name__ == "__main__":
    main()
