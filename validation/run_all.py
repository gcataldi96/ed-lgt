import os
import sys
import argparse
import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def set_threads_env(n: int | None) -> None:
    """
    Set thread env vars for NUMBA/OMP/MKL/OPENBLAS.

    If n is None: do nothing (use whatever the system/default chooses).
    """
    if n is None:
        return
    t = str(int(n))
    os.environ["NUMBA_NUM_THREADS"] = t
    os.environ["OMP_NUM_THREADS"] = t
    os.environ["MKL_NUM_THREADS"] = t
    os.environ["OPENBLAS_NUM_THREADS"] = t


def discover_test_files(root: Path) -> list[Path]:
    """
    Discover tests in all first-level subfolders of `root`.

    Rule:
      - each subfolder (e.g. su2/, qed/) is a suite
      - any file matching test*.py inside it is a test
    """
    tests: list[Path] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.startswith(".") or sub.name in {"__pycache__"}:
            continue
        tests.extend(sorted(sub.glob("test*.py")))
    return tests


def run_one_test(test_path: Path, env: dict) -> int:
    """Run one test file as a subprocess; return its exit code."""
    logger.info(f"Running {test_path.relative_to(test_path.parents[1])} ===")
    # ^ prints "su2/test01_..." nicely (parents[1] == validation/)
    r = subprocess.run([sys.executable, str(test_path)], env=env)
    return r.returncode


def main() -> int:
    # ----------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Run manual validation tests.")
    threads_msg = "Threads for NUMBA/OMP/MKL/OPENBLAS. Default: system/default."
    parser.add_argument("--threads", type=int, default=None, help=threads_msg)
    suite_msg = "Run only one suite (e.g. su2 or qed). If omitted, run all."
    parser.add_argument("--suite", type=str, default=None, help=suite_msg)
    args = parser.parse_args()
    # ----------------------------------------------------------------------------
    # IMPORTANT: set env vars before importing heavy stuff in child processes.
    set_threads_env(args.threads)
    env = os.environ.copy()
    root = Path(__file__).resolve().parent
    # ----------------------------------------------------------------------------
    # Optional: filter suites
    if args.suite is not None:
        suite_dir = root / args.suite
        if not suite_dir.is_dir():
            logger.info(f"Suite not found: {suite_dir}")
            return 2
        tests = sorted(suite_dir.glob("test*.py"))
    else:
        tests = discover_test_files(root)
    if not tests:
        logger.info("No validation tests found.")
        return 2
    failed = 0
    for test_path in tests:
        code = run_one_test(test_path, env)
        if code != 0:
            failed += 1
    # ----------------------------------------------------------------------------
    total = len(tests)
    logger.info("====================================================")
    logger.info(" Validation Summary ")
    logger.info("====================================================")
    if failed:
        logger.info(f"FAILED: {failed}/{total} tests")
        return 1
    logger.info(f"ALL PASSED: {total} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
