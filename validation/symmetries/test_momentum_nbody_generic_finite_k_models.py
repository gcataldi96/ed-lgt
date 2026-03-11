import copy
import logging

from edlgt.workflows.su2 import run_SU2_spectrum
from edlgt.workflows.qed import run_QED_spectrum
import edlgt.symmetries.sym_qmb_operations as sym_ops
from edlgt.symmetries.translational_sym import nbody_data_momentum

logger = logging.getLogger(__name__)


def _run_with_kernel_dispatch(run_fn, par: dict, force_generic_dispatch: bool):
    if not force_generic_dispatch:
        return run_fn(copy.deepcopy(par))
    old_2site = sym_ops.nbody_data_momentum_2sites
    old_4site = sym_ops.nbody_data_momentum_4sites
    sym_ops.nbody_data_momentum_2sites = nbody_data_momentum
    sym_ops.nbody_data_momentum_4sites = nbody_data_momentum
    try:
        return run_fn(copy.deepcopy(par))
    finally:
        sym_ops.nbody_data_momentum_2sites = old_2site
        sym_ops.nbody_data_momentum_4sites = old_4site


def _assert_ground_state_energy_close(
    check_name: str, ref_res: dict, gen_res: dict, atol: float = 1e-10
):
    ref_energy = float(ref_res["energy"][0])
    gen_energy = float(gen_res["energy"][0])
    abs_diff = abs(ref_energy - gen_energy)
    if abs_diff > atol:
        raise AssertionError(
            f"{check_name}: FAIL (|deltaE|={abs_diff:.3e}, atol={atol:.1e})"
        )
    logger.info(f"{check_name}: PASS (|deltaE|={abs_diff:.3e})")


def main():
    # For unit_cell_size=[1], k=pi corresponds to integer label L/2.
    su2_par = {
        "model": {
            "lvals": [6],
            "sectors": [6],
            "has_obc": [False],
            "spin": 0.5,
            "pure_theory": False,
            "ham_format": "sparse",
        },
        "hamiltonian": {
            "n_eigs": 1,
            "save_psi": False,
        },
        "momentum": {
            "get_momentum_basis": True,
            "unit_cell_size": [1],
            "momentum_k_vals": [3],  # pi in 1D with L=6
            "TC_symmetry": False,
        },
        "observables": {
            "measure_obs": False,
            "get_entropy": False,
            "get_state_configs": False,
            "get_overlap": False,
        },
        "ensemble": {
            "microcanonical": {"average": False},
            "diagonal": {"average": False},
            "canonical": {"average": False},
        },
        "g": 1.0,
        "m": 5.0,
    }
    logger.info("----------------------------------------------------")
    logger.info("Running SU2 finite-momentum check (k=pi)")
    logger.info("----------------------------------------------------")
    su2_res_ref = _run_with_kernel_dispatch(
        run_fn=run_SU2_spectrum,
        par=su2_par,
        force_generic_dispatch=False,
    )
    su2_res_gen = _run_with_kernel_dispatch(
        run_fn=run_SU2_spectrum,
        par=su2_par,
        force_generic_dispatch=True,
    )
    _assert_ground_state_energy_close(
        check_name="SU2 finite-k ground state",
        ref_res=su2_res_ref,
        gen_res=su2_res_gen,
    )

    qed_par = {
        "model": {
            "lvals": [2, 2, 2],
            "has_obc": [False, False, False],
            "spin": 1,
            "pure_theory": True,
            "ham_format": "sparse",
        },
        "hamiltonian": {
            "n_eigs": 1,
            "save_psi": False,
        },
        "momentum": {
            "get_momentum_basis": True,
            "unit_cell_size": [1, 1, 1],
            "momentum_k_vals": [1, 1, 1],  # (pi,pi,pi) for L=(2,2,2)
            "TC_symmetry": False,
        },
        "observables": {
            "measure_obs": False,
            "get_entropy": False,
            "get_state_configs": False,
            "get_overlap": False,
        },
        "g": 2.3,
        "theta": 0.41,
    }
    logger.info("----------------------------------------------------")
    logger.info("Running QED finite-momentum check (k=(pi,pi,pi))")
    logger.info("----------------------------------------------------")
    qed_res_ref = _run_with_kernel_dispatch(
        run_fn=run_QED_spectrum,
        par=qed_par,
        force_generic_dispatch=False,
    )
    qed_res_gen = _run_with_kernel_dispatch(
        run_fn=run_QED_spectrum,
        par=qed_par,
        force_generic_dispatch=True,
    )
    _assert_ground_state_energy_close(
        check_name="QED finite-k ground state",
        ref_res=qed_res_ref,
        gen_res=qed_res_gen,
    )

    logger.info("====================================================")
    logger.info("Finite-momentum model-level generic n-body validation: PASS")
    logger.info("====================================================")


if __name__ == "__main__":
    main()
