import logging
from ed_lgt.workflows.qed import run_QED_spectrum, check_observables

logger = logging.getLogger(__name__)


def main():
    par_OBC = {
        "model": {
            "lvals": [4, 2],
            "has_obc": [True, True],
            "spin": 1,
            "pure_theory": False,
            "ham_format": "sparse",
            "bg_list": None,
        },
        "hamiltonian": {"n_eigs": 1, "save_psi": False},
        "momentum": {
            "get_momentum_basis": False,
            "unit_cell_size": [1, 1],
            "momentum_k_vals": [0, 0],
        },
        "observables": {
            "measure_obs": True,
            "get_entropy": False,
            "get_state_configs": False,
            "get_overlap": False,
        },
        "g": 1.0,
        "m": 1.0,
    }
    res_OBC = run_QED_spectrum(par_OBC)
    ref_OBC = {
        "energy": -0.6862839217038432,
        "E2": 0.11506337997036731,
        "N": 0.06869112048269635,
        "C_px,py_C_py,mx_C_my,px_C_mx,my": 0.35977232322995706,
    }
    check_observables(res_OBC, ref_OBC, atol=1e-10, tag="QED 2D pure OBC test01")
    logger.info("****************************************************")
    logger.info("QED 2D pure OBC test01: PASS")
    logger.info("****************************************************")
    par_PBC = {
        "model": {
            "lvals": [4, 2],
            "has_obc": [False, False],
            "spin": 1,
            "pure_theory": False,
            "ham_format": "sparse",
            "bg_list": None,
        },
        "hamiltonian": {"n_eigs": 1, "save_psi": False},
        "momentum": {
            "get_momentum_basis": False,
            "unit_cell_size": [1, 1],
            "momentum_k_vals": [0, 0],
        },
        "observables": {
            "measure_obs": True,
            "get_entropy": False,
            "get_state_configs": False,
            "get_overlap": False,
        },
        "g": 1.0,
        "m": 1.0,
    }
    res_PBC = run_QED_spectrum(par_PBC)
    ref_PBC = {
        "energy": -0.8830087027925408,
        "E2": 0.1906093125650072,
        "N": 0.11057207253261524,
        "C_px,py_C_py,mx_C_my,px_C_mx,my": 0.36750298319023755,
    }
    check_observables(res_PBC, ref_PBC, atol=1e-10, tag="QED 2D with matter PBC test02")
    logger.info("****************************************************")
    logger.info("QED 2D with matter PBC test02: PASS")
    logger.info("****************************************************")


if __name__ == "__main__":
    main()
