import logging
from ed_lgt.workflows.qed import run_QED_spectrum, check_observables

logger = logging.getLogger(__name__)


def main():
    par_OBC = {
        "model": {
            "lvals": [4],
            "has_obc": [True],
            "spin": 1,
            "pure_theory": False,
            "ham_format": "sparse",
            "bg_list": None,
        },
        "hamiltonian": {"n_eigs": 1, "save_psi": False},
        "momentum": {
            "get_momentum_basis": False,
            "unit_cell_size": [1],
            "momentum_k_vals": [0],
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
        "energy": -0.5691411466075982,
        "E2": 0.03161445624423755,
        "N": 0.0474013240232069,
    }
    check_observables(res_OBC, ref_OBC, atol=1e-10, tag="QED 1D matter OBC test01")
    logger.info("****************************************************")
    logger.info("QED 1D matter OBC test01: PASS")
    logger.info("****************************************************")
    par_PBC = {
        "model": {
            "lvals": [4],
            "has_obc": [False],
            "spin": 1,
            "pure_theory": False,
            "ham_format": "sparse",
            "bg_list": None,
        },
        "hamiltonian": {"n_eigs": 1, "save_psi": False},
        "momentum": {
            "get_momentum_basis": False,
            "unit_cell_size": [1],
            "momentum_k_vals": [0],
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
        "energy": -0.5903881075118848,
        "E2": 0.029961500436415017,
        "N": 0.0598294699532885,
    }
    check_observables(res_PBC, ref_PBC, atol=1e-10, tag="QED 1D matter PBC test02")
    logger.info("****************************************************")
    logger.info("QED 1D matter PBC test02: PASS")
    logger.info("****************************************************")


if __name__ == "__main__":
    main()
