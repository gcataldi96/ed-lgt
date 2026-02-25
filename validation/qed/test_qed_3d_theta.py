from edlgt.workflows.qed import run_QED_spectrum, check_observables
import logging

logger = logging.getLogger(__name__)


def main():
    par = {
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
            "get_momentum_basis": False,
            "unit_cell_size": [1, 1, 1],
            "momentum_k_vals": [0, 0, 0],
        },
        "observables": {
            "measure_obs": True,
            "get_entropy": False,
            "entropy_partition": [0, 1, 2, 3],
            "get_state_configs": False,
            "get_overlap": False,
        },
        "g": 2.3,
        "theta": 0.41,
    }
    res = run_QED_spectrum(par)
    ref = {
        "energy": -0.37555632787741716,
        "E2": 0.548060559355979,
        "C_px,py_C_py,mx_C_my,px_C_mx,my": 0.05641112744051553,
        "C_px,pz_C_pz,mx_C_mz,px_C_mx,mz": 0.05641112744051553,
        "C_py,pz_C_pz,my_C_mz,py_C_my,mz": 0.05641112744051553,
        "EzC_px,py_C_py,mx_C_my,px_C_mx,my": 0.3875548826814126,
        "EyC_px,pz_C_pz,mx_C_mz,px_C_mx,mz": 0.3875548826814126,
        "ExC_py,pz_C_pz,my_C_mz,py_C_my,mz": 0.3875548826814126,
    }
    check_observables(res, ref, atol=1e-10, tag="QED 3D theta term test01")
    logger.info("****************************************************")
    logger.info("")
    logger.info("QED 3D theta term spectrum test01: PASS")
    logger.info("")
    logger.info("****************************************************")


if __name__ == "__main__":
    main()
