import numpy as np
import logging
from edlgt.workflows.qed import run_QED_spectrum, check_observables

logger = logging.getLogger(__name__)


def main():
    bg_list = [-1, 0, 0, 0, 0, +1]
    par = {
        "model": {
            "lvals": [3, 2],
            "has_obc": [True, True],
            "spin": 1,
            "pure_theory": False,
            "ham_format": "sparse",
            "bg_list": bg_list,
        },
        "hamiltonian": {"n_eigs": 1, "save_psi": False},
        "observables": {
            "measure_obs": True,
            "get_entropy": False,
            "get_state_configs": True,
            "get_overlap": False,
        },
        "g": 1,
        "m": 1,
    }
    res = run_QED_spectrum(par)
    ref = {
        "energy": -0.49619430243949,
        "E2": 0.4462942942081073,
        "N": 0.08360798337722186,
        "C_px,py_C_py,mx_C_my,px_C_mx,my": 0.4555429224294397,
    }
    check_observables(res, ref, atol=1e-10, tag="QED bg-string OBC test07")
    logger.info("QED bg-string OBC test07: PASS")


if __name__ == "__main__":
    main()
