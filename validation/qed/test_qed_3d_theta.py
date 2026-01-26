from ed_lgt.workflows.qed import run_QED_spectrum


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
        "g": 2.3277777777777775,
        "theta": 0.41,
    }
    res = run_QED_spectrum(par)
    ref = {
        "energy": -0.0892347822873874,
        "E2": 0.242194958774601,
    }
    atol = 1e-10
    obs_list = ["energy", "E2"]
    for obs in obs_list:
        if not abs(res[obs][0] - ref[obs]) < atol:
            raise ValueError(f"QED spectrum test01: FAIL on observable {obs}")
    print("QED spectrum test01: PASS")


if __name__ == "__main__":
    main()
