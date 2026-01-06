from ed_lgt.workflows.su2 import run_SU2_spectrum


def main():
    par = {
        "model": {
            "lvals": [6],
            "sectors": [6],
            "has_obc": [False],
            "spin": 0.5,
            "pure_theory": False,
            "background": 0,
            "ham_format": "sparse",
        },
        "hamiltonian": {
            "n_eigs": 1,
            "save_psi": False,
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
            "get_state_configs": True,
            "get_overlap": False,
        },
        "ensemble": {
            "microcanonical": {"average": False},
            "diagonal": {"average": False},
            "canonical": {"average": False},
        },
        "g": 1,
        "m": 5,
    }
    res = run_SU2_spectrum(par)
    ref = {
        "energy": -6.173752796132477,
        "entropy": 0.802957045143,
        "E2": 0.0553604161539725,
        "N_tot": 0.15535989251437904,
        "N_single": 0.14157936434695081,
        "N_pair": 0.006890264083714107,
        "N_zero": 0.8515303715693359,
    }
    atol = 1e-10
    obs_list = ["energy", "entropy", "E2", "N_tot", "N_single", "N_pair", "N_zero"]
    for obs in obs_list:
        if not abs(res[obs][0] - ref[obs]) < atol:
            raise ValueError(f"SU2 spectrum test01: FAIL on observable {obs}")
    print("SU2 spectrum test01: PASS")


if __name__ == "__main__":
    main()
