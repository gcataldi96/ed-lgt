import qtealeaves as qtl
from qtealeaves.convergence_parameters import TNConvergenceParameters
from qtealeaves.observables import Local, TNObservables, BondEntropy
from qtealeaves.emulator.ttn_simulator import TTN
from qredtea.torchapi import default_pytorch_backend
import torch as to
from pathlib import Path
from qtealeaves.tooling import HilbertCurveMap
import logging

logger = logging.getLogger(__name__)


def setup_run_dir(sim_name: str) -> Path:
    """
    Create results/<sim_name> and return its absolute Path.
    No chdir. No input/output subdirs.
    """
    run_dir = Path("results") / sim_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


# Shared model, operators and helpers
from QED_model import (
    get_QED_model,
    product_state_preparation,
    QED_Hamiltonian_couplings,
)


def main(
    tn_type=5,
    statics_method=2,
    bond_dim=100,
    g=0.1,
    L=4,
    alpha=20,
    local_dim=19,
    device="cpu",
    simulation_name="QED2D",
):
    sim_folder = setup_run_dir(simulation_name)
    # Acquire model and operator
    model, my_ops = get_QED_model()
    # Define the TTN convergence parameters
    my_conv = TNConvergenceParameters(
        max_iter=5,
        max_bond_dimension=bond_dim,
        statics_method=statics_method,
        device=device,
        cut_ratio=1e-12,
    )
    # Define the list of observables
    obs_list = ["E2"]
    obs_list += [f"E_{s}{d}" for s in "pm" for d in "xy"]
    obs_list += [f"E2_{s}{d}" for s in "pm" for d in "xy"]
    my_obs = TNObservables()
    for obs in obs_list:
        my_obs += Local(obs, obs)
    # Add Entropy
    my_obs.add_observable(BondEntropy())
    # Define the python backend
    py_tensor_backend = default_pytorch_backend(device=device, dtype=to.complex128)
    # Configure the simulation
    simulation = qtl.QuantumGreenTeaSimulation(
        model,
        my_ops,
        my_conv,
        my_obs,
        tn_type=tn_type,
        py_tensor_backend=py_tensor_backend,
        folder_name_output=sim_folder,
        store_checkpoints=False,
    )
    # Initial state
    initial_state = TTN.product_state_from_local_states_2d(
        product_state_preparation(L, local_dim),
        padding=[bond_dim, 1e-16],
        convergence_parameters=my_conv,
        tensor_backend=py_tensor_backend,
    )
    # params is a list of dictionaries (each dict is a simulation)
    # in practice I will only do a single simulation with this script
    params = []
    sim_dict = {
        "L": L,
        "g": g,
        "alpha": alpha,
        "device": device,
        "continue_file": initial_state,
        "exclude_from_hash": ["device", "t_x_even", "t_x_odd", "theta"],
    }
    sim_dict |= QED_Hamiltonian_couplings(dim=2, g=g, alpha=alpha)
    params.append(sim_dict)
    # Run the simulation
    simulation.run(params, delete_existing_folder=True, nthreads=1)
    # Acquire all the results of the simulations (which being unique is the first item of params)
    static_results = simulation.get_static_obs(params[0])
    # Save results results
    res = {"results": static_results, "entropies": {}}

    for obs in obs_list:
        res[obs] = static_results[obs]
    # Entropy
    res |= static_results["bond_entropy0"]
    # Define the filename and save the dictionary to a file
    data_filename = sim_folder / "static_data.npz"
    res2d = {}
    hilb = HilbertCurveMap(2, L)
    for obs in obs_list:
        res2d[obs] = hilb.backmapping_vector_observable(res[obs])
    for yy in range(L):
        for xx in range(L):
            if xx == 0:
                print(f"({xx}, {yy}) Emx {res2d['E_mx'][xx, yy]:.8f}")
            elif xx == L - 1:
                print(f"({xx}, {yy}) Epx {res2d['E_px'][xx, yy]:.8f}")
            else:
                diff = res2d["E_px"][xx, yy] + res2d["E_mx"][xx + 1, yy]
                print(f"({xx}, {yy})+({xx+1}, {yy}) Epx-Emx {diff:.8f}")
            if yy == 0:
                print(f"({xx}, {yy}) Emy {res2d['E_my'][xx, yy]:.8f}")
            elif yy == L - 1:
                print(f"({xx}, {yy}) Epy {res2d['E_py'][xx, yy]:.8f}")
            else:
                diff = res2d["E_py"][xx, yy] + res2d["E_my"][xx, yy + 1]
                print(f"({xx}, {yy})+({xx}, {yy+1}) Epy-Emy {diff:.8f}")
    for obs in obs_list[5:]:
        print(f"============ {obs} ===================")
        for xx in range(L):
            for yy in range(L):
                print(f"({xx}, {yy}) {res2d[obs][xx, yy]:.8f}")

    """np.savez_compressed(
        data_filename,
        E_square_px=res2d["E2"],
        E_px=res2d["E_px"],
        E_mx=res2d["E_mx"],
        E_py=res2d["E_py"],
        E_my=res2d["E_my"],
        entropies=res["entropies"],
        results=res["results"],
    )"""

    return


if __name__ == "__main__":
    main()
