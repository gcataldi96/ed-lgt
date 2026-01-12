# %%
from simsio import *
import numpy as np
from qtealeaves import QuantumGreenTeaSimulation, DynamicsQuench
from qtealeaves.convergence_parameters import TNConvergenceParameters
from qtealeaves.observables import (
    TNObsLocal,
    TNObservables,
    TNObsBondEntropy,
)
from qtealeaves.emulator import MPS
from qredtea.torchapi import default_pytorch_backend
import torch as to
from pathlib import Path
from ed_lgt.operators import SU2_dressed_site_operators, SU2_gauge_invariant_states
from qtealeaves.operators.tnoperators import TNOperators
from qtealeaves.modeling import QuantumModel, LocalTerm, TwoBodyTerm1D


def SU2_gauge_invariant_ops(spin, pure_theory, lattice_dim):
    in_ops = SU2_dressed_site_operators(spin, pure_theory, lattice_dim)
    gauge_basis, _ = SU2_gauge_invariant_states(spin, pure_theory, lattice_dim)
    ops = {}
    ops_size = gauge_basis["site"].shape[1]
    for op in in_ops.keys():
        ops[op] = np.array(
            (gauge_basis["site"].T @ in_ops[op] @ gauge_basis["site"]).todense()
        )
        ops["id"] = np.eye(ops_size)
    return ops


class TN_SU2_operators(TNOperators):
    def __init__(self, spin=0.5, pure_theory=False, lattice_dim=1):
        ops = SU2_gauge_invariant_ops(spin, pure_theory, lattice_dim)
        super().__init__()
        for key, value in ops.items():
            self[key] = value


def SU2_Hamiltonian_couplings(dim=1, g=1.0, m=3.0, theta=0.0):
    """
    This function provides the QED Hamiltonian coefficients
    starting from the gauge coupling g and the bare mass parameter m

    Args:
        pure_theory (bool): True if the theory does not include matter

        g (scalar): gauge coupling

        m (scalar, optional): bare mass parameter

    Returns:
        dict: dictionary of Hamiltonian coefficients
    """
    if dim == 1:
        E = 8 * g / 3
    elif dim == 2:
        E = g / 2
        B = -1 / (2 * g)
    else:
        E = g / 2
        B = -1 / (2 * g)
    # DICTIONARY WITH MODEL COEFFICIENTS
    coeffs = {
        "g": g,
        "E": E,  # ELECTRIC FIELD coupling
        "theta": -complex(0, theta * g),  # THETA TERM coupling
    }
    if dim > 1:
        coeffs["B"] = B  # MAGNETIC FIELD coupling
    if m is not None:
        t = 2 * np.sqrt(2)
        coeffs |= {
            "t_x_even": complex(0, t),  # x HOPPING (EVEN SITES)
            "m_even": m,
            "m_odd": -m,
        }
    return coeffs


def staggered_mask(site: str, params: dict):
    """Staggered mask function, params it's
    needed by qtl because it must be a callable"""

    length = params["L"]
    tmp = np.zeros(length, dtype=bool)
    if site == "even":
        for xx in range(length):
            if (xx) % 2 == 0:
                tmp[xx] = True
    elif site == "odd":
        for xx in range(length):
            if (xx) % 2 == 1:
                tmp[xx] = True
    return tmp


def staggered_even_mask(params):
    return staggered_mask("even", params)


def staggered_odd_mask(params):
    return staggered_mask("odd", params)


def get_SU2_model(has_obc=False):
    model_name = lambda params: "SU2_L%2.4f" % (params["L"])
    model = QuantumModel(1, "L", name=model_name)
    # ---------------------------------------------------------------------------
    # ELECTRIC ENERGY
    op_name = "E_square"
    model += LocalTerm(op_name, strength="E", prefactor=+1)
    # -----------------------------------------------------------------------
    # STAGGERED MASS TERM
    op_name = "N_tot"
    model += LocalTerm(op_name, strength="m", prefactor=+1, mask=staggered_even_mask)
    model += LocalTerm(op_name, strength="m", prefactor=-1, mask=staggered_odd_mask)
    # --------------------------------------------------------------------
    #  HOPPING
    op_names_list = ["Qpx_dag", "Qmx"]
    model += TwoBodyTerm1D(
        op_names_list, shift=1, strength=f"t_x_even", prefactor=-1, has_obc=has_obc
    )
    op_names_list = ["Qpx", "Qmx_dag"]
    model += TwoBodyTerm1D(
        op_names_list, shift=1, strength=f"t_x_even", has_obc=has_obc
    )
    # ---------------------------------------------------------------------------
    ops = TN_SU2_operators()
    return model, ops


def setup_run_dir(sim_name: str) -> Path:
    """
    Create results/<sim_name> and return its absolute Path.
    No chdir. No input/output subdirs.
    """
    run_dir = Path("WP_folder") / sim_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


def main(
    tn_type=6,
    statics_method=0,
    timesteps=10,
    bond_dim=200,
    dt=0.01,
    m=1,
    g=3,
    L=14,
    device="cpu",
    simulation_name="scattering",
    has_obc=False,
):
    sim_folder = setup_run_dir(simulation_name)
    # Acquire model and operator
    model, my_ops = get_SU2_model(has_obc=has_obc)
    # Define the MPS convergence parameters
    my_conv = TNConvergenceParameters(
        max_iter=0,
        max_bond_dimension=100,
        statics_method=statics_method,
        min_bond_dimension=bond_dim,
        device=device,
        rel_deviation=1e-6,
        cut_ratio=1e-3,
    )
    # Define the list of observables
    obs_list = ["N_single", "N_pair", "E_square"]
    my_obs = TNObservables()
    for obs in obs_list:
        my_obs += TNObsLocal(obs, obs)
    # Add Entropy
    my_obs += TNObsBondEntropy()
    # Define the python backend
    py_tensor_backend = default_pytorch_backend(device=device, dtype=to.complex128)
    # SET SIMULATION PARAMETERS
    quench = DynamicsQuench([dt] * timesteps, time_evolution_mode=4)
    quench["m"] = lambda tt, params: m
    quench["g"] = lambda tt, params: g
    simulation = QuantumGreenTeaSimulation(
        model,
        my_ops,
        my_conv,
        my_obs,
        tn_type=tn_type,
        py_tensor_backend=py_tensor_backend,
        folder_name_input=sim_folder,
        folder_name_output=sim_folder,
        store_checkpoints=False,
    )
    # params is a vector of dictionaries so that I can do more than 1 simulation at time
    params = []
    # Initial state
    initial_state = MPS.read_pickle("wavepacket_state.pklmps", py_tensor_backend)
    sim_dict = {
        "L": L,
        "m": m,
        "g": g,
        "device": device,
        "Quenches": [quench],
        "continue_file": initial_state,
        "exclude_from_hash": ["Quenches", "device", "t_x_even", "t_x_odd", "theta"],
    }
    sim_dict |= SU2_Hamiltonian_couplings()
    params.append(sim_dict)
    simulation.run(params, delete_existing_folder=True, nthreads=1)
    logger.info(f"----------------------------------------------------")
    logger.info("TIME EVOLUTION COMPLETED COMPLETED")
    # Acquire all the results of the simulations (which being unique is the first item of params)
    # "dynamics_results" is a list of dictionaries, one for each time step.
    # each dictionary contains all the information of the simulations
    dynamics_results = simulation.get_dynamic_obs(params[0])[0]
    # Acquire simulation results
    res = {"results": dynamics_results, "entropies": {}}
    # Print results
    time = [dt * t for t in range(timesteps)]
    for tidx, t in enumerate(time):
        logger.info(f"------------------  TIME {t}  -----------------------")
        for obs in obs_list:
            logger.info(f"----------------------------------------------------")
            logger.info(f"{obs}")
            logger.info(f"----------------------------------------------------")
            res[obs] = dynamics_results[obs][tidx]
            for ii in range(L):
                logger.info(f"({ii})  {res[obs][ii]:.16f}")
        logger.info(f"----------------------------------------------------")
    # Save entropies
    # Get the partition labels from the step 0
    for key in dynamics_results[0]["bond_entropy0"]:
        res["entropies"][key] = [
            S["bond_entropy0"][key] for S in dynamics_results[1::1]
        ]
    # Define the filename and save the dictionary to a file
    data_filename = sim_folder / "scattering_data.npz"
    np.savez_compressed(
        data_filename,
        Nsingle=res["N_single"],
        Npair=res["N_pair"],
        E2=res["E_square"],
        entropies=res["entropies"],
        results=res["results"],
    )
    return


# %%

if __name__ == "__main__":
    main()

# %%
