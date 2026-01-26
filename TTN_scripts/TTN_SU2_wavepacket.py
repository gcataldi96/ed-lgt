# %%
import numpy as np
from qtealeaves import QuantumGreenTeaSimulation
from qtealeaves.convergence_parameters import TNConvergenceParameters
from qtealeaves.observables import (
    Local,
    TNObservables,
    BondEntropy,
    State2File,
    GenericMPO,
)
from qtealeaves.mpos import DenseMPO
from qtealeaves.emulator import TTN
from qredtea.torchapi import default_pytorch_backend
import torch as to
from pathlib import Path

# from ed_lgt.operators import SU2_dressed_site_operators, SU2_gauge_invariant_states
from qtealeaves.operators.tnoperators import TNOperators
from qtealeaves.modeling import QuantumModel, LocalTerm, TwoBodyTerm1D


def product_state_preparation(L, local_dim):
    product_state = np.zeros((L, local_dim))
    # Start from the staggered bare vacuum
    for xx in range(L):
        if (xx) % 2 == 0:
            product_state[xx, 0] = 1
        else:
            product_state[xx, 4] = 1
    return product_state


"""def SU2_gauge_invariant_ops(spin, pure_theory, lattice_dim):
    in_ops = SU2_dressed_site_operators(spin, pure_theory, lattice_dim)
    gauge_basis, _ = SU2_gauge_invariant_states(spin, pure_theory, lattice_dim)
    ops = {}
    ops_size = gauge_basis["site"].shape[1]
    for op in in_ops.keys():
        ops[op] = np.array(
            (gauge_basis["site"].T @ in_ops[op] @ gauge_basis["site"]).todense()
        )
        ops["id"] = np.eye(ops_size)
    return ops"""


class TN_SU2_operators(TNOperators):
    def __init__(self, spin=0.5, pure_theory=False, lattice_dim=1):
        # ops = SU2_gauge_invariant_ops(spin, pure_theory, lattice_dim)
        ops = {
            "T2_mx": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.75, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.75, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.75],
                ],
                dtype=np.complex128,
            ),
            "id": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.complex128,
            ),
            "T2_px": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.75, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.75, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.75],
                ],
                dtype=np.complex128,
            ),
            "Qpx_dag": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.189207115003, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.840896415254, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.189207115003, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.840896415254, 0.0, 0.0],
                ],
                dtype=np.complex128,
            ),
            "Qmx_dag": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.840896415254, 0.0, 0.0, 0.0, 0.0],
                    [1.189207115003, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -1.189207115003, 0.0, 0.0],
                    [0.0, 0.0, 0.840896415254, 0.0, 0.0, 0.0],
                ],
                dtype=np.complex128,
            ),
            "Qpx": np.array(
                [
                    [0.0, 0.0, 1.189207115003, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.840896415254, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -1.189207115003, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.840896415254],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.complex128,
            ),
            "Qmx": np.array(
                [
                    [0.0, 0.0, 0.0, 1.189207115003, 0.0, 0.0],
                    [0.0, 0.0, 0.840896415254, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.840896415254],
                    [0.0, 0.0, 0.0, 0.0, -1.189207115003, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.complex128,
            ),
            "N_tot": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
                ],
                dtype=np.complex128,
            ),
            "N_single": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.complex128,
            ),
            "N_pair": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.complex128,
            ),
            "N_zero": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.complex128,
            ),
            "E_square": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.75, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.375, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.375, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.75],
                ],
                dtype=np.complex128,
            ),
        }
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
    """Staggered mask function, params it's needed by qtl because it must be a callable"""

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


def gaussian_wavepacket_coeffs(N, k, sigma):
    """
    Return coefficients c[x] = exp(-(x - x0)^2 / (2 sigma^2)) * exp(i k x)
    for a wavepacket of size N centered at x0 = (N-1)/2.

    Parameters
    ----------
    N : int
        Number of sites used to define the wavepacket envelope.
    k : float
        Momentum in radians.
    sigma : float
        Gaussian width (in site units).

    Returns
    -------
    c : np.ndarray of shape (N,), dtype=complex128
        Complex coefficients for the wavepacket.
    """
    xs = np.arange(N, dtype=float)
    x0 = (N - 1) / 2.0
    envelope = np.exp(-0.5 * ((xs - x0) / sigma) ** 2)
    phase = np.exp(1j * k * xs)
    coeffs = envelope * phase
    # Normalize so that sum |c|^2 = 1
    coeffs_norm = np.sqrt(np.sum(np.abs(coeffs) ** 2))
    return coeffs / coeffs_norm


"""def get_MPO(sim_name):
    config_filename = f"scattering/{sim_name}"
    match = SimsQuery(group_glob=config_filename)
    ugrid, vals = uids_grid(match.uids, ["g", "m"])
    MPO_list = []
    for ii in range(4):
        MPO_list.append(get_sim(ugrid[0, 0]).res[f"MPO[{ii}]"])
    return MPO_list"""


def get_MPO():
    data = np.load("MPO.npz")
    return [data[f"arr_{i}"] for i in range(len(data.files))]


def setup_run_dir(sim_name: str) -> Path:
    """
    Create results/<sim_name> and return its absolute Path.
    No chdir. No input/output subdirs.
    """
    run_dir = Path("WP_folder") / sim_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


def main(
    tn_type=5,
    statics_method=0,
    gs_max_bond_dim=200,
    m=3,
    g=1,
    L=64,
    local_dim=6,
    device="cpu",
    simulation_name="dynamics",
    has_obc=False,
):
    sim_folder = setup_run_dir(simulation_name)
    # Acquire model and operator
    model, my_ops = get_SU2_model(has_obc=has_obc)
    # Define the MPS convergence parameters
    my_conv = TNConvergenceParameters(
        max_iter=10,
        max_bond_dimension=gs_max_bond_dim,
        statics_method=statics_method,
        min_bond_dimension=5,
        device=device,
        rel_deviation=1e-6,
        cut_ratio=1e-3,
    )
    # Define the list of observables
    obs_list = ["N_single", "N_pair", "E_square", "T2_px", "T2_mx"]
    my_obs = TNObservables()
    for obs in obs_list:
        my_obs += Local(obs, obs)
    my_obs += State2File("GS_state", "U")
    # Add Entropy
    my_obs += BondEntropy()
    # Define the python backend
    py_tensor_backend = default_pytorch_backend(device=device, dtype=to.complex128)
    simulation = QuantumGreenTeaSimulation(
        model,
        my_ops,
        my_conv,
        my_obs,
        tn_type=tn_type,
        py_tensor_backend=py_tensor_backend,
        folder_name_output=sim_folder,
        store_checkpoints=False,
    )
    # params is a vector of dictionaries so that I can do more than 1 simulation at time
    params = []
    # Initial state
    initial_state = TTN.product_state_from_local_states(
        product_state_preparation(L, local_dim),
        padding=[10, 1e-16],
        convergence_parameters=my_conv,
        tensor_backend=py_tensor_backend,
    )
    sim_dict = {
        "L": L,
        "m": m,
        "g": g,
        "device": device,
        # "Quenches": [quench],
        "continue_file": initial_state,
        "exclude_from_hash": ["Quenches", "device", "t_x_even", "t_x_odd", "theta"],
    }
    sim_dict |= SU2_Hamiltonian_couplings()
    params.append(sim_dict)
    simulation.run(params, delete_existing_folder=True, nthreads=1)
    print(f"----------------------------------------------------")
    print("GROUND STATE SIMULATION COMPLETED")
    # Acquire all the results of the simulations (which being unique is the first item of params)
    static_results = simulation.get_static_obs(params[0])
    # Save results results
    res = {"results": static_results, "entropies": {}}
    for obs in obs_list:
        print(f"----------------------------------------------------")
        print(f"{obs}")
        print(f"----------------------------------------------------")
        res[obs] = static_results[obs]
        for ii in range(L):
            print(f"({ii})  {res[obs][ii]:.16f}")
    print(f"----------------------------------------------------")
    print(f"CHECK PENALTY")
    for ii in range(L - 1):
        print(f"({ii})  {res['T2_px'][ii]-res['T2_mx'][ii+1]}")
    print(f"----------------------------------------------------")
    # Entropy
    res |= static_results["bond_entropy0"]
    GSpsi = TTN.read_pickle("GS_state.pklttn", py_tensor_backend)
    for tensor in GSpsi._iter_tensors():
        print(f"{tensor.shape}")
    support = 4
    wp_size = 7
    offset = 2
    MPO = get_MPO()
    for i, W in enumerate(MPO):
        print(f"site {i}: shape {W.shape}")

    amplitudes = list(gaussian_wavepacket_coeffs(wp_size, k=np.pi / 4, sigma=1.0))
    list_states = []
    wp_conv = TNConvergenceParameters(
        max_bond_dimension=150,
        cut_ratio=1e-13,
    )
    for ii in range(0, L, 2):
        sites_list = np.arange(ii, ii + support + 1, 1, dtype=int).tolist()
        W_mpo_i = DenseMPO.from_tensor_list(
            MPO,
            conv_params=wp_conv,
            iso_center=None,
            tensor_backend=py_tensor_backend,
            sites=sites_list,
        )
    print("------------ OVERLAP between Wanniers ------------------")
    for ii in range(wp_size):
        print("----------------------------------------------------")
        print(f"Building wavepacket state {ii+1}/{wp_size}")
        site = int(2 * ii)
        sites_list = np.arange(
            offset + site, offset + site + support, 1, dtype=int
        ).tolist()
        print(f"sites list {sites_list}")
        state = GSpsi.copy()
        state.convergence_parameters = wp_conv
        W_mpo_i = DenseMPO.from_tensor_list(
            MPO,
            conv_params=wp_conv,
            iso_center=None,
            tensor_backend=py_tensor_backend,
            sites=sites_list,
        )
        state.eff_op = None  # eff_op is now outdated
        tot_singvals_cut = state.apply_mpo(W_mpo_i)
        state.normalize()
        print(f"{tot_singvals_cut} singular values cut")
        list_states.append(state)
        print(f"Single state tensor")
        for tensor in state._iter_tensors():
            print(f"{tensor.shape}")
    print("------------ OVERLAP between Wanniers ------------------")
    overlaps = np.zeros((len(list_states), len(list_states)))
    for ii in range(len(list_states)):
        for jj in range(len(list_states)):
            overlaps[ii, jj] = np.abs(list_states[ii].dot(list_states[jj]))
            print(f"<W{ii}|W{jj}> {overlaps[ii, jj]:.8f}")
    print(overlaps)
    print("SUMMING WAVEPACKET STATE")
    wavepacket_state = TTN.sum_approximate(
        sum_states=list_states,
        sum_amplitudes=amplitudes,
        convergence_parameters=wp_conv,
        max_iterations=100,
    )
    print(f"Wave packet tensor")
    for tensor in wavepacket_state._iter_tensors():
        print(f"{tensor.shape}")
    for ii, alpha in enumerate(amplitudes):
        square_overlap = np.abs(list_states[ii].dot(wavepacket_state)) ** 2
        square_alpha = np.abs(alpha) ** 2
        print(
            f"|<Wann{ii}|WAVEPACK>|^2={square_overlap} {square_alpha} {square_overlap-square_alpha: .6f}"
        )
    print(f"NORM {wavepacket_state.norm()}")
    wavepacket_state.save_pickle("wavepacket_state.pkl")
    # Compute the overlap with the ground state
    GSoverlap = wavepacket_state.dot(GSpsi)
    print(f"Overlap with the ground state: {GSoverlap:.16f}")
    return


# %%

if __name__ == "__main__":
    main()

# %%
