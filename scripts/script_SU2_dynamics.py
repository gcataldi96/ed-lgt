import numpy as np
from ed_lgt.models import SU2_Model
from ed_lgt.operators import SU2_Hamiltonian_couplings
from simsio import run_sim
from time import perf_counter
import logging

logger = logging.getLogger(__name__)
with run_sim() as sim:
    start_time = perf_counter()
    model = SU2_Model(**sim.par["model"])
    # GLOBAL SYMMETRIES
    if model.pure_theory:
        global_ops = None
        global_sectors = None
    else:
        global_ops = [model.ops[op] for op in sim.par["symmetries"]["sym_ops"]]
        global_sectors = sim.par["symmetries"]["sym_sectors"]
    # LINK SYMMETRIES
    link_ops = [
        [model.ops[f"T2_p{d}"], -model.ops[f"T2_m{d}"]] for d in model.directions
    ]
    link_sectors = [0 for _ in model.directions]
    # GET SYMMETRY SECTOR
    model.get_abelian_symmetry_sector(
        global_ops=global_ops,
        global_sectors=global_sectors,
        link_ops=link_ops,
        link_sectors=link_sectors,
    )
    # DEFAUL PARAMS
    model.default_params()
    # BUILD AND DIAGONALIZE HAMILTONIAN
    coeffs = SU2_Hamiltonian_couplings(
        model.dim, model.pure_theory, sim.par["g"], sim.par["m"]
    )
    model.build_Hamiltonian(coeffs)
    # model.diagonalize_Hamiltonian(n_eigs=sim.par["hamiltonian"]["n_eigs"])
    # LIST OF LOCAL OBSERVABLES
    local_obs = [f"T2_{s}{d}" for d in model.directions for s in "mp"]
    local_obs += ["E_square"]
    if not model.pure_theory:
        local_obs += [f"N_{label}" for label in ["r", "g", "tot", "single", "pair"]]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = [[f"P_p{d}", f"P_m{d}"] for d in model.directions]
    twobody_axes = [d for d in model.directions]
    # LIST OF PLAQUETTE OPERATORS
    plaquette_obs = []
    # DEFINE OBSERVABLES
    model.get_observables(
        local_obs, twobody_obs, plaquette_obs, twobody_axes=twobody_axes
    )
    # INITIAL STATE
    vacuum = np.array([0, 4, 0, 4, 0, 4, 0, 4, 0, 2])
    p_vacuum = np.array([1, 5, 1, 5, 1, 5, 1, 5, 1, 1])
    indx1 = np.where((model.sector_configs == vacuum).all(axis=1))[0]
    indx2 = np.where((model.sector_configs == p_vacuum).all(axis=1))[0]
    initial_state = np.zeros(model.sector_configs.shape[0])
    initial_state[indx1] = 1
    # MEASUREMENTS
    start = 0
    stop = 3
    delta_n = 0.01
    n_steps = int((stop - start) / delta_n)
    sim.res["entropy"] = np.zeros(n_steps, dtype=float)
    sim.res["fidelity"] = np.zeros(n_steps, dtype=float)
    # TIME EVOLUTION
    model.time_evolution_Hamiltonian(initial_state, start, stop, n_steps)
    for ii in range(n_steps):
        # ENTROPY
        sim.res["entropy"][ii] = model.H.psi_time[ii].entanglement_entropy(
            list(np.arange(0, int(model.lvals[0] / 2), 1)),
            sector_configs=model.sector_configs,
        )
        # FIDELITY WITH THE INITIAL STATE
        sim.res["fidelity"][ii] = np.abs(model.H.psi_time[ii].psi[indx1]) ** 2
        if sim.res["entropy"][ii] > 0.98:
            logger.info(sim.res["fidelity"][ii])
            model.H.psi_time[ii].get_state_configurations(
                threshold=1e-2, sector_configs=model.sector_configs
            )
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
