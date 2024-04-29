import numpy as np
from ed_lgt.models import SU2_Model
from ed_lgt.operators import SU2_Hamiltonian_couplings
from ed_lgt.symmetries import momentum_basis_k0
from simsio import run_sim
from time import perf_counter
import logging

logger = logging.getLogger(__name__)


def overlap_QMB_state(N, has_obc, name, config_state=None):
    if config_state == None:
        if name == "V":
            s1, s2, L, R = 0, 4, 0, 2
        elif name == "PV":
            s1, s2, L, R = 1, 5, 1, 1
        elif name == "M":
            s1, s2, L, R = 3, 2, 1, 1
        config_state = [s1 if ii % 2 == 0 else s2 for ii in range(N)]
        if has_obc:
            print(has_obc)
            config_state[0] = L
            config_state[-1] = R
        config_state = config_state
    return np.array(config_state)


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
    # -------------------------------------------------------------------------------
    # Project onto the momentum sector k=0
    if model.momentum_basis:
        model.momentum_basis_projection(logical_unit_size=2)
        # Get the momentum basis
        B = momentum_basis_k0(model.sector_configs, 2)
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
    model.diagonalize_Hamiltonian(n_eigs=sim.par["hamiltonian"]["n_eigs"])
    sim.res["energy"] = model.H.Nenergies
    # DETERMINE BETA of THERMALIZATION
    # model.get_thermal_beta()
    # -------------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------------
    # ENTROPY
    sim.res["entropy"] = np.zeros(model.n_eigs, dtype=float)
    # -------------------------------------------------------------------------------
    # OVERLAPS
    ov_info = {"config": {}, "ind": {}, "state": {}}
    for name in ["V", "PV", "M"]:
        # Initialize a null state
        sim.res[f"overlap_{name}"] = np.zeros(model.n_eigs, dtype=float)
        # Define the config_state associated to a specific axis
        config = overlap_QMB_state(model.n_sites, model.has_obc[0], name)
        ov_info["config"][name] = config
        # Get the corresponding QMB index
        ov_info["ind"][name] = np.where((model.sector_configs == config).all(axis=1))[0]
        # Initialize the state
        if model.momentum_basis:
            ov_info["state"][name] = np.zeros(len(model.sector_configs), dtype=float)
            ov_info["state"][name][ov_info["ind"][name]] = 1
            # Project the vacuum in the momentum sector
            ov_info["state"][name] = B.transpose().dot(ov_info["state"][name])
    # -------------------------------------------------------------------------------
    for ii in range(model.n_eigs):
        # PRINT ENERGY
        model.H.print_energy(ii)
        if not model.momentum_basis:
            # -----------------------------------------------------------------------
            # ENTROPY
            sim.res["entropy"][ii] = model.H.Npsi[ii].entanglement_entropy(
                list(np.arange(0, int(model.lvals[0] / 2), 1)),
                sector_configs=model.sector_configs,
            )
            # -----------------------------------------------------------------------
            # STATE CONFIGURATIONS
            model.H.Npsi[ii].get_state_configurations(
                threshold=1.5e-2, sector_configs=model.sector_configs
            )
            # -----------------------------------------------------------------------
            # MEASURE OBSERVABLES
            model.measure_observables(ii)
            # CHECK LINK SYMMETRIES
            model.check_symmetries()
            """
            if ii == 0:
                # SAVE RESULTS
                for measure in model.res.keys():
                    sim.res[measure] = model.res[measure]
            """
        # ---------------------------------------------------------------------------
        # OVERLAPS
        for name in ["V", "PV", "M"]:
            if model.momentum_basis:
                sim.res[f"overlap_{name}"][ii] = (
                    np.abs(ov_info["state"][name].conj().dot(model.H.Npsi[ii].psi)) ** 2
                )
            else:
                sim.res[f"overlap_{name}"][ii] = (
                    np.abs(model.H.Npsi[ii].psi[ov_info["ind"][name]]) ** 2
                )
        # ---------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
