import numpy as np
from ed_lgt.models import IsingModel
from ed_lgt.modeling import QMB_state

from simsio import run_sim
import logging

logger = logging.getLogger(__name__)

with run_sim() as sim:
    model = IsingModel(**sim.par["model"])
    # SYMMETRIES
    global_ops = [model.ops[op] for op in sim.par["symmetries"]["sym_ops"]]
    global_sectors = sim.par["symmetries"]["sym_sectors"]
    global_sym_type = sim.par["symmetries"]["sym_type"]
    model.get_abelian_symmetry_sector(
        global_ops=global_ops,
        global_sectors=global_sectors,
        global_sym_type=global_sym_type,
    )
    # DEFAUL PARAMS
    model.default_params()
    # -------------------------------------------------------------------------------
    # BUILD THE HAMILTONIAN
    coeffs = {"J": 1, "h": sim.par["h"]}
    model.build_Hamiltonian(coeffs)
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
    if sim.par["hamiltonian"]["diagonalize"]:
        model.diagonalize_Hamiltonian(
            n_eigs=sim.par["hamiltonian"]["n_eigs"],
            format=sim.par["hamiltonian"]["format"],
        )
        sim.res["energy"] = model.H.Nenergies
    # -------------------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = ["Sx", "Sz"]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = []
    # DEFINE OBSERVABLES
    model.get_observables(local_obs, twobody_obs)
    # -------------------------------------------------------------------------------
    sim.res["entropy"] = np.zeros(model.n_eigs, dtype=float)
    for obs in local_obs:
        sim.res[obs] = np.zeros((model.n_eigs, model.n_sites), dtype=float)
    # -------------------------------------------------------------------------------
    for ii in range(model.n_eigs):
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
                threshold=1e-1, sector_configs=model.sector_configs
            )
        # ---------------------------------------------------------------------------
        # MEASURE OBSERVABLES
        model.measure_observables(ii, dynamics=False)
        for obs in local_obs:
            sim.res[obs][ii, :] = model.res[obs]
