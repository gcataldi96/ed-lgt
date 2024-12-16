from ed_lgt.models import BoseHubbard_Model
from simsio import run_sim
from time import perf_counter
import logging

logger = logging.getLogger(__name__)

with run_sim() as sim:
    start_time = perf_counter()
    model = BoseHubbard_Model(**sim.par["model"])
    # -------------------------------------------------------------------------------
    # BUILD THE HAMILTONIAN
    coeffs = {"h": sim.par["h"]}
    model.build_Hamiltonian(coeffs)
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
    model.diagonalize_Hamiltonian(
        n_eigs=sim.par["hamiltonian"]["n_eigs"],
        format=sim.par["hamiltonian"]["format"],
    )
    sim.res["energy"] = model.H.Nenergies
    # BUILD AND DIAGONALIZE HAMILTONIAN
    coeffs = {"t": sim.par["t"], "U": sim.par["U"]}
    model.build_Hamiltonian(coeffs)
    model.diagonalize_Hamiltonian(**sim.par["hamiltonian"])
    # LIST OF LOCAL OBSERVABLES
    local_obs = ["N"]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = []
    # DEFINE OBSERVABLES
    model.get_observables(local_obs, twobody_obs)
    for ii in range(model.n_eigs):
        # PRINT ENERGY
        model.H.print_energy(ii)
        # PRINT STATE CONFIGURATIONS
        model.H.Npsi[ii].get_state_configurations(
            threshold=1e-3, sector_indices=model.sector_indices
        )
        # MEASURE OBSERVABLES
        model.measure_observables(ii)
        if ii == 0:
            for measure in model.res.keys():
                sim.res[measure] = model.res[measure]
