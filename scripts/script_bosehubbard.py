from ed_lgt.models import BoseHubbard_Model
from simsio import run_sim

with run_sim() as sim:
    sim.par["coeffs"] = {"t": sim.par["t"], "U": sim.par["U"]}
    model = BoseHubbard_Model(sim.par)
    # GET SYMMETRY SECTOR
    sym_sector = sim.par["sym_sector"]
    if sym_sector is not None:
        # GET OPERATORS
        model.get_operators(sparse=False)
        model.get_abelian_symmetry_sector(["N"], [sym_sector], sym_type="U")
    else:
        # GET OPERATORS
        model.get_operators()
    # BUILD AND DIAGONALIZE HAMILTONIAN
    model.build_Hamiltonian()
    model.diagonalize_Hamiltonian()
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
        model.H.Npsi[ii].get_state_configurations()
        # MEASURE OBSERVABLES
        model.measure_observables(ii)
        if ii == 0:
            for measure in model.res.keys():
                sim.res[measure] = model.res[measure]
