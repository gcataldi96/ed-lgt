from ed_lgt.models import BoseHubbard_Model
from simsio import run_sim

with run_sim() as sim:
    model = BoseHubbard_Model(**sim.par["model"])
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
