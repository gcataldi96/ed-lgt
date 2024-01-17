from ed_lgt.models import Z2_FermiHubbard_Model
from simsio import run_sim

with run_sim() as sim:
    sim.par["coeffs"]["U"] = sim.par["U"]
    model = Z2_FermiHubbard_Model(sim.par)
    # BUILD AND DIAGONALIZE HAMILTONIAN
    model.build_Hamiltonian()
    # LIST OF LOCAL OBSERVABLES
    local_obs = [f"n_{s}{d}" for d in model.directions for s in "mp"]
    local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
    local_obs += ["X_Cross"]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = [["P_px", "P_mx"], ["P_py", "P_my"]]
    # LIST OF PLAQUETTE CORRELATORS
    plaquette_obs = []  # ["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]
    # DEFINE OBSERVABLES
    model.get_observables(local_obs, twobody_obs, plaquette_obs)
    # MEASUREMENTS
    sim.res["obs"] = {}
    for ii in range(model.n_eigs):
        # PRINT ENERGY
        model.H.print_energy(ii)
        # MEASURE OBSERVABLES
        model.measure_observables(ii)
        # CHECK LINK SYMMETRIES
        model.check_symmetries()
        if ii == 0:
            # SAVE RESULTS
            for measure in model.res.keys():
                sim.res[measure] = model.res[measure]
