from ed_lgt.models import Z2_FermiHubbard_Model
from simsio import run_sim

with run_sim() as sim:
    sim.par["coeffs"]["U"] = sim.par["U"]
    model = Z2_FermiHubbard_Model(sim.par)
    # GET LOCAL SITE DIMENSION
    model.get_local_site_dimensions()
    # GET SYMMETRY SECTOR
    sym_sector = sim.par["sym_sector"]
    model.sector = sym_sector
    if sym_sector is not None:
        # GET OPERATORS
        model.get_operators(sparse=False)
        model.get_abelian_symmetry_sector(["N_tot"], [sym_sector], sym_type="U")
    else:
        # GET OPERATORS
        model.get_operators()
    # BUILD AND DIAGONALIZE HAMILTONIAN
    model.build_Hamiltonian()
    model.diagonalize_Hamiltonian()
    # LIST OF LOCAL OBSERVABLES
    local_obs = [f"n_{s}{d}" for d in model.directions for s in "mp"]
    local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
    local_obs += ["X_Cross"]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = [["P_px", "P_mx"], ["P_py", "P_my"]]
    # DEFINE OBSERVABLES
    model.get_observables(local_obs, twobody_obs)
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
