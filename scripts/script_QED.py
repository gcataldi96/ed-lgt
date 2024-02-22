from ed_lgt.models import QED_Model
from ed_lgt.operators import QED_Hamiltonian_couplings
from simsio import run_sim

with run_sim() as sim:
    model = QED_Model(**sim.par["model"])
    # SYMMETRIES
    global_ops = [model.ops[op] for op in sim.par["symmetries"]["sym_ops"]]
    global_sectors = sim.par["symmetries"]["sym_sectors"]
    link_ops = [
        [model.ops["P_px"], model.ops["P_mx"]],
        [model.ops["P_py"], model.ops["P_my"]],
    ]
    link_sectors = [1, 1]
    model.get_abelian_symmetry_sector(
        global_ops=global_ops,
        global_sectors=global_sectors,
        global_sym_type="U",
        link_ops=link_ops,
        link_sectors=link_sectors,
    )
    # DEFAUL PARAMS
    model.default_params()
    # BUILD AND DIAGONALIZE HAMILTONIAN
    coeffs = QED_Hamiltonian_couplings(
        sim.par["model"]["pure_theory"], sim.par["g"], sim.par["m"]
    )
    model.buil_Hamiltonian(coeffs)
    model.diagonalize_Hamiltonian(n_eigs=sim.par["hamiltonian"]["n_eigs"])
    # LIST OF LOCAL OBSERVABLES
    local_obs = [f"E_{s}{d}" for d in model.directions for s in "mp"]
    local_obs += ["N"]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = [["P_px", "P_mx"], ["P_py", "P_my"]]
    twobody_axes = ["x", "y"]
    # LIST OF PLAQUETTE OPERATORS
    plaquette_obs = [["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]]
    # DEFINE OBSERVABLES
    model.get_observables(
        local_obs, twobody_obs, plaquette_obs, twobody_axes=twobody_axes
    )
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
