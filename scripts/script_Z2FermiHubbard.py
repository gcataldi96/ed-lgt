from ed_lgt.models import Z2_FermiHubbard_Model
from simsio import run_sim
from ed_lgt.tools import analyze_correlator

with run_sim() as sim:
    model = Z2_FermiHubbard_Model(**sim.par["model"])
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
    coeffs = {"U": sim.par["U"], "t": sim.par["t"]}
    model.build_Hamiltonian(coeffs)
    model.diagonalize_Hamiltonian(n_eigs=sim.par["hamiltonian"]["n_eigs"])
    # LIST OF LOCAL OBSERVABLES
    # local_obs = [f"n_{s}{d}" for d in model.directions for s in "mp"]
    local_obs = [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
    local_obs += ["X_Cross", "S2"]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = [["Sz", "Sz"]]
    """[
        ["Sz_px,mx", "Sz_px,mx"],
        ["Sz_py,my", "Sz_py,my"],
        ["Sx_px,mx", "Sx_px,mx"],
        ["Sx_py,my", "Sx_py,my"],
    ]"""
    twobody_axes = None  # []  # ["x", "y", "x", "y"]
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
        model.H.Npsi[ii].get_state_configurations(sector_indices=model.sector_indices)
        # MEASURE OBSERVABLES
        model.measure_observables(ii)
        sim.res["Sz_Sz"] = analyze_correlator(model.res["Sz_Sz"])
        # CHECK LINK SYMMETRIES
        # model.check_symmetries()
        if ii == 0:
            # SAVE RESULTS
            for measure in model.res.keys():
                sim.res[measure] = model.res[measure]
