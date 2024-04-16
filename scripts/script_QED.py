from ed_lgt.models import QED_Model
from ed_lgt.operators import QED_Hamiltonian_couplings
from simsio import run_sim

with run_sim() as sim:
    par = sim.par["model"]
    par["spin"] = sim.par["spin"]
    model = QED_Model(**par)
    # GLOBAL SYMMETRIES
    if model.pure_theory:
        global_ops = None
        global_sectors = None
    else:
        global_ops = [model.ops[op] for op in sim.par["symmetries"]["sym_ops"]]
        global_sectors = sim.par["symmetries"]["sym_sectors"]
    # LINK SYMMETRIES
    link_ops = [[model.ops[f"E_p{d}"], model.ops[f"E_m{d}"]] for d in model.directions]
    link_sectors = [0 for _ in model.directions]
    # GET SYMMETRY SECTOR
    model.get_abelian_symmetry_sector(
        global_ops=global_ops,
        global_sectors=global_sectors,
        global_sym_type="U",
        link_ops=link_ops,
        link_sectors=link_sectors,
    )
    # DEFAULT PARAMS
    model.default_params()
    # BUILD AND DIAGONALIZE HAMILTONIAN
    coeffs = QED_Hamiltonian_couplings(
        model.dim, model.pure_theory, sim.par["g"], sim.par["m"]
    )
    model.build_Hamiltonian(coeffs)
    model.diagonalize_Hamiltonian(n_eigs=sim.par["hamiltonian"]["n_eigs"])
    # ---------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = [f"E_{s}{d}" for d in model.directions for s in "mp"] + ["E_square"]
    if not model.pure_theory:
        local_obs += ["N"]
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = [[f"P_p{d}", f"P_m{d}"] for d in model.directions]
    twobody_axes = [d for d in model.directions]
    # LIST OF PLAQUETTE OPERATORS
    plaquette_obs = [["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]]
    # DEFINE OBSERVABLES
    model.get_observables(
        local_obs, twobody_obs, plaquette_obs, twobody_axes=twobody_axes
    )
    for ii in range(model.n_eigs):
        # PRINT ENERGY
        model.H.print_energy(ii)
        # ENTROPY
        sim.res["entropy"] = float(
            model.H.Npsi[ii].entanglement_entropy(
                [0, 1], sector_configs=model.sector_configs
            )
        )
        # STATE CONFIGURATIONS
        model.H.Npsi[ii].get_state_configurations(sector_configs=model.sector_configs)
        # MEASURE OBSERVABLES
        model.measure_observables(ii)
        # CHECK LINK SYMMETRIES
        model.check_symmetries()
        if ii == 0:
            # SAVE RESULTS
            for measure in model.res.keys():
                sim.res[measure] = model.res[measure]
    for ii in range(model.n_eigs):
        model.H.print_energy(ii)
