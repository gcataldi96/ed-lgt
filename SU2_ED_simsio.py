from scipy.sparse import csr_matrix
from scipy.sparse import identity as IDD

from Error_Debugging.Checks import pause
from Hamitonian_Functions.LGT_Hamiltonians import *
from Hamitonian_Functions.LGT_Objects import *
from Hamitonian_Functions.Print_Observables import *
from Hamitonian_Functions.QMB_Operations.Density_Matrix_Tools import (
    Pure_State,
    entanglement_entropy,
    truncation,
)
from simsio import logger, run_sim

# ===================================================================================

with run_sim() as sim:
    sim.link("psi")
    # X-Lattice dimension
    nx = sim.par["nx"]
    # Y-Lattice dimension
    ny = sim.par["ny"]
    # String with Lattice size
    lattice_size_string = str(nx) + "x" + str(ny)
    # Total Number of lattice sites
    n = nx * ny
    # Debugging mode
    debug = False
    # Boundary Conditions
    PBC = sim.par["PBC"]
    if PBC:
        BC_string = "PBC"
    else:
        BC_string = "OBC"
    # FREE SU(2) THEORY
    pure_theory = sim.par["pure"]
    if pure_theory:
        theory_label = "Pure"
        phrase = "     PURE THEORY: NO MATTER FIELDS"
        # Local dimension of the theory
        loc_dim = 9
        # IMPORTING SINGLE SITE OPERATORS
        from old_operators.SU2_Free_Operators import (
            W_operators,
            gamma_operator,
            identity,
            penalties,
            plaquette,
        )
    else:
        theory_label = "Full"
        phrase = "     FULL THEORY: MATTER + GAUGE FIELDS"
        # Local dimension of the theory
        loc_dim = 30
        # IMPORTING SINGLE SITE OPERATORS
        from old_operators.SU2_Matter_Operators import (
            W_operators,
            gamma_operator,
            hopping,
            identity,
            matter_operator,
            number_operators,
            penalties,
            plaquette,
        )

        # GET MASS VALUE
        mass = sim.par["m"]
    # GET gSU2 VALUE
    g = sim.par["g"]

    # ===================================================================================
    logger.info("####################################################")
    logger.info(phrase)
    logger.info("####################################################")
    logger.info("")

    # ===================================================================================
    # Define the Printing phrase in debug mode
    phrase = "LOCAL OPERATORS"
    pause(phrase, debug)
    # IDENTITY OPERATOR
    ID = identity()
    # W OPERATORS
    W_Left, W_Right, W_Bottom, W_Top = W_operators()
    SU2_W_Link = Two_Body_Correlator(W_Left, W_Right, W_Bottom, W_Top)
    SU2_W_Link.add_Op_names("W_Left", "W_Right", "W_Top", "W_Bottom")
    SU2_W_Link.get_identity(ID)
    # C OPERATORS ON A PLAQUETTE
    C_Bottom_Left, C_Bottom_Right, C_Top_Left, C_Top_Right = plaquette()
    SU2_Plaq = Plaquette(C_Bottom_Left, C_Bottom_Right, C_Top_Left, C_Top_Right)
    SU2_Plaq.add_Op_names("BL", "BR", "TL", "TR")
    SU2_Plaq.get_identity(ID)
    # ELECTRIC GAUGE OPERATOR
    gamma = gamma_operator()
    Gamma = Local_Operator(gamma, "GAMMA")
    Gamma.get_identity(ID)
    if not pure_theory:
        # MATTER OPERATOR
        D_matter = matter_operator()
        Matter = Local_Operator(D_matter, "MASS")
        Matter.get_identity(ID)
        # HOPPING OPERATORS
        Q_Left_dagger, Q_Right_dagger, Q_Bottom_dagger, Q_Top_dagger = hopping()
        Q_Right = csr_matrix(Q_Right_dagger.conj().transpose())
        Q_Top = csr_matrix(Q_Top_dagger.conj().transpose())
        SU2_Hopping = Two_Body_Correlator(
            Q_Left_dagger, Q_Right, Q_Bottom_dagger, Q_Top
        )
        SU2_Hopping.get_identity(ID)
        # NUMBER OPERATORS
        N_single_occupancy, N_double_occupancy, N_total_occupancy = number_operators()
        N_SINGLE = Local_Operator(N_single_occupancy, "n_SINGLE")
        N_SINGLE.get_identity(ID)
        N_PAIR = Local_Operator(N_double_occupancy, "  n_PAIR")
        N_PAIR.get_identity(ID)
        N_TOTAL = Local_Operator(N_total_occupancy, " n_TOTAL")
        N_TOTAL.get_identity(ID)
    if not PBC:
        # CORNER PENALTIES
        P_left, P_right, P_bottom, P_top = penalties()
        Penalty = Rishon_Modes(P_left, P_right, P_bottom, P_top)
        Penalty.add_Op_names("  Left", " Right", "Bottom", "   Top")
        Penalty.get_identity(ID)

    # ===================================================================================
    # COMPUTE THE GAMMA HAMILTONIAN
    H_Gamma = Local_Hamiltonian(nx, ny, Gamma, debug)
    # COMPUTE THE PLAQUETTE HAMILTONIAN:
    # NOTE: WE NEED TO SUM THE COMPLEX CONJUGATE OF THE PLAQUETTES
    phrase = "PLAQUETTE HAMILTONIAN"
    H_Plaq, n_plaqs = Plaquette_Hamiltonian(
        nx, ny, SU2_Plaq, phrase, debug, periodicity=PBC, add_dagger=True
    )
    # COMPUTE THE LINK HAMILTONIAN
    phrase = "LINK-SYMMETRY HAMILTONIAN"
    H_Link, n_link_penalties = Two_Body_Hamiltonian(
        nx, ny, SU2_W_Link, phrase, debug, periodicity=PBC
    )
    # Define a variable that counts the number of penalties added to the Hamiltonian
    n_penalties = n_link_penalties
    if not pure_theory:
        # COMPUTE THE STAGGERED MATTER HAMILTONIAN
        H_Matter = Local_Hamiltonian(nx, ny, Matter, debug, staggered=True)
        # COMPUTE THE HOPPING (STAGGERED) HAMILTONIAN
        phrase = "HOPPING HAMILTONIAN"
        H_Hopping, n_hopping = Two_Body_Hamiltonian(
            nx,
            ny,
            SU2_Hopping,
            phrase,
            debug,
            periodicity=PBC,
            staggered=True,
            add_dagger=True,
            coeffs=[complex(0, -1.0), 1],
        )
    if not PBC:
        # COMPUTE THE BORDER PENALTY HAMILTONIAN
        phrase = "BORDER PENALTY HAMILTONIAN due to OBC"
        H_Border, n_border_penalties = Borders_Hamiltonian(
            nx, ny, Penalty, phrase, debug
        )
        n_penalties += n_border_penalties

    # DICTIONARY CONTAINING ALL THE OBSERVABLES FOR EACH SINGLE SITE
    Obs = {}
    Obs_list = [
        "    Gamma",
        "    n_SINGLE",
        "    n_PAIR",
        "    n_TOTAL",
        "    WW LINK SYMMETRY",
        "    BORDER RISHONS",
    ]
    # FIXING THE PARAMETERS FOR A SINGLE SIMULATION
    if not pure_theory:
        # CHOICE OF THE PENALTY
        eta = 10 * max((3 / 16) * (g**2), 4 / (g**2), mass)
    else:
        # CHOICE OF THE PENALTY
        eta = 10 * max((3 / 16) * (g**2), 4 / (g**2))
    # ----------------------------------------------------------------------------------
    # COMPUTE THE TOTAL HAMILTONIAN
    phrase = "TOTAL HAMILTONIAN"
    pause(phrase, debug)
    # PURE HAMILTONIAN IN PBC
    H = ((3 / 16) * (g**2)) * H_Gamma - eta * (H_Link) - (4 / (g**2)) * H_Plaq
    if not pure_theory:
        # Constraint the number of particles in the model
        fixN = sim.par["N"]
        H_fixN = Local_Hamiltonian(nx, ny, N_TOTAL, debug) - fixN * IDD(30 ** (nx * ny))
        H_fixN = H_fixN * H_fixN
        # ADD MATTER FIELDS
        H = H + mass * H_Matter + 0.5 * H_Hopping + 50 * H_fixN
    if not PBC:
        # ADD BORDER PENALTIES DUE TO OBC
        H = H - eta * H_Border
    """
    Check Hermitianity
    H_herm = csr_matrix.getH(H)
    check_matrix(H, H_herm)
    """
    # ----------------------------------------------------------------------------------
    # DIAGONALIZING THE HAMILTONIAN
    GS = Pure_State()
    GS.ground_state(H)
    # GS.get_first_n_eigs(H, n_eigs=2)
    # RESHIFT THE LOWEST EIGENVALUE OF THE HAMILTONIAN BY THE ENERGY PENALTIES
    # for ii in range(len(GS.N_energies)):
    #    GS.N_energies[ii] = (GS.N_energies[ii] + n_penalties * eta) / n
    # RESHIFT THE LOWEST EIGENVALUE OF THE HAMILTONIAN BY THE ENERGY PENALTIES
    # AND GET THE GROUND STATE ENERGY DENSITY (OF SINGLE SITE)
    GS.GSenergy = (GS.GSenergy + n_penalties * eta) / n
    # SAVE THE ENERGY VALUE in the dictionary
    sim.res["energy"] = GS.GSenergy
    # TRUNCATE THE ENTRIES OF THE GROUND STATE
    psi = truncation(GS.psi, 10 ** (-10))
    # MASURE ENTANGLEMENT OF A BIPARTITION
    sim.res["entropy"] = entanglement_entropy(
        psi=psi, loc_dim=loc_dim, partition_size=int(nx * ny / 2)
    )
    # ----------------------------------------------------------------------------------
    logger.info("----------------------------------------------------")
    logger.info(f"    ENERGY: {GS.GSenergy}")
    logger.info("----------------------------------------------------")
    # COMPUTE GAMMA EXPECTATION VALUE
    sim.res["gamma"], Obs["gamma"] = get_LOCAL_Observable(GS.psi, nx, ny, Gamma)
    # CHECK PLAQUETTE VALUES
    logger.info("----------------------------------------------------")
    logger.info("    PLAQUETTE VALUES")
    logger.info("----------------------------------------------------")
    logger.info("")
    sim.res["plaq"] = get_PLAQUETTE_Correlators(
        GS.psi,
        nx,
        ny,
        SU2_Plaq,
        periodicity=PBC,
        not_Hermitian=True,
        get_real=True,
        get_imag=False,
    )
    logger.info("")
    imag_plaquette = get_PLAQUETTE_Correlators(
        GS.psi,
        nx,
        ny,
        SU2_Plaq,
        periodicity=PBC,
        not_Hermitian=True,
        get_real=False,
        get_imag=True,
    )
    if not pure_theory:
        # COMPUTE THE NUMBER DENSITY OPERATORS
        # N_SINGLE
        (
            sim.res["n_single_ODD"],
            sim.res["n_single_EVEN"],
            Obs["n_single"],
        ) = get_LOCAL_Observable(GS.psi, nx, ny, N_SINGLE, staggered=True)
        # N_PAIR
        (
            sim.res["n_pair_ODD"],
            sim.res["n_pair_EVEN"],
            Obs["n_pair"],
        ) = get_LOCAL_Observable(GS.psi, nx, ny, N_PAIR, staggered=True)
        # N_TOT
        (
            sim.res["n_tot_ODD"],
            sim.res["n_tot_EVEN"],
            Obs["n_tot"],
        ) = get_LOCAL_Observable(GS.psi, nx, ny, N_TOTAL, staggered=True)
    # CHECK LINK SYMMETRY PENALTIES
    Obs["WW"] = get_TWO_BODY_Correlators(GS.psi, nx, ny, SU2_W_Link, periodicity=PBC)
    if PBC == False:
        # CHECK PENALTIES ON THE BORDERS
        Obs["Rishons"] = get_BORDER_Penalties(GS.psi, nx, ny, Penalty)

    for ii, (key, value) in enumerate(Obs.items()):
        logger.info(Obs_list[ii])
        logger.info("")
        logger.info("----------------------------------------------------")
        print_Observables(Obs[key])
        logger.info("----------------------------------------------------")
        logger.info("")


"""phrase = "SYMMETRY SECTORS OF THE HAMILTONIAN"
# GET THE SINGLE SITE SYMMETRY SECTORS WRT THE NUMBER OF FERMIONS
N_tot_sectors = [0, 1, 2]
N_tot_dim_sectors = [9, 12, 9]
single_site_syms = single_site_symmetry_sectors(
    loc_dim, sectors_list=N_tot_sectors, dim_sectors_list=N_tot_dim_sectors
)
# GET THE NBODY SYMMETRY SECTOR STATE
Nbody_syms = many_body_symmetry_sectors(single_site_syms, n) - 4


H_subsector = {}
for ii in range(-4, 5, 2):
    print(ii)
    # GET THE INDICES ASSOCIATED TO EACH SYMMETRY SECTOR
    indices = syms.get_indices_from_array(Nbody_syms, ii)
    indices = indices.tolist()
    print("Computing H subsector of ", ii)
    H_subsector[str(ii)] = syms.get_submatrix_from_sparse(H, indices, indices)
    sub_energy, sub_psi = get_ground_state_from_Hamiltonian(
        csr_matrix(H_subsector[str(ii)]), debug=False
    )
    sub_energy = (sub_energy + n_penalties * eta) / n
    print("SUBSECTOR", ii, "ENERGY", sub_energy)"""


# nohup bash -c "printf 'n%s\n' {0..20} | shuf | xargs -P6 -i python scripts/dmrgs.py massless/phases/weak {} 8" &>/dev/null &
