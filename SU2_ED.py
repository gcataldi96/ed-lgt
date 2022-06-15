import numpy as np
import os
from scipy.sparse import csr_matrix
import argparse

# ===================================================================================
from Data_Analysis.Manage_Data import store_results, save_dictionary
from Error_Debugging.Checks import pause
from Hamitonian_Functions.QMB_Operations.Density_Matrix_Tools import truncation
from Hamitonian_Functions.QMB_Operations.Density_Matrix_Tools import Pure_State
from Hamitonian_Functions.LGT_Objects import *
from Hamitonian_Functions.LGT_Hamiltonians import *
from Hamitonian_Functions.Print_Observables import *

# ===================================================================================
# DEFINE THE ARGUMENTS OF THE PARSER FUNCTION
parser = argparse.ArgumentParser(description="PARAMETERS OF SU2 HAMILTONIAN")
parser.add_argument("-x", nargs=1, type=int, help="x_SIZE OF THE LATTICE")
parser.add_argument("-y", nargs=1, type=int, help="y_SIZE OF THE LATTICE")
parser.add_argument("-pure", nargs=1, type=str, help="y to exclude MATTER FIELDS")
parser.add_argument("-pbc", nargs=1, type=str, help="y for PBC")
parser.add_argument("-m", nargs=1, type=float, help="m MASS of MATTER FIELDS")
parser.add_argument("-debug", nargs=1, type=int, help="If 0 Activate Debug")
# ===================================================================================
# ACQUIRING THE ARGUMENTS VIA PARSER
args = parser.parse_args()
# X-Lattice dimension
nx = int(args.x[0])
# Y-Lattice dimension
ny = int(args.y[0])
# String with Lattice size
lattice_size_string = str(nx) + "x" + str(ny)
# Total Number of lattice sites
n = nx * ny
# Debugging mode
if args.debug[0] == 0:
    debug = True
else:
    debug = False
# Boundary Conditions
if args.pbc[0] == "y":
    PBC = True
    BC_string = "PBC"
else:
    PBC = False
    BC_string = "OBC"
# FREE SU(2) THEORY
if args.pure[0] == "y":
    pure_theory = True
    theory_label = "Pure"
    phrase = "     PURE THEORY: NO MATTER FIELDS"
    # Local dimension of the theory
    local_dimension = 9
    # Labelling the simulation
    simulation_label = lattice_size_string
    # IMPORTING SINGLE SITE OPERATORS
    from Operators.SU2_Free_Operators import identity
    from Operators.SU2_Free_Operators import gamma_operator
    from Operators.SU2_Free_Operators import plaquette
    from Operators.SU2_Free_Operators import W_operators
    from Operators.SU2_Free_Operators import penalties
else:
    pure_theory = False
    theory_label = "Full"
    phrase = "     FULL THEORY: MATTER + GAUGE FIELDS"
    # Local dimension of the theory
    local_dimension = 30
    # Mass of Matter Fields
    mass = float(args.m[0])
    # Labelling the simulation
    simulation_label = "m_" + format(mass, ".5f")
    # IMPORTING SINGLE SITE OPERATORS
    from Operators.SU2_Matter_Operators import identity
    from Operators.SU2_Matter_Operators import gamma_operator
    from Operators.SU2_Matter_Operators import plaquette
    from Operators.SU2_Matter_Operators import W_operators
    from Operators.SU2_Matter_Operators import penalties
    from Operators.SU2_Matter_Operators import hopping
    from Operators.SU2_Matter_Operators import matter_operator
    from Operators.SU2_Matter_Operators import number_operators


# SU(2) Gauge Coupling set
gauge_coupling_set = np.logspace(start=-1, stop=1, num=30)
params_list = gauge_coupling_set.tolist()
# ===================================================================================
print("####################################################")
print(phrase)
print("####################################################")
print("")
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
    SU2_Hopping = Two_Body_Correlator(Q_Left_dagger, Q_Right, Q_Bottom_dagger, Q_Top)
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
    H_Border, n_border_penalties = Borders_Hamiltonian(nx, ny, Penalty, phrase, debug)
    n_penalties += n_border_penalties


simulation_label = "m_" + format(mass, ".5f")
# RESULT DICTIONARY
results = {
    "params": {
        "Lx": nx,
        "Ly": ny,
        "LxL": nx * ny,
        "g_SU2": {"values": params_list, "label": r"$g_{SU(2)}$"},
        "simulation_label": simulation_label,
    },
    "energy": {
        "label": r"$E_{GS}$",
        "values": [],
    },
    "gamma": {
        "label": r"$\avg{\Gamma}$",
        "values": [],
    },
    "plaq": {
        "label": r"$\avg{C}$",
        "values": [],
    },
    "n_single_EVEN": {
        "label": r"$\avg{n_{\uparrow}+n_{\downarrow}-2n_{\uparrow}n{\downarrow}}_{+}$",
        "values": [],
    },
    "n_single_ODD": {
        "label": r"$\avg{n_{\uparrow}+n_{\downarrow}-2n_{\uparrow}n{\downarrow}}_{-}$",
        "values": [],
    },
    "n_pair_EVEN": {
        "label": r"$\avg{n_{\uparrow}n{\downarrow}}_{+}$",
        "values": [],
    },
    "n_pair_ODD": {
        "label": r"$\avg{n_{\uparrow}n{\downarrow}}_{-}$",
        "values": [],
    },
    "n_tot_EVEN": {
        "label": r"$\avg{n_{\uparrow}+n_{\downarrow}}_{+}$",
        "values": [],
    },
    "n_tot_ODD": {
        "label": r"$\avg{n_{\uparrow}+n_{\downarrow}}_{-}$",
        "values": [],
    },
    "Correlators": {},
}

# ----------------------------------------------------------------------------------
# Run over different values of the PARAMETER
for g in params_list:
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
        # ADD MATTER FIELDS
        H = H + mass * H_Matter + 0.5 * H_Hopping
    if not PBC:
        # ADD BORDER PENALTIES DUE TO OBC
        H = H - eta * H_Border
    # ----------------------------------------------------------------------------------
    # DIAGONALIZING THE HAMILTONIAN
    GS = Pure_State()
    GS.ground_state(H, debug)
    # RESHIFT THE LOWEST EIGENVALUE OF THE HAMILTONIAN BY THE ENERGY PENALTIES
    # AND GET THE GROUND STATE ENERGY DENSITY (OF SINGLE SITE)
    GS.energy = (GS.energy + n_penalties * eta) / n
    # SAVE THE ENERGY VALUE IN A LIST
    results["energy"]["values"].append(GS.energy)
    # TRUNCATE THE ENTRIES OF THE GROUND STATE
    psi = truncation(GS.psi, 10 ** (-10))
    # ----------------------------------------------------------------------------------
    if not pure_theory:
        print("----------------------------------------------------")
        print("          EFFECTIVE MASS m ", format(mass, ".5f"))
    print("----------------------------------------------------")
    print("    SU(2) GAUGE COUPLING g ", format(g, ".3f"))
    print("----------------------------------------------------")
    print("    GROUND STATE ENERGY  E ", format(GS.energy, ".5f"))
    # COMPUTE GAMMA EXPECTATION VALUE
    results["gamma"]["values"].append(Print_LOCAL_Observable(GS.psi, nx, ny, Gamma))
    # CHECK PLAQUETTE VALUES
    print("----------------------------------------------------")
    print("    PLAQUETTE VALUES")
    print("----------------------------------------------------")
    print("")
    results["plaq"]["values"].append(
        Print_PLAQUETTE_Correlators(
            GS.psi,
            nx,
            ny,
            SU2_Plaq,
            periodicity=PBC,
            not_Hermitian=True,
            get_real=True,
            get_imag=False,
        )
    )
    print("")
    imag_plaquette = Print_PLAQUETTE_Correlators(
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
        avg_single_odd, avg_single_even = Print_LOCAL_Observable(
            GS.psi, nx, ny, N_SINGLE, staggered=True
        )
        results["n_single_ODD"]["values"].append(avg_single_odd)
        results["n_single_EVEN"]["values"].append(avg_single_even)
        # N_PAIR
        avg_pair_odd, avg_pair_even = Print_LOCAL_Observable(
            GS.psi, nx, ny, N_PAIR, staggered=True
        )
        results["n_pair_ODD"]["values"].append(avg_pair_odd)
        results["n_pair_EVEN"]["values"].append(avg_pair_even)
        # N_TOT
        avg_tot_odd, avg_tot_even = Print_LOCAL_Observable(
            GS.psi, nx, ny, N_TOTAL, staggered=True
        )
        results["n_tot_ODD"]["values"].append(avg_tot_odd)
        results["n_tot_EVEN"]["values"].append(avg_tot_even)
    # CHECK LINK SYMMETRY PENALTIES
    # DEFINE A VOCABULARY TO STORE THE EXPECTATION VALUES of 2BODY CORRELATORS
    correlators = {}
    print("----------------------------------------------------")
    print("    CHECK LINK SYMMETRY")
    print("----------------------------------------------------")
    correlators = Print_TWO_BODY_Correlators(
        correlators, GS.psi, nx, ny, SU2_W_Link, periodicity=PBC
    )
    if PBC == False:
        # CHECK PENALTIES ON THE BORDERS
        print("----------------------------------------------------")
        print("    CHECK PENALTIES ON THE BORDERS")
        print("----------------------------------------------------")
        correlators = Print_BORDER_Penalties(correlators, GS.psi, nx, ny, Penalty)

    print("")
    print("")
    print("")
    print("")
    print("")
    print("")

# Truncate the entries of the coupling list
params_list = ["%.3f" % elem for elem in params_list]
# Add the label for the values of the Hamiltonian parameter
params_list.insert(0, "g")
# Insert a label in each list of observables
results["energy"]["values"].insert(0, "Energy")
results["gamma"]["values"].insert(0, Gamma.Op_name)
results["plaq"]["values"].insert(0, "Plaq")
if not pure_theory:
    # N_single
    results["n_single_ODD"]["values"].insert(0, f"{N_SINGLE.Op_name}_odd")
    results["n_single_EVEN"]["values"].insert(0, f"{N_SINGLE.Op_name}_even")
    # N_pair
    results["n_pair_ODD"]["values"].insert(0, f"{N_PAIR.Op_name}_odd")
    results["n_pair_EVEN"]["values"].insert(0, f"{N_PAIR.Op_name}_even")
    # N_tot
    results["n_tot_ODD"]["values"].insert(0, f"{N_TOTAL.Op_name}_odd")
    results["n_tot_EVEN"]["values"].insert(0, f"{N_TOTAL.Op_name}_even")

# SAVE RESULTS ON TEXT FILES
RESULTS_dir = f"Results/SU2_{theory_label}_{BC_string}/"
Simulation_name = f"{RESULTS_dir}Simulation_{simulation_label}.txt"
if not os.path.exists(RESULTS_dir):
    os.makedirs(RESULTS_dir)
store_results(Simulation_name, params_list, results["energy"]["values"])
store_results(Simulation_name, params_list, results["gamma"]["values"])
store_results(Simulation_name, params_list, results["plaq"]["values"])
if not pure_theory:
    store_results(Simulation_name, params_list, results["n_single_EVEN"]["values"])
    store_results(Simulation_name, params_list, results["n_single_ODD"]["values"])
    store_results(Simulation_name, params_list, results["n_pair_EVEN"]["values"])
    store_results(Simulation_name, params_list, results["n_pair_ODD"]["values"])
    store_results(Simulation_name, params_list, results["n_tot_EVEN"]["values"])
    store_results(Simulation_name, params_list, results["n_tot_ODD"]["values"])

results.clear()
