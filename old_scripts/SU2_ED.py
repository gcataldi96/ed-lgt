from copy import deepcopy
import numpy as np
import os
from scipy.sparse import csr_matrix
import argparse

# ===================================================================================
from Data_Analysis.Manage_Data import save_dictionary
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
    # IMPORTING SINGLE SITE OPERATORS
    from old_operators.SU2_Free_Operators import identity
    from old_operators.SU2_Free_Operators import gamma_operator
    from old_operators.SU2_Free_Operators import plaquette
    from old_operators.SU2_Free_Operators import W_operators
    from old_operators.SU2_Free_Operators import penalties
else:
    pure_theory = False
    theory_label = "Full"
    phrase = "     FULL THEORY: MATTER + GAUGE FIELDS"
    # Local dimension of the theory
    local_dimension = 30
    # IMPORTING SINGLE SITE OPERATORS
    from old_operators.SU2_Matter_Operators import identity
    from old_operators.SU2_Matter_Operators import gamma_operator
    from old_operators.SU2_Matter_Operators import plaquette
    from old_operators.SU2_Matter_Operators import W_operators
    from old_operators.SU2_Matter_Operators import penalties
    from old_operators.SU2_Matter_Operators import hopping
    from old_operators.SU2_Matter_Operators import matter_operator
    from old_operators.SU2_Matter_Operators import number_operators


params = {
    "Lx": nx,
    "Ly": ny,
    "BC": BC_string,
    "theory": theory_label,
    "g_SU2": {"values": [0.1], "label": r"$g_{SU(2)}$"},
    "mass": {"values": [0.1, 0.2], "label": r"$m$"},
}


# ===================================================================================
print("####################################################")
print(phrase)
print("####################################################")
print("")


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


# ===================================================================================
# RESULT DICTIONARY
obs_labels = {
    "energy": r"$E_{GS}$",
    "gamma": r"$\avg{\Gamma}$",
    "plaq": r"$\avg{C}$",
    "n_single_EVEN": r"$\avg{n_{\uparrow}+n_{\downarrow}-2n_{\uparrow}n{\downarrow}}_{+}$",
    "n_single_ODD": r"$\avg{n_{\uparrow}+n_{\downarrow}-2n_{\uparrow}n{\downarrow}}_{-}$",
    "n_pair_EVEN": r"$\avg{n_{\uparrow}n{\downarrow}}_{+}$",
    "n_pair_ODD": r"$\avg{n_{\uparrow}n{\downarrow}}_{-}$",
    "n_tot_EVEN": r"$\avg{n_{\uparrow}+n_{\downarrow}}_{+}$",
    "n_tot_ODD": r"$\avg{n_{\uparrow}+n_{\downarrow}}_{-}$",
}


simulation_dict = {}
simulation_dict["params"] = deepcopy(params)

results = {}
for mass in params["mass"]["values"]:
    for g in params["g_SU2"]["values"]:
        results["mass"] = mass
        results["g_SU2"] = g
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
        # SAVE THE ENERGY VALUE in the dictionary
        results["energy"] = GS.energy
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
        results["gamma"] = Print_LOCAL_Observable(GS.psi, nx, ny, Gamma)
        # CHECK PLAQUETTE VALUES
        print("----------------------------------------------------")
        print("    PLAQUETTE VALUES")
        print("----------------------------------------------------")
        print("")
        results["plaq"] = Print_PLAQUETTE_Correlators(
            GS.psi,
            nx,
            ny,
            SU2_Plaq,
            periodicity=PBC,
            not_Hermitian=True,
            get_real=True,
            get_imag=False,
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
            results["n_single_ODD"], results["n_single_EVEN"] = Print_LOCAL_Observable(
                GS.psi, nx, ny, N_SINGLE, staggered=True
            )
            # N_PAIR
            results["n_pair_ODD"], results["n_pair_EVEN"] = Print_LOCAL_Observable(
                GS.psi, nx, ny, N_PAIR, staggered=True
            )
            # N_TOT
            results["n_tot_ODD"], results["n_tot_EVEN"] = Print_LOCAL_Observable(
                GS.psi, nx, ny, N_TOTAL, staggered=True
            )

        # CHECK LINK SYMMETRY PENALTIES
        print("----------------------------------------------------")
        print("    CHECK LINK SYMMETRY")
        print("----------------------------------------------------")
        Print_TWO_BODY_Correlators(GS.psi, nx, ny, SU2_W_Link, periodicity=PBC)
        if PBC == False:
            # CHECK PENALTIES ON THE BORDERS
            print("----------------------------------------------------")
            print("    CHECK PENALTIES ON THE BORDERS")
            print("----------------------------------------------------")
            Print_BORDER_Penalties(GS.psi, nx, ny, Penalty)
        # COLLECT ALL THE RESULTS IN A SLOT OF THE DICTIONARY
        simulation_dict[f"m_{mass}_g_{g}"] = deepcopy(results)
        results.clear()
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")


# SAVE RESULTS ON TEXT FILES
RESULTS_dir = f"Results/SU2_{theory_label}_{BC_string}/"
if not os.path.exists(RESULTS_dir):
    os.makedirs(RESULTS_dir)

simulation_label = "prova"
simulation_name = f"{RESULTS_dir}Simulation_{simulation_label}"

save_dictionary(simulation_dict, simulation_name)
