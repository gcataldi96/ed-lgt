import numpy as np
from scipy.sparse import csr_matrix
import argparse

# ===================================================================================
from Error_Debugging.Checks import pause

# ===================================================================================
from Hamitonian_Functions.QMB_Operations.Simple_Checks import check_commutator
from Hamitonian_Functions.LGT_Hamiltonians import Vocabulary_Local_Operator
from Hamitonian_Functions.LGT_Hamiltonians import Vocabulary_Two_Body_Operator
from Hamitonian_Functions.LGT_Hamiltonians import Vocabulary_Penalty_Operators
from Hamitonian_Functions.LGT_Hamiltonians import Vocabulary_Plaquette

# ===================================================================================
from Hamitonian_Functions.LGT_Objects import Local_Operator, Rishon_Modes
from Hamitonian_Functions.LGT_Objects import Two_Body_Correlator, Plaquette

# ===================================================================================
# DEFINE THE ARGUMENTS OF THE PARSER FUNCTION
parser = argparse.ArgumentParser(description="PARAMETERS OF SU2 HAMILTONIAN")
parser.add_argument("-nx", nargs=1, type=int, help="x_SIZE OF THE LATTICE")
parser.add_argument("-ny", nargs=1, type=int, help="y_SIZE OF THE LATTICE")
parser.add_argument("-pure", nargs=1, type=str, help="y to exclude MATTER FIELDS")
parser.add_argument("-pbc", nargs=1, type=str, help="y for PBC")
parser.add_argument("-d", "--debug", nargs=1, type=int, help="If 0 Activate Debug")
# ===================================================================================
# ACQUIRING THE ARGUMENTS VIA PARSER
args = parser.parse_args()
# X-Lattice dimension
nx = int(args.nx[0])
# Y-Lattice dimension
ny = int(args.ny[0])
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
else:
    PBC = False
# FREE or FULL SU(2) THEORY
if args.pure[0] == "y":
    pure_theory = True
else:
    pure_theory = False
# ===================================================================================
if pure_theory:
    phrase = "    FREE THEORY: NO MATTER FIELDS"
    # Local dimension of the theory
    local_dimension = 9
    # IMPORTING SINGLE SITE OPERATORS
    from old_operators.SU2_Free_Operators import identity
    from old_operators.SU2_Free_Operators import gamma_operator
    from old_operators.SU2_Free_Operators import plaquette
    from old_operators.SU2_Free_Operators import W_operators
    from old_operators.SU2_Free_Operators import penalties
else:
    phrase = "    FULL THEORY: MATTER + GAUGE FIELDS"
    # Local dimension of the theory
    local_dimension = 30
    # IMPORTING SINGLE SITE OPERATORS
    from old_operators.SU2_Matter_Operators import hopping
    from old_operators.SU2_Matter_Operators import matter_operator
    from old_operators.SU2_Matter_Operators import identity
    from old_operators.SU2_Matter_Operators import gamma_operator
    from old_operators.SU2_Matter_Operators import plaquette
    from old_operators.SU2_Matter_Operators import W_operators
    from old_operators.SU2_Matter_Operators import penalties
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
Gamma = Local_Operator(gamma, "Gamma")
Gamma.get_identity(ID)
if not pure_theory:
    # MATTER OPERATOR
    D_matter = matter_operator()
    Matter = Local_Operator(D_matter, "Mass")
    Matter.get_identity(ID)
    # HOPPING OPERATORS
    Q_Left_dagger, Q_Right_dagger, Q_Bottom_dagger, Q_Top_dagger = hopping()
    Q_Right = csr_matrix(Q_Right_dagger.conj().transpose())
    Q_Top = csr_matrix(Q_Top_dagger.conj().transpose())
    SU2_Hopping = Two_Body_Correlator(Q_Left_dagger, Q_Right, Q_Bottom_dagger, Q_Top)
    SU2_Hopping.get_identity(ID)
if not PBC:
    # CORNER PENALTIES
    P_left, P_right, P_bottom, P_top = penalties()
    Penalty = Rishon_Modes(P_left, P_right, P_bottom, P_top)
    Penalty.add_Op_names("  Left", " Right", "Bottom", "   Top")
    Penalty.get_identity(ID)
# ===================================================================================
# COMPUTE THE GAMMA HAMILTONIAN
Voc_Gamma, Voc_Gamma_names = Vocabulary_Local_Operator(nx, ny, Gamma, debug)
# COMPUTE THE PLAQUETTE TERMS
# NOTE: WE NEED TO SUM THE COMPLEX CONJUGATE OF THE PLAQUETTES
phrase = "Plaq"
Voc_Plaq, Voc_Plaq_names = Vocabulary_Plaquette(
    nx, ny, SU2_Plaq, phrase, debug, periodicity=PBC, add_dagger=True
)
# COMPUTE THE LINK HAMILTONIAN
phrase = "WW"
Voc_W_Link, Voc_W_link_names = Vocabulary_Two_Body_Operator(
    nx, ny, SU2_W_Link, phrase, debug, periodicity=PBC
)
if not pure_theory:
    # COMPUTE THE STAGGERED MATTER HAMILTONIAN
    Voc_Matter, Voc_Matter_names = Vocabulary_Local_Operator(
        nx, ny, Matter, debug, staggered=True
    )
    # COMPUTE THE HOPPING (STAGGERED) HAMILTONIAN
    phrase = "Q_Corr"
    Voc_Q_Hop, Voc_Q_Hop_names = Vocabulary_Two_Body_Operator(
        nx,
        ny,
        SU2_Hopping,
        phrase,
        debug,
        periodicity=PBC,
        staggered=True,
        add_dagger=True,
        coeffs=[complex(0.0, -1.0), 1.0],
    )
if not PBC:
    # COMPUTE THE BORDER PENALTY HAMILTONIAN
    phrase = "BORDER PENALTY HAMILTONIAN due to OBC"
    Voc_Penalty, Voc_Penalty_names = Vocabulary_Penalty_Operators(
        nx, ny, Penalty, phrase, debug
    )
# ===================================================================================
phrase = " GAMMA  VS  PLAQUETTES"
pause(phrase, debug)

for gamma_name in Voc_Gamma_names:
    for plaq_name in Voc_Plaq_names:
        print(f"{gamma_name} , {plaq_name}")
        check_commutator(Voc_Gamma[gamma_name], Voc_Plaq[plaq_name])

phrase = " GAMMA  VS  W_LINKS"
pause(phrase, debug)

for gamma_name in Voc_Gamma_names:
    for W_link_name in Voc_W_link_names:
        print(f"{gamma_name} , {W_link_name}")
        check_commutator(Voc_Gamma[gamma_name], Voc_W_Link[W_link_name])

if not PBC:
    phrase = " GAMMA  VS  PENALTIES"
    pause(phrase, debug)

    for gamma_name in Voc_Gamma_names:
        for penalty_name in Voc_Penalty_names:
            print(f"{gamma_name} , {penalty_name}")
            check_commutator(Voc_Gamma[gamma_name], Voc_Penalty[penalty_name])

    phrase = " W_LINKS  VS  PENALTIES"
    pause(phrase, debug)

    for W_link_name in Voc_W_link_names:
        for penalty_name in Voc_Penalty_names:
            print(f"{W_link_name} , {penalty_name}")
            check_commutator(Voc_W_Link[W_link_name], Voc_Penalty[penalty_name])

    phrase = " PLAQUETTES  VS  PENALTIES"
    pause(phrase, debug)

    for plaq_name in Voc_Plaq_names:
        for penalty_name in Voc_Penalty_names:
            print(f"{plaq_name} , {penalty_name}")
            check_commutator(Voc_Plaq[plaq_name], Voc_Penalty[penalty_name])

phrase = " PLAQUETTE  VS  W_LINKS"
pause(phrase, debug)

for plaq_name in Voc_Plaq_names:
    for W_link_name in Voc_W_link_names:
        print(f"{plaq_name} , {W_link_name}")
        check_commutator(Voc_Plaq[plaq_name], Voc_W_Link[W_link_name])


if not pure_theory:
    phrase = " MATTER  VS  Q_HOPPING"
    pause(phrase, debug)

    for matter_name in Voc_Matter_names:
        for hop_name in Voc_Q_Hop_names:
            print(f"{matter_name} , {hop_name}")
            check_commutator(Voc_Matter[matter_name], Voc_Q_Hop[hop_name])

    phrase = " MATTER  VS  GAMMA"
    pause(phrase, debug)

    for matter_name in Voc_Matter_names:
        for gamma_name in Voc_Gamma_names:
            print(f"{matter_name} , {gamma_name}")
            check_commutator(Voc_Matter[matter_name], Voc_Gamma[gamma_name])

    phrase = " HOPPING  VS  GAMMA"
    pause(phrase, debug)

    for hop_name in Voc_Q_Hop_names:
        for gamma_name in Voc_Gamma_names:
            print(f"{hop_name} , {gamma_name}")
            check_commutator(Voc_Q_Hop[hop_name], Voc_Gamma[gamma_name])

    phrase = " MATTER  VS  W_LINKS"
    pause(phrase, debug)

    for matter_name in Voc_Matter_names:
        for W_link_name in Voc_W_link_names:
            print(f"{matter_name} , {W_link_name}")
            check_commutator(Voc_Matter[matter_name], Voc_W_Link[W_link_name])

    phrase = " HOPPING  VS  W_LINKS"
    pause(phrase, debug)

    for hop_name in Voc_Q_Hop_names:
        for W_link_name in Voc_W_link_names:
            print(f"{hop_name} , {W_link_name}")
            check_commutator(Voc_Q_Hop[hop_name], Voc_W_Link[W_link_name])

    phrase = " MATTER  VS  PLAQUETTES"
    pause(phrase, debug)

    for matter_name in Voc_Matter_names:
        for plaq_name in Voc_Plaq_names:
            print(f"{matter_name} , {plaq_name}")
            check_commutator(Voc_Matter[matter_name], Voc_Plaq[plaq_name])

    phrase = " HOPPING  VS  PLAQUETTES"
    pause(phrase, debug)

    for hop_name in Voc_Q_Hop_names:
        for plaq_name in Voc_Plaq_names:
            print(f"{hop_name} , {plaq_name}")
            check_commutator(Voc_Q_Hop[hop_name], Voc_Plaq[plaq_name])

    if not PBC:
        phrase = " MATTER  VS  PENALTY"
        pause(phrase, debug)

        for matter_name in Voc_Matter_names:
            for penalty_name in Voc_Penalty_names:
                print(f"{matter_name} , {penalty_name}")
                check_commutator(Voc_Matter[matter_name], Voc_Penalty[penalty_name])

        phrase = " HOPPING  VS  PENALTY"
        pause(phrase, debug)

        for hop_name in Voc_Q_Hop_names:
            for penalty_name in Voc_Penalty_names:
                print(f"{hop_name} , {penalty_name}")
                check_commutator(Voc_Q_Hop[hop_name], Voc_Penalty[penalty_name])
