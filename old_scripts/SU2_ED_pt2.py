import numpy as np
import os
from scipy.sparse import csr_matrix
import argparse

# ===================================================================================
from Data_Analysis.Manage_Data import store_results
from Error_Debugging.Checks import pause
from Hamitonian_Functions.QMB_Operations.Mappings_1D_2D import coords

# ===================================================================================
from Hamitonian_Functions.QMB_Operations.Density_Matrix_Tools import (
    get_reduced_density_matrix,
)
from Hamitonian_Functions.QMB_Operations.Density_Matrix_Tools import (
    truncation,
    get_projector,
)
from Hamitonian_Functions.QMB_Operations.Density_Matrix_Tools import (
    Pure_State,
    get_ground_state_from_Hamiltonian,
)

# ===================================================================================
from Hamitonian_Functions.LGT_Objects import Local_Operator, Rishon_Modes
from Hamitonian_Functions.LGT_Objects import Two_Body_Correlator, Plaquette

# ===================================================================================
from Hamitonian_Functions.LGT_Hamiltonians import Local_Hamiltonian
from Hamitonian_Functions.LGT_Hamiltonians import Two_Body_Hamiltonian
from Hamitonian_Functions.LGT_Hamiltonians import Plaquette_Hamiltonian
from Hamitonian_Functions.LGT_Hamiltonians import Borders_Hamiltonian

# ===================================================================================
from Hamitonian_Functions.Print_Observables import Print_LOCAL_Observable
from Hamitonian_Functions.Print_Observables import Print_TWO_BODY_Correlators
from Hamitonian_Functions.Print_Observables import Print_PLAQUETTE_Correlators
from Hamitonian_Functions.Print_Observables import Print_BORDER_Penalties

# ===================================================================================
from Hamitonian_Functions import Symmetry_sectors as syms

# ===================================================================================
# DEFINE THE ARGUMENTS OF THE PARSER FUNCTION
parser = argparse.ArgumentParser(description="PARAMETERS OF SU2 HAMILTONIAN")
parser.add_argument("-nx", nargs=1, type=int, help="x_SIZE OF THE LATTICE")
parser.add_argument("-ny", nargs=1, type=int, help="y_SIZE OF THE LATTICE")
parser.add_argument("-pure", nargs=1, type=str, help="y to exclude MATTER FIELDS")
parser.add_argument("-pbc", nargs=1, type=str, help="y for PBC")
parser.add_argument("-op", nargs=1, type=str, help="m: vary MASS, g: the GAUGE")
parser.add_argument("-g", nargs=1, type=float, help="g SU(2) COUPLING VALUES")
parser.add_argument("-m", nargs=1, type=float, help="m MASS of MATTER FIELDS")
parser.add_argument("-debug", nargs=1, type=int, help="If 0 Activate Debug")
# ===================================================================================
# ACQUIRING THE ARGUMENTS VIA PARSER
args = parser.parse_args()
# X-Lattice dimension
nx = int(args.nx[0])
# Y-Lattice dimension
ny = int(args.ny[0])
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
# FREE or FULL SU(2) THEORY
if args.pure[0] == "y":
    pure_theory = True
    theory_label = "Pure"
else:
    pure_theory = False
    theory_label = "Full"
# ===================================================================================
if pure_theory:
    phrase = "          PURE THEORY: NO MATTER FIELDS"
    # Local dimension of the theory
    local_dimension = 9
    # IMPORTING SINGLE SITE OPERATORS
    from old_operators.SU2_Free_Operators import identity
    from old_operators.SU2_Free_Operators import gamma_operator
    from old_operators.SU2_Free_Operators import plaquette
    from old_operators.SU2_Free_Operators import W_operators
    from old_operators.SU2_Free_Operators import penalties
else:
    phrase = "          FULL THEORY: MATTER + GAUGE FIELDS"
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
# ===================================================================================
print("####################################################")
print("")
print(phrase)
print("")
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
# COMPUTE THE PLAQUETTE HAMILTONIAN: NOTE: WE NEED TO SUM THE COMPLEX CONJUGATE OF THE PLAQUETTES
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

# ----------------------------------------------------------------------------------
if not pure_theory:
    option = args.op[0]
    if option == "m":
        mass_fixed = False
        # Mass Coupling set
        mass_coupling_set = np.logspace(start=-1, stop=0, num=30)
        set_parameters_list = mass_coupling_set.tolist()
    elif option == "g":
        mass_fixed = True
        # SU(2) Gauge Coupling set
        gauge_coupling_set = np.logspace(start=-1, stop=1, num=30)
        set_parameters_list = gauge_coupling_set.tolist()
else:
    # SU(2) Gauge Coupling set
    gauge_coupling_set = np.logspace(start=-1, stop=1, num=30)
    set_parameters_list = gauge_coupling_set.tolist()

# DEFINE 3 LISTS FOR THE EXPECTATION VALUES WE WANT TO COMPUTE
energy_set = list()  # GS ENERGY
avg_gamma = list()  # AVG GAMMA OPERATOR
avg_plaquette = list()  # AVG PLAQUETTE OPERATOR
avg_N_SINGLE_ODD_site = list()  # AVG Single particle in ODD SITES
avg_N_SINGLE_EVEN_site = list()  # AVG Single particle in EVEN SITES
avg_N_PAIR_ODD_site = list()  # AVG PAIR particle in ODD SITES
avg_N_PAIR_EVEN_site = list()  # AVG PAIR particle in EVEN SITES
avg_N_TOT_ODD_site = list()  # AVG TOT particle in ODD SITES
avg_N_TOT_EVEN_site = list()  # AVG TOT particle in EVEN SITE

# ----------------------------------------------------------------------------------
# Run over different values of the PARAMETER
for ii in set_parameters_list:
    # FIXING THE PARAMETERS FOR A SINGLE SIMULATION
    if not pure_theory:
        if mass_fixed:
            # ACQUIRE FROM INPUT THE VALUE OF MASS
            mass = float(args.m[0])
            # g_SU2 IS THE VARYING PARAMETER
            g = ii
        else:
            # ACQUIRE FROM INPUT THE VALUE OF g_SU2
            g = float(args.g[0])
            # MASS IS THE VARYING PARAMETER
            mass = ii
        # CHOICE OF THE PENALTY
        eta = 10 * max((3 / 16) * (g**2), 4 / (g**2), mass)
    else:
        # g_SU2 IS THE VARYING PARAMETER
        g = ii
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
    H = csr_matrix(H)
    # ----------------------------------------------------------------------------------
    # DIAGONALIZING THE HAMILTONIAN
    GS = Pure_State()
    GS.ground_state(H, debug)
    # RESHIFT THE LOWEST EIGENVALUE OF THE HAMILTONIAN BY THE ENERGY PENALTIES
    # AND GET THE GROUND STATE ENERGY DENSITY (OF SINGLE SITE)
    GS.energy = (GS.energy + n_penalties * eta) / n
    # SAVE THE ENERGY VALUE IN A LIST
    energy_set.append(GS.energy)
    # TRUNCATE THE ENTRIES OF THE GROUND STATE
    psi = truncation(GS.psi, 10 ** (-10))
    # ----------------------------------------------------------------------------------
    if not pure_theory:
        print("----------------------------------------------------")
        print("          EFFECTIVE MASS m ", format(mass, ".3f"))
    print("----------------------------------------------------")
    print("    SU(2) GAUGE COUPLING g ", format(g, ".3f"))
    print("----------------------------------------------------")
    print("    GROUND STATE ENERGY  E ", format(GS.energy, ".5f"))
    # COMPUTE GAMMA EXPECTATION VALUE
    avg_gamma.append(Print_LOCAL_Observable(GS.psi, nx, ny, Gamma))
    # CHECK PLAQUETTE VALUES
    print("----------------------------------------------------")
    print("    PLAQUETTE VALUES")
    print("----------------------------------------------------")
    print("")
    avg_plaquette.append(
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
        avg_N_SINGLE_ODD_site.append(avg_single_odd)
        avg_N_SINGLE_EVEN_site.append(avg_single_even)
        # N_PAIR
        avg_pair_odd, avg_pair_even = Print_LOCAL_Observable(
            GS.psi, nx, ny, N_PAIR, staggered=True
        )
        avg_N_PAIR_ODD_site.append(avg_pair_odd)
        avg_N_PAIR_EVEN_site.append(avg_pair_even)
        # N_TOT
        avg_tot_odd, avg_tot_even = Print_LOCAL_Observable(
            GS.psi, nx, ny, N_TOTAL, staggered=True
        )
        avg_N_TOT_ODD_site.append(avg_tot_odd)
        avg_N_TOT_EVEN_site.append(avg_tot_even)
    # CHECK LINK SYMMETRY PENALTIES
    # DEFINE A VOCABULARY TO STORE THE EXPECTATION VALUES of 2BODY CORRELATORS
    results = {}
    print("----------------------------------------------------")
    print("    CHECK LINK SYMMETRY")
    print("----------------------------------------------------")
    results = Print_TWO_BODY_Correlators(
        results, GS.psi, nx, ny, SU2_W_Link, periodicity=PBC
    )
    if PBC == False:
        # CHECK PENALTIES ON THE BORDERS
        print("----------------------------------------------------")
        print("    CHECK PENALTIES ON THE BORDERS")
        print("----------------------------------------------------")
        results = Print_BORDER_Penalties(results, GS.psi, nx, ny, Penalty)
    # ----------------------------------------------------------------------------------
    """
    # COMPUTE AND PRINT SINGLE SITE REDUCED DENSITY MATRICES
    single_site_rho={}
    for ii in range(n):
        single_site_rho[str(ii)]=get_reduced_density_matrix(GS.psi,local_dimension,nx,ny,ii)
        # COMPUTE THE PROJECTOR ALONG THE MOST RELEVANT STATES OF THE SINGLE SITE BASIS
        # DEFINE THE PATH WHERE THE PROJECTOR IS GOING TO BE SAVED
        if mass_fixed:
            param='g'
        else:
            param='m'
        projector_directory=f'Results/SU2/{theory_label}/{BC_string}/Projectors/{param}_{format(ii,'.2f')}/'
        if not os.path.exists(projector_directory):
            os.makedirs(projector_directory)
        projector_label=projector_directory+f'TLB_site_{str(ii)}'
        # FIX A THRESHOLD WITH WHICH YOU WANT TO TRASH OUT EIGENVALUES
        threshold=10**(-4)
        Proj=get_projector(single_site_rho[str(ii)],local_dimension,threshold,debug,projector_label)
    """
    # ----------------------------------------------------------------------------------
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    # ----------------------------------------------------------------------------------
"""    if not pure_theory:
        phrase='SYMMETRY SECTORS OF THE HAMILTONIAN'
        # GET THE SINGLE SITE SYMMETRY SECTORS WRT THE NUMBER OF FERMIONS
        single_site_syms=syms.single_site_symmetry_sectors(local_dimension,[0,1,2],[9,12,9])
        # GET THE NBODY SYMMETRY SECTOR STATE
        Nbody_syms=syms.many_body_symmetry_sectors(single_site_syms,n)-4
        print(Nbody_syms)
        
        H_subsector={}
        for ii in range(-4,5,2):
            print(ii)
            # GET THE INDICES ASSOCIATED TO EACH SYMMETRY SECTOR
            indices=syms.get_indices_from_array(Nbody_syms,ii)
            indices=indices.tolist()
            print('Computing H subsector of ', ii)
            H_subsector[str(ii)]=syms.get_submatrix_from_sparse(H,indices,indices)
            sub_energy,sub_psi=get_ground_state_from_Hamiltonian(csr_matrix(H_subsector[str(ii)]),debug=False)
            sub_energy=(sub_energy+n_penalties*eta)/n
            print('SUBSECTOR', ii, 'ENERGY',sub_energy)"""
# ----------------------------------------------------------------------------------

# Truncate the entries of the coupling list
set_parameters_list = ["%.3f" % elem for elem in set_parameters_list]

if pure_theory:
    # Add a label to the simulation
    simulation_label = lattice_size_string
    # Add the label for the values of the Hamiltonian parameter
    set_parameters_list.insert(0, "g_{SU(2)}")
else:
    if mass_fixed:
        # Add the label for the values of the Hamiltonian parameter
        set_parameters_list.insert(0, "g_{SU(2)}")
        # Add a label to the simulation
        simulation_label = "m_" + format(mass, ".2f")
    else:
        # Add the label for the values of the Hamiltonian parameter
        set_parameters_list.insert(0, "m")
        # Add a label to the simulation
        simulation_label = "g_" + format(g, ".2f")


# Insert a label in each list of observables
energy_set.insert(0, "Energy")
avg_gamma.insert(0, Gamma.Op_name)
avg_plaquette.insert(0, "Plaquette")
if not pure_theory:
    # N_single
    avg_N_SINGLE_ODD_site.insert(0, f"{N_SINGLE.Op_name}_odd")
    avg_N_SINGLE_EVEN_site.insert(0, f"{N_SINGLE.Op_name}_even")
    # N_pair
    avg_N_PAIR_ODD_site.insert(0, f"{N_PAIR.Op_name}_odd")
    avg_N_PAIR_EVEN_site.insert(0, f"{N_PAIR.Op_name}_even")
    # N_tot
    avg_N_TOT_ODD_site.insert(0, f"{N_TOTAL.Op_name}_odd")
    avg_N_TOT_EVEN_site.insert(0, f"{N_TOTAL.Op_name}_even")

# SAVE RESULTS ON TEXT FILES
RESULTS_directory = f"Results/SU2/{theory_label}/{BC_string}/"
if not os.path.exists(RESULTS_directory):
    os.makedirs(RESULTS_directory)
if len(set_parameters_list) > 3:
    store_results(
        f"{RESULTS_directory}Simulation_{simulation_label}.txt",
        set_parameters_list,
        energy_set,
    )
    store_results(
        f"{RESULTS_directory}Simulation_{simulation_label}.txt",
        set_parameters_list,
        avg_gamma,
    )
    store_results(
        f"{RESULTS_directory}Simulation_{simulation_label}.txt",
        set_parameters_list,
        avg_plaquette,
    )
    if not pure_theory:
        store_results(
            f"{RESULTS_directory}Simulation_{simulation_label}.txt",
            set_parameters_list,
            avg_N_SINGLE_EVEN_site,
        )
        store_results(
            f"{RESULTS_directory}Simulation_{simulation_label}.txt",
            set_parameters_list,
            avg_N_SINGLE_ODD_site,
        )
        store_results(
            f"{RESULTS_directory}Simulation_{simulation_label}.txt",
            set_parameters_list,
            avg_N_PAIR_EVEN_site,
        )
        store_results(
            f"{RESULTS_directory}Simulation_{simulation_label}.txt",
            set_parameters_list,
            avg_N_PAIR_ODD_site,
        )
        store_results(
            f"{RESULTS_directory}Simulation_{simulation_label}.txt",
            set_parameters_list,
            avg_N_TOT_EVEN_site,
        )
        store_results(
            f"{RESULTS_directory}Simulation_{simulation_label}.txt",
            set_parameters_list,
            avg_N_TOT_ODD_site,
        )


"""
base_30=np.zeros(30,dtype=float)
state_1=base_30
state_1[0]=1.
state_22=base_30
state_22[21]=1.

state_7=base_30
state_7[6]=1.
state_25=base_30
state_25[24]=1.

state_28=base_30
state_28[27]=1.
state_2=base_30
state_2[1]=1.

ket_1=np.kron(np.kron(np.kron(state_1,state_22),state_22),state_1)
bra_1=np.kron(np.kron(np.kron(state_7,state_25),state_28),state_2)


state_12=base_30
state_12[11]=1.
state_10=base_30
state_10[9]=1.

state_13=base_30
state_13[12]=1.
state_11=base_30
state_11[10]=1.

ket_2=np.kron(np.kron(np.kron(state_12,state_10),state_12),state_10)
bra_2=np.kron(np.kron(np.kron(state_13,state_13),state_11),state_11)


print('entry_1')
print(np.real(np.dot(bra_1,H_Plaq.dot(ket_1))))
print('entry_2')
print(np.real(np.dot(bra_2,H_Plaq.dot(ket_2)))) 
print('entry_1')
print(np.real(np.dot(bra_1,H.dot(ket_1))))
print('entry_2')
print(np.real(np.dot(bra_2,H.dot(ket_2))))
"""


# ----------------------------------------------------------------------------------
"""
# COMPUTE AND PRINT SINGLE SITE REDUCED DENSITY MATRICES
single_site_rho={}
for ii in range(n):
    single_site_rho[str(ii)]=get_reduced_density_matrix(GS.psi,local_dimension,nx,ny,ii)
    # COMPUTE THE PROJECTOR ALONG THE MOST RELEVANT STATES OF THE SINGLE SITE BASIS
    # DEFINE THE PATH WHERE THE PROJECTOR IS GOING TO BE SAVED
    if mass_fixed:
        param='g'
    else:
        param='m'
    projector_directory=f'Results/SU2/{theory_label}/{BC_string}/Projectors/{param}_{format(ii,'.2f')}/'
    if not os.path.exists(projector_directory):
        os.makedirs(projector_directory)
    projector_label=projector_directory+f'TLB_site_{str(ii)}'
    # FIX A THRESHOLD WITH WHICH YOU WANT TO TRASH OUT EIGENVALUES
    threshold=10**(-4)
    Proj=get_projector(single_site_rho[str(ii)],local_dimension,threshold,debug,projector_label)
"""
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
"""
phrase='SYMMETRY SECTORS OF THE HAMILTONIAN'
# GET THE SINGLE SITE SYMMETRY SECTORS WRT THE NUMBER OF FERMIONS
single_site_syms=syms.single_site_symmetry_sectors(local_dimension,[0,1,2],[9,12,9])
# GET THE NBODY SYMMETRY SECTOR STATE
Nbody_syms=syms.many_body_symmetry_sectors(single_site_syms,n)-4
print(Nbody_syms)

H_subsector={}
for ii in range(-4,5,1):
    # GET THE INDICES ASSOCIATED TO EACH SYMMETRY SECTOR
    indices=syms.get_indices_from_array(Nbody_syms,ii)
    indices=indices.tolist()
    H_subsector[str(ii)]=syms.get_submatrix_from_sparse(H,indices,indices)
"""
# ----------------------------------------------------------------------------------
