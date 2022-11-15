import numpy as np
from copy import deepcopy
from scipy import sparse
from scipy.sparse import kron
import argparse

# IMPORT FUNCTIONS FROM A FILE
from Hubbard_functions import jordan_wigner_operators
from Hubbard_functions import local_operator
from Hubbard_functions import hopping_terms
from Hubbard_functions import two_body_operator
from Hubbard_functions import sparse_diagonalization
from Hubbard_functions import store_results

# from Data_Analysis.Manage_Data import save_dictionary

# ========================================================================================
# DEFINE THE ARGUMENTS OF THE PARSER FUNCTION
parser = argparse.ArgumentParser(description="PARAMETERS OF THE HUBBARD HAMILTONIAN.")
parser.add_argument("-x", "--x", nargs=1, type=int, help="x SIZE OF THE LATTICE")
parser.add_argument("-y", "--y", nargs=1, type=int, help="y SIZE OF THE LATTICE")
parser.add_argument("-t", "--t", nargs=1, type=float, help="HOPPING TERM")
parser.add_argument("-U", "--U", nargs=3, type=float, help="COULOMB INTERACTION")
parser.add_argument("-N", "--N", nargs=1, type=int, help="NUMBER OF PARTICLES")
parser.add_argument("-mu", "--mu", nargs=1, type=float, help="CHEMICAL POTENTIAL")
# ========================================================================================
# ACQUIRING THE ARGUMENTS VIA PARSER
args = parser.parse_args()
nx = int(args.x[0])
ny = int(args.y[0])
# ====================================================================================
# WE CONSIDER THE HUBBARD MODEL WITH A FIXED NUMBER OF PARTICLES
# WE FIX THIS NUMBER BY MAKING USE OF a CHEMICAL POTENTIAL
# WHICH IS TYPICALLY MUCH LARGER THAN U,t
N = int(args.N[0])
t = float(args.t[0])
mu = float(args.mu[0])
# U_values = np.arange(float(args.U[0]), float(args.U[1]), float(args.U[2]))
# dU = float(args.U[2])


params = {
    "Lx": nx,
    "Ly": ny,
    "LxL": nx * ny,
    "t": t,
    "U": {"values": [0.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "label": r"$U/t$"},
    "N": N,
    "rho": N / (nx * ny),
}


results = {
    "params": {},
    "energy": {
        "label": r"$E_{GS}$",
        "values": [],
    },
    "n_up": {
        "label": r"$\avg{n_{\uparrow}}$",
        "values": [],
    },
    "n_down": {
        "label": r"$\avg{n_{\downarrow}}$",
        "values": [],
    },
    "n_pair": {
        "label": r"$\avg{n_{\uparrow}n_{\downarrow}}$",
        "values": [],
    },
    "n_tot": {
        "label": r"$\avg{n_{\uparrow}+n_{\downarrow}}$",
        "values": [],
    },
}

results["params"] = deepcopy(params)

# 2x2 IDENTITY
ID_2 = np.array([[1.0, 0.0], [0.0, 1.0]])
# COMPUTE THE JORDAN WIGNER OPERATORS:
# CREATION AND ANNIHILATION OPERATORS, NUMBER OPERATOR, JW TERM
creator, annihilator, n, JW = jordan_wigner_operators()
# TRANSFORM THESE 2x2 OPERATORS INTO COMPRESSED SPARSE ROW MATRICES
creator = sparse.csr_matrix(creator)
annihilator = sparse.csr_matrix(annihilator)
n = sparse.csr_matrix(n)
ID_2 = sparse.csr_matrix(ID_2)
JW = sparse.csr_matrix(JW)


# CREATION AND ANNIHILATION OPERATORS IN THE SINGLE LATTICE SITE
up = kron(annihilator, ID_2)
up_dag = kron(creator, ID_2)
down = kron(JW, annihilator)
down_dag = kron(JW, creator)
# NUMBER OPERATORS IN THE LATTICE SITE
n_up = kron(n, ID_2)
n_down = kron(ID_2, n)
n_pair = n_up * n_down
# 4x4 IDENTITY
ID = kron(ID_2, ID_2)
# JW TERM FOR A LATTICE SITE
JW_4 = kron(JW, JW)


# DEFINE A DICTIONARY dict WHERE ALL THE NEEDED OPERATORS ARE STORED
# TO CALL AN ELEMENT OF THE DICTIONARY WRITE: dict['name_element']
operators = {}
# ========================================================================================
operators["ID_%s" % (str(nx * ny))] = local_operator(ID, ID, 1, nx * ny)
# INTIALIZE TO ZERO ALL THE MAIN CONTRIBUTION OF THE HUBBARD HAMILTONIAN
null = 0 * operators["ID_" + str(nx * ny)]
H_t_up = null.copy()  # HOPPING OF UP PARTICLES
H_t_down = null.copy()  # HOPPING OF DOWN PARTICLES
H_U = null.copy()  # ON SITE COULOMB POTENTIAL
H_mu_up = null.copy()  # CHEM. POTENTIAL FOR UP PARTICLES
H_mu_down = null.copy()  # CHEM. POTENTIAL FOR DOWN PARTICLES

for ii in range(nx * ny):
    operators["N_up_%s" % (str(ii + 1))] = local_operator(n_up, ID, ii + 1, nx * ny)
    operators["N_down_%s" % (str(ii + 1))] = local_operator(n_down, ID, ii + 1, nx * ny)
    operators["N_pair_%s" % (str(ii + 1))] = local_operator(n_pair, ID, ii + 1, nx * ny)
    # COMPUTE THE ON SITE COULOMB POTENTIAL AND A TEMPORARY CHEMICAL POTENTIAL
    H_U += operators["N_pair_" + str(ii + 1)]
    H_mu_up += operators["N_up_" + str(ii + 1)]
    H_mu_down += operators["N_down_" + str(ii + 1)]

H_mu_TMP = H_mu_up + H_mu_down
# DEFINE ALL THE HOPPING TERMS INVOLVED IN THE HOPPING HAMILTONIAN
hops = hopping_terms(nx, ny, 4)


for ii in range(hops.shape[1]):
    H_t_up += two_body_operator(up_dag, up, ID, JW_4, hops[0][ii], hops[1][ii], nx * ny)
    H_t_down += two_body_operator(
        down_dag, down, ID, JW_4, hops[0][ii], hops[1][ii], nx * ny
    )

# CHEMICAL POTENTIAL OPERATOR FIXING THE NUMBER OF PARTICLES
H_mu = H_mu_TMP - N * operators["ID_" + str(nx * ny)]
H_mu = H_mu * H_mu

simulation_label = f'Hubbard_{nx}x{ny}_rho_{format(params["rho"],".1f")}'

for ii, U in enumerate(params["U"]["values"]):
    print("    ----NUMBER OF PARTICLES  N  ", N)
    print("    ----COUPLING CONSTANT    U  ", U)
    # ------------------------------------------------------------
    # TOTAL HAMILTONIAN AT t=1
    H = -t * H_t_up - t * H_t_down + U * H_U + mu * H_mu
    # DIAGONALIZATION
    energy, avg_pair = sparse_diagonalization(H, nx, ny, operators)
    results["energy"]["values"].append(energy)
    results["n_pair"]["values"].append(avg_pair)

# SAVE ENERGY VALUES OF THE LIST IN AN ALREADY EXISTING FILE
# CONTAINING THE RESULTS OF OTHER SIMULATIONS WITH THE SAME
# VALUE OF THE TOTAL NUMBER BUT OTHER CHOICES OF THE REMAINING
# PARAMETERS OR OTHER METHODS

params["U"]["values"].insert(0, "U/t")
results["energy"]["values"].insert(0, "energy")
results["n_pair"]["values"].insert(0, "n_pair")

store_results(
    f"Results/{simulation_label}.txt",
    params["U"]["values"],
    results["energy"]["values"],
)

store_results(
    f"Results/{simulation_label}.txt",
    params["U"]["values"],
    results["n_pair"]["values"],
)

# save_dictionary(results, f"Results/{simulation_label}")
