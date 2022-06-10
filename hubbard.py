import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse import kron
import os
import re
import argparse
import subprocess as sub
import itertools

# IMPORT FUNCTIONS FROM A FILE
from functions import pause
from functions import derivative
from functions import jordan_wigner_operators
from functions import local_operator
from functions import hopping_terms
from functions import two_body_operator
from functions import sparse_diagonalization


from functions import store_results
from functions import check_array
from functions import check_number
from functions import check_hermitianity
from functions import commutator
# ========================================================================================
# DEFINE THE ARGUMENTS OF THE PARSER FUNCTION
parser = argparse.ArgumentParser(description='PARAMETERS OF THE HUBBARD HAMILTONIAN.')
parser.add_argument('-x',  '--x',     nargs=1,  type=int,   help='x SIZE OF THE LATTICE')
parser.add_argument('-y',  '--y',     nargs=1,  type=int,   help='y SIZE OF THE LATTICE')
parser.add_argument('-t',  '--t',     nargs=3,  type=float, help='HOPPING TERM')
parser.add_argument('-U',  '--U',     nargs=3,  type=float, help='COULOMB INTERACTION')
parser.add_argument('-N',  '--N',     nargs=3,  type=int,   help='NUMBER OF PARTICLES')
parser.add_argument('-mu', '--mu',    nargs=3,  type=float, help='CHEMICAL POTENTIAL')
parser.add_argument('-op', '--option',nargs=1,  type=int,   help='1 N FIXED; 2 IF FREE')
# ========================================================================================
# ACQUIRING THE ARGUMENTS VIA PARSER
args = parser.parse_args()
nx=int(args.x[0])
ny=int(args.y[0])
# GET THE MOOD OF THE SIMLATION: BELOW, THE TWO MAIN SITUATIONS
op=int(args.option[0])
if op==1:
    # ====================================================================================
    # OPTION 1: WE CONSIDER THE HUBBARD MODEL WITH A FIXED NUMBER OF PARTICLES
    #           WE FIX THIS NUMBER BY MAKING USE OF THE 1st CHEMICAL POTENTIAL
    #           WHICH IS TYPICALLY MUCH LARGER THAN U,t
    NN=np.arange(int(args.N[0]), int(args.N[1]), int(args.N[2]))
    t=float(args.t[0])
    mu=float(args.mu[0])
    UU=np.arange(float(args.U[0]), float(args.U[1]), float(args.U[2]))
    dU=float(args.U[2])
elif op==2:
    # ====================================================================================
    # OPTION 2: WE DO NOT FIX THE NUMBER OF PARTICLES, BUT LET THE 2nd CHEMICAL
    #           POTENTIAL TO RULE THE ADDITION OF PARTICLES IN THE MODEL
    #           WE TYPICALLY KEEP U AND t FIXED AND LET mu2 VARY IN ORDER TO
    #           STUDY THE MOTT GAP IN THE DENSITY OF PARTICLES
    mu=np.arange(float(args.mu[0]), float(args.mu[1]), float(args.mu[2]))
    tt=np.arange(float(args.t[0]), float(args.t[1]), float(args.t[2]))
    U=float(args.U[0])
elif op==3:
    # ====================================================================================
    # OPTION 3: WE FOCUS ON THE HUBBARD MODEL WITH A SPECIFIC SET OF {U,N,t,mu}
    #           WE COMPUTE THE ENERGY AND COMPARE IT WITH SOME LITERATURE RESULTS.
    #           WE ALSO CHECK THE MAGNETIC CONFIGURATION OF THE SYSTEM BY PLOTTING
    #           THE MAGNETIZATION ALONG THE Z AXIS
    N=int(args.N[0])
    U=float(args.U[0])
    t=float(args.t[0])
    mu=float(args.mu[0])
# ========================================================================================


# 2x2 IDENTITY
ID_2=np.array([[1., 0.], [0., 1.]])
# COMPUTE THE JORDAN WIGNER OPERATORS:
# CREATION AND ANNIHILATION OPERATORS, NUMBER OPERATOR, JW TERM
creator,annihilator,n,JW = jordan_wigner_operators()
# TRANSFORM THESE 2x2 OPERATORS INTO COMPRESSED SPARSE ROW MATRICES
creator=sparse.csr_matrix(creator)
annihilator=sparse.csr_matrix(annihilator)
n=sparse.csr_matrix(n)
ID_2=sparse.csr_matrix(ID_2)
JW=sparse.csr_matrix(JW)


# CREATION AND ANNIHILATION OPERATORS IN THE SINGLE LATTICE SITE
up=kron(annihilator, ID_2)
up_dag=kron(creator, ID_2)
down=kron(JW, annihilator)
down_dag=kron(JW, creator)
# NUMBER OPERATORS IN THE LATTICE SITE
n_up=kron(n, ID_2)
n_down=kron(ID_2, n)
n_pair=n_up*n_down
# 4x4 IDENTITY
ID=kron(ID_2, ID_2)
# JW TERM FOR A LATTICE SITE
JW_4=kron(JW,JW)



# DEFINE A DICTIONARY dict WHERE ALL THE NEEDED OPERATORS ARE STORED
# TO CALL AN ELEMENT OF THE DICTIONARY WRITE: dict['name_element']
operators={}
# ========================================================================================
operators['ID_%s' % (str(nx*ny))] = local_operator(ID, ID, 1, nx*ny)
# INTIALIZE TO ZERO ALL THE MAIN CONTRIBUTION OF THE HUBBARD HAMILTONIAN
null=0*operators['ID_'+str(nx*ny)]
H_t_up=null.copy()                                 # HOPPING OF UP PARTICLES
H_t_down=null.copy()                               # HOPPING OF DOWN PARTICLES
H_U=null.copy()                         # ON SITE COULOMB POTENTIAL
H_mu_up=null.copy()                     # CHEM. POTENTIAL FOR UP PARTICLES
H_mu_down=null.copy()                   # CHEM. POTENTIAL FOR DOWN PARTICLES

for ii in range(nx*ny):
    operators['N_up_%s'   % (str(ii+1))] = local_operator(n_up,   ID, ii+1, nx*ny)
    operators['N_down_%s' % (str(ii+1))] = local_operator(n_down, ID, ii+1, nx*ny)
    operators['N_pair_%s' % (str(ii+1))] = local_operator(n_pair, ID, ii+1, nx*ny)
    # COMPUTE THE ON SITE COULOMB POTENTIAL AND A TEMPORARY CHEMICAL POTENTIAL
    H_U       += operators['N_pair_'+str(ii+1)]
    H_mu_up   += operators['N_up_'  +str(ii+1)]
    H_mu_down += operators['N_down_'+str(ii+1)]

H_mu_TMP=H_mu_up+H_mu_down
# DEFINE ALL THE HOPPING TERMS INVOLVED IN THE HOPPING HAMILTONIAN
hops=hopping_terms(nx,ny,4)


for ii in range(hops.shape[1]):
    H_t_up   +=two_body_operator(up_dag,  up,   ID, JW_4,hops[0][ii],hops[1][ii],nx*ny)
    H_t_down +=two_body_operator(down_dag,down, ID, JW_4,hops[0][ii],hops[1][ii],nx*ny)


if op==1:
    # DEFINING THE ARRAY CONTAINING ALL THE GROUND STATE
    # ENERGY VALUES OBTAINED FOR EACH VALUE OF N AND U
    energy=np.zeros((NN.shape[0], UU.shape[0]), dtype=float)
    for ii in range(NN.shape[0]):
        N=NN[ii]
        # GENERATE A LIST WHERE THE FIRST ENTRY IS THE LABEL
        # OF THE SIMULATION AND ALL THE OTHERS ARE ENERGY VALUES.
        # AN ANALOGOUS LIST IS MADE FOR U VALUES
        data_eng=list()
        data_eng.append(str(nx)+'x'+str(ny)+' HUBBARD')
        data_U=list()
        data_U.append('U/t')
        for jj in range(UU.shape[0]):
            U=UU[jj]
            data_U.append(U)
            print('    ----NUMBER OF PARTICLES  N  ', N)
            print('    ----COUPLING CONSTANT    U  ', U)
            # ------------------------------------------------------------
            # COMPLETE THE CREATION OF THE CHEMICAL POTENTIAL OPERATOR
            H_mu = H_mu_TMP - N*operators['ID_'+str(nx*ny)]
            H_mu = H_mu*H_mu
            # -------------------------------------------------------------
            # TOTAL HAMILTONIAN AT t=1
            H=null
            H= -t*H_t_up -t*H_t_down + U*H_U + mu*H_mu

            energy[ii][jj], rho = sparse_diagonalization(H,nx,ny,operators)
            # ALLOCATE THE ENERGY VALUE IN THE LIST data
            data_eng.append(energy[ii][jj])

        # SAVE ENERGY VALUES OF THE LIST IN AN ALREADY EXISTING FILE
        # CONTAINING THE RESULTS OF OTHER SIMULATIONS WITH THE SAME
        # VALUE OF THE TOTAL NUMBER BUT OTHER CHOICES OF THE REMAINING
        # PARAMETERS OR OTHER METHODS
        name_results_eng='energy_rho_'+str(format(rho,'.1f'))+'_Hubbard.txt'
        store_results(name_results_eng,data_U,data_eng)
        # ===============================================================
        # EMPTY THE EXISTING LIST IN ORDER FOR IT TO BE REFILLED WITH
        # ENERGY VALUES ASSOCIATED TO A DIFFERENT N
        del data_eng[:]
        del data_U[:]

elif op==2:
    # DEFINING THE ARRAY CONTAINING ALL THE GROUND STATE
    # ENERGY VALUES OBTAINED FOR EACH VALUE OF t AND mu2
    energy=np.zeros((tt.shape[0],mu.shape[0]), dtype=float)
    # DEFINE A SIMILAR ARRAY FOR DENSITY VALUES
    rho=np.zeros((tt.shape[0],mu.shape[0]), dtype=float)
    # GENERATE A LIST WHERE THE FIRST ENTRY IS THE LABEL
    # OF THE SIMULATION AND ALL THE OTHERS ARE rho VALUES.
    # THE LIST IS DONE FOR EACH VALUE OF THE TOTAL
    # NUMBER OF PARTICLES WE SIMULATE
    data_rho=list()
    # DATA FOR mu VALUES (x AXIS)
    data_mu=list()
    for ii in range(tt.shape[0]):
        t=tt[ii]
        # FIRST LINE
        data_rho.append('t='+str(format(t,'.2f')))
        # CREATE A FILE FOR STORING THE VALUES OF U
        data_mu.append('\mu')
        for jj in range(mu.shape[0]):
            data_mu.append(mu[jj])
            print('    ----HOPPING COUPLING   t  ',format(t,'.3f')+'      |')
            print('    ----COULOMB POTENTIAL  U  ',format(U,'.3f')+'      |')
            print('    ----CHEMICAL POTENTIAL mu ',format(mu[jj],'.3f')+'      |')
            # ------------------------------------------------------------------
            # TOTAL HAMILTONIAN
            H = null
            H = -t*H_t_up -t*H_t_down + U*H_U -(U/2)*H_mu_TMP\
                        +(U/4)*operators['ID_'+str(nx*ny)] - mu[jj]*H_mu_TMP
            # ------------------------------------------------------------------
            energy[ii][jj], rho[ii][jj] = sparse_diagonalization(H,nx,ny,operators)
            data_rho.append(rho[ii][jj])
        # STORE RESULTS
        name_results='density_U_'+str(U)+'.txt'
        store_results(name_results,data_mu,data_rho)
        del data_mu[:]
        del data_rho[:]

elif op==3:
    # WITH THIS OPTION WE FOCUS ON THE HUBBARD MODEL WITH A SPECIFIC
    # SET OF {U,N,t,mu}. WE COMPUTE THE ENERGY AND COMPARE IT WITH
    # SOME LITERATURE RESULTS. WE ALSO CHECK THE MAGNETIC CONFIGURATION
    # OF THE SYSTEM BY PLOTTING THE MAGNETIZATION ALONG THE Z AXIS
    print('-----------------------------------------------------')
    print('    ----NUMBER OF PARTICLES  N   '+str(N)+'                  |')
    print('    ----COULOMB COUPLING     U   '+str(U)+'                |')
    print('    ----HOPPING COUPLING     t   '+str(t)+'                |')
    print('    ----CHEMICAL POTENTIAL  mu   '+str(mu)+'               |')
    # ------------------------------------------------------------
    # COMPLETE THE CREATION OF THE CHEMICAL POTENTIAL OPERATOR
    H_mu = H_mu_TMP - N*operators['ID_'+str(nx*ny)]
    H_mu = H_mu*H_mu
    # -------------------------------------------------------------
    # TOTAL HAMILTONIAN AT t=1
    H=null
    H= -t*H_t_up -t*H_t_down + U*H_U + mu*H_mu
    # DIAGONALIZATION
    energy, rho = sparse_diagonalization(H,nx,ny,operators)


# export OMP_NUM_THREADS=1
