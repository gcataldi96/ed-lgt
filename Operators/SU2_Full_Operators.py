import numpy as np
from numpy import sqrt
from scipy.sparse import csr_matrix
from .Manage_Data import acquire_data

# ===========================================================================================
# NOTE: OPERATORS FOR THE FULL THEORY: 
#       * 30_DIM SINGLE SITE BASIS
#       * BOTH FERMIONIC MATTER AND GAUGE FIELDS        
# ===========================================================================================
def identity():
    data=np.ones(30)
    x=np.arange(1,31,1)
    ID=csr_matrix((data, (x-1, x-1)), shape=(30, 30))
    return ID


def matter_operator():
    # * DIAGONAL OPERATOR
    # * The coefficients 2 are referred to DOUBLY OCCUPIED fermionic states
    data_D=np.array([1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2], dtype=float)
    x_D=np.arange(10,31,1)
    D_operator=csr_matrix((data_D, (x_D-1, x_D-1)), shape=(30, 30))
    return D_operator


def gamma_operator():
    # * DIAGONAL OPERATOR
    # The first 8 coefficients are for the EMPTY MATTER STATES: 
    #   * state 0 is not present as it represent the VACUUM
    #   * state 8 and 9 are TRICKY and this explains the factor 4
    # The second 12 coefficients are for the SINGLY OCCUPIED STATES:
    #   * from state 14 to state 22 we have TRICKY definition and a factor 3
    # The last 9 coefficients are for the DOUBLY OCCUPIED STATES
    #   * state 29 and 30 have TRICKY DEFINITIONS and the factor 4
    data_G=0.5*np.array([2,2,2,2,2,2,4,4,
                    1,1,1,1,3,3,3,3,3,3,3,3,\
                        2,2,2,2,2,2,4,4], dtype=float)
    x_G=np.array([2,3,4,5,6,7,8,9,
                    10,11,12,13,14,15,16,17,18,19,20,21,\
                        23,24,25,26,27,28,29,30])
    Gamma=csr_matrix((data_G, (x_G-1, x_G-1)), shape=(30, 30))
    return Gamma



def W_operators():
    # * DIAGONAL OPERATORS
    x_coords=np.arange(1,31,1)
    # ======================================================================================
    data_Left=np.array([1, 1,-1, 1,-1, 1,-1,-1,-1,\
                     1, 1,-1, 1,-1,-1, 1,\
                     1,-1,-1,-1,-1, 1, 1,\
                    -1, 1,-1, 1,-1,-1,-1], dtype=float)
    W_Left=csr_matrix((data_Left, (x_coords-1, x_coords-1)), shape=(30, 30))
    # ======================================================================================    
    data_Right=np.array([1,-1,-1,-1, 1, 1, 1,-1,-1,\
                    -1, 1, 1, 1,-1,-1,-1,\
                    -1,-1,-1, 1, 1, 1,-1,\
                    -1,-1, 1, 1, 1,-1,-1], dtype=float)
    W_Right=csr_matrix((data_Right, (x_coords-1, x_coords-1)),shape=(30, 30))
    # ======================================================================================    
    data_Bottom=np.array([1, 1, 1,-1, 1,-1,-1,-1,-1,\
                     1, 1, 1,-1, 1, 1,-1,\
                    -1,-1,-1,-1,-1, 1, 1,\
                     1,-1, 1,-1,-1,-1,-1], dtype=float)
    W_Bottom=csr_matrix((data_Bottom, (x_coords-1, x_coords-1)), shape=(30, 30))
    # ======================================================================================
    data_Top=np.array([1,-1, 1, 1,-1,-1, 1,-1,-1,\
                     1,-1, 1, 1,-1,-1,-1,\
                    -1, 1, 1,-1,-1, 1,-1,\
                     1, 1,-1,-1, 1,-1,-1], dtype=float)
    W_Top=csr_matrix((data_Top, (x_coords-1, x_coords-1)), shape=(30, 30))
    # ======================================================================================
    return W_Left, W_Right, W_Bottom, W_Top 



def plaquette():
    path='/Users/giovannicataldi/Dropbox/PhD/Models/Operators/Op_SU2_Full/'
    # ======================================================================================
    Bottom_Left=acquire_data(path+'Corner_Bottom_Left.txt')
    # COORDINATES
    x_Bottom_Left=Bottom_Left['0']
    y_Bottom_Left=Bottom_Left['1']
    # DATA
    data_Bottom_Left=Bottom_Left['2']
    # GENERATE MATRIX
    C_Bottom_Left=csr_matrix((data_Bottom_Left, (x_Bottom_Left-1, y_Bottom_Left-1)), shape=(30,30))
    # ======================================================================================
    Bottom_Right=acquire_data(path+'Corner_Bottom_Right.txt')
    # COORDINATES
    x_Bottom_Right=Bottom_Right['0']
    y_Bottom_Right=Bottom_Right['1']
    # DATA
    data_Bottom_Right=Bottom_Right['2']
    # GENERATE MATRIX
    C_Bottom_Right=csr_matrix((data_Bottom_Right, (x_Bottom_Right-1, y_Bottom_Right-1)), shape=(30,30))
    # ======================================================================================
    Top_Left=acquire_data(path+'Corner_Top_Left.txt')
    # COORDINATES
    x_Top_Left=Top_Left['0']
    y_Top_Left=Top_Left['1']
    # DATA
    data_Top_Left=Top_Left['2']
    # GENERATE MATRIX
    C_Top_Left=csr_matrix((data_Top_Left, (x_Top_Left-1, y_Top_Left-1)), shape=(30, 30))    
    # ======================================================================================
    Top_Right=acquire_data(path+'Corner_Top_Right.txt')
    # COORDINATES
    x_Top_Right=Top_Right['0']
    y_Top_Right=Top_Right['1']
    # DATA
    data_Top_Right=Top_Right['2']
    # GENERATE MATRIX
    C_Top_Right=csr_matrix((data_Top_Right, (x_Top_Right-1, y_Top_Right-1)), shape=(30, 30))
    # ======================================================================================
    return C_Bottom_Left, C_Bottom_Right, C_Top_Left, C_Top_Right



def hopping():
    path='/Users/giovannicataldi/Dropbox/PhD/Models/Operators/Op_SU2_Full/'
    # ======================================================================================
    Left=acquire_data(path+'Q_Left_dagger.txt')
    # COORDINATES
    x_Left=Left['0']
    y_Left=Left['1']
    # DATA
    data_Left=Left['2']
    # GENERATE MATRIX
    Q_Left_dagger=csr_matrix((data_Left, (x_Left-1, y_Left-1)), shape=(30, 30))
    #Q_Left       =csr_matrix((data_Left, (y_Left-1, x_Left-1)), shape=(30, 30))
    # ======================================================================================
    Right=acquire_data(path+'Q_Right_dagger.txt')
    # COORDINATES
    x_Right=Right['0']
    y_Right=Right['1']
    # DATA
    data_Right=Right['2']
    # GENERATE MATRIX
    Q_Right_dagger=csr_matrix((data_Right, (x_Right-1, y_Right-1)), shape=(30, 30))
    #Q_Right       =csr_matrix((data_Right, (y_Right-1, x_Right-1)), shape=(30, 30))
    # ======================================================================================
    Bottom=acquire_data(path+'Q_Bottom_dagger.txt')
    # COORDINATES
    x_Bottom=Bottom['0']
    y_Bottom=Bottom['1']
    # DATA
    data_Bottom=Bottom['2']
    # GENERATE MATRIX
    Q_Bottom_dagger=csr_matrix((data_Bottom, (x_Bottom-1, y_Bottom-1)), shape=(30, 30))
    #Q_Bottom       =csr_matrix((data_Bottom, (y_Bottom-1, x_Bottom-1)), shape=(30, 30))
    # ======================================================================================
    Top=acquire_data(path+'Q_Top_dagger.txt')
    # COORDINATES
    x_Top=Top['0']
    y_Top=Top['1']
    # DATA
    data_Top=Top['2']
    # GENERATE MATRIX
    Q_Top_dagger=csr_matrix((data_Top, (x_Top-1, y_Top-1)), shape=(30, 30))
    #Q_Top       =csr_matrix((data_Top, (y_Top-1, x_Top-1)), shape=(30, 30))
    # ======================================================================================
    #return Q_Left, Q_Left_dagger, Q_Right, Q_Right_dagger, Q_Bottom, Q_Bottom_dagger, Q_Top, Q_Top_dagger
    return Q_Left_dagger, Q_Right_dagger, Q_Bottom_dagger, Q_Top_dagger


def number_operators():
    data_PAIR=np.ones(9, dtype=float)
    x_PAIR=np.arange(22,31,1)
    n_PAIR=csr_matrix((data_PAIR, (x_PAIR-1, x_PAIR-1)), shape=(30, 30))
    # ======================================================================================
    data_SINGLE=np.ones(12, dtype=float)
    x_SINGLE=np.arange(10,22,1)
    n_SINGLE=csr_matrix((data_SINGLE, (x_SINGLE-1, x_SINGLE-1)), shape=(30, 30))
    # ======================================================================================
    data_TOTAL=np.array([1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2], dtype=float)
    x_TOTAL=np.arange(10,31,1)
    n_TOTAL=csr_matrix((data_TOTAL, (x_TOTAL-1, x_TOTAL-1)), shape=(30, 30))
    # ======================================================================================
    return n_SINGLE, n_PAIR, n_TOTAL


def penalties():
    # Penalties we impose on the corner of the lattice to use
    # OBC and not to consider the action of external Rishon modes
    data_L=np.ones(13, dtype=float)
    x_L=np.array([1,5,6,7,11,12,13,20,21,22,26,27,28])
    P_left=csr_matrix((data_L, (x_L-1, x_L-1)), shape=(30, 30))
    # ======================================================================================
    data_R=np.ones(13, dtype=float)
    x_R=np.array([1,2,4,6,10,11,13,16,17,22,23,25,27])
    P_right=csr_matrix((data_R, (x_R-1, x_R-1)), shape=(30, 30))
    # ======================================================================================
    data_B=np.ones(13, dtype=float)
    x_B=np.array([1,3,4,7,10,12,13,18,19,22,24,25,28])
    P_bottom=csr_matrix((data_B, (x_B-1, x_B-1)), shape=(30, 30))
    # ======================================================================================
    data_T=np.ones(13, dtype=float)
    x_T=np.array([1,2,3,5,10,11,12,14,15,22,23,24,26])
    P_top=csr_matrix((data_T, (x_T-1, x_T-1)), shape=(30, 30))
    # ======================================================================================
    return P_left, P_right, P_bottom, P_top