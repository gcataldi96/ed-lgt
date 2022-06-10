from scipy.sparse.base import isspmatrix
import numpy as np
from scipy.sparse.linalg import norm
# =====================================================================================
def pause(phrase,debug):
    if debug == True:
        # IT PROVIDES A PAUSE (with a phrase) in a given point of the PYTHON CODE
        print('----------------------------------------------------')
        #Press the <ENTER> key to continue
        programPause = input(phrase)        
        print('----------------------------------------------------')
        print('')

# =====================================================================================
def alert(phrase,debug):
    if debug == True:
        # IT PRINTS A PHRASE IN A GIVEN POINT OF A PYTHON CODE
        print('')
        print(phrase)

# =====================================================================================
def check_commutator(A,B):
    # THIS FUNCTION CHECK THE COMMUTATION RELATIONS BETWEEN THE OPERATORS A AND B
    if not isspmatrix(A):
        raise TypeError(f'A should be a csr_matrix, not a {type(A)}')
    if not isspmatrix(B):
        raise TypeError(f'B should be a csr_matrix, not a {type(B)}')
    norma=norm(A*B-B*A)
    norma_max=max(norm(A*B+B*A),norm(A),norm(B))
    ratio=norma/norma_max
    #check=(AB!=BA).nnz
    if ratio>10**(-15):
        print('    ERROR: A and B do NOT COMMUTE')
        print('    NORM', norma)
        print('    RATIO', ratio)
    print('')
# =====================================================================================
def check_matrix(A,B):
    # THIS FUNCTION CHECK THE DIFFERENCE BETWEEN TWO SPARSE MATRICES
    # CHECK TYPE
    if not isspmatrix(A):
        raise TypeError(f'A should be a csr_matrix, not a {type(A)}')
    if not isspmatrix(B):
        raise TypeError(f'B should be a csr_matrix, not a {type(B)}')
    norma=norm(A-B)
    norma_max=max(norm(A+B),norm(A),norm(B))
    ratio=norma/norma_max
    if ratio>10**(-15):
        print('    ERROR: A and B are DIFFERENT MATRICES')
        print('    NORM', norma)
        print('    RATIO', ratio)
    print('')
# =====================================================================================