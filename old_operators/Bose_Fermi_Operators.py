import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import identity

def hubbard_operators():
    data_c_UP=np.array([1.,1.])
    x_c_UP=np.array([3,4])
    y_c_UP=np.array([1,2])
    c_UP=csr_matrix((data_c_UP,(x_c_UP-1,y_c_UP-1)),shape=(4,4))
    # ---------------------------------------------------------------
    data_c_DOWN=np.array([-1.,1.])
    x_c_DOWN=np.array([2,4])
    y_c_DOWN=np.array([1,3])
    c_DOWN=csr_matrix((data_c_DOWN,(x_c_DOWN-1,y_c_DOWN-1)),shape=(4,4))
    # ---------------------------------------------------------------
    data_n_UP=np.array([1.,1.])
    x_n_UP=np.array([1,2])
    y_n_UP=np.array([1,2])
    n_UP=csr_matrix((data_n_UP,(x_n_UP-1,y_n_UP-1)),shape=(4,4))
    # ---------------------------------------------------------------
    data_n_DOWN=np.array([1.,1.])
    x_n_DOWN=np.array([1,3])
    y_n_DOWN=np.array([1,3])
    n_DOWN=csr_matrix((data_n_DOWN,(x_n_DOWN-1,y_n_DOWN-1)),shape=(4,4))
    # ---------------------------------------------------------------
    data_n_PAIR=np.array([1.])
    x_n_PAIR=np.array([1])
    y_n_PAIR=np.array([1])
    n_PAIR=csr_matrix((data_n_PAIR,(x_n_PAIR-1,y_n_PAIR-1)),shape=(4,4))
    # ---------------------------------------------------------------
    data_JW=np.array([1.,-1.,-1.,1.])
    x_JW=np.array([1,2,3,4])
    y_JW=np.array([1,2,3,4])
    JW=csr_matrix((data_JW,(x_JW-1,y_JW-1)),shape=(4,4))
    # ---------------------------------------------------------------
    ID=identity(4)
    return c_UP, c_DOWN, n_UP, n_DOWN, n_PAIR, JW, ID


def bose_operators(n_max):
    entries=np.arange(1,n_max+1,1)
    entries=np.sqrt(entries)
    x_coords=np.arange(0,n_max,1)
    y_coords=np.arange(1,n_max+1,1)
    b_dagger_Op=csr_matrix((entries,(x_coords,y_coords)),shape=(n_max+1,n_max+1))
    b_Op=csr_matrix(b_dagger_Op.conj().transpose())
    num_Op=b_dagger_Op*b_Op
    ID=identity(n_max+1)
    return b_dagger_Op, b_Op, num_Op, ID
    