import numpy as np
from scipy.sparse import csr_matrix

__all__=["spin_12_operators"]

def spin_12_operators():
    data_sx=np.array([1,1])
    x_sx=np.array([1,2])
    y_sx=np.array([2,1])
    sigma_x=csr_matrix((data_sx,(x_sx-1,y_sx-1)),shape=(2,2))
    # -------------------------------------------------------
    data_sy=np.array([complex(0,-1),complex(0,1)])
    x_sy=np.array([1,2])
    y_sy=np.array([2,1])
    sigma_y=csr_matrix((data_sy,(x_sy-1,y_sy-1)),shape=(2,2))
    # -------------------------------------------------------
    data_sz=np.array([1,-1])
    x_sz=np.array([1,2])
    y_sz=np.array([1,2])
    sigma_z=csr_matrix((data_sz,(x_sz-1,y_sz-1)),shape=(2,2))
    # -------------------------------------------------------
    data_ID=np.array([1,1])
    x_ID=np.array([1,2])
    y_ID=np.array([1,2])
    ID=csr_matrix((data_ID,(x_ID-1,y_ID-1)),shape=(2,2))
    # -------------------------------------------------------
    return ID, sigma_x, sigma_y, sigma_z