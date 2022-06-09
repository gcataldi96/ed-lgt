import numpy as np
from scipy.sparse import isspmatrix
from scipy.sparse import lil_matrix


#===========================================================================
def single_site_symmetry_sectors(loc_dim,list_of_sectors,list_of_dim_sectors):
    # CHECK ON TYPES
    if not np.isscalar(loc_dim) and not isinstance(loc_dim, int):
        raise TypeError(f'loc_dim must be a SCALAR & INTEGER, not a {type(loc_dim)}')
    if not isinstance(list_of_sectors,list):
        raise TypeError(f'list_of_sectors must be a LIST, not a {type(list_of_sectors)}')
    if not isinstance(list_of_dim_sectors,list):
        raise TypeError(f'list_of_dim_sectors must be a LIST, not a {type(list_of_dim_sectors)}')
    # Generate a single-site vector 
    single_site_vec=np.zeros(loc_dim,dtype=float)
    # TMP variable
    tmp=0
    # fill the entries of single_site_vec with the values of the charge for each state of the local_basis
    for jj, dim in enumerate(list_of_dim_sectors):
        for ii in range(dim):
            single_site_vec[tmp]=int(list_of_sectors[jj])
            tmp+=1
    return single_site_vec


#===========================================================================
def QMB_local_state(single_site_sym_sector_state,state_site,n_sites):
    # CHECK ON TYPES
    if not np.isscalar(state_site) and not isinstance(state_site, int):
        raise TypeError(f'state_site must be a SCALAR & INTEGER, not a {type(state_site)}')
    if not isinstance(single_site_sym_sector_state,np.ndarray):
        raise TypeError(f'single_site_sym_sector_state must be a ndarray, not a {type(single_site_sym_sector_state)}')
    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f'n_sites must be a SCALAR & INTEGER, not a {type(n_sites)}')
    # Make a copy of the single site symm sector state
    tmp=single_site_sym_sector_state
    # Make the tensor products with Identities
    ID=np.ones(single_site_sym_sector_state.shape[0])
    for ii in range(state_site):
        tmp=np.kron(ID,tmp)
    for ii in range(n_sites-state_site-1):
        tmp=np.kron(tmp,ID)
    #print(tmp)
    return tmp


#===========================================================================
def many_body_symmetry_sectors(single_site_state,n_sites):
    # CHECK ON TYPES
    if not isinstance(single_site_state,np.ndarray):
        raise TypeError(f'single_site_state must be a ndarray, not a {type(single_site_state)}')
    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f'n_sites must be a SCALAR & INTEGER, not a {type(n_sites)}')
    QMB_sym_sec_state=np.zeros(single_site_state.shape[0]**n_sites)
    for ii in range(n_sites):
        QMB_sym_sec_state+=QMB_local_state(single_site_state,ii,n_sites)

    return QMB_sym_sec_state


#===========================================================================
def get_submatrix_from_sparse(matrix,rows_list,cols_list):
    # CHECK ON TYPES
    if not isspmatrix(matrix):
        raise TypeError(f'matrix must be a SPARSE MATRIX, not a {type(matrix)}')
    if not isinstance(rows_list,list):
        raise TypeError(f'rows_list must be a LIST, not a {type(rows_list)}')
    if not isinstance(cols_list,list):
        raise TypeError(f'cols_list must be a LIST, not a {type(cols_list)}')
    matrix=lil_matrix(matrix)
    # Get the Submatrix out of the list of rows and columns
    sub_matrix=matrix[rows_list, :][:, cols_list]
    return sub_matrix


#===========================================================================
def get_indices_from_array(array,value):
    if not isinstance(array,np.ndarray):
        raise TypeError(f'array must be a ndarray, not a {type(array)}')
    index_list=np.nonzero((value-10**(-10)<array) & (array<value+10**(-10)))[0]
    return index_list




"""
prova=single_site_symmetry_sectors(3,[0,1,2],[1,1,1])
print(prova)

N_sym_sectors=many_body_symmetry_sectors(prova,3)
print(N_sym_sectors)

indices=get_indices_from_array(N_sym_sectors,3)
print(indices)

A=random(27,27,density=0.03)
print(A)

print('_________')

B=get_submatrix_from_sparse(A,indices.tolist(),indices.tolist())
print(B)
"""