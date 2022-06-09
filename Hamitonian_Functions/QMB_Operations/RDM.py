import numpy as np
# ===================================================================
def get_reduced_density_matrix(psi,loc_dim,n_sites,site,print_rho=False):
    """
    Parameters
    ----------
    psi : [type: ndarray]
        [state of the QMB system]
    loc_dim : [type: int]
        [local dimension of each single site of the QMB system]
    n_sites : [type: int]
        [total number of sites in the QMB system]
    site : [type: int]
        [position of the site wrt which we want to get the reduced density matrix]
    print_rho : bool, optional
        [If True, it prints the obtained reduced density matrix], by default False

    Returns
    -------
    [type: ndarray]
        [Reduced density matrix of the single site]

    """
    if not isinstance(psi, np.ndarray):
        raise TypeError(f'density_mat should be an ndarray, not a {type(psi)}')
    if not np.isscalar(loc_dim) and not isinstance(loc_dim, int):
        raise TypeError(f'loc_dim must be an SCALAR & INTEGER, not a {type(loc_dim)}')
    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f'n_sites must be an SCALAR & INTEGER, not a {type(n_sites)}')
    if not np.isscalar(site) and not isinstance(site, int):
        raise TypeError(f'site must be an SCALAR & INTEGER, not a {type(site)}')
    # RESHAPE psi
    psi_copy=psi.reshape(*[loc_dim for ii in range(n_sites)])
    # DEFINE A LIST OF SITES WE WANT TO TRACE OUT
    indices=np.arange(0,n_sites)
    indices=indices.tolist()
    # The index we remove is the one wrt which we get the reduced DM
    indices.remove(site)
    # COMPUTE THE REDUCED DENSITY MATRIX
    rho=np.tensordot(psi_copy,np.conjugate(psi_copy), axes=(indices,indices))
    # PRINT RHO
    if print_rho:
        print('----------------------------------------------------')
        print(f'DENSITY MATRIX OF SITE ({str(site)})')
        print('----------------------------------------------------')
        print(rho)
    return rho