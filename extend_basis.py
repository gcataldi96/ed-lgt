import numpy as np
from ed_lgt.operators.bose_fermi_operators import bose_operators


def extend_basis(basis,eigVal,op,pi):

    # assert that basis is a unitary
    tol = 1e-12
    assert np.allclose(
        np.matmul(basis, basis.conj().T),
        np.identity(len(basis[:, 0])),
        atol=tol,
    )
    assert np.allclose(
        np.matmul(basis.conj().T, basis),
        np.identity(len(basis[:, 0])),
        atol=tol,
    )
    
    # apply operator
    basis_p=np.matmul(op, basis)

    #we choose cutoff
    error=1e-16
    eigVal=[val for val in eigVal if val>= error]

    #stack matrices
    ext_basis=np.hstack((basis[:,:len(eigVal)],basis_p[:,:len(eigVal)]))

    U, S, Vh = np.linalg.svd(ext_basis, full_matrices=False)

    #"unitary" check
    assert np.allclose(
        np.matmul(U.conj().T,U),
        np.identity(2*len(eigVal)),
        atol=tol,
    )

    #normalize singular values and truncate, not needed
    S_norm=[val/(sum(S)) for val in S.tolist() if val/(sum(S))>=error]

    return U[:,:len(S_norm)]
   

#load basis
rho1_eigVec = np.loadtxt("rho1_eigVec.txt", delimiter=" ")
rho1_eigVal = np.loadtxt("rho1_eigVal.txt", delimiter=" ")

ops = bose_operators(n_max=rho1_eigVec.shape[0]-1)
ops_dens = {k: it.toarray() for k, it in ops.items()}

#field operator
phi=1/(np.sqrt(2))*(ops_dens["b"]+ops_dens["b_dagger"])
pi=1j/(np.sqrt(2))*(-ops_dens["b"]+ops_dens["b_dagger"])

ext_basis=extend_basis(rho1_eigVec,rho1_eigVal,phi,pi)
np.savetxt("rho1_eigVec_ext.txt", ext_basis, delimiter=" ")
