import numpy as np
from ed_lgt.operators.bose_fermi_operators import bose_operators


def extend_basis(basis, ps, loc_ops: dict):
    """
    basis: complete basis
    ps: probabilities
    loc: local operators

    I have to pass
    """
    # assert that basis is a unitary
    tol = 1e-10
    assert np.allclose(
        np.matmul(basis, basis.conj().T),
        np.identity(len(eigenVectors[:, 0])),
        atol=tol,
    )
    assert np.allclose(
        np.matmul(basis.conj().T, basis),
        np.identity(len(eigenVectors[:, 0])),
        atol=tol,
    )

    # extend vectors to basis with d_loc -> d_loc+1
    zero_row = np.zeros((1, basis.shape[1]))
    ext_basis = np.vstack([basis, zero_row])

    e_n = np.zeros(basis.shape[1] + 1)
    e_n[-1] = 1.0
    ext_basis = np.hstack([ext_basis, e_n.reshape(-1, 1)])

    # transorm operators
    ops = {
        k: np.matmul(ext_basis, np.matmul(it, ext_basis.conj().T))
        for k, it in loc_ops.items()
    }

    # apply operators, like v'=b v, or v'=b_dagger n
    ext_basis_p = np.matmul(ops["b"], ext_basis)

    # 
    pass


rho1 = np.loadtxt("rho1.txt", delimiter=" ")
eigval, eigvec = np.linalg.eigh(rho1)

# Sort decending
idx = eigval.argsort()[::-1]
eigenValues = eigval[idx]
eigenVectors = eigvec[:, idx]

# Operators
d_loc = len(eigenVectors[:, 0])
ops = bose_operators(n_max=d_loc)
ops_dens = {k: it.toarray() for k, it in ops.items()}

for k, it in ops_dens.items():
    print(k, it.shape)


extend_basis(eigenVectors, eigenValues, ops_dens)
