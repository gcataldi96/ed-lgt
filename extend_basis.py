# %%
import numpy as np
from ed_lgt.operators.bose_fermi_operators import bose_operators
from numpy.linalg import norm, qr,matrix_power


def truncate(array, threshold=1e-15):
    return np.where(np.abs(array) > threshold, array, 0)

def gram_schmidt_columns(X):
    Q, R =qr(X)
    return Q

def extend_basis(basis, op,norm_err):

    col_norms=norm(basis, axis=0)
    basis = basis / col_norms

    #new states
    basis_new=np.matmul(op,basis)
    #normalize basis_new
    col_norms = norm(basis_new, axis=0)
    basis_new = basis_new / col_norms

    new_vec=[]
    new_vec_norms=[]
    for v_p in basis_new.T:
        r=v_p.copy() 
        for v in basis.T:
            r-=np.dot(v,v_p)*v

        norm_r=norm(r)
        if norm_r>norm_err:
            new_vec_norms.append(norm_r)
            r /= norm_r 
            new_vec.append(r)    
            basis = np.column_stack((basis, r))
    
    return np.array(new_vec).T,np.array(new_vec_norms)


# %%
# load basis
rho_vecs = np.loadtxt("d_loc44mu2-2lambda_0.6rho0_eigVec.txt", delimiter=" ")
rho_vals = np.loadtxt("d_loc44mu2-2lambda_0.6rho0_eigVal.txt", delimiter=" ")


A=np.matmul(rho_vecs,rho_vecs.T.conj())
assert np.allclose(A,np.identity(A.shape[0]),atol=1e-14)
assert norm(A-np.identity(A.shape[0]))<1e-14

#check if all vectors are normalized, done 
# acquire operators
ops = bose_operators(n_max=rho_vecs.shape[0] - 1)
ops_dens = {k: it.toarray() for k, it in ops.items()}
# field operator
phi = 1 / (np.sqrt(2)) * (ops_dens["b"] + ops_dens["b_dagger"])
pi = 1j / (np.sqrt(2)) * (-ops_dens["b"] + ops_dens["b_dagger"])

# %%
#lets use one operator after the other.
error=1e-9
norm_err=1e-2
eigvals=[vals for vals in rho_vals if vals>error]
eigVec=rho_vecs[:,:len(eigvals)]

B=np.matmul(eigVec.T.conj(),eigVec)
assert np.allclose(B,np.identity(B.shape[0]),atol=1e-14)
assert norm(B-np.identity(B.shape[0]))<1e-14, norm

basis=eigVec.copy()
print("start shape",basis.shape)
for op in [phi,matrix_power(pi,2),matrix_power(phi,2),matrix_power(phi,4)]:

    expand_basis,norms_r=extend_basis(basis,op,norm_err)
    
    #order according to norms
    idx = norms_r.argsort()[::-1]

    basis=np.column_stack((basis,expand_basis[:,idx]))
    basis=gram_schmidt_columns(basis)


print("start shape",basis.shape)
C=np.matmul(basis.T.conj(),basis)
assert np.allclose(C,np.identity(C.shape[0]),atol=1e-14)
assert norm(C-np.identity(C.shape[0]))<1e-14, norm


with open("output.txt", "w") as f:
    for row in basis:
        f.write(" ".join(str(val) for val in row) + "\n")

