# %%
import numpy as np
from ed_lgt.operators.bose_fermi_operators import bose_operators


def truncate(array, threshold=1e-15):
    return np.where(np.abs(array) > threshold, array, 0)


def extend_basis(basis: np.ndarray, eigvals: np.ndarray, op: np.ndarray, tol: float):
    # select the meaningful eigenvalues associated to the states of the basis
    good_eigvals = [val for val in eigvals if val >= tol]
    # normalize the associated meaningful vectors
    good_basis = basis[:,:len(good_eigvals)]
    print(np.matmul(good_basis.T.conj(), good_basis))
    new_states = np.matmul(op, good_basis)
    print("AAAA",norm(np.matmul(good_basis.T.conj(),good_basis)-np.identity(good_basis.shape[1])))


    # Extract an augmented basis
    U = gram_schmidt_augment(good_basis, new_states, tol)#
    #print(U.shape)
    A = np.matmul(U.T.conj(), U)

    print("test, after extend",norm(A-np.identity(A.shape[0])))

    return U


def gram_schmidt_augment(
    old_basis: np.ndarray, new_vectors: np.ndarray, tol=1e-16#set overall basis
) -> np.ndarray:
    """
    old_basis: ndarray of shape (n, k)
               Columns are already orthonormal.
    new_vectors: ndarray of shape (n, m)
    tol: threshold below which a vector is considered "zero" and discarded.

    Returns: augmented_basis of shape (n, k + r)
             where r <= m is the number of new vectors that survive.
    """
    # Copy over the original orthonormal basis
    augmented = [old_basis[:, ii] for ii in range(old_basis.shape[1])]
    # Process each new vector
    for ii in range(new_vectors.shape[1]):
        new_vec = new_vectors[:, ii].copy()
        # Subtract projections onto all existing basis vectors in 'augmented'
        for jj in range(old_basis.shape[1]):
            old_vec=old_basis[:,jj]
            #wrong, phases! 
            new_vec -= np.dot(old_vec.conj(), new_vectors[:, ii]) * old_vec
        # Check if v is effectively zero
        norm_v = np.linalg.norm(new_vec)
        print("norm, gram",norm_v)

        if norm_v > tol: #should same threshold as before, when selecting eigenvectors!
            # Normalize and add to augmented set
            new_vec = new_vec / norm_v
            augmented.append(new_vec)
    # Convert list of vectors back to a 2D array
    return np.column_stack(augmented)


# %%
# load basis
rho_vecs = np.loadtxt("rho1_eigVec.txt", delimiter=" ")
rho_vals = np.loadtxt("rho1_eigVal.txt", delimiter=" ")

for ii in range(rho_vecs.shape[0]):
    print(norm(rho_vecs[:,ii]))

A=np.matmul(rho_vecs,rho_vecs.T.conj())
from numpy.linalg import norm
print("norm",norm(A-np.identity(A.shape[0])))
assert np.allclose(A,np.identity(A.shape[0]),atol=1e-15)

#check if all vectors are normalized, done 
# acquire operators
ops = bose_operators(n_max=rho_vecs.shape[0] - 1)
ops_dens = {k: it.toarray() for k, it in ops.items()}
# field operator
phi = 1 / (np.sqrt(2)) * (ops_dens["b"] + ops_dens["b_dagger"])
pi = 1j / (np.sqrt(2)) * (-ops_dens["b"] + ops_dens["b_dagger"])

# %%
#lets use one operator after the other.
ext_basis = extend_basis(rho_vecs, rho_vals, phi, 1e-16)



#np.savetxt("output.txt", ext_basis, fmt="%d", delimiter=",")

with open("output.txt", "w") as f:
    for row in ext_basis:
        f.write(" ".join(str(val) for val in row) + "\n")

load_matrix=[]
with open("output.txt", "r") as f:
    for line in f:
        # Split the line into complex number strings and convert them to complex numbers
        row = [complex(num) for num in line.strip().split()]
        load_matrix.append(row)

# Convert the list of lists to a numpy array
complex_matrix = np.array(load_matrix)


assert np.array_equal(ext_basis, complex_matrix)
# %%
