import numpy as np
from scipy.linalg import eigh as array_eigh
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from scipy.sparse.linalg import eigsh as sparse_eigh
from scipy.sparse.base import isspmatrix
from scipy.sparse.csr import isspmatrix_csr

# ===================================================================
from .Mappings_1D_2D import zig_zag
from .Simple_Checks import pause

# ===================================================================
class Pure_State:
    def ground_state(self, Hamiltonian, debug_choice):
        self.energy, self.psi = get_ground_state_from_Hamiltonian(
            Hamiltonian, debug=debug_choice
        )

    def psi_truncate(self, threshold):
        if not np.isscalar(threshold) and not isinstance(threshold, float):
            raise TypeError(
                f"threshold should be a SCALAR FLOAT, not a {type(threshold)}"
            )
        self.psi = truncation(self.psi, threshold)


# =====================================================================================
def truncation(array, threshold):
    if not isinstance(array, np.ndarray):
        raise TypeError(f"array should be an ndarray, not a {type(array)}")
    if not np.isscalar(threshold) and not isinstance(threshold, float):
        raise TypeError(f"threshold should be a SCALAR FLOAT, not a {type(threshold)}")
    array = np.where(np.abs(array) > threshold, array, 0)
    if np.all(np.imag(array)) < 10 ** (-15):
        array = np.real(array)
    return array


# ===================================================================
def get_ground_state_from_Hamiltonian(Ham, sparse=True, debug=True):
    if not isinstance(sparse, bool):
        raise TypeError(f"sparse should be BOOL, not a {type(sparse)}")
    if not isinstance(debug, bool):
        raise TypeError(f"debug should be a BOOL, not a {type(debug)}")
    # DIAGONALIZE THE HAMILTONIAN AND GET THE GROUND STATE |psi> + ITS ENERGY
    phrase = "HAMILTONIAN DIAGONALIZATION"
    pause(phrase, debug)
    # CHECK ON TYPES
    if sparse:
        if not isspmatrix_csr(Ham):
            raise TypeError(f"Ham should be a csr_matrix, not a {type(Ham)}")
        # COMPUTE THE LOWEST ENERGY VALUE AND ITS EIGENSTATE
        eig_vals, eig_vecs = sparse_eigh(Ham, k=1, which="SA")
    else:
        if not isinstance(Ham, np.ndarray):
            raise TypeError(f"Ham should be a np.ndarray, not a {type(Ham)}")
        eig_vals, eig_vecs = array_eigh(Ham, subset_by_index=[0, 0])
    # GROUND STATE ENERGY
    energy = eig_vals[0]
    # GROUND STATE
    psi = np.zeros(Ham.shape[0], dtype=complex)
    psi[:] = eig_vecs[:, 0]
    return energy, psi


# ===================================================================
def diagonalize_density_matrix(rho, debug):
    # Diagonalize the reduced density matrix which STILL is a HERMITIAN COMPLEX MATRIX
    if isinstance(rho, np.ndarray):
        rho_eigvals, rho_eigvecs = array_eigh(rho)
    elif isspmatrix(rho):
        rho_eigvals, rho_eigvecs = array_eigh(rho.toarray())
    if not isinstance(debug, bool):
        raise TypeError(f"debug should be a BOOL, not a {type(debug)}")

    print("EIGENVALUES")
    # Print eigenvalues in descending order
    print(rho_eigvals[::-1])
    print("")
    print("SUM OF EIGENVALUES    " + str(np.sum(rho_eigvals)))
    print("")
    # Print eigenvectors
    # print(csr_matrix(np.real(rho_eigvecs[:,::-1])))
    # print('')
    return rho_eigvals, rho_eigvecs


# ===================================================================
def get_reduced_density_matrix(psi, loc_dim, nx, ny, site):
    if not isinstance(psi, np.ndarray):
        raise TypeError(f"density_mat should be an ndarray, not a {type(psi)}")
    if not np.isscalar(loc_dim) and not isinstance(loc_dim, int):
        raise TypeError(f"loc_dim must be an SCALAR & INTEGER, not a {type(loc_dim)}")
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f"nx must be an SCALAR & INTEGER, not a {type(nx)}")
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f"ny must be an SCALAR & INTEGER, not a {type(ny)}")
    if not np.isscalar(site) and not isinstance(site, int):
        raise TypeError(f"site must be an SCALAR & INTEGER, not a {type(site)}")
    # GET THE TOTAL NUMBER OF SITES
    n_sites = nx * ny
    # GET THE COORDINATES OF THE SITE site
    x, y = zig_zag(nx, ny, site)
    print("----------------------------------------------------")
    print(f"DENSITY MATRIX OF SITE ({str(x+1)},{str(y+1)})")
    print("----------------------------------------------------")
    # RESHAPE psi
    psi_copy = psi.reshape(*[loc_dim for ii in range(n_sites)])
    # DEFINE A LIST OF SITES WE WANT TO TRACE OUT
    indices = np.arange(0, n_sites)
    indices = indices.tolist()
    # The index we remove is the one wrt which we get the reduced DM
    indices.remove(site)
    # COMPUTE THE REDUCED DENSITY MATRIX
    rho = np.tensordot(psi_copy, np.conjugate(psi_copy), axes=(indices, indices))
    # TRUNCATE RHO
    rho = truncation(rho, 10 ** (-10))
    # PROMOTE RHO TO A SPARSE MATRIX
    rho = csr_matrix(rho)
    return rho


# ===================================================================
def get_projector(rho, loc_dim, threshold, debug, name_save):
    # THIS FUNCTION CONSTRUCTS THE PROJECTOR OPERATOR TO REDUCE
    # THE SINGLE SITE DIMENSION ACCORDING TO THE EIGENVALUES
    # THAT MOSTLY CONTRIBUTES TO THE REDUCED DENSITY MATRIX OF THE SINGLE SITE
    if not isinstance(loc_dim, int) and not np.isscalar(loc_dim):
        raise TypeError(f"loc_dim should be INT & SCALAR, not a {type(loc_dim)}")
    if not isinstance(threshold, float) and not np.isscalar(threshold):
        raise TypeError(f"threshold should be FLOAT & SCALAR, not a {type(threshold)}")
    if not isinstance(debug, bool):
        raise TypeError(f"debug should be a BOOL, not a {type(debug)}")
    if not isinstance(name_save, str):
        raise TypeError(f"name_save should be a STR, not a {type(name_save)}")
    # ------------------------------------------------------------------
    # 1) DIAGONALIZE THE SINGLE SITE DENSITY MATRIX rho
    phrase = "DIAGONALIZE THE SINGLE SITE DENSITY MATRIX"
    pause(phrase, debug)
    rho_eigvals, rho_eigvecs = diagonalize_density_matrix(rho, debug)
    # ------------------------------------------------------------------
    # 2) COMPUTE THE PROJECTOR STARTING FROM THE DIAGONALIZATION of rho
    phrase = "COMPUTING THE PROJECTOR"
    pause(phrase, debug)
    # Counts the number of eigenvalues larger than <threshold>
    P_columns = (rho_eigvals > threshold).sum()
    # PREVENT THE CASE OF A SINGLE RELEVANT EIGENVALUE:
    while P_columns < 2:
        threshold = threshold / 10
        # Counts the number of eigenvalues larger than <threshold>
        P_columns = (rho_eigvals > threshold).sum()
    phrase = "TOTAL NUMBER OF SIGNIFICANT EIGENVALUES " + str(P_columns)
    pause(phrase, debug)

    column_indx = -1
    # Define the projector operator Proj: it has dimension (loc_dim,P_columns)
    Proj = np.zeros((loc_dim, P_columns), dtype=complex)
    # Now, recalling that eigenvalues in <reduced_dm> are stored in increasing order,
    # in order to compute the columns of P_proj we proceed as follows
    for ii in range(loc_dim):
        if rho_eigvals[ii] > threshold:
            column_indx += 1
            Proj[:, column_indx] = rho_eigvecs[:, ii]
    # Truncate to 0 all the entries of Proj that are below a certain threshold
    Proj = truncation(Proj, 10 ** (-14))
    # Promote the Projector Proj to a CSR_MATRIX
    Proj = csr_matrix(Proj)
    print(Proj)
    # Save the Projector on a file
    save_npz(f"{name_save}.npz", Proj)
    return Proj


# ===================================================================
def projection(Proj, Operator):
    # THIS FUNCTION PERFORMS THE PROJECTION OF Operator
    # WITH THE PROJECTOR Proj: Op' = Proj^{dagger} Op Proj
    # NOTE: both Proj and Operator have to be CSR_MATRIX type!
    if not isspmatrix_csr(Proj):
        raise TypeError(f"Proj should be an CSR_MATRIX, not a {type(Proj)}")
    if not isspmatrix_csr(Operator):
        raise TypeError(f"Operator should be an CSR_MATRIX, not a {type(Operator)}")
    # -----------------------------------------------------------------------
    Proj_dagger = csr_matrix(csr_matrix(Proj).conj().transpose())
    Operator = Operator * Proj
    Operator = Proj_dagger * Operator
    return Operator
