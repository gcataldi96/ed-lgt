import numpy as np
from scipy.linalg import svd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

def rand_state(d):
    """
    Argument:
    int d, which gives length of state
    Returns:
    Random normalized state of length d
    """
    state_init = np.random.rand(d)
    return state_init / np.linalg.norm(state_init)

def decomp_2body(C, m_A, n_A, m_B, n_B, error=1e-16):
    """
    Input
    A two body operator C.

    Returns:
    [[A1,B1,],[A2,B2],...]
    Such that C=∑_i Ai ⊗ Bi 

    """
    C_reshaped = C.reshape((m_A, n_A, m_B, n_B))
    C_flat = C_reshaped.transpose(0, 2, 1, 3).reshape(m_A * m_B, n_A * n_B)
    U, S, Vh = svd(C_flat)

    ops = [
        [
            np.sqrt(Si) * U[:, ii].reshape(m_A, m_B),
            np.sqrt(Si) * Vh[ii, :].reshape(n_A, n_B),
        ]
        for ii, Si in enumerate(S)
        if Si >= error
    ]
    return ops

def decomp_2body_sparse(C, m_A, n_A, m_B, n_B, error=1e-16, k=None):
    """
    Input
    Matrix C sparse

    Returns:
    [[A1,B1,],[A2,B2],...]
    Such that C=∑_i Ai ⊗ Bi 

    """
    assert sp.issparse(C), "C must be sparse."
    
    # Convert C to a dense matrix temporarily and reshape 
    # TODO: circumvent to change to dense matrix
    C_dense = C.toarray().reshape((m_A, n_A, m_B, n_B))
    C_flat = C_dense.transpose(0, 2, 1, 3).reshape(m_A * m_B, n_A * n_B)
    U, S, Vh = svds(sp.csr_matrix(C_flat), k=k if k else min(C_flat.shape) - 1)
    
    # order of scipy.sparse.linalg.svds is ascending
    U, S, Vh = U[:, ::-1], S[::-1], Vh[::-1, :]
    
    ops = []
    for ii, Si in enumerate(S):
        if Si >= error:
            A_i = sp.csr_matrix(np.sqrt(Si) * U[:, ii].reshape(m_A, m_B))
            B_i = sp.csr_matrix(np.sqrt(Si) * Vh[ii, :].reshape(n_A, n_B))
            ops.append([A_i, B_i])

    return ops


def Ham_eff(ops, state):
    """
    Assuming a Hamiltonian H= ∑ H_{i,i+1}
    Step 1:
    With an SVD decomposition we find A_i, B_i, 
    such that H_{i,i+1}=∑ A_i ⊗ B_i.

    Step 2:
    Do the contraction with the state with:
    Id ⊗ A_i ⊗ B_i and A_i ⊗ B_i ⊗ Id
    
    Step 3:
    Calculate: H_eff

    Input:
    Operators and state

    Output: 
    Effective operator
    """

    v_L,v_R=0,0 
    H_l,H_r=0,0
    H=0
    d_loc=10
    Id_A = np.identity(d_loc)
    Id_B = np.identity(d_loc)
    for Ai, Bi in ops:
        v_L+= np.inner(state, np.dot(np.kron(Id_B, Ai), state))
        v_R+= np.inner(state, np.dot(np.kron(Bi, Id_A), state))

    for Ai,Bi in ops:
        H_l+=v_L*np.kron(Bi,Id_A)
        H_r+=v_R*np.kron(Id_B,Ai)
        H+=np.kron(Ai,Bi)
                         
    return H_l+H+H_r


def Ham_eff_sparse(ops, state):
    """
    Assuming a Hamiltonian H= ∑ H_{i,i+1}
    Step 1:
    With an SVD decomposition we find A_i, B_i, 
    such that H_{i,i+1}=∑ A_i ⊗ B_i.

    Step 2:
    Do the contraction with the state with:
    Id ⊗ A_i ⊗ B_i and A_i ⊗ B_i ⊗ Id
    
    Step 3:
    Calculate: H_eff

    Input:
    Operators and state

    Output: 
    Effective operator
    """
    v_L, v_R = 0, 0
    H_l, H_r, H = 0, 0, 0

    d_loc=10
    Id_A = sp.eye(d_loc, format="csr")
    Id_B = sp.eye(d_loc, format="csr")

    for Ai, Bi in ops:
        v_L += (state.T @ (sp.kron(Id_B, Ai) @ state)).item()  
        v_R += (state.T @ (sp.kron(Bi, Id_A) @ state)).item()

    for Ai, Bi in ops:
        H_l += v_L * sp.kron(Bi, Id_A)
        H_r += v_R * sp.kron(Id_B, Ai)
        H += sp.kron(Ai, Bi)

    return H_l + H + H_r

def test_decomp_sparse(ops_decomp,ops,atol=1e-13, rtol=1e-13):
    C_rec = 0
    for A,B in ops_decomp:
        C_rec += np.kron(A.toarray(), B.toarray())
    assert np.allclose(ops.toarray(), C_rec, atol=1e-13, rtol=1e-13)

    return C_rec

def test_decomp(ops_decomp,ops,atol=1e-13, rtol=1e-13):
    C_rec = 0
    for A,B in ops_decomp:
        C_rec += np.kron(A, B)
    assert np.allclose(ops, C_rec, atol=1e-13, rtol=1e-13)

    return C_rec

def sim(ops,error_mean,error_dec):
    
    """
    Step 1:
    Decompose operators
    
    """
    d_loc=10
    ops_decomp=decomp_2body_sparse(ops, m_A=d_loc, n_A=d_loc, m_B=d_loc, n_B=d_loc, error=error_dec)

    #test decomp
    t1=test_decomp_sparse(ops_decomp,ops,atol=1e-16, rtol=1e-16)
    
    
    # #get random state
    state=rand_state(d_loc**2)

    op_test=decomp_2body(ops.toarray(), m_A=d_loc, n_A=d_loc, m_B=d_loc, n_B=d_loc, error=error_dec)
    t2=test_decomp(op_test,ops.toarray(),atol=1e-16, rtol=1e-16)
    
    #before contraction test if clost
    print(type(t1),type(t2))
    t1=np.allclose(t1,t2,atol=1e-18, rtol=1e-18)

    #do contraction
    h_sparse=Ham_eff_sparse(ops_decomp, state)
    h=Ham_eff(op_test,state)

    #Something foes wrong in the contraction
    assert np.allclose(h_sparse.toarray(), h, atol=1e-5, rtol=1e-22)






    # diff=1+error_mean
    # E=[10]
    # ii=1
    # while diff>error_mean:

    #     #contraction
    #     H_eff=Ham_eff(ops=ops_decomp,state=state)
        
    #     #diagonalize
    #     Eg=1
        
    #     E.append(Eg)
    #     diff=abs(E[ii]-E[ii-1])
    #     ii+=1

    # res={"E_conv":E,
    #      "state":state
    #      }

    # return res 


