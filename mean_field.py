import numpy as np
from scipy.linalg import svd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import copy


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


def Ham_eff(ops, state, d_loc):
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

    H_l, H_r = 0, 0  # Initialize H_l and H_r as scalars (will become arrays)
    H = 0  # Initialize H as scalar (will become array)
    Id_A = np.identity(d_loc)
    Id_B = np.identity(d_loc)

    for Ai, Bi in ops:
        # Compute contractions for the current operator pair
        kron_IdB_Ai = np.kron(Id_B, Ai)  # (Id_B ⊗ A_i)
        kron_Bi_IdA = np.kron(Bi, Id_A)  # (B_i ⊗ Id_A)

        # Ensure state is a 1D array
        state = state.flatten()

        # Compute <state | (Id_B ⊗ A_i) | state>, <state | (B_i ⊗ Id_A) | state>
        v_L_i = np.inner(state, kron_IdB_Ai @ state)
        v_R_i = np.inner(state, kron_Bi_IdA @ state)

        H_l += v_L_i * np.kron(Bi, Id_A)
        H_r += v_R_i * np.kron(Id_B, Ai)
        H += np.kron(Ai, Bi)

    return H_l + H + H_r


# def contract(ops, state, d_loc):
#     """
#     ops: [[A1,B1,],[A2,B2],...]
#     state:
#     d_loc:
#     """
#     Id = np.identity(d_loc)
#     T_r = np.zeros(4 * [d_loc**2])
#     H = np.zeros(2 * [d_loc**2])

#     for Ai, Bi in ops:
#         T_r += np.einsum("ij,kl,mn,uv->ikmujlnv", Id, Ai, Bi, Id).reshape(
#             4 * [d_loc**2]
#         )
#         H += np.kron(Ai, Bi)

#     # two types of contractions
#     C1 = np.tensordot(copy(T_r), state, axes=([2], [0]))
#     H_l = np.tensordot(C1, state, axes=([0], [0]))

#     C3 = np.tensordot(T_r, state, axes=([3], [0]))
#     H_r = np.tensordot(C3, state, axes=([1], [0]))

#     return H_l + H + H_r


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

    d_loc = 10
    Id_A = sp.eye(d_loc, format="csr")
    Id_B = sp.eye(d_loc, format="csr")

    for Ai, Bi in ops:
        v_L += (state.T @ (sp.kron(Id_B, Ai) @ state)).item()
        v_R += (state.T @ (sp.kron(Bi, Id_A) @ state)).item()

    for Ai, Bi in ops:
        H_l += v_L * sp.kron(Bi, Id_A)
        H_r += v_R * sp.kron(Id_B, Ai)
        H += sp.kron(Ai, Bi)

    return H_l, H_r


def test_decomp_sparse(ops_decomp, ops, atol=1e-13, rtol=1e-13):
    C_rec = 0
    for A, B in ops_decomp:
        C_rec += np.kron(A.toarray(), B.toarray())
    assert np.allclose(ops.toarray(), C_rec, atol=1e-13, rtol=1e-13)


def test_decomp(ops_decomp, ops, atol=1e-12, rtol=1e-12):
    C_rec = 0
    for A, B in ops_decomp:
        C_rec += np.kron(A, B)
    assert np.allclose(ops, C_rec, atol, rtol)


def sim(Hij, par, error_mean, error_dec):
    """
    Step 1:
    Decompose operators

    """
    d_loc = par["n_max"] + 1

    # #get random state
    state = rand_state(d_loc**2)

    op_decomp_dens = decomp_2body(
        Hij.toarray(), m_A=d_loc, n_A=d_loc, m_B=d_loc, n_B=d_loc, error=error_dec
    )

    test_decomp(op_decomp_dens, Hij.toarray(), atol=1e-12, rtol=1e-12)

    eigval, eigvec = np.linalg.eigh(Hij.toarray())
    print("gs", eigval[0] / 2)

    # do contraction
    # h = contract(op_decomp_dens, state, d_loc)
    h = Ham_eff(op_decomp_dens, state, d_loc)

    eigval, eigvec = np.linalg.eigh(h)

    # # #Something goes wrong in the contraction
    # eigval, eigvec = np.linalg.eig(h)

    diff = 1 + error_mean
    E = [eigval[0] / 2]
    conv = [error_mean + 1]
    ii = 1
    while diff > error_mean:

        # contraction
        h = Ham_eff(op_decomp_dens, eigvec[:, 0], d_loc)

        # diagonalize
        eigval, eigvec = np.linalg.eigh(h)

        E.append(eigval[0] / 6)
        diff = abs(E[ii] - E[ii - 1])
        conv.append(abs(E[ii] - E[ii - 1]))
        ii += 1

    for item in E:
        print(item)

    return {"E_conv": E, "state": state, "conv": 0}
