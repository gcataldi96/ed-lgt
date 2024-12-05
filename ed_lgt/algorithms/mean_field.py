import numpy as np
from scipy.linalg import svd


class mean_field:
    """
    Assumption: Hamiltonian H= ∑ h_{i,i+1}

    (In genrall h_{i,i+1} is not the same for all i.
    Due to the staggering we have for LGT for example,
    alternating terms

    ...h_{i,i+1}^{A}+h_{i+1,i+2}^{B}+h_{i+2,i+3}^{A}+...

    Step 1:
    With an SVD decomposition we find A_i, B_i,
    such that h_{i,i+1}=∑_j A_i^{j} ⊗ B_i+1^{j}

    Step 2:
    Perfom the contraction with the state with:
    Id ⊗ A_i ⊗ B_i and A_i ⊗ B_i ⊗ Id

    Step 3:
    Calculate: H_eff

    Input:
    Matrix h_{i,i+1}

    Output:
    Effective operator

    TODO:
    -Generalize to alternating Hamiltonian terms
    -Observables (outside of this class)
    -Sparse representation
    -Generalize to n-side mf
    """

    def __init__(self, Hij: list, mf_error, decomp_error):

        self.Hij = Hij
        self.mf_error = mf_error
        self.decomp_error = decomp_error

    @staticmethod
    def rand_state(d: int):
        """
        Argument:
        d: length of state
        Returns:
        Random normalized state of length d
        """
        state_init = np.random.rand(d)
        return state_init / np.linalg.norm(state_init)

    def test_decomp(ops_decomp, ops, atol=1e-12, rtol=1e-12):
        C_rec = sum([np.kron(A, B) for A, B in ops_decomp])
        assert np.allclose(ops, C_rec, atol, rtol)

    def decomp_2body(C: list, d_loc, par_m, error=1e-16):
        """
        Input
        C: list of operators
        d_loc: local dim of H_i
        error:

        Returns:
        [[A1,B1,],[A2,B2],...]
        Such that C=∑_i Ai ⊗ Bi
        """
        for op in C:  # generalize for case where I have more than one item in list
            op_reshaped = op.reshape(4 * [d_loc])
            C_flat = op_reshaped.transpose(0, 2, 1, 3).reshape(
                d_loc ** par_m["n_side_mf"], d_loc ** par_m["n_side_mf"]
            )
            U, S, Vh = svd(C_flat)

            ops = [
                [
                    np.sqrt(Si) * U[:, ii].reshape(d_loc, d_loc),
                    np.sqrt(Si) * Vh[ii, :].reshape(d_loc, d_loc),
                ]
                for ii, Si in enumerate(S)
                if Si >= error
            ]
        return ops

    def Ham_eff(ops, state, d_loc):
        """
        Input:
        Operators and state

        Output:
        Effective operator
        """

        H_l, H_r = 0, 0
        H = 0
        Id_A = np.identity(d_loc)
        Id_B = np.identity(d_loc)

        state = state.flatten()  # do I need this?

        for Ai, Bi in ops:
            # Compute contractions for the current operator pair
            kron_IdB_Ai = np.kron(Id_B, Ai)  # (Id_B ⊗ A_i)
            kron_Bi_IdA = np.kron(Bi, Id_A)  # (B_i ⊗ Id_A)

            # Compute <state | (Id_B ⊗ A_i) | state>, <state | (B_i ⊗ Id_A) | state>
            v_L_i = np.inner(state, kron_IdB_Ai @ state)
            v_R_i = np.inner(state, kron_Bi_IdA @ state)

            H_l += v_L_i * np.kron(Bi, Id_A)
            H_r += v_R_i * np.kron(Id_B, Ai)
            H += np.kron(Ai, Bi)

        return H_l + H + H_r

    def sim(self, par_m: dict):
        d_loc = par_m["d_loc"]
        state = mean_field.rand_state(d_loc ** par_m["n_side_mf"])

        op_decomp_dens = mean_field.decomp_2body(
            [x.toarray() for x in self.Hij],
            d_loc,
            par_m,
            error=self.decomp_error,
        )

        mean_field.test_decomp(
            op_decomp_dens,
            [x.toarray() for x in self.Hij],
            atol=1e-12,
            rtol=1e-12,
        )  # NOTE remove in final version

        eigval, eigvec = np.linalg.eigh(self.Hij[0].toarray())
        h = mean_field.Ham_eff(op_decomp_dens, state, d_loc)
        eigval, eigvec = np.linalg.eigh(h)

        diff = 1 + self.mf_error
        E = [eigval[0] / (3 * par_m["n_side_mf"])]
        conv = [self.mf_error + 1]
        ii = 1
        while diff > self.mf_error:
            h = mean_field.Ham_eff(op_decomp_dens, eigvec[:, 0], d_loc)
            eigval, eigvec = np.linalg.eigh(h)

            E.append(eigval[0] / ((3 * par_m["n_side_mf"])))
            diff = abs(E[ii] - E[ii - 1])
            conv.append(abs(E[ii] - E[ii - 1]))
            ii += 1

        self.res = {"E_conv": E, "state": eigvec[:, 0], "conv": conv}

    def get_result(self) -> dict:
        return self.res
