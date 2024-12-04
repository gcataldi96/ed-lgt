import numpy as np


class mean_field:
    """
    Assumption: Hamiltonian H= ∑ h_{i,i+1}

    Step 1:
    With an SVD decomposition we find A_i, B_i,
    such that h_{i,i+1}=∑_j A_i^{j} ⊗ B_i+1^{j}

    Step 2:
    Perfom the contraction with the state with:
    Id ⊗ A_i ⊗ B_i and A_i ⊗ B_i ⊗ Id

    Step 3:
    Calculate: H_eff

    Input:
    Sparse matrix h_{i,i+1}

    Output:
    Effective operator
    """

    def __init__(self, Hij, mf_error, decomp_error):

        self.Hij = Hij
        self.mf_error = mf_error
        self.decomp_error = decomp_error

    def reshape(self):
        pass

    def sim(self):

        pass
