# %%
import numpy as np
from numpy.linalg import matrix_rank
from itertools import product
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import norm


def change_of_basis(N, eigvecs, eigvals):
    """_summary_

    Args:
        N (scalar, int or semi int): value of the maximal spin representation
        eigvecs (np.ndarray): set of vectors forming the Electric Basis. Every vector should have size 2N+1
        eigvals (np.ndarray): set of the corresponding eigenvalues of the Electric
        operator evaluated on the Electric basis

    Returns:
        np.ndarray: array of the
    """
    prefactor = 1 / np.sqrt(2 * np.pi)
    basis_size = 2 * N + 1
    vector = np.zeros(basis_size)
    for i in range(-N, N + 1):
        vector += (
            prefactor * np.exp(complex(0, 1) * eigvals[i] / basis_size) * eigvecs[i]
        )
    return vector


def QED_rishon_operators(spin, U="ladder", basis="E"):
    if not np.isscalar(spin):
        raise TypeError(f"spin must be SCALAR & (semi)INTEGER, not {type(spin)}")
    if not isinstance(U, str):
        raise TypeError(f"U must be str, not {type(U)}")
    """
    This function computes the SU2 the Rishon modes adopted
    for the U(1) Lattice Gauge Theory for the chosen spin-s irrepresentation

    Args:
        spin (scalar, real): spin value, assumed to be integer or semi-integer
        U (str, optional): which version of U you want to use to obtain rishons

    Returns:
        dict: dictionary with the rishon operators and the parity
    """
    # Size of the spin/rishon matrix
    size = int(2 * spin + 1)
    shape = (size, size)
    # Based on the U definition, define the diagonal entries of the rishon modes
    if U == "ladder":
        zm_diag = [(-1) ** (i + 1) for i in range(size - 1)][::-1]
        U_diag = np.ones(size - 1)
    elif U == "spin":
        sz_diag = np.arange(-spin, spin + 1)[::-1]
        U_diag = (np.sqrt(spin * (spin + 1) - sz_diag[:-1] * (sz_diag[:-1] - 1))) / spin
        zm_diag = [U_diag[i] * ((-1) ** (i + 1)) for i in range(size - 1)][::-1]
    else:
        raise ValueError(f"U can only be 'ladder' or 'spin', not {U}")
    ops = {}
    ops["U"] = diags(U_diag, -1, shape)
    # RISHON MODES
    ops["xi_p"] = diags(np.ones(size - 1), 1, shape)
    ops["xi_m"] = diags(zm_diag, 1, shape)
    # PARITY OPERATOR
    ops["P_xi"] = diags([(-1) ** i for i in range(size)], 0, shape)
    # IDENTITY OPERATOR
    ops["ID_xi"] = identity(size)
    # ELECTRIC FIELD OPERATORS
    ops["n"] = diags(np.arange(size), 0, shape)
    ops["E0"] = ops["n"] - 0.5 * (size - 1) * identity(size)
    ops["E0_square"] = ops["E0"] ** 2
    for side in ["p", "m"]:
        # GENERATE THE DAGGER OPERATORS
        ops[f"xi_{side}_dag"] = ops[f"xi_{side}"].transpose()
    return ops


# %%
