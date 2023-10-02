# %%
import numpy as np
from numpy.linalg import matrix_rank
from itertools import product
from scipy.sparse import csr_matrix, diags, identity, kron
from scipy.sparse.linalg import norm


def change_of_basis(N):
    prefactor = 1 / np.sqrt(2 * np.pi)
    basis_size = int(2 * N + 1)
    F = np.zeros((basis_size, basis_size), dtype=complex)
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            F[i, j] = prefactor * np.exp(complex(0, 2) * i * j * np.pi / basis_size)
    return F


def qmb_op(ops, op_list, add_dagger=False, get_real=False, get_imag=False):
    """
    This function performs the QMB operation of an arbitrary long list
    of operators of arbitrary dimensions.

    Args:
        ops (dict): dictionary storing all the single site operators

        op_list (list): list of the names of the operators involved in the qmb operator
        the list is assumed to be stored according to the zig-zag order on the lattice

        strength (scalar): real/complex coefficient applied in front of the operator

        add_dagger (bool, optional): if true, yields the hermitian conjugate. Defaults to False.

        get_real (bool, optional):  if true, yields only the real part. Defaults to False.

        get_imag (bool, optional): if true, yields only the imaginary part. Defaults to False.
    Returns:
        csr_matrix: QMB sparse operator
    """
    # CHECK ON TYPES
    if not isinstance(ops, dict):
        raise TypeError(f"ops must be a DICT, not a {type(ops)}")
    if not isinstance(op_list, list):
        raise TypeError(f"op_list must be a LIST, not a {type(op_list)}")
    if not isinstance(add_dagger, bool):
        raise TypeError(f"add_dagger should be a BOOL, not a {type(add_dagger)}")
    if not isinstance(get_real, bool):
        raise TypeError(f"get_real should be a BOOL, not a {type(get_real)}")
    if not isinstance(get_imag, bool):
        raise TypeError(f"get_imag should be a BOOL, not a {type(get_imag)}")
    tmp = ops[op_list[0]]
    for op in op_list[1:]:
        tmp = kron(tmp, ops[op])
    if add_dagger:
        tmp = csr_matrix(tmp + tmp.conj().transpose())
    if get_real:
        tmp = csr_matrix(tmp + tmp.conj().transpose()) / 2
    elif get_imag:
        tmp = complex(0.0, -0.5) * (csr_matrix(tmp - tmp.conj().transpose()))
    return tmp


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


def truncation(array, threshold):
    if not isinstance(array, np.ndarray):
        raise TypeError(f"array should be an ndarray, not a {type(array)}")
    if not np.isscalar(threshold) and not isinstance(threshold, float):
        raise TypeError(f"threshold should be a SCALAR FLOAT, not a {type(threshold)}")
    return np.where(np.abs(array) > threshold, array, 0)


# %%
L = 1
alpha = 10
QED_ops = QED_rishon_operators(L)
# Rishon Number operators
for op in ["E0"]:
    QED_ops[f"{op}_mx"] = qmb_op(QED_ops, [op, "ID_xi", "ID_xi", "ID_xi"])
    QED_ops[f"{op}_my"] = qmb_op(QED_ops, ["ID_xi", op, "ID_xi", "ID_xi"])
    QED_ops[f"{op}_px"] = qmb_op(QED_ops, ["ID_xi", "ID_xi", op, "ID_xi"])
    QED_ops[f"{op}_py"] = qmb_op(QED_ops, ["ID_xi", "ID_xi", "ID_xi", op])
# Gauss Law Operator
QED_ops["G"] = 0
for d in ["x", "y"]:
    for i, s in enumerate(["p", "m"]):
        QED_ops["G"] += alpha * ((-1) ** (i) * QED_ops[f"E0_{s}{d}"])
main_diagonal = QED_ops["G"].diagonal()
zero_entries = np.where(main_diagonal == 0)[0]
# Fourier Transform of the Gauss Law Operator
QED_ops["F"] = change_of_basis(L)
QED_ops["F4"] = qmb_op(QED_ops, ["F", "F", "F", "F"])
QED_ops["G_mag"] = QED_ops["F4"] * QED_ops["G"] * QED_ops["F4"].conj().transpose()

# %%
A = csr_matrix(truncation(QED_ops["G_mag"].toarray(), 1e-10))

# %%
