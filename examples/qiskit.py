# %%
from symtable import Class
import numpy as np

# generate the pauli string for dressed hamiltonian and store in txt files.
# the coefficients are specified in Main.py

from qiskit.quantum_info import SparsePauliOp

# trivial mapping 0 -> 000 1 -> 001 2 -> 010 3 -> 011 4 -> 100 5 -> 101 6


def map2site(i, j, width=3):
    """
    Combine two bitstrings (or integers) of length `width` into the decimal
    value of their concatenation.

    Examples:
      mapping(2, 6)         -> 22   # '010' + '110' -> '010110' -> 22
      mapping('010', '110') -> 22
    """

    def to_bits(x, w):
        if isinstance(x, str):
            s = x
            if s.startswith("0b"):
                s = s[2:]
            return s.zfill(w)
        if isinstance(x, int):
            return format(x, f"0{w}b")
        raise TypeError("silly mistake")

    b1 = to_bits(i, width)
    b2 = to_bits(j, width)
    return int(b1 + b2, 2)


# %%
def hopping_term():
    h = np.zeros((64, 64), dtype=complex)
    entries = [
        ((0, 2), (2, 1), 1.0j),
        ((0, 4), (2, 3), -1.4142135623730956j),
        ((1, 3), (3, 0), 1.0j),
        ((1, 5), (3, 2), 0.7071067811865478j),
        ((2, 1), (0, 2), -1.0j),
        ((2, 3), (0, 4), 1.4142135623730956j),
        ((2, 3), (4, 0), -1.4142135623730956j),
        ((2, 5), (4, 2), -1.0j),
        ((3, 0), (1, 3), -1.0j),
        ((3, 2), (1, 5), -0.7071067811865478j),
        ((3, 2), (5, 1), 0.7071067811865478j),
        ((3, 4), (5, 3), -1.0j),
        ((4, 0), (2, 3), 1.4142135623730956j),
        ((4, 2), (2, 5), 1.0j),
        ((5, 1), (3, 2), -0.7071067811865478j),
        ((5, 3), (3, 4), 1.0j),
    ]
    for (i1, i2), (j1, j2), coeff in entries:
        h[map2site(i1, i2), map2site(j1, j2)] = coeff
    return h


from scipy.sparse import csr_matrix


def mass_term():
    m = np.zeros((8, 8), dtype=complex)
    m[2, 2] = 1.0
    m[3, 3] = 1.0
    m[4, 4] = 2.0
    m[5, 5] = 2.0
    return m


def casimir_term():
    c = np.zeros((8, 8), dtype=complex)
    c[1, 1] = 3.0 / 4.0
    c[2, 2] = 3.0 / 8.0
    c[3, 3] = 3.0 / 8.0
    c[4, 4] = 3.0 / 4.0
    return c


def get_pauli_string(matrix):
    pauli_op = SparsePauliOp.from_operator(matrix)
    return pauli_op


# %%
def set_sites_1qb(pauli_op, index, Nsites=2):
    # add "III" suffix to one qubit operators on each site
    str_new = []
    for string, coeffs in zip(pauli_op.paulis, pauli_op.coeffs):
        str_new.append(
            [(index * "III") + str(string) + ((Nsites - 1 - index) * "III"), coeffs]
        )
    return str_new


def set_sites_2qb(pauli_op, index, Nsites=2):
    # add "III" suffix to two qubit operators on each site
    # implement the periodic boundary condition later if needed
    str_new = []
    for string, coeffs in zip(pauli_op.paulis, pauli_op.coeffs):
        str_new.append(
            [(index * "III") + str(string) + ((Nsites - 2 - index) * "III"), coeffs]
        )
    return str_new


# %%
class dressed_hamiltonian:
    def __init__(self, mass_coeff, casimir_coeff, hopping_coeff, Nsite: int):
        self.mass_coeff = mass_coeff
        self.casimir_coeff = casimir_coeff
        self.hopping_coeff = hopping_coeff
        self.Nsite = Nsite

    def sum_hopping_term(self):
        h = hopping_term()
        pauli_op = get_pauli_string(h)
        str_new = []
        for index in range(self.Nsite - 1):
            str_new += set_sites_2qb(pauli_op, index, self.Nsite)
        return str_new

    def sum_mass_term(self):
        m = mass_term()
        pauli_op = get_pauli_string(m)
        str_new = []
        for index in range(self.Nsite):
            str_new += set_sites_1qb(pauli_op, index, self.Nsite)
        return str_new

    def sum_casimir_term(self):
        c = casimir_term()
        pauli_op = get_pauli_string(c)
        str_new = []
        for index in range(self.Nsite):
            str_new += set_sites_1qb(pauli_op, index, self.Nsite)
        return str_new

    def export_pauli_strings(self, operator, textfile):
        with open(textfile, "w") as file:
            for i in range(len(operator)):
                coeff = operator[i][1]
                if np.real(coeff) > 1.0e-8 or np.imag(coeff) > 1.0e-8:
                    file.write(f"{operator[i][0]}")
                    file.write("\t")
                    file.write(f"{coeff}")
                    file.write("\n")
        return None


# %%
