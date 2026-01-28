# %%
from symtable import Class
import numpy as np
import itertools
from scipy.sparse import csr_matrix
from qiskit.quantum_info import SparsePauliOp


def map2site(i, j, width: int = 3, encoding=None) -> int:
    """
    Map two *site-local* labels to a single integer index representing the
    concatenation of their corresponding `width`-bit qubit encodings.

    Context
    -------
    - Each lattice site is a 6-level system (physical labels 0..5).
    - We embed it into `width=3` qubits, i.e. an 8-dimensional space (0..7).
    - An `encoding` specifies which 3-bit computational basis state represents
      each physical label. Example (identity / trivial):
          encoding = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}
      meaning:
          0 -> 000, 1 -> 001, ..., 5 -> 101
      leaving 6->110 and 7->111 unused (unphysical).

    Purpose
    -------
    Your two-site hopping operator is built as a matrix on 2*width qubits
    (here 6 qubits -> 64 states). To place a matrix element between two
    physical two-site configurations (i1,i2) and (f1,f2), you need a consistent
    map:
        (site1_label, site2_label) -> integer in [0, 2**(2*width)).

    How it works
    ------------
    1) Convert each site label into a `width`-bit string:
         b1 = bits(i),  b2 = bits(j)
       If `encoding` is provided and the input is an int, we *first* remap the
       physical label (0..5) to an encoded qubit basis state (0..7 or "010", etc.).
    2) Concatenate: b1 + b2   (site 1 bits followed by site 2 bits)
    3) Interpret as binary -> integer.

    Returns
    -------
    int
        The basis index of the two-site computational basis state.

    Examples
    --------
    With width=3, encoding={0:0,1:1,2:2,3:3,4:4,5:5}:
        map2site(2, 5) -> int('010101', 2) = 21

    With a custom encoding (permutes the codes):
        encoding = {0:0, 1:3, 2:1, 3:2, 4:4, 5:5}
        map2site(1, 2) uses:
            1 -> 011
            2 -> 001
        so it returns int('011001', 2) = 25

    Notes / caveats
    ---------------
    - This function defines an ordering convention for the 2-site basis:
        "site-1 bits are the most significant block"
        "site-2 bits are the least significant block"
      This is fine as long as you use the same convention everywhere.
    - `encoding` must be injective on {0..5}. If two physical labels map to the
      same code, the embedding is invalid.
    - If you pass strings as i/j (like "010"), `encoding` is not applied; the
      string is treated as already-encoded bits.
    """

    def to_bits(x, w: int) -> str:
        """
        Convert `x` into a binary string of length `w`.

        Allowed inputs:
        - int:
            If `encoding` is not None, this int is interpreted as a *physical*
            label (0..5) and remapped via encoding[x]. After remapping, the
            result must fit in [0, 2**w).
            If `encoding` is None, the int is interpreted directly as an
            encoded basis index (0..2**w-1).
        - str:
            Treated as already-encoded bits (optionally prefixed with '0b').
            It is zero-padded to length w.

        Returns:
        - str: bitstring of length w.
        """
        # 1) Remap physical label -> encoded code if requested.
        if isinstance(x, int) and encoding is not None:
            # Expect physical labels 0..5 here (you can add explicit checks if you want).
            x = encoding[x]  # may become int (0..7) or str like "010"
        # 2) If x is a string, interpret as bitstring.
        if isinstance(x, str):
            s = x[2:] if x.startswith("0b") else x
            # Optional safety checks (recommended while debugging):
            # if any(ch not in "01" for ch in s):
            #     raise ValueError(f"Invalid bitstring '{x}'")
            # if len(s) > w:
            #     raise ValueError(f"Bitstring '{s}' longer than width={w}")
            return s.zfill(w)
        # 3) If x is an int, interpret as an encoded integer and format to w bits.
        if isinstance(x, int):
            # Optional safety checks:
            # if x < 0 or x >= (1 << w):
            #     raise ValueError(f"Encoded int {x} outside [0, {1<<w})")
            return format(x, f"0{w}b")
        raise TypeError(f"Unsupported type {type(x)}")

    b1 = to_bits(i, width)
    b2 = to_bits(j, width)
    return int(b1 + b2, 2)


# %%
def hopping_term(encoding=None):
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
        h[
            map2site(i1, i2, width=3, encoding=encoding),
            map2site(j1, j2, width=3, encoding=encoding),
        ] = coeff
    return h


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


def num_pauli_terms_from_operator(H, atol=1e-12):
    op = SparsePauliOp.from_operator(H)
    # prune tiny coefficients
    return int(np.count_nonzero(np.abs(op.coeffs) > atol))


def find_best_encodings_by_terms(
    atol=1e-12, width=3, max_lenght=100, restrict_physical_bits=None
):
    """
    Returns a list of the best encodings:
      [(n_terms, encoding_dict), ...] where best means such that the number of
      Pauli terms in the hopping term is <max_lenght.

    encoding_dict maps physical labels 0..5 -> encoded integer in [0, 2**width)
    where each integer corresponds to a width-bit computational basis state.

    restrict_physical_bits: optional iterable subset of {0..7} to choose codes from.
      Example: restrict_physical_bits=[0,1,2,3,4,5] forces using only 000..101.
    """
    codes = (
        list(restrict_physical_bits)
        if restrict_physical_bits is not None
        else list(range(2**width))
    )

    best = []  # min-heap-like list kept sorted by n_terms ascending
    # generate all injective maps 0..5 -> codes
    for chosen in itertools.permutations(codes, 6):
        enc = {s: chosen[s] for s in range(6)}  # physical state s -> code chosen[s]
        H = hopping_term(encoding=enc)
        n_terms = num_pauli_terms_from_operator(H, atol=atol)
        # check if the encoding generates a small enough amount of Pauli strings
        if n_terms <= max_lenght:
            best.append((n_terms, enc))
            best.sort(key=lambda x: x[0])
    return best


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
best = find_best_encodings_by_terms(atol=1e-10, max_lenght=100)
for ii in best:
    print(ii)
# %%
