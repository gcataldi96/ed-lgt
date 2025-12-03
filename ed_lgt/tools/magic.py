import numpy as np
from numpy.linalg import matrix_power
import logging
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

__all__ = ["pauli_string", "pauli_operators"]


"""
N: number of qudits
d: dimension of local Hilbert space of a qudit


This is to generate the elements of Pauli group consisting of all possible Pauli strings of N d-dimensional qudits.
Notes:
    1. The cardinality of the Pauli group would be N to the power h [# (non-commuting) Pauli operators] in a given dimension d
    2. Each element has a form of Pk = X_1^p1Z_1^q1 otimes X_2^p2Z_2^q2 otimes ... otimes X_N^pNZ_N^qN
        2.1. pi,qi = 0,1,2,...,h-1
        2.2. X_iZ_i: Pauli operators on the ith qudit

"""


def pauli_string(N: int, d: int):
    # -----------------------------------------------------------------------------------
    # call the script for generating Pauli operators with the associated keys
    PO, PO_keys = pauli_operators(d, "non-commute")
    # -----------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------
    # construct Pauli strings
    PO_num = len(PO)
    PG = dict()
    PG_keys = []
    n = 0
    while n < N**PO_num:
        # ===============================================================================
        # create an array indicating which Pauli operator will be called on each site at each loop
        sites_ele = np.zeros((N), dtype=int)
        for j in range(N):
            sites_ele[j] = np.floor((n / (PO_num) ** j) % PO_num)
        # ===============================================================================
        PS = 1
        for a in range(N):
            PS = np.kron(PS, PO[PO_keys[sites_ele[a]]])
        PG["P" + str(n)] = PS
        PG_keys.append("P" + str(n))
        n += 1
    # -----------------------------------------------------------------------------------

    return PG, PG_keys


"""
Notes:
    1. d refers to the dimension of the Hilbert space of a qudit
    2. setType sets the type of the Pauli operator set. Default is set to 'full'. If you set it to 'non-commute', it gives the non-commuting Pauli operator set
       3.1. for d being a prime, the size of the set is d + 1
       3.2. for d being a composite number, the size of th set is given by the Dedekind psi function
            Psi(d) = d*prod(1 + 1/p_j) from j = 0 to m - 1, where pj is the prime factor of d
       
    E.g.
        - For d = 2, there are in total 2^2 = 4 Pauli operators possible; the set of non-commuting operators is
          {I,Z,X,XZ}
        - For d = 6, there are in total 6^2 = 36 Pauli operators possible; the set of non-commuting operators is
          {I,Z,X,XZ,XZ2,XZ3,XZ4,XZ5,X2Z,X2Z3,X2Z5,X3Z,X3Z2}
"""


def pauli_operators(d: int, setType="full"):
    # -----------------------------------------------------------------------------------
    # construct d-dimensional Pauli X and Z operator
    X = np.zeros((d, d))
    Z = np.zeros((d, d), dtype=complex)
    for i in range(d):
        X[(i + 1) % d, i] = 1
        Z[i, i] = np.exp(1j * 2 * np.pi * i / d)
    # -----------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------
    # construct all possible Pauli operators in the form X^mZ^n according to setType
    PO = dict(I=np.identity(d, dtype=complex))
    keys = ["I"]
    XZ_powers = [[0, 0]]
    for m in range(d):
        for n in range(d):
            if m == 0 and n == 0:
                continue
            else:
                if m == 0:
                    PO["Z" + str(n)] = np.matmul(matrix_power(X, m), matrix_power(Z, n))
                    keys.append("Z" + str(n))
                elif n == 0:
                    PO["X" + str(m)] = np.matmul(matrix_power(X, m), matrix_power(Z, n))
                    keys.append("X" + str(m))
                else:
                    PO["X" + str(m) + "Z" + str(n)] = np.matmul(
                        matrix_power(X, m), matrix_power(Z, n)
                    )
                    keys.append("X" + str(m) + "Z" + str(n))
                XZ_powers.append([m, n])
    for key, op in PO.items():
        PO[key] = csr_matrix(op)
    # -----------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------
    # determine if non-commuting set is needed
    if setType == "full":
        return PO, keys
    elif setType == "non-commute":
        XZ_powers = np.array(XZ_powers)
        comm_OP = []
        a = 1
        b = 2
        while a < len(PO):
            while b < len(PO):
                pd = (
                    XZ_powers[b, 0] * XZ_powers[a, 1]
                    - XZ_powers[b, 1] * XZ_powers[a, 0]
                )
                if pd % d == 0:
                    comm_OP.append(b)
                b += 1
            for i in range(len(comm_OP)):
                del PO[keys[comm_OP[i] - i]]
                keys.remove(keys[comm_OP[i] - i])
                XZ_powers = np.delete(XZ_powers, comm_OP[i] - i, 0)
            a += 1
            b = a + 1
            comm_OP = []
        return PO, keys
    else:
        raise ValueError(f"setType expeceted 'full' or 'non-commute'. Got {setType}.")
    # -----------------------------------------------------------------------------------


"""
# This is for sanity check if the generated Pauli group has the correct Pauli strings


import math
import matplotlib.pyplot as plt

N = 2
d = 3
PG,PG_keys = pauli_string(N,d)

for n in range(len(PG)):
    A = PG[PG_keys[n]]
    s = 600/d
    plt.figure(n)
    for pr in range(A.shape[0]):
        for pc in range(A.shape[1]):
            if math.isclose(np.real(A[pr,pc]),1,abs_tol=1e-8) == True:
                plt.scatter(pc,A.shape[0]-pr,s,marker='$1$',c='black')
            elif np.imag(A[pr,pc]) > 0:
                plt.scatter(pc,A.shape[0]-pr,s,marker='$\omega$',c='black')
            elif np.imag(A[pr,pc]) < 0:
                plt.scatter(pc,A.shape[0]-pr,s,marker='$\omega^2$',c='black')
            else:
                plt.scatter(pc,A.shape[0]-pr,s,marker='o',c='black')
    plt.title(PG_keys[n])
"""
