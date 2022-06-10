import numpy as np
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse import kron
from scipy.sparse import csr_matrix
import os
import re
import sys
import subprocess as sub

# ========================================================================================
# ========================================================================================
def pause():
    # IT PROVIDES A PAUSE WHEN LAUNCHING A PYTHON CODE
    programPause = input("Press the <ENTER> key to continue")


# ========================================================================================
# ========================================================================================
def derivative(x, f, dx, f_der, option):
    # COMPUTE THE 1st OR THE 2nd DERIVATIVE
    # f_der OF A FUNCTION f WRT A VARIABLE x
    f_der = np.zeros(x.shape[0] - 2)
    if option == 1:
        # COMPUTE THE 1ST ORDER CENTRAL DERIVATIVE
        for ii in range(f_der.shape[0]):
            jj = ii + 1
            f_der[ii] = (f[jj + 1] - f[jj - 1]) / dx
    elif option == 2:
        # COMPUTE THE 2ND ORDER CENTRAL DERIVATIVE
        for ii in range(f_der.shape[0]):
            jj = ii + 1
            f_der[ii] = (f[jj + 1] - 2 * f[jj] + f[jj - 1]) / (dx**2)
    # USE AN UPDATE VERSION OF X WHERE THE FIRST
    # AND THE LAST ENTRY ARE ELIMINATED IN ORDER
    # TO GET AN ARRAY OF THE SAME DIMENSION OF
    # THE ONE WITH THE DERIVATIVE OF F
    x_copy = np.zeros(f_der.shape[0])
    for ii in range(f_der.shape[0]):
        x_copy[ii] = x[ii + 1]
    return x_copy, f_der


# ========================================================================================
# ========================================================================================
def kronecker(a, b):
    # HAND WRITTEN TENSOR PRODUCT BETWEEN TWO MATRICES
    c = np.zeros((a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]), dtype=float)
    for ii in range(a.shape[0]):
        for jj in range(a.shape[1]):
            for kk in range(b.shape[0]):
                for ll in range(b.shape[1]):
                    c[kk + ii * b.shape[0]][ll + jj * b.shape[1]] = (a[ii][jj]) * (
                        b[kk][ll]
                    )
    return c


# ========================================================================================
# ========================================================================================
def jordan_wigner_operators():
    # MAPS BOSE SPIN OPERATORS INTO FERMION OPERATORS THROUGH THE KNOWN
    # JORDAN WIGNER TRANSFORMATION. THE BTAINED OPERATORS SHOULD SATISFY
    # ALL THE ANTI-COMMUTATION RELATIONS OF FERMION OPERATORS
    ID2 = np.array([[1, 0], [0, 1]])
    sx = np.array([[0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]])
    sy = np.array([[0 + 0j, 0 - 1j], [0 + 1j, 0 + 0j]])
    sz = np.array([[1, 0], [0, -1]])
    # S+ and S- BOSON OPERATORS
    sp = csr_matrix(np.real((sx + (0 + 1j) * sy) / 2))
    sm = csr_matrix(np.real((sx - (0 + 1j) * sy) / 2))
    # OCCUPATION NUMBER OPERATOR
    n = csr_matrix((sz + ID2) / 2)
    # JORDAN WIGNER TERM
    JW = csr_matrix(np.array([[-1, 0], [0, 1]]))
    return sp, sm, n, JW


# ========================================================================================
# ========================================================================================
def local_operator(a, ID, pos, dim):
    # a    IS THE LOCAL OPERATOR
    # ID   IS THE LOCAL IDENTITY MATRIX
    # pos  IS THE SITE WHERE <a> HAS TO BE APPLIED
    # dim  IS THE NUMBER OF SITES OF THE SYSTEM
    # GLOBALLY, THE # OF TENSOR PRODUCT HAS TO BE dim-1
    c = a
    for ii in range(pos - 1):
        if dim > 4:
            print("TENSOR PRODUCT")
        c = kron(ID, c)
    for ii in range(dim - pos):
        if dim > 4:
            print("TENSOR PRODUCT")
        c = kron(c, ID)
    return c


# ========================================================================================
# ========================================================================================
def hopping_terms(x, y, dim):
    couples = np.zeros((2, 2 * (2 * x * y - x - y)), dtype=int)
    ii = 0
    for yy in range(y):
        for xx in range(x):
            x1 = xx
            y1 = yy
            if x1 < x - 1:
                x2 = x1 + 1
                y2 = y1
                d1 = inverse_snake(x, x1, y1) + 1
                d2 = inverse_snake(x, x2, y2) + 1
                couples[0][ii] = d1
                couples[1][ii] = d2
                ii += 1
                couples[0][ii] = d2
                couples[1][ii] = d1
                ii += 1
            if y1 < y - 1:
                x3 = x1
                y3 = y1 + 1
                d1 = inverse_snake(x, x1, y1) + 1
                d3 = inverse_snake(x, x3, y3) + 1
                couples[0][ii] = d1
                couples[1][ii] = d3
                ii += 1
                couples[0][ii] = d3
                couples[1][ii] = d1
                ii += 1
    print("THE LIST OF PAIR OF SITES INVOLVED IN THE HOPPING HAMILTONIAN")
    print(couples)
    if dim == 4:
        return couples
    elif dim == 2:
        # DEFINE A NEW ARRAY WHERE THE HOPS ARE DIFFERENTLY DEFINED FOR UP AND DOWN SITES
        hops = np.zeros((2, couples.shape[0], couples.shape[1]), dtype=int)
        for ii in range(couples.shape[1]):
            for jj in range(couples.shape[0]):
                hops[0][jj][ii] = 2 * couples[jj][ii] - 2
                hops[1][jj][ii] = 2 * couples[jj][ii] - 1
        return hops


# ========================================================================================
# ========================================================================================
def two_body_operator(a, b, ID, JW, pos1, pos2, dim):
    # IMPLEMENT A 2 BODY FERMION OPERATOR ON A CHAIN
    # OF dim SITES VIA TENSOR PRODUCT OPERATIONS
    # GLOBALLY, THE # OF TENSOR PRODUCT HAS TO BE dim-1
    #  a     IS THE 1st LOCAL OPERATOR
    #  b     IS THE 2nd LOCAL OPERATOR
    #  ID    IS THE LOCAL IDENTITY MATRIX
    #  JW    IS THE JORDAN WIGNER TERM
    #  pos1  IS THE SITE WHERE <a> HAS TO BE APPLIED
    #  pos2  IS THE SITE WHERE <b> HAS TO BE APPLIED
    #  dim   IS THE NUMBER OF SITES OF THE SYSTEM
    if pos2 < pos1:
        aa = JW * b
        bb = a
        d2 = pos1
        d1 = pos2
    else:
        aa = a * JW
        bb = b
        d1 = pos1
        d2 = pos2
    c = aa
    for ii in range(d1 - 1):
        if dim > 9:
            print("TENSOR PRODUCT")
        c = kron(ID, c)
    for ii in range(d2 - d1 - 1):
        if dim > 9:
            print("TENSOR PRODUCT")
        c = kron(c, JW)
    if dim > 9:
        print("TENSOR PRODUCT")
    c = kron(c, bb)
    for ii in range(dim - d2):
        if dim > 9:
            print("TENSOR PRODUCT")
        c = kron(c, ID)
    return c


# ========================================================================================
# ========================================================================================
def sparse_diagonalization(H, x, y, operators):
    # DIAGONALIZE THE HAMILTONIAN ASSOCIATED TO A DISCRETE LATTICE OF SIZES x,y.
    # ACQUIRE THE LOWEST ALGEBRAIC EIGENVALUE (GROUND STATE ENERGY) AND
    # PROJECT ALL THE SINGLE SITE OCCUPATION NUMBER OPERATORS ONTO THE GROUND
    # STATE psi ASSOCIATED TO THE OBTAINED GROUND STATE ENERGY.
    # ALL THE OPERATORS WHOSE ZERO POINT AVERAGE HAS TO BE COMPUTED ARE
    # STORED IN A VOCABULARY NAMED <operators>
    # AS RESULTS IT PROVIDES THE GROUND STATE ENERGY, THE DENSITY OF PARTICLES
    # AND THE DENSITY OF DOUBLE OCCUPANCIES
    print("-----------------------------------------------------")
    print("|               DIAGONALIZATION                     |")
    print("-----------------------------------------------------")
    # GET EIGENVALUES AND EIGENVECTORS OF THE HAMILTONIAN:
    # (w IS THE ARRAY OF EIGENVALUES, v THE MATRIX OF EIGENVECTORS)
    w, v = linalg.eigsh(H, k=1, which="SA")
    # SINCE WE ARE TALKING ABOUT ENERGY (WHICH IS AN OBSERVABLE)
    # ALL THE EIGENVALUES AND EIGENVECTORS HAVE TO BE REAL
    w = np.real(w)
    v = np.real(v)
    # SAVE THE LOWEST EIGENVALUE OF THE HAMILTONIAN
    energy = (w[0]) / (x * y)
    print("    ----GROUND STATE ENERGY  E ", format(energy, ".3f"))
    # SAVE THE GROUND STATE ASSOCIATED TO THE ENERGY MINIMUM
    psi = np.zeros(H.shape[0], dtype=float)
    psi[:] = v[:, 0]
    # COMPUTE THE AVERAGE NUMBER OF UP/DOWN/DOUBLE PARTICLES IN
    # EACH SITE OF THE LATTICE PRINT THESE VALUES
    n_up_tot = 0.0
    n_down_tot = 0.0
    n_pair_tot = 0.0
    print("=====================================================")
    print("      # UP        # DOWN      # PAIR                |")
    print("-----------------------------------------------------")

    for kk in range(x * y):
        operators["up_%s" % (str(kk + 1))] = np.dot(
            psi, operators["N_up_" + str(kk + 1)].dot(psi)
        )
        operators["down_%s" % (str(kk + 1))] = np.dot(
            psi, operators["N_down_" + str(kk + 1)].dot(psi)
        )
        operators["pair_%s" % (str(kk + 1))] = np.dot(
            psi, operators["N_pair_" + str(kk + 1)].dot(psi)
        )
        print(
            "     ",
            format(operators["up_" + str(kk + 1)], ".3f"),
            "     ",
            format(operators["down_" + str(kk + 1)], ".3f"),
            "     ",
            format(operators["pair_" + str(kk + 1)], ".3f"),
            "                |",
        )

        n_up_tot += operators["up_" + str(kk + 1)]
        n_down_tot += operators["down_" + str(kk + 1)]
        n_pair_tot += operators["pair_" + str(kk + 1)]

    print("-----------------------------------------------------")
    n_sum = n_up_tot + n_down_tot
    print("    THE TOTAL UP PARTICLES   IS  " + str(format(n_up_tot, ".3f")) + "    |")
    print(
        "    THE TOTAL DOWN PARTICLES IS  " + str(format(n_down_tot, ".3f")) + "    |"
    )
    print(
        "    THE TOTAL COUPLES        IS  " + str(format(n_pair_tot, ".3f")) + "    |"
    )
    print("    THE TOTAL PARTICLES      IS  " + str(format(n_sum, ".3f")) + "    |")
    print("=====================================================")
    rho = n_sum / (x * y)
    return energy, rho


# ========================================================================================
# ========================================================================================
def check_array(a, b):
    # CHECK ONE BY ONE THE ENTRIES OF TWO ARRAYS a AND b
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # THE 2 ARRAYS MUST HAVE THE SAME SHAPE
    if a.shape == b.shape:
        if a.shape == (a.shape[0],):
            print("                                                   %%")
            print("CHECK VECTORS                                      %%")
            for ii in range(a.shape[0]):
                c = abs(a[ii] - b[ii])
                if c > 10 ** (-16):
                    print("ERROR:                                             %%")
                    print("THE 2 VECTORS DIFFER IN THE ENTRY [" + str(ii) + "]")
                    print("a[" + str(ii) + "]=" + str(a[ii]))
                    print("b[" + str(ii) + "]=" + str(b[ii]))
                    if a.shape[0] < 10:
                        print("VECTOR a")
                        print(a)
                        print("VECTOR b")
                        print(b)
                    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                    sys.exit()
            print("THE 2 VECTORS ARE EQUAL                            %%")
            print("                                                   %%")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        elif a.shape == (a.shape[0], a.shape[1]):
            print("                                                   %%")
            print("CHECK MATRICES                                     %%")
            for ii in range(a.shape[0]):
                for jj in range(a.shape[1]):
                    c = abs(a[ii][jj] - b[ii][jj])
                    if c > 10 ** (-16):
                        print("ERROR:                                             %%")
                        print(
                            "THE TWO MATRICES DIFFER IN THE ENTRY "
                            + "["
                            + str(ii)
                            + "]["
                            + str(jj)
                            + "]        %%"
                        )
                        print(
                            "a["
                            + str(ii)
                            + "]["
                            + str(jj)
                            + "]="
                            + str(a[ii][jj])
                            + "                                       %%"
                        )
                        print(
                            "b["
                            + str(ii)
                            + "]["
                            + str(jj)
                            + "]="
                            + str(b[ii][jj])
                            + "                                       %%"
                        )
                        if a.shape[0] < 10:
                            print("MATRIX a")
                            print(a)
                            print("MATRIX b")
                            print(b)
                        print("                                                   %%")
                        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                        sys.exit()
            print("THE TWO MATRICES ARE EQUAL                         %%")
            print("                                                   %%")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


# ========================================================================================
# ========================================================================================
def check_number(a, b, threshold):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("CHECK NUMBER")
    if type(a) == type(b):
        # CASE OF FLOAT OR INT TYPE
        if type(a) == float or type(a) == int:
            if abs(a) > 10 ** (-10):
                c = abs(abs(a - b) / a)  # RELATIVE CONVERGENCE
            else:
                c = abs(a - b)  # ABSOLUTE CONVERGENCE
            if c > threshold:
                print("THE 2 NUMBERS ARE DIFFERENT WITH A " + threshold + " PRECISION")
                print("a=" + str(a))
                print("b=" + str(b))
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                sys.exit()
    else:
        print("ERROR: a & b ARE NOT OF THE SAME TYPE")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        sys.exit()


# ========================================================================================
# ========================================================================================
def commutator(a, b, spin, c):
    # THIS FUNCTION CHECKS THE USUAL (ANTI)COMMUTATION RELATIONS BETWEEN
    # OPERATORS, MAKING USE OF THE FUNCTION check_array()
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    if spin == "b":
        print("CHECK COMMUTATION REALATIONS OF BOSON OPERATORS")
        print(" ")
        comm = a * b - b * a
    elif spin == "f":
        print("CHECK ANTI-COMMUTATION REALATIONS OF FERMION OPERATORS")
        print(" ")
        comm = a * b + b * a
    check_array(comm, c)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")


# ========================================================================================
# ========================================================================================
def check_hermitianity(a):
    # CHECK THE HERMITIANITY OF A MATRIX
    print("---------------------------------------------------------------")
    print("CHECK HERMITIANITY")
    b = a.transpose()
    check_array(a, b)
    print("THE MATRIX IS HERMITIAN")
    print("---------------------------------------------------------------")


# ========================================================================================
# ========================================================================================
def store_results(data_file, x_data, new_data):
    # THIS FUNCTION STORES SOME VALUES IN NEW COLUMN OF A FILE
    # WITH ALREADY EXISTING COLUMNS OF RESULTS. THESE WOULD
    # BE THEN EASILY COMPARED IN A PLOT, ALL TOGETHER.
    #    data_file   IS THE NAME OF THE FILE WHERE ALL THE
    #                ALREADY EXISTING SIMULATIONS LIE
    #    x_data      LIST CONTAINING X VALUES (FIRST ENTRY = LABEL OF x AXIS)
    #    new data    IS A LIST CONTAINING ALL THE NEW DATA (y VALUES).
    #                THE FIRST ELEMENT OF THE LIST IS A STRING
    #                LABELING THE NAME OF THE SIMULATION
    # STORE X VALUES
    if not os.path.exists(data_file):
        g = open(data_file, "w+")
        for ii in range(len(x_data)):
            g.write(str(x_data[ii]) + "\n")
        g.close()
    # STORE NEW Y VALUES
    f = open(data_file, "r+")
    line = f.readlines()
    f.close()
    h = open(data_file, "w+")
    for ii in range(len(line)):
        line[ii] = line[ii].rstrip()
        h.write(line[ii] + "," + str(new_data[ii]) + "\n")
    h.close()


# ========================================================================================
# ========================================================================================
#                                    SNAKE MAPPING
# ========================================================================================
# ========================================================================================
def snake(n, d):
    # GIVEN THE 1D POINT OF THE SNAKE CURVE IN A nxn DISCRETE LATTICE
    # IT PROVIDES THE CORRESPONDING 2D COORDINATES (x,y) OF THE POINT
    # NOTE: THE SNAKE CURVE IS BUILT BY ALWAYS COUNTING FROM 0 (NOT 1)
    #       HENCE THE POINTS OF THE 1D CURVE START FROM 0 TO (n**2)-1
    #       AND THE COORD.S x AND y ARE SUPPOSED TO GO FROM 0 TO n-1
    #       FOR MATTER OF CODE AT THE END OF THE PROCEEDING EITHER
    #       THE COORDS (x,y) EITHER THE POINTS OF THE CURVE HAVE TO
    #       BE SHIFTED BY ADDING 1
    if d == 0:
        x = 0
        y = 0
    elif d < n:
        y = 0
        x = d
    else:
        # COMPUTE THE REST OF THE DIVISION
        tmp1 = d % n
        # COMPUTE THE INTEGER PART OF THE DIVISION
        tmp2 = int(d / n)
        tmp3 = (tmp2 + 1) % 2
        y = tmp2
        if tmp3 == 0:
            x = n - 1 - tmp1
        else:
            x = tmp1
    return x, y


# ========================================================================================
# ========================================================================================
def inverse_snake(n, x, y):
    # INVERSE SNAKE CURVE MAPPING (from coords to the 1D points)
    # NOTE: GIVEN THE SIZE L of A SQUARE LATTICE, THE COORDS X,Y HAS TO
    #       START FROM 0 AND ARRIVE TO L-1. AT THE END, THE POINTS OF THE
    #       SNAKE CURVE START FROM 0. ADD 1 IF YOU WANT TO START FROM 1
    d = 0
    tmp1 = (y + 1) % 2
    # notice that the first (and hence odd) column is the 0^th column
    if tmp1 == 0:
        # EVEN COLUMNS (1,3,5...n-1)
        d = (y * n) + n - 1 - x
    else:
        # ODD COLUMNS (0,2,4,...n-2)
        d = (y * n) + x
    return d


# ========================================================================================
# ========================================================================================
#                                   HILBERT MAPPING
# ========================================================================================
# ========================================================================================
def regions(num, x, y, s):
    if num == 0:
        # BOTTOM LEFT: CLOCKWISE ROTATE THE COORDS (x,y) OF 90 DEG
        #              THE ROTATION MAKES (x,y) INVERT (y,x)
        t = x
        x = y
        y = t
    elif num == 1:
        # TOP LEFT: TRANSLATE UPWARDS (x,y) OF THE PREVIOUS LEVE
        x = x
        y = y + s
    elif num == 2:
        # TOP RIGHT: TRANSLATE UPWARDS AND RIGHTFORWARD (x,y)
        x = x + s
        y = y + s
    elif num == 3:
        # BOTTOM RIGHT: COUNTER CLOCKWISE ROTATE OF 90 DEG THE (x,y)
        t = x
        x = (s - 1) - y + s
        y = (s - 1) - t
    return x, y


# ========================================================================================
# ========================================================================================
def bitconv(num):
    # GIVEN THE POSITION OF THE HILBERT CURVE IN A 2x2 SQUARE,
    # IT RETURNS THE CORRESPONDING PAIR OF COORDINATES (rx,ry)
    if num == 0:
        # BOTTOM LEFT
        rx = 0
        ry = 0
    elif num == 1:
        # TOP LEFT
        rx = 0
        ry = 1
    elif num == 2:
        # TOP RIGHT
        rx = 1
        ry = 1
    elif num == 3:
        # BOTTOM RIGHT
        rx = 1
        ry = 0
    return rx, ry


# ========================================================================================
# ========================================================================================
def hilbert(n, d):
    # MAPPING THE POSITION d OF THE HILBERT CURVE
    # LIVING IN A nxn SQUARE LATTIVE INTO THE
    # CORRESPONDING 2D (x,y) COORDINATES
    # OF A S
    s = 1  # FIX THE INITIAL LEVEL OF DESCRIPTION
    n1 = d & 3  # FIX THE 2 BITS CORRESPONDING TO THE LEVEL
    x = 0
    y = 0
    # CONVERT THE POSITION OF THE POINT IN THE CURVE AT LEVEL 0 INTO
    # THE CORRESPONDING (x,y) COORDINATES
    x, y = bitconv(n1)
    s *= 2  # UPDATE THE LEVEL OF DESCRIPTION
    tmp = d  # COPY THE POINT d OF THE HILBERT CURVE
    while s < n:
        tmp = tmp >> 2  # MOVE TO THE RIGHT THE 2 BITS OF THE POINT dÅ
        n2 = tmp & 3  # FIX THE 2 BITS CORRESPONDING TO THE LEVEL
        x, y = regions(n2, x, y, s)  # UPDATE THE COORDINATES OF THAT LEVEL
        s *= 2  # UPDATE THE LEVEL OF DESCRIPTION
        s = int(s)
    return x, y


# ========================================================================================
# ========================================================================================
def inverse_regions(num, x, y, s):
    if num == 0:
        # BOTTOM LEFT
        t = x
        x = y
        y = t
    elif num == 1:
        # TOP LEFT
        x = x
        y = y - s
    elif num == 2:
        # TOP RIGHT
        x = x - s
        y = y - s
    elif num == 3:
        # BOTTOM RIGHT
        tt = x
        x = (s - 1) - y
        y = (s - 1) - tt + s
    return x, y


# ========================================================================================
# ========================================================================================
def inverse_bitconv(rx, ry):
    # GIVEN A PAIR OF COORDINATES (x,y) IN A 2x2 LATTICE, IT
    # RETURNS THE POINT num OF THE CORRESPONDING HILBERT CURVE
    if rx == 0:
        if ry == 0:
            # BOTTOM LEFT
            num = 0
        elif ry == 1:
            # TOP LEFT
            num = 1
    elif rx == 1:
        if ry == 0:
            # BOTTOM RIGHT
            num = 3
        elif ry == 1:
            # TOP RIGHT
            num = 2
    return num


# ========================================================================================
# ========================================================================================
def inverse_hilbert(n, x, y):
    # MAPPING THE 2D (x,y) OF A nxn SQUARE INTO THE POSITION d
    # OF THE HILBERT CURVE. REMEMBER THAT THE FINAL POINT
    # HAS TO BE SHIFTED BY 1
    d = 0
    n0 = 0
    s = int(n / 2)
    while s > 1:
        rx = int(x / s)
        ry = int(y / s)
        n0 = inverse_bitconv(rx, ry)
        x, y = inverse_regions(n0, x, y, s)
        d += n0
        d = d << 2
        s /= 2
        s = int(s)
    n0 = inverse_bitconv(x, y)
    d += n0
    return d


# ========================================================================================
# ========================================================================================
