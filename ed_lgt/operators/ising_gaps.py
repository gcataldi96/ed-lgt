import numpy as np
from math import prod
from itertools import product
from scipy.linalg import eigh

__all__ = [
    "energy_gap",
    "get_M_operator",
    "get_N_operator",
    "get_P_operator",
    "get_Q_operator",
]


def energy_gap(lvals, has_obc, h_terms, coeffs):
    Q = get_Q_operator(lvals, has_obc, h_terms, degree=2)
    P = get_P_operator(lvals, has_obc, h_terms, coeffs, degree=2)
    w = {}
    for ii, jj in product(range(2), repeat=2):
        w[f"{ii}{jj}"] = eigh(a=P[ii, jj], b=Q[ii, jj], eigvals_only=True)
        print(w[f"{ii}{jj}"][0])
    return w


def get_N_operator(lvals, ht):
    n = prod(lvals)
    N = np.zeros((n, n), dtype=float)
    for ii in range(n):
        N[ii, ii] += ht["Sz"].obs[ii]
    return N


def get_M_operator(lvals, has_obc, ht, coeffs):
    n = prod(lvals)
    M = np.zeros((n, n), dtype=complex)
    for ii, jj in product(range(n), repeat=2):
        nn_condition = [
            all([ii > 0, jj == ii - 1]),
            all([ii < n - 1, jj == ii + 1]),
            all([not has_obc, ii == 0, jj == n - 1]),
            all([not has_obc, ii == n - 1, jj == 0]),
        ]
        if any(nn_condition):
            M[ii, jj] += coeffs["J"] * ht["Sz_Sz"].corr[ii, jj]
        elif jj == ii:
            M[ii, jj] += 2 * coeffs["h"] * ht["Sz"].obs[ii]
            if 0 < ii < n - 1 or all([(ii == 0 or ii == n - 1), not has_obc]):
                M[ii, jj] += complex(0, 0.5 * coeffs["J"]) * (
                    ht["Sm_Sx"].corr[ii, (ii + 1) % n]
                    - ht["Sp_Sx"].corr[ii, (ii + 1) % n]
                    + ht["Sx_Sm"].corr[(ii - 1) % n, ii]
                    - ht["Sx_Sp"].corr[(ii - 1) % n, ii]
                )
    return M


def get_Q_operator(lvals, has_obc, ht, degree=2):
    n = prod(lvals)
    Q = np.zeros((degree, degree), dtype=object)
    for alpha in range(degree):
        for beta in range(degree):
            Q[alpha, beta] = np.zeros((n, n), dtype=float)
            # ---------------------- CASE Q_00 ----------------------------------------
            if alpha == beta == 0:
                Q[alpha, beta] = get_N_operator(lvals, ht)
            # ---------------------- CASE Q_01 ----------------------------------------
            elif alpha == 0 and beta == 1:
                for ii, jj in product(range(n), repeat=2):
                    if jj == ii and any([ii < n - 1, all([ii == n - 1, not has_obc])]):
                        print(ii, jj, "Sz_Sm", ii, (ii + 1) % n)
                        Q[alpha, beta][ii, jj] += ht["Sz_Sm"].corr[ii, (ii + 1) % n]
                    if jj == ii - 1 and any([ii > 0, all([ii == 0, not has_obc])]):
                        print(ii, jj, "Sm_Sz", jj % n, ii)
                        Q[alpha, beta][ii, jj] += ht["Sm_Sz"].corr[jj % n, ii]
            # ---------------------- CASE Q_10 = Q_01 dag -----------------------------
            elif alpha == 1 and beta == 0:
                Q[alpha, beta] = Q[beta, alpha].transpose().conj()
            # ---------------------- CASE Q_11 ----------------------------------------
            elif alpha == 1 and beta == 1:
                for ii, jj in product(range(n), repeat=2):
                    if jj == ii - 1 and any(
                        [
                            0 < ii < n - 1,
                            all([any([ii == 0, ii == n - 1]), not has_obc]),
                        ]
                    ):
                        print(ii, jj, "Sm_Sz_Sp", jj % n, ii, (ii + 1) % n)
                        Q[alpha, beta][ii, jj] += ht["Sm_Sz_Sp"].corr[
                            jj % n, ii, (ii + 1) % n
                        ]
                    if jj == ii and any([ii < n - 1, all([ii == n - 1, not has_obc])]):
                        print(ii, jj, "Sz_Sz", ii, (ii + 1) % n)
                        Q[alpha, beta][ii, jj] += ht["Sz_Sz"].corr[ii, (ii + 1) % n]
                    if jj == ii + 1 and any(
                        [ii < n - 2, all([ii >= n - 2, not has_obc])]
                    ):
                        print(ii, jj, "Sp_Sz_Sm", ii, jj % n, (jj + 1) % n)
                        Q[alpha, beta][ii, jj] += ht["Sp_Sz_Sm"].corr[
                            ii, jj % n, (jj + 1) % n
                        ]
    return Q


def get_P_operator(lvals, has_obc, ht, coeffs, degree=2):
    n = prod(lvals)
    P = np.zeros((degree, degree), dtype=object)
    for alpha in range(degree):
        for beta in range(degree):
            P[alpha, beta] = np.zeros((n, n), dtype=complex)
            # ---------------------- CASE P_00 ----------------------------------------
            if alpha == beta == 0:
                P[alpha, beta] = get_M_operator(
                    lvals=lvals, ht=ht, coeffs=coeffs, has_obc=has_obc
                )
            # ---------------------- CASE P_01 = P_10 ---------------------------------
            elif any([all([alpha == 0, beta == 1]), all([alpha == 1, beta == 0])]):
                for ii, jj in product(range(n), repeat=2):
                    # -------------------------------------------------------------
                    if ii == jj and any([ii < n - 1, ii == n - 1 and not has_obc]):
                        P[alpha, beta][ii, jj] += coeffs["J"] * (
                            ht["Sp_Sz"].corr[jj, (jj + 1) % n]
                            + ht["Sm_Sz"].corr[jj, (jj + 1) % n]
                        ) + coeffs["h"] * (
                            ht["Sz_Sp"].corr[jj, (jj + 1) % n]
                            + ht["Sz_Sm"].corr[jj, (jj + 1) % n]
                        )
                    if ii == jj and any(
                        [
                            0 < ii < n - 1,
                            all([(0 == ii or ii == n - 1), not has_obc]),
                        ]
                    ):
                        P[alpha, beta][ii, jj] += complex(0, 1) * (
                            ht["Sx_Sp_Sp"].corr[(jj - 1) % n, jj, (jj + 1) % n]
                            - ht["Sx_Sm_Sm"].corr[(jj - 1) % n, jj, (jj + 1) % n]
                        )
                    # -------------------------------------------------------------
                    if jj == ii + 1 and any([ii < n - 1, ii == n - 1 and not has_obc]):
                        P[alpha, beta][ii, jj] += (
                            -coeffs["J"] * ht["Sz_Sz"].corr[ii, jj % n]
                        )
                    # -------------------------------------------------------------
                    if jj == ii - 2 and any([ii > 1, ii <= 1 and not has_obc]):
                        P[alpha, beta][ii, jj] += -complex(coeffs["J"] * 0.5, 0) * (
                            ht["Sp_Sz_Sz"].corr[jj % n, (jj + 1) % n, ii]
                            + ht["Sm_Sz_Sz"].corr[jj % n, (jj + 1) % n, ii]
                        )
                    # -------------------------------------------------------------
                    if jj == ii - 1 and any([0 < ii, (0 == ii and not has_obc)]):
                        P[alpha, beta][ii, jj] += coeffs["J"] * (
                            ht["Sz_Sp"].corr[jj % n, ii] + ht["Sz_Sm"].corr[jj % n, ii]
                        )
                    # -------------------------------------------------------------
                    if jj == ii - 1 and any(
                        [0 < ii < n - 1, all([0 == ii or ii == n - 1, not has_obc])]
                    ):
                        P[alpha, beta][ii, jj] += complex(0, coeffs["J"]) * (
                            ht["Sp_Sz_Sx"].corr[jj % n, ii, (ii + 1) % n]
                            - ht["Sm_Sz_Sx"].corr[jj % n, ii, (ii + 1) % n]
                        )
            # ---------------------- CASE P_11 ----------------------------------------
            elif alpha == beta == 1:
                for ii, jj in product(range(n), repeat=2):
                    # -------------------------------------------------------------
                    if jj == ii - 2 and any(
                        [
                            1 < ii < n - 1,
                            (ii <= 1 or ii == n - 1) and not has_obc,
                        ]
                    ):
                        P[alpha, beta][ii, jj] += -complex(0.5 * coeffs["J"], 0) * (
                            ht["Sp_Sz_Sz_Sm"].corr[
                                jj % n, (jj + 1) % n, ii, (ii + 1) % n
                            ]
                            + ht["Sm_Sz_Sz_Sp"].corr[
                                jj % n, (jj + 1) % n, ii, (ii + 1) % n
                            ]
                        )
                    # -------------------------------------------------------------
                    if jj == ii - 1 and any(
                        [
                            0 < ii < n - 1,
                            (0 == ii or ii == n - 1) and not has_obc,
                        ]
                    ):
                        P[alpha, beta][ii, jj] += coeffs["J"] * (
                            ht["Sz_Sp_Sm"].corr[jj % n, ii, (ii + 1) % n]
                            + ht["Sz_Sm_Sp"].corr[jj % n, ii, (ii + 1) % n]
                            + ht["Sp_Sp_Sz"].corr[jj % n, ii, (ii + 1) % n]
                            + ht["Sm_Sm_Sz"].corr[jj % n, ii, (ii + 1) % n]
                        ) + 2 * coeffs["h"] * (
                            ht["Sp_Sz_Sm"].corr[jj % n, ii, (ii + 1) % n]
                            + ht["Sm_Sz_Sp"].corr[jj % n, ii, (ii + 1) % n]
                        )
                    # -------------------------------------------------------------
                    if jj == ii - 1 and any(
                        [
                            0 < ii < n - 2,
                            (0 == ii or ii >= n - 2) and not has_obc,
                        ]
                    ):
                        P[alpha, beta][ii, jj] += -complex(0, 0.5 * coeffs["J"]) * (
                            ht["Sp_Sz_Sz_Sx"].corr[
                                jj % n, (jj + 1) % n, (ii + 1) % n, (ii + 2) % n
                            ]
                            - ht["Sm_Sz_Sz_Sx"].corr[
                                jj % n, (jj + 1) % n, (ii + 1) % n, (ii + 2) % n
                            ]
                        )
                    # -------------------------------------------------------------
                    if jj == ii and any(
                        [
                            0 < ii < n - 1,
                            (0 == ii or ii == n - 1) and not has_obc,
                        ]
                    ):
                        P[alpha, beta][ii, jj] += complex(0, coeffs["J"]) * (
                            ht["Sx_Sp_Sz"].corr[(ii - 1) % n, ii, (ii + 1) % n]
                            - ht["Sx_Sm_Sz"].corr[(ii - 1) % n, ii, (ii + 1) % n]
                        )
                    # -------------------------------------------------------------
                    if jj == ii and any([ii < n - 2, ii >= n - 2 and not has_obc]):
                        P[alpha, beta][ii, jj] += complex(0, coeffs["J"]) * (
                            +ht["Sz_Sp_Sx"].corr[ii, (ii + 1) % n, (ii + 2) % n]
                            - ht["Sz_Sm_Sx"].corr[ii, (ii + 1) % n, (ii + 2) % n]
                        )
                    # -------------------------------------------------------------
                    if jj == ii and any([ii < n - 1, ii >= n - 1 and not has_obc]):
                        P[alpha, beta][ii, jj] += (
                            complex(0, coeffs["J"])
                            * (
                                ht["Sp_Sp"].corr[ii, (ii + 1) % n]
                                + ht["Sm_Sm"].corr[ii, (ii + 1) % n]
                            )
                            + 4 * coeffs["h"] * ht["Sz_Sz"].corr[ii, (ii + 1) % n]
                        )
                    # -------------------------------------------------------------
                    if jj == ii + 1 and any([ii < n - 2, ii >= n - 2 and not has_obc]):
                        P[alpha, beta][ii, jj] += (
                            complex(0, 0.5 * coeffs["J"])
                            * (
                                ht["Sx_Sz_Sz_Sm"].corr[
                                    (ii - 1) % n, ii, jj % n, (jj + 1) % n
                                ]
                                - ht["Sx_Sz_Sz_Sp"].corr[
                                    (ii - 1) % n, ii, jj % n, (jj + 1) % n
                                ]
                                + 2
                                * (
                                    ht["Sp_Sm_Sz"].corr[ii, jj % n, (jj + 1) % n]
                                    - ht["Sm_Sp_Sz"].corr[ii, jj % n, (jj + 1) % n]
                                )
                            )
                            + coeffs["J"]
                            * (
                                ht["Sz_Sm_Sm"].corr[ii, jj % n, (jj + 1) % n]
                                - ht["Sz_Sp_Sp"].corr[ii, jj % n, (jj + 1) % n]
                            )
                            + (2 * coeffs["h"])
                            * (
                                ht["Sp_Sz_Sm"].corr[ii, jj % n, (jj + 1) % n]
                                + ht["Sm_Sz_Sp"].corr[ii, jj % n, (jj + 1) % n]
                            )
                        )
                    # -------------------------------------------------------------
                    if jj == ii + 2 and any([ii < n - 3, ii >= n - 3 and not has_obc]):
                        P[alpha, beta][ii, jj] += -(0.5 * coeffs["J"]) * (
                            ht["Sp_Sz_Sz_Sm"].corr[
                                ii, (ii + 1) % n, jj % n, (jj + 1) % n
                            ]
                            + ht["Sm_Sz_Sz_Sp"].corr[
                                ii, (ii + 1) % n, jj % n, (jj + 1) % n
                            ]
                        )
    return P
