import numpy as np
from math import prod

__all__ = ["get_M_operator", "get_N_operator", "get_P_operator", "get_Q_operator"]


def get_N_operator(lvals, ht):
    n = prod(lvals)
    N = np.zeros((n, n), dtype=float)
    for ii in range(n):
        N[ii, ii] += ht["Sz"].obs[ii]
    return N


def get_M_operator(lvals, ht, coeffs, has_obc):
    n = prod(lvals)
    M = np.zeros((n, n), dtype=complex)
    for ii in range(n):
        for jj in range(n):
            nn_condition = [
                (ii > 0) and (jj == ii - 1),
                (ii < n - 1) and (jj == ii + 1),
                np.all([not has_obc, ii == 0, jj == n - 1]),
                np.all([not has_obc, ii == n - 1, jj == 0]),
            ]
            if np.any(nn_condition):
                M[ii, jj] += coeffs["J"] * ht["Sz_Sz"].corr[ii, jj]
            elif jj == ii:
                M[ii, jj] += 2 * coeffs["h"] * ht["Sz"].obs[ii]
                if ii < n - 1 and ii > 0:
                    M[ii, jj] += (
                        complex(0, 0.5)
                        * coeffs["J"]
                        * (
                            ht["Sm_Sx"].corr[ii, ii + 1]
                            - ht["Sp_Sx"].corr[ii, ii + 1]
                            + ht["Sx_Sm"].corr[ii - 1, ii]
                            - ht["Sx_Sp"].corr[ii - 1, ii]
                        )
                    )
                elif ii == n - 1 and not has_obc:
                    M[ii, jj] += (
                        complex(0, 0.5)
                        * coeffs["J"]
                        * (ht["Sm_Sx"].corr[ii, 0] - ht["Sp_Sx"].corr[ii, 0])
                    )
                elif ii == 0 and not has_obc:
                    M[ii, jj] += (
                        complex(0, 0.5)
                        * coeffs["J"]
                        * (ht["Sx_Sm"].corr[n - 1, ii] - ht["Sx_Sp"].corr[n - 1, ii])
                    )
    return M


def get_Q_operator(lvals, ht, has_obc, degree=2):
    n = prod(lvals)
    Q = np.zeros((degree, degree), dtype=object)
    for alpha in range(degree):
        for beta in range(degree):
            Q[alpha, beta] = np.zeros((n, n), dtype=float)
            # ---------------------- CASE Q_00 ----------------------------------------
            if alpha == beta == 0:
                Q[alpha, beta] += get_N_operator(lvals, ht)
            # ---------------------- CASE Q_01 ----------------------------------------
            elif alpha == 0 and beta == 1:
                for ii in range(n):
                    for jj in range(n):
                        if jj == ii < n - 1:
                            Q[alpha, beta][ii, jj] += ht["Sm_Sz"].corr[ii, jj + 1]
                        elif jj == ii == n - 1 and not has_obc:
                            Q[alpha, beta][ii, jj] += ht["Sm_Sz"].corr[ii, 0]
                        elif jj == ii - 1 and ii > 0:
                            Q[alpha, beta][ii, jj] += ht["Sm_Sz"].corr[jj, ii]
                        elif jj == ii - 1 and ii == 0 and not has_obc:
                            Q[alpha, beta][ii, jj] += ht["Sm_Sz"].corr[n - 1, ii]
            # ---------------------- CASE Q_10 = Q_01 dag -----------------------------
            elif alpha == 1 and beta == 0:
                Q[alpha, beta] = Q[beta, alpha].transpose()
            # ---------------------- CASE Q_11 ----------------------------------------
            elif alpha == 1 and beta == 1:
                for ii in range(n):
                    for jj in range(n):
                        if jj == ii - 1 and ii > 0:
                            Q[alpha, beta][ii, jj] += ht["Sm_Sz_Sp"].corr[
                                jj, ii, ii + 1
                            ]
                        elif jj == ii - 1 and ii == 0 and not has_obc:
                            Q[alpha, beta][ii, jj] += ht["Sm_Sz_Sp"].corr[
                                n - 1, ii, ii + 1
                            ]
                        elif jj == ii < n - 1:
                            Q[alpha, beta][ii, jj] += ht["Sz_Sz"].corr[ii, ii + 1]
                        elif jj == ii == n - 1 and not has_obc:
                            Q[alpha, beta][ii, jj] += ht["Sz_Sz"].corr[ii, 0]
                        elif jj == ii + 1 and ii < n - 2:
                            Q[alpha, beta][ii, jj] += ht["Sp_Sz_Sz"].corr[
                                ii, jj, jj + 1
                            ]
                        elif jj == ii + 1 and ii == n - 2 and not has_obc:
                            Q[alpha, beta][ii, jj] += ht["Sp_Sz_Sz"].corr[ii, jj, 0]
                        elif jj == ii + 1 and ii == n - 1 and not has_obc:
                            Q[alpha, beta][ii, jj] += ht["Sp_Sz_Sz"].corr[ii, 0, 1]
    return Q


def get_P_operator(lvals, has_obc, ht, coeffs, degree=2):
    n = prod(lvals)
    P = np.zeros((degree, degree), dtype=object)
    for alpha in range(degree):
        for beta in range(degree):
            P[alpha, beta] = np.zeros((n, n), dtype=complex)
            # ---------------------- CASE P_00 ----------------------------------------
            if alpha == beta == 0:
                P[alpha, beta] += get_M_operator(lvals, ht, coeffs, has_obc)
            # ---------------------- CASE P_01 = P_10 ---------------------------------
            elif (alpha == 0 and beta == 1) or (alpha == 1 and beta == 0):
                for ii in range(n):
                    for jj in range(n):
                        # -------------------------------------------------------------
                        if ii == jj and any[ii < n - 1, ii == n - 1 and not has_obc]:
                            P[alpha, beta][ii, jj] += (
                                complex(0, 2 * coeffs["J"])
                                * (
                                    ht["Sm_Sm"].corr[jj, (jj + 1) % n]
                                    - ht["Sp_Sp"].corr[jj, (jj + 1) % n]
                                )
                                + coeffs["J"]
                                * (
                                    ht["Sp_Sz"].corr[jj, (jj + 1) % n]
                                    + ht["Sm_Sz"].corr[jj, (jj + 1) % n]
                                )
                                + coeffs["h"]
                                * (
                                    ht["Sz_Sp"].corr[jj, (jj + 1) % n]
                                    + ht["Sz_Sm"].corr[jj, (jj + 1) % n]
                                )
                            )
                        if (
                            ii == jj
                            and any[
                                0 < ii < n - 1,
                                (0 == ii or ii == n - 1) and not has_obc,
                            ]
                        ):
                            P[alpha, beta][ii, jj] += complex(0, 1) * (
                                ht["Sx_Sp_Sp"].corr[(jj - 1) % n, ii, (jj + 1) % n]
                                - ht["Sx_Sm_Sm"].corr[(jj - 1) % n, ii, (jj + 1) % n]
                            )
                        # -------------------------------------------------------------
                        if (
                            jj == ii + 1
                            and any[jj < n - 1, jj >= n - 1 and not has_obc]
                        ):
                            P[alpha, beta][ii, jj] += (
                                ht["Sz_Sp_Sp"].corr[ii, jj % n, (jj + 1) % n]
                                - ht["Sz_Sm_Sm"].corr[ii, jj % n, (jj + 1) % n]
                                - coeffs["J"] * ht["Sz_Sz"].corr[ii, jj]
                            )
                        # -------------------------------------------------------------
                        if jj == ii - 2 and any[ii > 1, ii <= 1 and not has_obc]:
                            P[alpha, beta][ii, jj] += (
                                -coeffs["J"]
                                * 0.5
                                * (
                                    ht["Sp_Sz_Sz"].corr[jj % n, (jj + 1) % n, ii]
                                    + ht["Sm_Sz_Sz"].corr[jj % n, (jj + 1) % n, ii]
                                )
                            )
                        # -------------------------------------------------------------
                        elif (
                            jj == ii - 1
                            and any[
                                0 < ii < n - 1,
                                (0 == ii or ii == n - 1) and not has_obc,
                            ]
                        ):
                            P[alpha, beta][ii, jj] += coeffs["J"] * (
                                ht["Sz_Sp"].corr[jj % n, ii]
                                + ht["Sz_Sp"].corr[jj % n, ii]
                                + complex(0, 1)
                                * (
                                    ht["Sp_Sz_Sx"].corr[jj % n, ii, (ii + 1) % n]
                                    - ht["Sm_Sz_Sx"].corr[jj % n, ii, (ii + 1) % n]
                                )
                            )
            # ---------------------- CASE P_11 ----------------------------------------
            elif alpha == beta == 1:
                for ii in range(n):
                    for jj in range(n):
                        if (
                            jj == ii - 2
                            and any[
                                1 < ii < n - 1,
                                (1 == ii or ii == n - 1) and not has_obc,
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
                        if (
                            jj == ii - 1
                            and any[
                                0 < ii < n - 1,
                                (0 == ii or ii == n - 1) and not has_obc,
                            ]
                        ):
                            P[alpha, beta][ii, jj] += coeffs["J"] * (
                                ht["Sz_Sp_Sm"].corr[jj % n, ii, (ii + 1) % n]
                                + ht["Sz_Sm_Sp"].corr[jj % n, ii, (ii + 1) % n]
                                + ht["Sp_Sp_Sz"].corr[jj % n, ii, (ii + 1) % n]
                                + ht["Sm_Sm_Sz"].corr[jj % n, ii, (ii + 1) % n]
                                - complex(0, 0.5)
                                * (
                                    ht["Sp_Sz_Sz_Sx"].corr[
                                        jj % n, (jj + 1) % n, (ii + 1) % n, (ii + 2) % n
                                    ]
                                    + ht["Sm_Sz_Sz_Sx"].corr[
                                        jj % n, (jj + 1) % n, (ii + 1) % n, (ii + 2) % n
                                    ]
                                )
                            ) + 2 * coeffs["h"] * (
                                ht["Sp_Sz_Sm"].corr[jj % n, ii, (ii + 1) % n]
                                + ht["Sm_Sz_Sp"].corr[jj % n, ii, (ii + 1) % n]
                            )
                        if (
                            jj == ii
                            and any[
                                0 < ii < n - 1,
                                (0 == ii or ii == n - 1) and not has_obc,
                            ]
                        ):
                            P[alpha, beta][ii, jj] += complex(0, coeffs["J"]) * (
                                ht["Sx_Sp_Sz"].corr[(ii - 1) % n, ii, (ii + 1) % n]
                                - ht["Sx_Sm_Sz"].corr[(ii - 1) % n, ii, (ii + 1) % n]
                            )
                        if jj == ii and any[ii < n - 1, ii == n - 1 and not has_obc]:
                            P[alpha, beta][ii, jj] += (
                                complex(0, coeffs["J"])
                                * (
                                    +ht["Sz_Sp_Sx"].corr[ii, (ii + 1) % n, (ii + 2) % n]
                                    - ht["Sz_Sm_Sx"].corr[
                                        ii, (ii + 1) % n, (ii + 2) % n
                                    ]
                                    + ht["Sp_Sp"].corr[ii, (ii + 1) % n]
                                    + ht["Sm_Sm"].corr[ii, (ii + 1) % n]
                                )
                                + 4 * coeffs["h"] * ht["Sz_Sz"].corr[ii, (ii + 1) % n]
                            )
                        if (
                            jj == ii + 1
                            and any[ii < n - 1, ii == n - 1 and not has_obc]
                        ):
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
                        if (
                            jj == ii + 2
                            and any[ii < n - 2, ii >= n - 2 and not has_obc]
                        ):
                            P[alpha, beta][ii, jj] += -(0.5 * coeffs["J"]) * (
                                ht["Sp_Sz_Sz_Sm"].corr[
                                    ii, (ii + 1) % n, jj % n, (jj + 1) % n
                                ]
                                + ht["Sm_Sz_Sz_Sp"].corr[
                                    ii, (ii + 1) % n, jj % n, (jj + 1) % n
                                ]
                            )
    return P
