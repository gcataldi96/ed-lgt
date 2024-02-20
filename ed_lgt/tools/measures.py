import numpy as np
from math import prod
from itertools import product
from scipy.linalg import eigh
from .mappings_1D_2D import zig_zag

__all__ = ["structure_factor", "get_charge", "get_density"]


def structure_factor(corr, lvals):
    # DEFINE THE BRILLUOIN ZONE
    Lx = lvals[0]
    Ly = lvals[1]
    str_factor = np.zeros((Lx, Ly), dtype=complex)
    for nx in range(Lx):
        for ny in range(Ly):
            kx = 2 * np.pi * nx / Lx
            ky = 2 * np.pi * ny / Ly
            str_factor[nx, ny] = single_structure_factor(lvals, kx, ky, corr)
    return str_factor


def single_structure_factor(lvals, kx, ky, corr):
    # DEFINE THE BRILLUOIN ZONE
    Lx = lvals[0]
    Ly = lvals[1]
    n_sites = Lx * Ly
    counter = 0
    sum = 0.0
    for i in range(n_sites):
        for j in range(n_sites):
            if i != j:
                counter += 1
                # get the coordinates of the lattice points
                ix, iy = zig_zag(Lx, Ly, i)
                jx, jy = zig_zag(Lx, Ly, j)
                exp_factor = kx * (ix - jx) + ky * (iy - jy)
                sum += np.exp(complex(0.0, 1.0) * exp_factor) * corr[ix, iy, jx, jy]
    return sum / counter


def get_density(N_plus, N_minus):
    return N_plus - N_minus + 2


def get_charge(N_plus, N_minus):
    return N_plus + N_minus - 2


def get_energy_gap(self, parity_sector):
    if parity_sector:
        Den = get_Q_operator(self.lvals, self.res)
        Num = get_P_operator(self.lvals, self.has_obc[0], self.res, self.coeffs)
    else:
        Den = get_N_operator(self.lvals, self.res)
        Num = get_M_operator(self.lvals, self.has_obc[0], self.res, self.coeffs)
    self.res["th_gap"] = eigh(a=Num, b=Den, eigvals_only=True)[0]


# ====================================================================================
# EXCITATIONS that breaks the parity sectors
def get_M_operator(lvals, has_obc, obs, coeffs):
    n_sites = prod(lvals)
    M = np.zeros((n_sites, n_sites), dtype=complex)
    for ii, jj in product(range(n_sites), repeat=2):
        nn_condition = [
            all([ii > 0, jj == ii - 1]),
            all([ii < n_sites - 1, jj == ii + 1]),
            all([not has_obc, ii == 0, jj == n_sites - 1]),
            all([not has_obc, ii == n_sites - 1, jj == 0]),
        ]
        if any(nn_condition):
            M[ii, jj] += coeffs["J"] * obs["Sz_Sz"][ii, jj]
        elif jj == ii:
            M[ii, jj] += 2 * coeffs["h"] * obs["Sz"][ii]
            if 0 < ii < n_sites - 1 or all(
                [(ii == 0 or ii == n_sites - 1), not has_obc]
            ):
                M[ii, jj] += complex(0, 0.5 * coeffs["J"]) * (
                    obs["Sm_Sx"][ii, (ii + 1) % n_sites]
                    - obs["Sp_Sx"][ii, (ii + 1) % n_sites]
                    + obs["Sx_Sm"][(ii - 1) % n_sites, ii]
                    - obs["Sx_Sp"][(ii - 1) % n_sites, ii]
                )
    return M


def get_N_operator(lvals, obs):
    n_sites = prod(lvals)
    N = np.zeros((n_sites, n_sites), dtype=float)
    for ii in range(n_sites):
        N[ii, ii] += obs["Sz"][ii]
    return N


# ====================================================================================
# Parity preserving excitations


def get_P_operator(lvals, has_obc, obs, coeffs):
    n_sites = prod(lvals)
    B = np.zeros((n_sites, n_sites), dtype=complex)

    for ii, jj in product(range(n_sites), repeat=2):
        if ii == jj - 1 and any(
            [
                0 < jj < n_sites - 1,
                jj == 0 and not has_obc,
                ii == n_sites - 1 and not has_obc,
            ]
        ):
            B[ii, jj] += complex(2 * coeffs["J"], 0) * (
                obs["Sz_Sp_Sm"][ii % n_sites, jj, (jj + 1) % n_sites]
                - obs["Sm_Sp_Sz"][ii % n_sites, jj, (jj + 1) % n_sites]
            )
            B[jj, ii] = -complex(0, 1) * B[ii, jj]
        if ii == jj - 2 and any(
            [
                1 < jj < n_sites - 1,
                jj <= 1 and not has_obc,
                ii == n_sites - 1 and not has_obc,
            ]
        ):
            B[ii, jj] += (
                -coeffs["J"]
                * obs["Sm_Sz_Sz_Sm"][
                    ii % n_sites, (ii + 1) % n_sites, jj, (jj + 1) % n_sites
                ]
            )
            B[jj, ii] = -complex(0, 1) * B[ii, jj]
        elif ii == jj:
            if any([ii < n_sites - 1, ii == n_sites - 1 and not has_obc]):
                B[ii, jj] += 4 * (
                    -coeffs["J"] * obs["Sm_Sp"][ii, (ii + 1) % n_sites]
                    + coeffs["h"] * obs["Sz_Sz"][ii, (ii + 1) % n_sites]
                )
            if any(
                [
                    0 < ii < n_sites - 1,
                    ii == 0 and not has_obc,
                    ii == n_sites - 1 and not has_obc,
                ]
            ):
                B[ii, jj] += complex(0, 2 * coeffs["J"]) * (
                    obs["Sx_Sm_Sz"][(ii - 1) % n_sites, ii, (ii + 1) % n_sites]
                )
            if any([ii < n_sites - 2, ii >= n_sites - 2 and not has_obc]):
                B[ii, jj] += complex(0, 2 * coeffs["J"]) * (
                    obs["Sz_Sp_Sx"][ii, (ii + 1) % n_sites, (ii + 2) % n_sites]
                )

    Q = 0.5 * (B + np.conjugate(B.T))
    return Q


def get_Q_operator(lvals, obs):
    n_sites = prod(lvals)
    P = np.zeros((n_sites, n_sites), dtype=float)
    for ii in range(n_sites):
        P[ii, ii] += (
            obs["SpSm_Sz"][ii, (ii + 1) % n_sites]
            - obs["Sz_SpSm"][ii, (ii + 1) % n_sites]
        )
    return P
