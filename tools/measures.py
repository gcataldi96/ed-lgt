import numpy as np
from .mappings_1D_2D import zig_zag
from simsio import logger

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
