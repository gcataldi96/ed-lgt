from simsio import logger
from itertools import product
import numpy as np
import pickle
from .mappings_1D_2D import inverse_zig_zag

__all__ = ["save_dictionary", "load_dictionary", "get_energy_density"]


def save_dictionary(dict, filename):
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(dict, outp, pickle.HIGHEST_PROTOCOL)
    outp.close


def load_dictionary(filename):
    with open(filename, "rb") as outp:
        return pickle.load(outp)


def get_energy_density(
    tot_energy,
    lvals,
    penalty,
    border_penalty=False,
    link_penalty=False,
    plaquette_penalty=False,
    has_obc=True,
):
    # ACQUIRE LATTICE DIMENSIONS
    Lx = lvals[0]
    Ly = lvals[1]
    n_borders = 0
    n_links = 0
    n_plaquettes = 0
    if has_obc:
        if border_penalty:
            n_borders += 2 * Lx + 2 * Ly
        if link_penalty:
            n_links += Lx * (Ly - 1) + Ly * (Lx - 1)
        if plaquette_penalty:
            n_plaquettes += (Lx - 1) * (Ly - 1)
    else:
        if link_penalty:
            n_links += 2 * (Lx * Ly)
        if plaquette_penalty:
            n_plaquettes += Lx * Ly
    # COUNTING THE TOTAL NUMBER OF PENALTIES
    n_penalties = n_borders + n_links + n_plaquettes
    # RESCALE ENERGY
    energy_density = (tot_energy - n_penalties * penalty) / (Lx * Ly)
    return energy_density
