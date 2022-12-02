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
    PBC=False,
):
    # ACQUIRE LATTICE DIMENSIONS
    Lx = lvals[0]
    Ly = lvals[1]
    n_borders = 0
    n_links = 0
    n_plaquettes = 0
    if not PBC:
        if border_penalty:
            n_borders += 2 * Lx + 2 * Ly
        if link_penalty:
            n_links += Lx * (Lx - 1) + Ly * (Ly - 1)
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


def map_1D_to_2D_LocalObs(L, Obs_1D):
    """
    Convert Obs_1D from a 1D array (len L**2), where sites are ordered acconding
    to the zig_zag curve, to a 2D array with shape of the original lattice (L,L)
    """
    if not isinstance(Obs_1D, np.ndarray):
        raise TypeError(f"Obs_1D must be an ARRAY, not a {type(Obs_1D)}")
    else:
        if Obs_1D.shape[0] != L**2:
            raise ValueError(
                f"Obs_1D.shape[0] must be equal to L**2, not to {Obs_1D.shape[0]}"
            )
    Obs_2D = np.zeros(L**2)
    for ii, (x, y) in enumerate(product(range(L), range(L))):
        point = inverse_zig_zag(L, x, y)
        Obs_2D[ii] = Obs_1D[point]
    return Obs_2D.reshape((L, L))


def map_1D_to_2D_TwoBodyObs(L, Corr1D):
    """
    Convert Corr1D from an array (Lx*Ly, Lx*Ly), where sites are ordered
    acconding to the zig_zag curve to an array (Lx, Ly, Lx, Ly)
    where sites are stored as in the original lattice
    """
    if not isinstance(Corr1D, np.ndarray):
        raise TypeError(f"Corr1D must be an ARRAY, not a {type(Corr1D)}")
    else:
        for ii in range(2):
            if Corr1D.shape[ii] != L**2:
                raise ValueError(
                    f"Corr1D.shape[{ii}] must be L**2, not a {Corr1D.shape[ii]}"
                )
    Corr2D = np.zeros((L**2, L**2))
    for indx1, (x1, y1) in enumerate(product(range(L), range(L))):
        p1 = inverse_zig_zag(L, x1, y1)
        for indx2, (x2, y2) in enumerate(product(range(L), range(L))):
            p2 = inverse_zig_zag(L, x2, y2)
            Corr2D[indx1, indx2] = np.real(Corr1D[p1, p2])
    return Corr2D.reshape(L, L, L, L)


def check_border(Op, border, value=1, threshold=10 ** (-8)):
    tmp = 0
    if border == "mx":
        if np.any(np.abs(Op[0, :] - value) > threshold):
            tmp += 1
            logger.info(Op[0, :])
    elif border == "px":
        if np.any(np.abs(Op[-1, :] - value) > threshold):
            tmp += 1
            logger.info(Op[-1, :])
    if border == "my":
        if np.any(np.abs(Op[:, 0] - value) > threshold):
            tmp += 1
            logger.info(Op[:, 0])
    elif border == "py":
        if np.any(np.abs(Op[:, -1] - value) > threshold):
            tmp += 1
            logger.info(Op[:, -1])
    if tmp > 0:
        logger.info(f"The P_{border} penalty is not satisfied")
    else:
        logger.info(f"The P_{border} penalty is satisfied")
    return tmp


def check_link_symm(L, Corr, value=1, axis="x", threshold=10 ** (-8), has_obc=True):
    tmp = 0
    if axis == "x":
        for y in range(L):
            for x in range(L):
                if x == L - 1:
                    if not has_obc:
                        if np.abs(Corr[x, y, 0, y] - value) > threshold:
                            tmp += 1
                            logger.info(f"W{axis}_({x},{y})-({0},{y})={Corr[x,y,0,y]}")
                    else:
                        continue
                else:
                    if np.abs(Corr[x, y, x + 1, y] - value) > threshold:
                        tmp += 1
                        logger.info(f"W{axis}_({x},{y})-({x+1},{y})={Corr[x,y,x+1,y]}")
    if axis == "y":
        for x in range(L):
            for y in range(L):
                if y == L - 1:
                    if not has_obc:
                        if np.abs(Corr[x, y, x, 0] - value) > threshold:
                            tmp += 1
                            logger.info(f"W{axis}_({x},{y})-({x},{0})={Corr[x,y,x,0]}")
                    else:
                        continue
                else:
                    if np.abs(Corr[x, y, x, y + 1] - value) > threshold:
                        tmp += 1
                        logger.info(f"W{axis}_({x},{y})-({x},{y+1})={Corr[x,y,x,y+1]}")
    if tmp > 0:
        logger.info(f"{tmp} Link Symmetries are not Satisfied")
    else:
        logger.info(f"All the Link Symmetries are satisfied")
    return tmp


def check_sym_sector(Op, value, threshold=10 ** (-8)):
    tmp = 0
    if np.abs(np.sum(Op) - value) > threshold:
        logger.info("The U(1) Symmetry sector is not satisfied")
        logger.info(f"Expected Sector {value} - Actual sector {np.sum(Op)}")
    else:
        logger.info("The U(1) Symmetry sector is satisfied")
    return tmp


def make_checks(params, res):
    if not isinstance(params, dict):
        raise TypeError("params has to be a dict")
    if not isinstance(res, dict):
        raise TypeError("res has to be a dict")
    """
    Make the checks for border penalties, link symmetries and symmetry sectors
    """
    tmp = 0
    # CHECK BORDER PENALTIES
    if params["has_obc"]:
        for border in ["mx", "px", "my", "py"]:
            # MOVE FROM 1D zig_zag ORDERING TO 2D LATTICE
            res[f"P_{border}"] = map_1D_to_2D_LocalObs(
                params["lvals"], res[f"P_{border}"]
            )
            # CHECK THE SINGLE BORDER VALUES
            tmp += check_border(res[f"P_{border}"], border)
    # CHECK LINK SYMMETRIES
    for d in ["x", "y"]:
        # MOVE FROM 1D zig_zag ORDERING TO 2D LATTICE
        res[f"W_{d}_link"] = map_1D_to_2D_TwoBodyObs(
            params["lvals"], res[f"W_{d}_link"]
        )
        # CHECK LINK SYMMETRY
        tmp += check_link_symm(
            params["lvals"],
            res[f"W_{d}_link"],
            axis=d,
            has_obc=params["has_obc"],
        )
    # CHECK SYMMETRY SECTORS
    for sym, value in zip(
        params.get("SymmetryGenerators", []), params.get("SymmetrySectors", [])
    ):
        tmp += check_sym_sector(res[sym], value)
    if tmp > 0:
        logger.info(f"{tmp} constraints are not satisfied")
