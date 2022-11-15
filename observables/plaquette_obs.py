import numpy as np
from simsio import logger

from tools import inverse_zig_zag, zig_zag
from modeling import four_body_op

__all__ = ["PlaquetteObs"]


class PlaquetteObs:
    def __init__(self, psi, nx, ny, Plaq, has_obc=True, get_imag=False):
        # CHECK ON TYPES
        if not isinstance(psi, np.ndarray):
            raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
        if not np.isscalar(nx) and not isinstance(nx, int):
            raise TypeError(f"nx must be a SCALAR & INTEGER, not a {type(nx)}")
        if not np.isscalar(ny) and not isinstance(ny, int):
            raise TypeError(f"ny must be a SCALAR & INTEGER, not a {type(ny)}")
        if not isinstance(has_obc, bool):
            raise TypeError(f"has_obc should be a BOOL, not a {type(has_obc)}")
        if not isinstance(get_imag, bool):
            raise TypeError(f"get_imag should be a BOOL, not a {type(get_imag)}")

        # ADVERTISE OF THE CHOSEN PART OF THE PLAQUETTE YOU WANT TO COMPUTE
        logger.info(f"      -----------------------------")
        if get_imag:
            chosen_part = "IMAG"
        else:
            chosen_part = "REAL"
        logger.info(f"      PLAQUETTE: {chosen_part} PART")
        logger.info(f"      -----------------------------")
        # Compute the total number of particles
        n = nx * ny
        self.nx = nx
        self.ny = ny
        # Compute the complex_conjugate of the ground state psi
        psi_dag = np.conjugate(psi)
        # Define a list with the Four Operators involved in the Plaquette:
        Op_list = [Plaq.BL, Plaq.BR, Plaq.TL, Plaq.TR]
        # DEFINE A VOCABULARY FOR THE STORED VALUES
        self.plaq_obs = {}
        # DEFINE A VECTOR FOR THE STORED VALUES
        values = np.zeros(n, dtype=float)
        for ii in range(n):
            x, y = zig_zag(nx, ny, ii)
            if x < nx - 1 and y < ny - 1:
                # List of sites where to apply Operators
                sites_list = [ii + 1, ii + 2, ii + nx + 1, ii + nx + 2]
                # DEFINE A STRING LABELING THE PLAQUETTE
                plaq_label = f"({x+1},{y+1})_({x+2},{y+1})_({x+1},{y+2})_({x+2},{y+2})"
            else:
                if not has_obc:
                    if x < nx - 1 and y == ny - 1:
                        # UPPER BORDER
                        jj = inverse_zig_zag(nx, ny, x, 0)
                        # List of Sites where to apply Operators
                        sites_list = [ii + 1, ii + 2, jj + 1, jj + 2]
                        # DEFINE A STRING LABELING THE PLAQUETTE on the UPPER BORDER
                        plaq_label = (
                            f"({x+1},{y+1})_({x+2},{y+1})_({x+1},{1})_({x+2},{1})"
                        )
                    elif x == nx - 1 and y < ny - 1:
                        # RIGHT BORDER
                        # List of Sites where to apply Operators
                        sites_list = [ii + 1, ii + 2 - nx, ii + nx + 1, ii + 2]
                        # DEFINE A STRING LABELING THE PLAQUETTE on the RIGHT BORDER
                        plaq_label = (
                            f"({x+1},{y+1})_({1},{y+1})_({x+1},{y+2})_({1},{y+2})"
                        )
                    else:
                        # TOP RIGHT CORNER
                        # List of Sites where to apply Operators
                        sites_list = [ii + 1, ii + 2 - nx, nx, 1]
                        # DEFINE A STRING LABELING THE PLAQUETTE on the TOP RIGHT CORNER
                        plaq_label = f"({x+1},{y+1})_({1},{y+1})_({x+1},{1})_({1},{1})"
            # COMPUTE THE PLAQUETTE
            self.plaq_obs[f"Plaq_{plaq_label}"] = np.real(
                np.dot(
                    psi_dag,
                    four_body_op(
                        Op_list,
                        sites_list,
                        n,
                        get_only_part=chosen_part,
                    ).dot(psi),
                )
            )
