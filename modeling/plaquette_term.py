import numpy as np
from scipy.sparse import isspmatrix_csr, isspmatrix, csr_matrix
from tools import zig_zag, inverse_zig_zag
from .qmb_operations import four_body_op
from simsio import logger

__all__ = ["PlaquetteTerm2D"]


class PlaquetteTerm2D:
    def __init__(self, op_list, op_name_list, staggered_basis=False, site_basis=None):
        if not isinstance(op_list, list):
            raise TypeError(f"op_list should be a list, not a {type(op_list)}")
        else:
            for ii, op in enumerate(op_list):
                if not isspmatrix_csr(op):
                    raise TypeError(
                        f"op_list[{ii}] should be a CSR_MATRIX, not {type(op)}"
                    )
        if not isinstance(op_name_list, list):
            raise TypeError(
                f"op_name_list should be a list, not a {type(op_name_list)}"
            )
        else:
            for ii, name in enumerate(op_name_list):
                if not isinstance(name, str):
                    raise TypeError(
                        f"op_name_list[{ii}] should be a STRING, not {type(name)}"
                    )
        self.stag_basis = staggered_basis
        self.site_basis = site_basis
        self.BL = op_list[0]
        self.BR = op_list[1]
        self.TL = op_list[2]
        self.TR = op_list[3]
        self.BL_name = op_name_list[0]
        self.BR_name = op_name_list[1]
        self.TL_name = op_name_list[2]
        self.TR_name = op_name_list[3]
        # logger.info(
        #    f"PLAQUETTE {op_name_list[0]}-{op_name_list[1]}-{op_name_list[2]}-{op_name_list[3]}"
        # )
        # Define a list with the Four Operators involved in the Plaquette:
        self.op_list = [self.BL, self.BR, self.TL, self.TR]

    def get_Hamiltonian(
        self, lvals, strength, has_obc=True, add_dagger=False, mask=None
    ):
        # CHECK ON TYPES
        if not isinstance(lvals, list):
            raise TypeError(f"lvals should be a list, not a {type(lvals)}")
        else:
            for ii, ll in enumerate(lvals):
                if not isinstance(ll, int):
                    raise TypeError(f"lvals[{ii}] should be INTEGER, not {type(ll)}")
        if not np.isscalar(strength):
            raise TypeError(f"strength must be SCALAR not a {type(strength)}")
        if not isinstance(has_obc, bool):
            raise TypeError(f"has_obc must be a BOOL, not a {type(has_obc)}")
        if not isinstance(add_dagger, bool):
            raise TypeError(f"add_dagger must be a BOOL, not a {type(add_dagger)}")
        # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
        nx = lvals[0]
        ny = lvals[1]
        n = nx * ny
        # Define the Hamiltonian
        H_plaq = 0
        for ii in range(n):
            # Compute the corresponding (x,y) coords
            x, y = zig_zag(nx, ny, ii)
            if x < nx - 1 and y < ny - 1:
                # List of Sites where to apply Operators
                sites_list = [ii, ii + 1, ii + nx, ii + nx + 1]
            else:
                if not has_obc:
                    # PERIODIC BOUNDARY CONDITIONS
                    if x < nx - 1 and y == ny - 1:
                        # UPPER BORDER
                        jj = inverse_zig_zag(nx, ny, x, 0)
                        # List of Sites where to apply Operators
                        sites_list = [ii, ii + 1, jj, jj + 1]
                    elif x == nx - 1 and y < ny - 1:
                        # RIGHT BORDER
                        # List of Sites where to apply Operators
                        sites_list = [ii, ii + 1 - nx, ii + nx, ii + 1]
                    else:
                        # UPPER RIGHT CORNER
                        # List of Sites where to apply Operators
                        sites_list = [ii, ii + 1 - nx, nx - 1, 0]
                else:
                    continue
            # Add the Plaquette to the Hamiltonian
            if mask is None:
                mask_conditions = True
            else:
                if mask[x, y] == True:
                    mask_conditions = True
                else:
                    mask_conditions = False
            if mask_conditions:
                # logger.info(sites_list)
                H_plaq += strength * four_body_op(
                    self.op_list,
                    sites_list,
                    lvals,
                    has_obc,
                    self.stag_basis,
                    self.site_basis,
                )
        if not isspmatrix(H_plaq):
            H_plaq = csr_matrix(H_plaq)
        if add_dagger:
            H_plaq += csr_matrix(H_plaq.conj().transpose())
        return H_plaq

    def get_expval(self, psi, lvals, has_obc=True, get_imag=False, site=None):
        # CHECK ON TYPES
        if not isinstance(psi, np.ndarray):
            raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
        if not isinstance(lvals, list):
            raise TypeError(f"lvals should be a list, not a {type(lvals)}")
        else:
            for ii, ll in enumerate(lvals):
                if not isinstance(ll, int):
                    raise TypeError(f"lvals[{ii}] should be INTEGER, not {type(ll)}")
        if not isinstance(has_obc, bool):
            raise TypeError(f"has_obc should be a BOOL, not a {type(has_obc)}")
        if not isinstance(get_imag, bool):
            raise TypeError(f"get_imag should be a BOOL, not a {type(get_imag)}")
        # ADVERTISE OF THE CHOSEN PART OF THE PLAQUETTE YOU WANT TO COMPUTE
        logger.info(f"-----------------------")
        if get_imag:
            chosen_part = "IMAG"
        else:
            chosen_part = "REAL"
        logger.info(f"PLAQUETTE: {chosen_part} PART")
        logger.info(f"----------------------")
        # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
        nx = lvals[0]
        ny = lvals[1]
        n = nx * ny
        # Compute the complex_conjugate of the ground state psi
        psi_dag = np.conjugate(psi)
        plaq_obs = []
        delta_plaq_obs = []
        for ii in range(n):
            # Compute the corresponding (x,y) coords
            x, y = zig_zag(nx, ny, ii)
            if x < nx - 1 and y < ny - 1:
                # List of Sites where to apply Operators
                sites_list = [ii, ii + 1, ii + nx, ii + nx + 1]
                plaq_string = [
                    f"{x+1},{y+1}",
                    f"{x+2},{y+1}",
                    f"{x+1},{y+2}",
                    f"{x+2},{y+2}",
                ]
            else:
                if not has_obc:
                    # PERIODIC BOUNDARY CONDITIONS
                    if x < nx - 1 and y == ny - 1:
                        # UPPER BORDER
                        jj = inverse_zig_zag(nx, ny, x, 0)
                        # List of Sites where to apply Operators
                        sites_list = [ii, ii + 1, jj, jj + 1]
                        plaq_string = [
                            f"{x+1},{y+1}",
                            f"{x+2},{y+1}",
                            f"{x+1},{1}",
                            f"{x+2},{1}",
                        ]
                    elif x == nx - 1 and y < ny - 1:
                        # RIGHT BORDER
                        # List of Sites where to apply Operators
                        sites_list = [ii, ii + 1 - nx, ii + nx, ii + 1]
                        plaq_string = [
                            f"{x+1},{y+1}",
                            f"{1},{y+1}",
                            f"{x+1},{y+2}",
                            f"{1},{y+2}",
                        ]
                    else:
                        # UPPER RIGHT CORNER
                        # List of Sites where to apply Operators
                        sites_list = [ii, ii + 1 - nx, nx - 1, 0]
                        plaq_string = [
                            f"{x+1},{y+1}",
                            f"{1},{y+1}",
                            f"{x+1},{1}",
                            f"{1},{1}",
                        ]
                else:
                    continue
            # COMPUTE THE PLAQUETTE only for the appropriate site
            stag = (-1) ** (x + y)
            site_conditions = [
                site is None,
                (site == "even" and stag > 0),
                (site == "odd" and stag < 0),
            ]
            if any(site_conditions):
                plaq = np.real(
                    np.dot(
                        psi_dag,
                        four_body_op(
                            self.op_list,
                            sites_list,
                            lvals,
                            has_obc,
                            self.stag_basis,
                            self.site_basis,
                            get_real=True,
                        ).dot(psi),
                    )
                )
                delta_plaq = (
                    np.real(
                        np.dot(
                            psi_dag,
                            (
                                four_body_op(
                                    self.op_list,
                                    sites_list,
                                    lvals,
                                    has_obc,
                                    self.stag_basis,
                                    self.site_basis,
                                    get_real=True,
                                )
                                ** 2
                            ).dot(psi),
                        )
                    )
                    - plaq**2
                )
                # PRINT THE PLAQUETTE
                self.print_Plaquette(plaq_string, plaq)
                # self.print_Plaquette(plaq_string, delta_plaq)
                plaq_obs.append(plaq)
                delta_plaq_obs.append(delta_plaq)
        plaq_obs = np.array(plaq_obs)
        delta_plaq_obs = np.array(delta_plaq_obs)
        self.avg = np.sum(plaq_obs) / plaq_obs.shape[0]
        self.std = np.sqrt(np.sum(delta_plaq_obs) / plaq_obs.shape[0])

    def print_Plaquette(self, sites_list, value):
        if not isinstance(sites_list, list):
            raise TypeError(f"sites_list should be a LIST, not a {type(sites_list)}")
        if len(sites_list) != 4:
            raise ValueError(
                f"sites_list should have 4 elements, not {str(len(sites_list))}"
            )
        if not isinstance(value, float):
            raise TypeError(
                f"sites_list should be a FLOAT REAL NUMBER, not a {type(value)}"
            )
        if value > 0:
            value = format(value, ".10f")
        else:
            if np.abs(value) < 10 ** (-10):
                value = format(np.abs(value), ".10f")
            else:
                value = format(value, ".9f")
        logger.info(f"({sites_list[2]})------------({sites_list[3]})")
        logger.info(f"  |                |")
        logger.info(f"  |  {value}  |")
        logger.info(f"  |                |")
        logger.info(f"({sites_list[0]})------------({sites_list[1]})")
        logger.info("")
