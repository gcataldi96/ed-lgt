import numpy as np
from simsio import logger

# =========================================================================================================
from .QMB_Operations.Mappings_1D_2D import zig_zag
from .QMB_Operations.Mappings_1D_2D import inverse_zig_zag
from .QMB_Operations.QMB_Operators import local_op
from .QMB_Operations.QMB_Operators import two_body_op
from .QMB_Operations.QMB_Operators import four_body_operator

# =========================================================================================================
from .LGT_Objects import Local_Operator
from .LGT_Objects import Two_Body_Correlator
from .LGT_Objects import Plaquette
from .LGT_Objects import Rishon_Modes

# =========================================================================================================
def get_LOCAL_Observable(psi, nx, ny, Operator, staggered=False):
    # CHECK ON TYPES
    if not isinstance(psi, np.ndarray):
        raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f"nx must be a SCALAR & INTEGER, not a {type(nx)}")
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f"ny must be a SCALAR & INTEGER, not a {type(ny)}")
    if not isinstance(Operator, Local_Operator):
        raise TypeError(
            f"Operator should be an instance of <class: Local_Operator>, not a {type(Operator)}"
        )
    if not isinstance(staggered, bool):
        raise TypeError(f"staggered should be a BOOL, not a {type(staggered)}")
    # Compute the total number of particles
    n = nx * ny
    # Compute the complex_conjugate of the ground state psi
    psi_dag = np.conjugate(psi)
    # DEFINE A VOCABULARY FOR THE STORED VALUES
    exp_val = {}
    # DEFINE A VECTOR FOR THE STORED VALUES
    if staggered:
        tmp_odd = -1
        tmp_even = -1
        odd_site_values = np.zeros(int(n / 2))
        even_site_values = np.zeros(int(n / 2))
    else:
        values = np.zeros(n)
    # RUN OVER THE LATTICE SITES
    for ii in range(n):
        # Given the 1D point on the lattice, get the corresponding (x,y) coords
        x, y = zig_zag(nx, ny, ii)
        # Define a string with the coords of each lattice point
        coords = f"({x + 1},{y + 1})"
        # Compute and print the (REAL PART OF THE) OBSERVABLE
        exp_val[f"{Operator.Op_name}_{coords}"] = np.real(
            np.dot(psi_dag, local_op(Operator.Op, Operator.ID, ii + 1, n).dot(psi))
        )
        # UPDATE THE VECTOR OF VALUES
        if staggered:
            # Compute the staggered factor depending on the site
            staggered_factor = (-1) ** (x + y)
            if staggered_factor < 1:
                tmp_odd += 1
                odd_site_values[tmp_odd] = exp_val[f"{Operator.Op_name}_{coords}"]
            else:
                tmp_even += 1
                even_site_values[tmp_even] = exp_val[f"{Operator.Op_name}_{coords}"]
        else:
            values[ii] = exp_val[f"{Operator.Op_name}_{coords}"]
    if staggered:
        return np.mean(odd_site_values), np.mean(even_site_values), exp_val
    else:
        return np.mean(values), exp_val


# =========================================================================================================
def get_TWO_BODY_Correlators(psi, nx, ny, Corr, periodicity=False):
    # CHECK ON TYPES
    if not isinstance(psi, np.ndarray):
        raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f"nx must be a SCALAR & INTEGER, not a {type(nx)}")
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f"ny must be a SCALAR & INTEGER, not a {type(ny)}")
    if not isinstance(Corr, Two_Body_Correlator):
        raise TypeError(f"C should be a CLASS Two_Body_Correlator, not a {type(Corr)}")
    if not isinstance(periodicity, bool):
        raise TypeError(f"periodicity should be a BOOL, not a {type(periodicity)}")
    # Compute the total number of particles
    n = nx * ny
    # Compute the complex_conjugate of the ground state psi
    psi_dag = np.conjugate(psi)
    # Make two lists for Single-Site Operators involved in TwoBody Operators
    Op_HH_list = [Corr.Left, Corr.Right]
    Op_VV_list = [Corr.Bottom, Corr.Top]
    # Define a dictionary
    exp_val = {}
    for ii in range(n):
        # Given the 1D point on the lattice, get the corresponding (x,y) coords
        x, y = zig_zag(nx, ny, ii)
        # ---------------------------------------------------------------------------------------
        if x < nx - 1:
            Op_sites_list = [ii + 1, ii + 2]
            # Define a string with the coords of the 2 points involved in the HORIZONTAL CORRELATOR
            h_coords = f"({x+1},{y+1})----({x+2},{y+1})"
            # Compute and print the HORIZONTAL 2BODY CORR
            exp_val[f"{Corr.Left_name}|{Corr.Right_name}_{h_coords}"] = np.real(
                np.dot(
                    psi_dag, two_body_op(Op_HH_list, Corr.ID, Op_sites_list, n).dot(psi)
                )
            )
        else:
            if periodicity:
                Op_sites_list = [ii + 1, ii - nx + 2]
                # Define a string with the coords of the 2 points involved in the HORIZONTAL CORRELATOR
                h_coords = f"({x+1},{y+1})----({1},{y+1})"
                # Compute and print the HORIZONTAL 2BODY CORR
                exp_val[f"{Corr.Left_name}|{Corr.Right_name}_{h_coords}"] = np.real(
                    np.dot(
                        psi_dag,
                        two_body_op(Op_HH_list, Corr.ID, Op_sites_list, n).dot(psi),
                    )
                )
        if y < ny - 1:
            Op_sites_list = [ii + 1, ii + nx + 1]
            # Define a string with the coords of the 2 points involved in the VERTICAL CORRELATOR
            v_coords = f"({x+1},{y+1})----({x+1},{y+2})"
            # Compute and print the VERTICAL 2BODY CORR
            exp_val[f"{Corr.Bottom_name}|{Corr.Top_name}_{v_coords}"] = np.real(
                np.dot(
                    psi_dag, two_body_op(Op_VV_list, Corr.ID, Op_sites_list, n).dot(psi)
                )
            )
        else:
            if periodicity:
                # Define a string with the coords of the 2 points involved in the VERTICAL CORRELATOR
                v_coords = f"({x+1},{y+1})----({x+1},{1})"
                jj = inverse_zig_zag(nx, ny, x, 0)
                Op_sites_list = [ii + 1, jj + 1]
                # Compute and print the VERTICAL 2BODY CORR
                exp_val[f"{Corr.Bottom_name}|{Corr.Top_name}_{v_coords}"] = np.real(
                    np.dot(
                        psi_dag,
                        two_body_op(Op_VV_list, Corr.ID, Op_sites_list, n).dot(psi),
                    )
                )
    return exp_val


# =========================================================================================================
def get_PLAQUETTE_Correlators(
    psi,
    nx,
    ny,
    Plaq,
    periodicity=False,
    not_Hermitian=False,
    get_real=False,
    get_imag=False,
):
    # CHECK ON TYPES
    if not isinstance(psi, np.ndarray):
        raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f"nx must be a SCALAR & INTEGER, not a {type(nx)}")
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f"ny must be a SCALAR & INTEGER, not a {type(ny)}")
    if not isinstance(Plaq, Plaquette):
        raise TypeError(f"Plaq should be a CLASS <Plaquette>, not a {type(Plaq)}")
    if not isinstance(periodicity, bool):
        raise TypeError(f"periodicity should be a BOOL, not a {type(periodicity)}")
    if not isinstance(not_Hermitian, bool):
        raise TypeError(f"not_Hermitian should be a BOOL, not a {type(not_Hermitian)}")
    if not isinstance(get_real, bool):
        raise TypeError(f"get_real should be a BOOL, not a {type(get_real)}")
    if not isinstance(get_imag, bool):
        raise TypeError(f"get_imag should be a BOOL, not a {type(get_imag)}")
    if not_Hermitian == True:
        if get_real and not get_imag:
            chosen_part = "REAL"
        elif get_imag and not get_real:
            chosen_part = "IMAG"
        else:
            raise ValueError(
                "CANNOT HAVE get_real and get_imag CONCURRENTLY TRUE or CONCURRENTLY FALSE"
            )
        # ADVERTISE OF THE CHOSEN PART OF THE PLAQUETTE YOU WANT TO COMPUTE
        logger.info(f"      -------------")
        logger.info(f"------| {chosen_part} PART |---------------------------------")
        logger.info(f"      -------------")
    # Compute the total number of particles
    n = nx * ny
    # Compute the complex_conjugate of the ground state psi
    psi_dag = np.conjugate(psi)
    # Define a list with the Four Operators involved in the Plaquette:
    Operator_list = [Plaq.BL, Plaq.BR, Plaq.TL, Plaq.TR]
    # DEFINE A VOCABULARY FOR THE STORED VALUES
    exp_val = {}
    # DEFINE A VECTOR FOR THE STORED VALUES
    values = np.zeros(n, dtype=float)
    for ii in range(n):
        # Given the 1D point on the lattice, get the corresponding (x,y) coords
        x, y = zig_zag(nx, ny, ii)
        if x < nx - 1 and y < ny - 1:
            # List of Sites where to apply Operators
            Sites_list = [ii + 1, ii + 2, ii + nx + 1, ii + nx + 2]
            # DEFINE A STRING LABELING THE PLAQUETTE
            plaq_string = f"({x+1},{y+1})__({x+2},{y+1})__({x+1},{y+2})__({x+2},{y+2})"
            plaq_string = [
                f"{x+1},{y+1}",
                f"{x+2},{y+1}",
                f"{x+1},{y+2}",
                f"{x+2},{y+2}",
            ]
            # COMPUTE THE PLAQUETTE
            exp_val[f"Plaq_{plaq_string}"] = np.real(
                np.dot(
                    psi_dag,
                    four_body_operator(
                        Operator_list, Plaq.ID, Sites_list, n, get_only_part=chosen_part
                    ).dot(psi),
                )
            )
            values[ii] = exp_val[f"Plaq_{plaq_string}"]
            # PRINT THE PLAQUETTE
            # logger.info(f'Plaq_{plaq_string}    ', format(exp_val[f'Plaq_{plaq_string}'],'.12f'))
            Plaq.print_Plaquette(plaq_string, exp_val[f"Plaq_{plaq_string}"])
        else:
            if periodicity:
                if x < nx - 1 and y == ny - 1:
                    # DEFINE A STRING LABELING THE PLAQUETTE on the UPPER BORDER
                    plaq_string = (
                        f"({x+1},{y+1})__({x+2},{y+1})__({x+1},{1})__({x+2},{1})"
                    )
                    plaq_string = [
                        f"{x+1},{y+1}",
                        f"{x+2},{y+1}",
                        f"{x+1},{1}",
                        f"{x+2},{1}",
                    ]
                    # On the UPPER BORDER
                    jj = inverse_zig_zag(nx, ny, x, 0)
                    # List of Sites where to apply Operators
                    Sites_list = [ii + 1, ii + 2, jj + 1, jj + 2]
                    # COMPUTE THE PLAQUETTE
                    exp_val[f"Plaq_{plaq_string}"] = np.real(
                        np.dot(
                            psi_dag,
                            four_body_operator(
                                Operator_list,
                                Plaq.ID,
                                Sites_list,
                                n,
                                get_only_part=chosen_part,
                            ).dot(psi),
                        )
                    )
                    values[ii] = exp_val[f"Plaq_{plaq_string}"]
                    # PRINT THE PLAQUETTE
                    # logger.info(f'Plaq_{plaq_string}    ', format(exp_val[f'Plaq_{plaq_string}'],'.12f'))
                    Plaq.print_Plaquette(plaq_string, exp_val[f"Plaq_{plaq_string}"])
                elif x == nx - 1 and y < ny - 1:
                    # DEFINE A STRING LABELING THE PLAQUETTE on the RIGHT BORDER
                    plaq_string = (
                        f"({x+1},{y+1})__({1},{y+1})__({x+1},{y+2})__({1},{y+2})"
                    )
                    plaq_string = [
                        f"{x+1},{y+1}",
                        f"{1},{y+1}",
                        f"{x+1},{y+2}",
                        f"{1},{y+2}",
                    ]
                    # On the RIGHT BORDER
                    # List of Sites where to apply Operators
                    Sites_list = [ii + 1, ii + 2 - nx, ii + nx + 1, ii + 2]
                    # COMPUTE THE PLAQUETTE
                    exp_val[f"Plaq_{plaq_string}"] = np.real(
                        np.dot(
                            psi_dag,
                            four_body_operator(
                                Operator_list,
                                Plaq.ID,
                                Sites_list,
                                n,
                                get_only_part=chosen_part,
                            ).dot(psi),
                        )
                    )
                    values[ii] = exp_val[f"Plaq_{plaq_string}"]
                    # PRINT THE PLAQUETTE
                    # logger.info(f'Plaq_{plaq_string}    ', format(exp_val[f'Plaq_{plaq_string}'],'.12f'))
                    Plaq.print_Plaquette(plaq_string, exp_val[f"Plaq_{plaq_string}"])
                else:
                    # DEFINE A STRING LABELING THE PLAQUETTE on the TOP RIGHT CORNER
                    plaq_string = f"({x+1},{y+1})__({1},{y+1})__({x+1},{1})__({1},{1})"
                    plaq_string = [
                        f"{x+1},{y+1}",
                        f"{1},{y+1}",
                        f"{x+1},{1}",
                        f"{1},{1}",
                    ]
                    # List of Sites where to apply Operators
                    Sites_list = [ii + 1, ii + 2 - nx, nx, 1]
                    # COMPUTE THE PLAQUETTE
                    exp_val[f"Plaq_{plaq_string}"] = np.real(
                        np.dot(
                            psi_dag,
                            four_body_operator(
                                Operator_list,
                                Plaq.ID,
                                Sites_list,
                                n,
                                get_only_part=chosen_part,
                            ).dot(psi),
                        )
                    )
                    values[ii] = exp_val[f"Plaq_{plaq_string}"]
                    # PRINT THE PLAQUETTE
                    # logger.info(f'Plaq_{plaq_string}    ', format(exp_val[f'Plaq_{plaq_string}'],'.12f'))
                    Plaq.print_Plaquette(plaq_string, exp_val[f"Plaq_{plaq_string}"])
    return np.mean(values)


# =========================================================================================================
def get_BORDER_Penalties(psi, nx, ny, Rishon):
    # CHECK ON TYPES
    if not isinstance(psi, np.ndarray):
        raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
    if not np.isscalar(nx) and not isinstance(nx, int):
        raise TypeError(f"nx must be a SCALAR & INTEGER, not a {type(nx)}")
    if not np.isscalar(ny) and not isinstance(ny, int):
        raise TypeError(f"ny must be a SCALAR & INTEGER, not a {type(ny)}")
    if not isinstance(Rishon, Rishon_Modes):
        raise TypeError(
            f"Rishon should be a CLASS <Rishon_Modes>, not a {type(Rishon)}"
        )
    # Compute the total number of particles
    n = nx * ny
    # Compute the complex_conjugate of the ground state psi
    psi_dag = np.conjugate(psi)
    # Define a Dictionary where all the expectation values are collected
    exp_val = {}
    for ii in range(n):
        # Given the 1D point on the lattice, get the corresponding (x,y) coords
        x, y = zig_zag(nx, ny, ii)
        # Define a string with the coords of the lattice point
        coords = f"({x + 1},{y + 1})"
        if y == 0:
            if x == 0:
                # Compute and print LEFT & BOTTOM Rishon modes
                exp_val[f"{Rishon.Left_name}_{coords}"] = np.real(
                    np.dot(
                        psi_dag, local_op(Rishon.Left, Rishon.ID, ii + 1, n).dot(psi)
                    )
                )
                #
                exp_val[f"{Rishon.Bottom_name}_{coords}"] = np.real(
                    np.dot(
                        psi_dag, local_op(Rishon.Bottom, Rishon.ID, ii + 1, n).dot(psi)
                    )
                )
            elif x == nx - 1:
                # Compute and print RIGHT & BOTTOM Rishon modes
                exp_val[f"{Rishon.Right_name}_{coords}"] = np.real(
                    np.dot(
                        psi_dag, local_op(Rishon.Right, Rishon.ID, ii + 1, n).dot(psi)
                    )
                )
                #
                exp_val[f"{Rishon.Bottom_name}_{coords}"] = np.real(
                    np.dot(
                        psi_dag, local_op(Rishon.Bottom, Rishon.ID, ii + 1, n).dot(psi)
                    )
                )
            else:
                # Compute and print BOTTOM Rishon modes
                exp_val[f"{Rishon.Bottom_name}_{coords}"] = np.real(
                    np.dot(
                        psi_dag, local_op(Rishon.Bottom, Rishon.ID, ii + 1, n).dot(psi)
                    )
                )
        elif y == ny - 1:
            if x == 0:
                # Compute and print LEFT & TOP Rishon modes
                exp_val[f"{Rishon.Left_name}_{coords}"] = np.real(
                    np.dot(
                        psi_dag, local_op(Rishon.Left, Rishon.ID, ii + 1, n).dot(psi)
                    )
                )
                #
                exp_val[f"{Rishon.Top_name}_{coords}"] = np.real(
                    np.dot(psi_dag, local_op(Rishon.Top, Rishon.ID, ii + 1, n).dot(psi))
                )
            elif x == nx - 1:
                # Compute and print RIGHT & TOP Rishon modes
                exp_val[f"{Rishon.Right_name}_{coords}"] = np.real(
                    np.dot(
                        psi_dag, local_op(Rishon.Right, Rishon.ID, ii + 1, n).dot(psi)
                    )
                )
                #
                exp_val[f"{Rishon.Top_name}_{coords}"] = np.real(
                    np.dot(psi_dag, local_op(Rishon.Top, Rishon.ID, ii + 1, n).dot(psi))
                )
            else:
                # Compute and print TOP Rishon modes
                exp_val[f"{Rishon.Top_name}_{coords}"] = np.real(
                    np.dot(psi_dag, local_op(Rishon.Top, Rishon.ID, ii + 1, n).dot(psi))
                )
        else:
            if x == 0:
                # Compute and print LEFT Rishon modes
                exp_val[f"{Rishon.Left_name}_{coords}"] = np.real(
                    np.dot(
                        psi_dag, local_op(Rishon.Left, Rishon.ID, ii + 1, n).dot(psi)
                    )
                )
            elif x == nx - 1:
                # Compute and print RIGHT Rishon modes
                exp_val[f"{Rishon.Right_name}_{coords}"] = np.real(
                    np.dot(
                        psi_dag, local_op(Rishon.Right, Rishon.ID, ii + 1, n).dot(psi)
                    )
                )
    return exp_val


# =========================================================================================================


def print_Observables(exp_val):
    # CHECK exp_val to be a DICTIONARY
    if not isinstance(exp_val, dict):
        raise TypeError(f"exp_val should be an dict, not a {type(exp_val)}")
    for key, value in exp_val.items():
        logger.info(f"{key}    {value}")
