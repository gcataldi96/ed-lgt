"""
This module provides utility functions for manipulating quantum many-body operators and matrices.
"""
import numpy as np
from math import prod
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import norm

__all__ = [
    "validate_parameters",
    "pause",
    "alert",
    "commutator",
    "anti_commutator",
    "check_commutator",
    "check_matrix",
    "check_hermitian",
]


def validate_parameters(
    lvals=None,
    loc_dims=None,
    has_obc=None,
    axes=None,
    site_label=None,
    coords=None,
    ops_dict=None,
    op_list=None,
    op_names_list=None,
    op_sites_list=None,
    add_dagger=None,
    get_real=None,
    get_imag=None,
    staggered_basis=None,
    staggered=None,
    stag_label=None,
    all_sites_equal=None,
    site_basis=None,
    dictionary=None,
    filename=None,
    phrase=None,
    debug=None,
    psi=None,
    spmatrix=None,
    index=None,
    threshold=None,
    print_plaq=None,
):
    """
    This is a function for type validation of parameters widely used in the library
    """
    # -----------------------------------------------------------------------------
    if lvals is not None and (
        not isinstance(lvals, list) or not all(isinstance(x, int) for x in lvals)
    ):
        raise TypeError(f"lvals should be a LIST of INTs, not {type(lvals)}")
    if loc_dims is not None:
        if isinstance(loc_dims, int):
            loc_dims = np.full(prod(lvals), loc_dims)
        elif isinstance(loc_dims, list):
            loc_dims = np.asarray(loc_dims)
        elif not isinstance(loc_dims, np.ndarray):
            raise TypeError(
                f"loc_dims must be INT, LIST, or np.ndarray, not {type(loc_dims)}"
            )
    if has_obc is not None and (
        not isinstance(has_obc, list) or not all(isinstance(x, bool) for x in has_obc)
    ):
        raise TypeError(f"has_obc should be a LIST of BOOLs, not {type(has_obc)}")
    if axes is not None and (
        not isinstance(axes, list) or not all(isinstance(ax, str) for ax in axes)
    ):
        raise TypeError(f"axes should be a LIST of STRs, not {type(axes)}")
    if site_label is not None and not isinstance(site_label, str):
        raise TypeError(f"site_label should be a STRING, not {type(site_label)}")
    if coords is not None and not (
        (isinstance(coords, (tuple, list)) and all(isinstance(x, int) for x in coords))
    ):
        raise TypeError(f"coords must be a TUPLE or LIST of INTs, not {type(coords)}")
    # -----------------------------------------------------------------------------
    if ops_dict is not None and not isinstance(ops_dict, dict):
        raise TypeError(f"ops_dict must be a DICT, not {type(ops_dict)}")
    if op_list is not None and not any(
        [
            isinstance(axes, list),
            any(
                [
                    all([isspmatrix(op) for op in op_list]),
                    all([isinstance(op, np.ndarray) for op in op_list]),
                ]
            ),
        ]
    ):
        raise TypeError(
            f"op_list must be a LIST of SPARSE/Numpy matrices, not {type(op_list)}"
        )
    if op_sites_list is not None and (
        not isinstance(op_sites_list, list)
        or not all(isinstance(x, int) for x in op_sites_list)
    ):
        raise TypeError(
            f"op_sites_list must be a LIST of INTs, not {type(op_sites_list)}"
        )
    if op_names_list is not None and (
        not isinstance(op_names_list, list)
        or not all(isinstance(x, str) for x in op_names_list)
    ):
        raise TypeError(
            f"op_names_list must be a LIST of INTs, not {type(op_names_list)}"
        )
    # -----------------------------------------------------------------------------
    if add_dagger is not None and not isinstance(add_dagger, bool):
        raise TypeError(f"add_dagger should be a BOOL, not {type(add_dagger)}")
    if get_real is not None and not isinstance(get_real, bool):
        raise TypeError(f"get_real should be a BOOL, not {type(get_real)}")
    if get_imag is not None and not isinstance(get_imag, bool):
        raise TypeError(f"get_imag should be a BOOL, not {type(get_imag)}")
    # -----------------------------------------------------------------------------
    if staggered is not None and not isinstance(staggered, bool):
        raise TypeError(f"staggered should be a BOOL, not {type(staggered)}")
    if staggered_basis is not None and not isinstance(staggered_basis, bool):
        raise TypeError(f"staggered_basis must be a BOOL, not {type(staggered_basis)}")
    if stag_label is not None and not any([stag_label == "even", stag_label == "odd"]):
        raise ValueError(f"stag_label must be 'even' or 'odd', not {stag_label}")
    if all_sites_equal is not None and not isinstance(all_sites_equal, bool):
        raise TypeError(
            f"all_sites_equal should be a BOOL, not {type(all_sites_equal)}"
        )
    if site_basis is not None and not isinstance(site_basis, dict):
        raise TypeError(f"site_basis must be a DICT, not {type(site_basis)}")
    # -----------------------------------------------------------------------------
    if dictionary is not None and not isinstance(dictionary, dict):
        raise TypeError(f"dictionary should be a DICT, not {type(dictionary)}")
    if filename is not None and not isinstance(filename, str):
        raise TypeError(f"filename should be a STRING, not {type(filename)}")
    # -----------------------------------------------------------------------------
    if phrase is not None and not isinstance(phrase, str):
        raise TypeError(f"phrase should be a STRING, not {type(phrase)}")
    if debug is not None and not isinstance(debug, bool):
        raise TypeError(f"debug should be a BOOL, not {type(debug)}")
    # -----------------------------------------------------------------------------
    if psi is not None and not isinstance(psi, np.ndarray):
        raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
    # -----------------------------------------------------------------------------
    if spmatrix is not None and not isspmatrix(spmatrix):
        raise TypeError(f"spmatrix should be a BOOL, not {type(spmatrix)}")
    if index is not None and not isinstance(index, int):
        raise TypeError(f"index should be a SCALAR INT, not {type(index)}")
    if threshold is not None and not isinstance(threshold, float):
        raise TypeError(f"threshold should be a SCALAR FLOAT, not {type(threshold)}")
    # -----------------------------------------------------------------------------
    if print_plaq is not None and not isinstance(print_plaq, bool):
        raise TypeError(f"print_plaq must be a BOOL, not a {type(print_plaq)}")


def pause(phrase, debug):
    """
    Pause the execution of the program and display a message.

    Args:
        phrase (str): The message to display.

        debug (bool): If ``True``, the pause and message are executed; if ``False``, they are skipped.

    Raises:
        TypeError: If the input arguments are of incorrect types or formats.
    """
    # Validate type of parameters
    validate_parameters(phrase=phrase, debug=debug)
    if debug == True:
        # IT PROVIDES A PAUSE in a given point of the PYTHON CODE
        print("----------------------------------------------------")
        # Press the <ENTER> key to continue
        programPause = input(phrase)
        print("----------------------------------------------------")
        print("")


def alert(phrase, debug):
    """
    Display an alert message during program execution.

    Args:
        phrase (str): The alert message to display.

        debug (bool): If ``True``, the alert and message are executed; if ``False``, they are skipped.

    Raises:
        TypeError: If the input arguments are of incorrect types or formats.
    """
    # Validate type of parameters
    validate_parameters(phrase=phrase, debug=debug)
    if debug == True:
        # IT PRINTS A PHRASE IN A GIVEN POINT OF A PYTHON CODE
        print("")
        print(phrase)


def commutator(A, B):
    """
    Compute the commutator of two sparse matrices.

    Args:
        A (scipy.sparse.csr_matrix): First matrix

        B (scipy.sparse.csr_matrix): Second matrix

    Raises:
        TypeError: If the input arguments are of incorrect types or formats.

    Returns:
        scipy.sparse.csr_matrix: The commutator of matrices A and B.
    """
    validate_parameters(spmatrix=A)
    validate_parameters(spmatrix=B)
    return A * B - B * A


def anti_commutator(A, B):
    """
    Compute the anti-commutator of two sparse matrices.

    Args:
        A (scipy.sparse.csr_matrix): First matrix

        B (scipy.sparse.csr_matrix): Second matrix

    Raises:
        TypeError: If the input arguments are of incorrect types or formats.

    Returns:
        scipy.sparse.csr_matrix: The anti-commutator of matrices A and B.
    """
    validate_parameters(spmatrix=A)
    validate_parameters(spmatrix=B)
    return A * B + B * A


def check_commutator(A, B):
    """
    Check the commutation relations between two operators A and B.

    Args:
        A (scipy.sparse.csr_matrix): First matrix

        B (scipy.sparse.csr_matrix): Second matrix

    Raises:
        TypeError: If the input arguments are of incorrect types or formats.

        ValueError: If the commutation ratio is greater than a threshold.
    """
    # CHECKS THE COMMUTATION RELATIONS BETWEEN THE OPERATORS A AND B
    validate_parameters(spmatrix=A)
    validate_parameters(spmatrix=B)
    norma = norm(A * B - B * A)
    norma_max = max(norm(A * B + B * A), norm(A), norm(B))
    ratio = norma / norma_max
    # check=(AB!=BA).nnz
    if ratio > 10 ** (-15):
        print("    ERROR: A and B do NOT COMMUTE")
        print("    NORM", norma)
        print("    RATIO", ratio)
    print("")


def check_matrix(A, B):
    """
    Check the difference between two sparse matrices A and B computing the Frobenius Norm

    Args:
        A (scipy.sparse.csr_matrix): First matrix
        B (scipy.sparse.csr_matrix): Second matrix

    Raises:
        TypeError: If the input arguments are of incorrect types or formats.
        ValueError: If the matrices have different shapes or the difference ratio is above a threshold.
    """
    # CHEKS THE DIFFERENCE BETWEEN TWO SPARSE MATRICES
    validate_parameters(spmatrix=A)
    validate_parameters(spmatrix=B)
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch between : A {A.shape} & B: {B.shape}")
    norma = norm(A - B)
    norma_max = max(norm(A + B), norm(A), norm(B))
    ratio = norma / norma_max
    if ratio > 1e-15:
        print("    ERROR: A and B are DIFFERENT MATRICES")
        raise ValueError(f"    NORM {norma}, RATIO {ratio}")


def check_hermitian(A):
    """
    Check if a sparse matrix A is Hermitian.

    Args:
        A (scipy.sparse.csr_matrix): The sparse matrix to check for Hermiticity.

    Raises:
        TypeError: If the input matrix is not in the correct format.
    """
    validate_parameters(spmatrix=A)
    A_dag = A.getH()
    check_matrix(A, A_dag)
    # Get the Hermitian
    print("HERMITICITY VALIDATED")
