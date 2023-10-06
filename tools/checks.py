"""
This module provides utility functions for manipulating quantum many-body operators and matrices.
"""
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import norm

__all__ = [
    "pause",
    "alert",
    "commutator",
    "anti_commutator",
    "check_commutator",
    "check_matrix",
    "check_hermitian",
]


def pause(phrase, debug):
    """
    Pause the execution of the program and display a message.

    Args:
        phrase (str): The message to display.
        debug (bool): If ``True``, the pause and message are executed; if ``False``, they are skipped.

    Raises:
        TypeError: If the input arguments are of incorrect types or formats.
    """
    if not isinstance(phrase, str):
        raise TypeError(f"phrase should be a STRING, not a {type(phrase)}")
    if not isinstance(debug, bool):
        raise TypeError(f"debug should be a BOOL, not a {type(debug)}")
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
    if not isinstance(phrase, str):
        raise TypeError(f"phrase should be a STRING, not a {type(phrase)}")
    if not isinstance(debug, bool):
        raise TypeError(f"debug should be a BOOL, not a {type(debug)}")
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
    # THIS FUNCTION COMPUTES THE COMMUTATOR of TWO SPARSE MATRICES
    if not isspmatrix(A):
        raise TypeError(f"A should be a csr_matrix, not a {type(A)}")
    if not isspmatrix(B):
        raise TypeError(f"B should be a csr_matrix, not a {type(B)}")
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
    # THIS FUNCTION COMPUTES THE ANTI_COMMUTATOR of TWO SPARSE MATRICES
    if not isspmatrix(A):
        raise TypeError(f"A should be a csr_matrix, not a {type(A)}")
    if not isspmatrix(B):
        raise TypeError(f"B should be a csr_matrix, not a {type(B)}")
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
    if not isspmatrix(A):
        raise TypeError(f"A should be a csr_matrix, not a {type(A)}")
    if not isspmatrix(B):
        raise TypeError(f"B should be a csr_matrix, not a {type(B)}")
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
    if not isspmatrix(A):
        raise TypeError(f"A should be a csr_matrix, not a {type(A)}")
    if not isspmatrix(B):
        raise TypeError(f"B should be a csr_matrix, not a {type(B)}")
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
    if not isspmatrix(A):
        raise TypeError(f"A should be a csr_matrix, not a {type(A)}")
    # Get the Hermitian
    print("CHECK HERMITICITY")
    A_dag = A.getH()
    check_matrix(A, A_dag)
