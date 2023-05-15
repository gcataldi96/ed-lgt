from scipy.sparse import isspmatrix
import numpy as np
from simsio import logger
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
    if not isinstance(phrase, str):
        raise TypeError(f"phrase should be a STRING, not a {type(phrase)}")
    if not isinstance(debug, bool):
        raise TypeError(f"debug should be a BOOL, not a {type(debug)}")
    if debug == True:
        # IT PROVIDES A PAUSE in a given point of the PYTHON CODE
        logger.info("----------------------------------------------------")
        # Press the <ENTER> key to continue
        programPause = input(phrase)
        logger.info("----------------------------------------------------")
        logger.info("")


def alert(phrase, debug):
    if not isinstance(phrase, str):
        raise TypeError(f"phrase should be a STRING, not a {type(phrase)}")
    if not isinstance(debug, bool):
        raise TypeError(f"debug should be a BOOL, not a {type(debug)}")
    if debug == True:
        # IT PRINTS A PHRASE IN A GIVEN POINT OF A PYTHON CODE
        logger.info("")
        logger.info(phrase)


def commutator(A, B):
    # THIS FUNCTION COMPUTES THE COMMUTATOR of TWO SPARSE MATRICES
    if not isspmatrix(A):
        raise TypeError(f"A should be a csr_matrix, not a {type(A)}")
    if not isspmatrix(B):
        raise TypeError(f"B should be a csr_matrix, not a {type(B)}")
    return A * B - B * A


def anti_commutator(A, B):
    # THIS FUNCTION COMPUTES THE ANTI_COMMUTATOR of TWO SPARSE MATRICES
    if not isspmatrix(A):
        raise TypeError(f"A should be a csr_matrix, not a {type(A)}")
    if not isspmatrix(B):
        raise TypeError(f"B should be a csr_matrix, not a {type(B)}")
    return A * B + B * A


def check_commutator(A, B):
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
        logger.info("    ERROR: A and B do NOT COMMUTE")
        logger.info("    NORM", norma)
        logger.info("    RATIO", ratio)
    logger.info("")


def check_matrix(A, B):
    # CHEKS THE DIFFERENCE BETWEEN TWO SPARSE MATRICES
    if not isspmatrix(A):
        raise TypeError(f"A should be a csr_matrix, not a {type(A)}")
    if not isspmatrix(B):
        raise TypeError(f"B should be a csr_matrix, not a {type(B)}")
    if A.shape != B.shape:
        raise ValueError(
            f"A and B should have the same shape but A is {A.shape}, B is {B.shape}"
        )
    norma = norm(A - B)
    norma_max = max(norm(A + B), norm(A), norm(B))
    ratio = norma / norma_max
    if ratio > 1e-15:
        logger.info("    ERROR: A and B are DIFFERENT MATRICES")
        raise ValueError(f"    NORM {norma}, RATIO {ratio}")


def check_hermitian(A):
    if not isspmatrix(A):
        raise TypeError(f"A should be a csr_matrix, not a {type(A)}")
    # Get the Hermitian
    logger.info("CHECK HERMITICITY")
    A_dag = A.getH()
    check_matrix(A, A_dag)
