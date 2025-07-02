import numpy as np
from .spin_operators import get_spin_operators


__all__ = ["get_SU2_spin_network_operators"]


def get_SU2_spin_network_operators(spin):
    """
    This function computes the SU(2) spin network operators for a given spin value.
    It returns a dictionary containing operators.

    Args:
        spin (scalar, real): Spin value, assumed to be integer or half-integer.

    Returns:
        dict: Dictionary with the SU(2) spin network operators.
    """
    return get_spin_operators(spin)
