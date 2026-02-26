"""Spin and SU(2)-generator matrix constructors."""

import numpy as np
from scipy.sparse import diags, block_diag, identity, csr_matrix
import logging
from edlgt.tools import validate_parameters

logger = logging.getLogger(__name__)

__all__ = [
    "spin_space",
    "m_values",
    "get_spin_operators",
    "get_Pauli_operators",
    "SU2_generators",
]


def spin_space(spin):
    """Return the Hilbert-space dimension of a spin irrep."""
    validate_parameters(spin_list=[spin])
    return int(2 * spin + 1)


def m_values(spin):
    """Return the allowed ``m`` quantum numbers for a spin irrep (descending)."""
    validate_parameters(spin_list=[spin])
    return np.arange(-spin, spin + 1)[::-1]


def get_spin_operators(spin):
    """Construct sparse spin matrices for an arbitrary spin representation.

    Parameters
    ----------
    spin : scalar
        Spin value (integer or half-integer).

    Returns
    -------
    dict
        Sparse matrices ``Sz``, ``Sp``, ``Sm``, ``Sx``, ``Sy``, and ``S2``.
    """
    validate_parameters(spin_list=[spin])
    # Size of the spin matrix
    size = spin_space(spin)
    shape = (size, size)
    # Diagonal entries of the Sz matrix
    sz_diag = m_values(spin)
    # Diagonal entries of the S+ matrix
    sp_diag = np.sqrt(spin * (spin + 1) - sz_diag[1:] * (sz_diag[1:] + 1))
    ops = {}
    ops["Sz"] = diags(sz_diag, 0, shape)
    ops["Sp"] = diags(sp_diag, 1, shape)
    ops["Sm"] = ops["Sp"].transpose()
    ops["Sx"] = (ops["Sp"] + ops["Sm"]) / 2
    ops["Sy"] = complex(0, -0.5) * (ops["Sp"] - ops["Sm"])
    ops["S2"] = diags([spin * (spin + 1) for i in range(size)], 0, shape)
    return ops


def get_Pauli_operators():
    """Return Pauli-operator matrices in the package normalization."""
    shape = (2, 2)
    ops = {}
    ops["Sz"] = diags([1, -1], 0, shape)
    ops["Sp"] = diags([1], 1, shape)
    ops["Sm"] = ops["Sp"].transpose()
    ops["Sx"] = ops["Sp"] + ops["Sm"]
    ops["Sy"] = complex(0, -1) * (ops["Sp"] - ops["Sm"])
    return ops


def SU2_generators(spin, matter=False):
    """Construct SU(2) generators for rishon or matter sectors.

    Parameters
    ----------
    spin : scalar
        Maximum spin irrep used in the block-diagonal construction.
    matter : bool, optional
        If ``True``, build the generators acting on the matter sector;
        otherwise build the rishon-sector generators.

    Returns
    -------
    dict
        Dictionary of sparse SU(2) generators and related composites.
    """
    validate_parameters(spin_list=[spin], matter=matter)
    largest_spin_size = int(2 * spin + 1)
    matrices = {"Tz": [0], "Tp": [0], "T2": [0]}
    if not matter:
        tot_shape = 0
        for spin_size in range(1, largest_spin_size):
            tot_shape += spin_size
            j_spin = spin_size / 2
            spin_ops = get_spin_operators(j_spin)
            for op in ["z", "p", "2"]:
                matrices[f"T{op}"].append(spin_ops[f"S{op}"])
        tot_shape += largest_spin_size
        matrices["T0"] = np.zeros((tot_shape, tot_shape), dtype=np.float64)
        matrices["T0"][0, 0] = 1.0
        SU2_gen = {}
        SU2_gen["T0"] = csr_matrix(matrices["T0"])
        for op in ["Tz", "Tp", "T2"]:
            SU2_gen[op] = block_diag(tuple(matrices[op]), format="csr")
        SU2_gen["Tm"] = SU2_gen["Tp"].transpose()
        SU2_gen["Tx"] = 0.5 * (SU2_gen["Tp"] + SU2_gen["Tm"])
        SU2_gen["Ty"] = complex(0, -0.5) * (SU2_gen["Tp"] - SU2_gen["Tm"])
        SU2_gen["T4"] = SU2_gen["T2"] ** 2
        # Introduce the effective Casimir operator on which a single rishon is acting
        gen_size = SU2_gen["T2"].shape[0]
        ID = identity(gen_size)
        SU2_gen["T2_root"] = 0.5 * (csr_matrix(ID + 4 * SU2_gen["T2"]).sqrt() - ID)
    else:
        spin_ops = get_spin_operators(1 / 2)
        for op in ["z", "p", "2"]:
            matrices[f"T{op}"] = [0, spin_ops[f"S{op}"], 0]
        SU2_gen = {}
        for op in ["z", "p", "2"]:
            SU2_gen[f"S{op}_psi"] = block_diag(tuple(matrices[f"T{op}"]), format="csr")
        SU2_gen["Sm_psi"] = SU2_gen["Sp_psi"].transpose()
        SU2_gen["Sx_psi"] = 0.5 * (SU2_gen["Sp_psi"] + SU2_gen["Sm_psi"])
        SU2_gen["Sy_psi"] = complex(0, -0.5) * (SU2_gen["Sp_psi"] - SU2_gen["Sm_psi"])
    return SU2_gen
