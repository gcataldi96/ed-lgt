"""Helpers to assemble lattice operators in the full many-body Hilbert space.

This module provides low-level functions used by term classes to construct
local, two-body, and four-body sparse operators on a lattice, including support
for site-dependent basis projections (e.g. gauge bases).
"""

import numpy as np
from scipy.sparse import csr_matrix, identity, kron
from edlgt.tools import validate_parameters
from edlgt.dtype_config import coerce_matrix_dtype
from .lattice_mappings import zig_zag
from .lattice_geometry import get_site_label
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "qmb_operator",
    "local_op",
    "two_body_op",
    "four_body_op",
    "construct_operator_list",
    "apply_basis_projection",
]


def qmb_operator(
    ops,
    op_names_list,
    add_dagger=False,
    get_real=False,
    get_imag=False,
):
    """Build a many-body sparse operator from an ordered list of site operators.

    Parameters
    ----------
    ops : dict
        Dictionary mapping operator names to single-site sparse matrices.
    op_names_list : list
        Ordered list of operator names to tensor together, one per lattice site,
        following the project site ordering convention.
    add_dagger : bool, optional
        If ``True``, symmetrize by adding the Hermitian conjugate.
    get_real : bool, optional
        If ``True``, return only the Hermitian (real) part.
    get_imag : bool, optional
        If ``True``, return only the anti-Hermitian-derived imaginary part.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse many-body operator in the full Hilbert space.
    """
    # Validate type of parameters
    validate_parameters(
        ops_dict=ops,
        op_names_list=op_names_list,
        add_dagger=add_dagger,
        get_real=get_real,
        get_imag=get_imag,
    )
    tmp = ops[op_names_list[0]]
    for op in op_names_list[1:]:
        tmp = kron(tmp, ops[op])
    if add_dagger:
        tmp = csr_matrix(tmp + tmp.conj().transpose())
    if get_real:
        tmp = csr_matrix(tmp + tmp.conj().transpose()) / 2
    elif get_imag:
        tmp = complex(0.0, -0.5) * (csr_matrix(tmp - tmp.conj().transpose()))
    return coerce_matrix_dtype(tmp, name="qmb_operator output")


def local_op(
    operator,
    op_site,
    lvals,
    has_obc,
    staggered_basis=False,
    gauge_basis=None,
):
    """Construct a single-site operator embedded in the full lattice Hilbert space.

    Parameters
    ----------
    operator : scipy.sparse.spmatrix
        Single-site operator.
    op_site : int
        Site index where the operator acts.
    lvals : list[int]
        Lattice sizes along each axis.
    has_obc : list[bool]
        Boundary-condition flags for each axis (``True`` for OBC).
    staggered_basis : bool, optional
        Whether a staggered basis is used.
    gauge_basis : dict, optional
        Site-label-dependent basis projectors.

    Returns
    -------
    scipy.sparse.csr_matrix
        Embedded local operator.
    """
    # Validate type of parameters
    validate_parameters(
        op_list=[operator],
        op_sites_list=[op_site],
        lvals=lvals,
        has_obc=has_obc,
        staggered_basis=staggered_basis,
        gauge_basis=gauge_basis,
    )
    # Construct the dictionary of operators and their names needed to construct the QMB local term
    ops, op_names_list = construct_operator_list(
        [operator], [op_site], lvals, has_obc, staggered_basis, gauge_basis
    )
    return qmb_operator(ops, op_names_list)


def two_body_op(
    op_list,
    op_sites_list,
    lvals,
    has_obc,
    staggered_basis=False,
    gauge_basis=None,
):
    """Construct a two-site operator embedded in the full lattice Hilbert space.

    Parameters
    ----------
    op_list : list
        Two single-site operators.
    op_sites_list : list[int]
        Two site indices where the operators act.
    lvals : list[int]
        Lattice sizes along each axis.
    has_obc : list[bool]
        Boundary-condition flags for each axis.
    staggered_basis : bool, optional
        Whether a staggered basis is used.
    gauge_basis : dict, optional
        Site-label-dependent basis projectors.

    Returns
    -------
    scipy.sparse.csr_matrix
        Embedded two-body operator.
    """
    # Validate type of parameters
    validate_parameters(
        op_list=op_list,
        op_sites_list=op_sites_list,
        lvals=lvals,
        has_obc=has_obc,
        staggered_basis=staggered_basis,
        gauge_basis=gauge_basis,
    )
    # Construct the dictionary of operators and their names needed to construct the QMB twobody term
    ops, op_names_list = construct_operator_list(
        op_list, op_sites_list, lvals, has_obc, staggered_basis, gauge_basis
    )
    return qmb_operator(ops, op_names_list)


def four_body_op(
    op_list,
    op_sites_list,
    lvals,
    has_obc,
    staggered_basis=False,
    gauge_basis=None,
    get_real=False,
):
    """Construct a four-site (plaquette-like) operator in the full Hilbert space.

    Parameters
    ----------
    op_list : list
        Four single-site operators.
    op_sites_list : list[int]
        Four site indices where the operators act.
    lvals : list[int]
        Lattice sizes along each axis.
    has_obc : list[bool]
        Boundary-condition flags for each axis.
    staggered_basis : bool, optional
        Whether a staggered basis is used.
    gauge_basis : dict, optional
        Site-label-dependent basis projectors.
    get_real : bool, optional
        If ``True``, return only the Hermitian (real) part.

    Returns
    -------
    scipy.sparse.csr_matrix
        Embedded four-body operator.
    """
    # Validate type of parameters
    validate_parameters(
        op_list=op_list,
        op_sites_list=op_sites_list,
        lvals=lvals,
        has_obc=has_obc,
        get_real=get_real,
        staggered_basis=staggered_basis,
        gauge_basis=gauge_basis,
    )
    # Construct the dictionary of operators and their names needed to construct the QMB twobody term
    ops, op_names_list = construct_operator_list(
        op_list, op_sites_list, lvals, has_obc, staggered_basis, gauge_basis
    )
    return qmb_operator(ops, op_names_list, get_real=get_real)


def construct_operator_list(
    op_list, op_sites_list, lvals, has_obc, staggered_basis, gauge_basis
):
    """Create per-site operator labels/matrices for a lattice operator product.

    Parameters
    ----------
    op_list : list
        Operators to place on the lattice.
    op_sites_list : list[int]
        Lattice sites where the operators in ``op_list`` act.
    lvals : list[int]
        Lattice sizes along each axis.
    has_obc : list[bool]
        Boundary-condition flags for each axis.
    staggered_basis : bool
        Whether a staggered basis is used.
    gauge_basis : dict or None
        Site-label-dependent basis projectors, or ``None`` for a uniform basis.

    Returns
    -------
    tuple
        ``(ops_dict, op_names_list)`` where ``ops_dict`` is a dictionary of
        projected operators and ``op_names_list`` is the ordered list of names
        passed to :func:`qmb_operator`.
    """
    # Validate type of parameteres
    validate_parameters(
        op_list=op_list,
        op_sites_list=op_sites_list,
        lvals=lvals,
        has_obc=has_obc,
        staggered_basis=staggered_basis,
        gauge_basis=gauge_basis,
    )
    all_sites_equal = True if gauge_basis is None else False
    # Define the identity operator
    ID = identity(op_list[0].shape[0])
    # Empty dictionary of operators and list of their names
    ops_dict = {}
    op_names_list = []
    # Assign a name to the operator at position ii
    tmp = 0
    for ii in range(np.prod(lvals)):
        if ii in op_sites_list:
            tmp += 1
            op_name = f"op{tmp}"
            op = op_list[op_sites_list.index(ii)]
        else:
            op_name = "ID"
            op = ID
        # Get the coordinates of ii in the d-dim lattice
        coords = zig_zag(lvals, ii)
        # Get the site label according to its position in the lattice (border, corner, core)
        basis_label = get_site_label(
            coords, lvals, has_obc, staggered_basis, all_sites_equal
        )
        # Apply projection of the operator on the proper basis for that lattice position (and update its name)
        op, op_name = apply_basis_projection(op, op_name, basis_label, gauge_basis)
        # Save operator and its name
        ops_dict[op_name] = op
        op_names_list.append(op_name)
    return ops_dict, op_names_list


def apply_basis_projection(op, op_name, basis_label, gauge_basis):
    """Project a single-site operator into a site-dependent basis if needed.

    Parameters
    ----------
    op : scipy.sparse.spmatrix
        Operator to project.
    op_name : str
        Base operator name.
    basis_label : str
        Site label used to select the projector.
    gauge_basis : dict or None
        Dictionary of projectors keyed by ``basis_label``.

    Returns
    -------
    tuple
        Projected operator and its (possibly updated) name.
    """
    # Validate type of parameteres
    validate_parameters(
        op_list=[op],
        op_names_list=[op_name],
        gauge_basis=gauge_basis,
    )
    if len(basis_label) > 0 and gauge_basis is not None:
        # Apply projection of the operator on the proper basis for that lattice position
        op = gauge_basis[basis_label].transpose() * op * gauge_basis[basis_label]
        # Update the name of the operator with the label
        op_name = f"{op_name}_{basis_label}"
    return op, op_name
