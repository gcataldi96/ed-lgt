import numpy as np
from copy import deepcopy
from math import prod
from itertools import product
from scipy.sparse import csr_matrix, identity, isspmatrix, kron
from ed_lgt.tools import zig_zag, inverse_zig_zag

__all__ = [
    "local_op",
    "two_body_op",
    "three_body_op",
    "four_body_op",
    "qmb_operator",
    "lattice_base_configs",
    "get_site_label",
    "get_close_sites_along_direction",
]


# Helper function for parameter validation
def validate_parameters(operator=None, positions=None, lvals=None, has_obc=None):
    if operator is not None and not isspmatrix(operator):
        raise TypeError(f"operator should be SPARSE, not a {type(operator)}")
    if positions is not None and (
        not isinstance(positions, list)
        or not all(isinstance(x, int) for x in positions)
    ):
        raise TypeError(
            f"positions must be a list of integers, not a {type(positions)}"
        )
    if lvals is not None and (
        not isinstance(lvals, list) or not all(isinstance(x, int) for x in lvals)
    ):
        raise TypeError(f"lvals should be a list of integers, not a {type(lvals)}")
    if has_obc is not None and not isinstance(has_obc, bool):
        raise TypeError(f"has_obc should be a BOOL, not a {type(has_obc)}")


# Unified function for operator list construction
def construct_operator_list(
    operators, positions, lvals, has_obc, staggered_basis, site_basis
):
    validate_parameters(positions=positions, lvals=lvals, has_obc=has_obc)
    ID = identity(operators[0].shape[0])
    ops_list = []
    for ii in range(np.prod(lvals)):
        op = operators[positions.index(ii)] if ii in positions else ID
        coords = zig_zag(lvals, ii)
        basis_label = get_site_label(coords, lvals, has_obc, staggered_basis)
        op = apply_basis_projection(op, basis_label, site_basis)
        ops_list.append(op)
    return ops_list


def apply_basis_projection(operator, basis_label, site_basis):
    # Helper function for basis projection
    if len(basis_label) > 0 and site_basis is not None:
        operator = (
            site_basis[basis_label].transpose() * operator * site_basis[basis_label]
        )
    return operator


def qmb_operator(ops, op_list, add_dagger=False, get_real=False, get_imag=False):
    """
    This function performs the QMB operation of an arbitrary long list
    of operators of arbitrary dimensions.

    Args:
        ops (dict): dictionary storing all the single site operators

        op_list (list): list of the names of the operators involved in the qmb operator
        the list is assumed to be stored according to the zig-zag order on the lattice

        strength (scalar): real/complex coefficient applied in front of the operator

        add_dagger (bool, optional): if true, yields the hermitian conjugate. Defaults to False.

        get_real (bool, optional):  if true, yields only the real part. Defaults to False.

        get_imag (bool, optional): if true, yields only the imaginary part. Defaults to False.
    Returns:
        csr_matrix: QMB sparse operator
    """
    # CHECK ON TYPES
    if not isinstance(ops, dict):
        raise TypeError(f"ops must be a DICT, not a {type(ops)}")
    if not isinstance(op_list, list):
        raise TypeError(f"op_list must be a LIST, not a {type(op_list)}")
    if not isinstance(add_dagger, bool):
        raise TypeError(f"add_dagger should be a BOOL, not a {type(add_dagger)}")
    if not isinstance(get_real, bool):
        raise TypeError(f"get_real should be a BOOL, not a {type(get_real)}")
    if not isinstance(get_imag, bool):
        raise TypeError(f"get_imag should be a BOOL, not a {type(get_imag)}")
    tmp = ops[op_list[0]]
    for op in op_list[1:]:
        tmp = kron(tmp, ops[op])
    if add_dagger:
        tmp = csr_matrix(tmp + tmp.conj().transpose())
    if get_real:
        tmp = csr_matrix(tmp + tmp.conj().transpose()) / 2
    elif get_imag:
        tmp = complex(0.0, -0.5) * (csr_matrix(tmp - tmp.conj().transpose()))
    return tmp


def lattice_base_configs(base, lvals, has_obc=True, staggered=False):
    """
    This function associate the basis to each lattice site and the corresponding dimension.

    Args:
        base (dict of sparse matrices): dict with the proper hilbert basis
        of a given LGT for each lattice site

        lvals (list of ints): lattice dimensions

        has_obc (bool, optional): true for OBC, false for PBC

        staggered (bool, optional): if True, a staggered basis is required. Default to False.

    Returns:
        (np.array((lvals)),np.array((lvals))): the d-dimensional array with the labels of
            the site and the d-dimensional array with the corresponding site-basis dimensions
    """
    if not isinstance(lvals, list):
        raise TypeError(f"lvals should be a list, not a {type(lvals)}")
    elif not all(np.isscalar(dim) and isinstance(dim, int) for dim in lvals):
        raise TypeError("All items of lvals must be scalar integers.")
    if not isinstance(has_obc, bool):
        raise TypeError(f"has_obc should be a BOOL, not a {type(has_obc)}")
    if not isinstance(staggered, bool):
        raise TypeError(f"staggered should be a BOOL, not a {type(staggered)}")
    lattice_base = np.zeros(tuple(lvals), dtype=object)
    loc_dims = np.zeros(tuple(lvals), dtype=int)
    for coords in product(*[range(l) for l in lvals]):
        # PROVIDE A LABEL TO THE LATTICE SITE
        label = get_site_label(coords, lvals, has_obc, staggered, all_sites_equal=False)
        lattice_base[tuple(coords)] = label
        loc_dims[tuple(coords)] = base[label].shape[1]
    return lattice_base, loc_dims


def get_site_label(coords, lvals, has_obc, staggered=False, all_sites_equal=True):
    """
    This function associate a label to each lattice site according
    to the presence of a staggered basis, the choice of the boundary
    conditions and the position of the site in the lattice.

    Args:
        coords (tuple of ints): d-dimensional coordinates of a point in the lattice

        lvals (list of ints): lattice dimensions

        has_obc (bool, optional): true for OBC, false for PBC

        staggered (bool, optional): if True, a staggered basis is required. Defaults to False.

        all_sites_equal (bool, optional): if False, a different basis can be used for sites
            on borders and corners of the lattice

    Returns:
        (np.array((lvals)),np.array((lvals))): the d-dimensional array with the labels of the site and the corresponding site-basis dimensions
    """
    if not all(np.isscalar(c) and isinstance(c, int) for c in coords):
        raise TypeError("All coords must be scalar integers.")
    if not isinstance(lvals, list):
        raise TypeError(f"lvals should be a list, not a {type(lvals)}")
    elif not all(np.isscalar(dim) and isinstance(dim, int) for dim in lvals):
        raise TypeError("All items of lvals must be scalar integers.")
    if not isinstance(has_obc, bool):
        raise TypeError(f"has_obc should be a BOOL, not a {type(has_obc)}")
    if not isinstance(staggered, bool):
        raise TypeError(f"staggered should be a BOOL, not a {type(staggered)}")
    if not isinstance(all_sites_equal, bool):
        raise TypeError(
            f"all_sites_equal should be a BOOL, not a {type(all_sites_equal)}"
        )
    dim = len(lvals)
    dimension = "xyz"[:dim]
    # STAGGERED LABEL
    stag = (-1) ** sum(coords)
    if staggered:
        if stag > 0:
            stag_label = "even"
        else:
            stag_label = "odd"
    else:
        stag_label = "site"
    # SITE LABEL
    site_label = ""
    if not all_sites_equal:
        if has_obc:
            for ii, c in enumerate(coords):
                if c == 0:
                    site_label += f"_m{dimension[ii]}"
                elif c == lvals[ii] - 1:
                    site_label += f"_p{dimension[ii]}"
                elif c < 0 or c > lvals[ii] - 1:
                    raise ValueError(
                        f"coords[{ii}] must be in betweem 0 and {lvals[ii]}: got {c}"
                    )
    label = f"{stag_label}{site_label}"
    return label


def local_op(
    operator, op_1D_site, lvals, has_obc, staggered_basis=False, site_basis=None
):
    """
    This function compute the single local operator term on the lattice where the operator
    acts on a specific site (the rest is occupied by identities).

    Args:
        operator (scipy.sparse): A single site sparse operator matrix.

        op_1D_site (scalar int): position of the site along a certain 1D ordering in the 2D lattice

        lvals (list): Dimensions (# of sites) of a d-dimensional lattice

        has_obc (bool): It specifies the type of boundary conditions. If False, the topology is a thorus

        staggered_basis (bool, optional): Whether the lattice has staggered basis. Defaults to False.

        site_basis (dict, optional): Dictionary of Basis Projectors (sparse matrices) for lattice sites
            (corners, borders, lattice core, even/odd sites). Defaults to None.

    Raises:
        TypeError: If the input arguments are of incorrect types or formats.

    Returns:
        scipy.sparse.matrix: QMB Hamiltonian in sparse form
    """
    # CHECK ON TYPES
    if not isspmatrix(operator):
        raise TypeError(f"operator should be SPARSE, not a {type(operator)}")
    if not np.isscalar(op_1D_site) and not isinstance(op_1D_site, int):
        raise TypeError(
            f"op_1D_site must be SCALAR & INTEGER, not a {type(op_1D_site)}"
        )
    if not isinstance(lvals, list):
        raise TypeError(f"lvals should be a list, not a {type(lvals)}")
    elif not all(np.isscalar(dim) and isinstance(dim, int) for dim in lvals):
        raise TypeError("All items of lvals must be scalar integers.")
    if not isinstance(has_obc, bool):
        raise TypeError(f"has_obc should be a BOOL, not a {type(has_obc)}")
    if not isinstance(staggered_basis, bool):
        raise TypeError(
            f"staggered_basis should be a BOOL, not a {type(staggered_basis)}"
        )
    # DEFINE an IDENTITY operator
    ID = identity(operator.shape[0])
    # GENERATE empty dictionary and list to be filled with operators and names
    ops = {}
    ops_list = []
    for ii in range(prod(lvals)):
        if ii == op_1D_site:
            op = operator
            op_name = "op"
        else:
            op = ID
            op_name = "ID"
        # GET THE LABEL OF THE SITE
        coords = zig_zag(lvals, ii)
        all_sites_equal = False if site_basis is not None else True
        basis_label = get_site_label(
            coords, lvals, has_obc, staggered_basis, all_sites_equal
        )
        # PROJECT THE OPERATOR ON THE CORRESPONDING SITE BASIS
        op = apply_basis_projection(op, basis_label, site_basis)
        op_name = f"{op_name}_{basis_label}" if len(basis_label) > 0 else op_name
        # UPDATE THE LIST OF OPERATORS
        ops_list.append(op_name)
        ops[op_name] = op
    return qmb_operator(ops, ops_list)


def two_body_op(
    op_list, op_sites_list, lvals, has_obc, staggered_basis=False, site_basis=None
):
    """
    This function compute the single twobody operator term on the lattice with 2 operators
    acting on two specific lattice sites (the rest is occupied by identities).

    Args:
        op_list (list of 2 scipy.sparse.matrices): list of the 4 operators involved in the Plaquette Term

        op_sites_list (list of 2 int): list of the positions of two operators in the 1D chain ordering 2d lattice sites

        lvals (list): Dimensions (# of sites) of a d-dimensional lattice

        has_obc (bool): It specifies the type of boundary conditions. If False, the topology is a thorus

        staggered_basis (bool, optional): Whether the lattice has staggered basis. Defaults to False.

        site_basis (dict, optional): Dictionary of Basis Projectors (sparse matrices) for lattice sites
            (corners, borders, lattice core, even/odd sites). Defaults to None.

    Raises:
        TypeError: If the input arguments are of incorrect types or formats.

    Returns:
        scipy.sparse.matrix: QMB Hamiltonian in sparse form
    """
    # CHECK ON TYPES
    if not isinstance(op_list, list):
        raise TypeError(f"op_list must be a LIST, not a {type(op_list)}")
    if not isinstance(op_sites_list, list):
        raise TypeError(f"op_sites_list must be a LIST, not a {type(op_sites_list)}")
    if not isinstance(lvals, list):
        raise TypeError(f"lvals should be a list, not a {type(lvals)}")
    elif not all(np.isscalar(dim) and isinstance(dim, int) for dim in lvals):
        raise TypeError("All items of lvals must be scalar integers.")
    if not isinstance(has_obc, bool):
        raise TypeError(f"has_obc should be a BOOL, not a {type(has_obc)}")
    if not isinstance(staggered_basis, bool):
        raise TypeError(
            f"staggered_basis should be a BOOL, not a {type(staggered_basis)}"
        )
    # DEFINE an IDENTITY operator
    ID = identity(op_list[0].shape[0])
    # GENERATE empty dictionary and list to be filled with operators and names
    ops = {}
    ops_list = []
    # Run over all the lattice sites
    for ii in range(prod(lvals)):
        if ii == op_sites_list[0]:
            op = op_list[0]
            op_name = "op1"
        elif ii == op_sites_list[1]:
            op = op_list[1]
            op_name = "op2"
        else:
            op = ID
            op_name = "ID"
        # GET THE LABEL OF THE SITE
        coords = zig_zag(lvals, ii)
        all_sites_equal = False if site_basis is not None else True
        basis_label = get_site_label(
            coords, lvals, has_obc, staggered_basis, all_sites_equal
        )
        # PROJECT THE OPERATOR ON THE CORRESPONDING SITE BASIS
        op = apply_basis_projection(op, basis_label, site_basis)
        op_name = f"{op_name}_{basis_label}" if len(basis_label) > 0 else op_name
        # UPDATE THE LIST OF OPERATORS
        ops_list.append(op_name)
        ops[op_name] = op
    return qmb_operator(ops, ops_list)


def three_body_op(op_list, op_sites_list, lvals, has_obc, site_basis=None):
    """
    This function compute the single twobody operator term on the lattice with 2 operators
    acting on two specific lattice sites (the rest is occupied by identities).

    Args:
        op_list (list of 2 scipy.sparse.matrices): list of the 4 operators involved in the Plaquette Term

        op_sites_list (list of 2 int): list of the positions of two operators in the 1D chain ordering 2d lattice sites

        lvals (list): Dimensions (# of sites) of a d-dimensional lattice

        has_obc (bool): It specifies the type of boundary conditions. If False, the topology is a thorus

        staggered_basis (bool, optional): Whether the lattice has staggered basis. Defaults to False.

        site_basis (dict, optional): Dictionary of Basis Projectors (sparse matrices) for lattice sites
            (corners, borders, lattice core, even/odd sites). Defaults to None.

    Raises:
        TypeError: If the input arguments are of incorrect types or formats.

    Returns:
        scipy.sparse.matrix: QMB Hamiltonian in sparse form
    """
    # CHECK ON TYPES
    if not isinstance(op_list, list):
        raise TypeError(f"op_list must be a LIST, not a {type(op_list)}")
    if not isinstance(op_sites_list, list):
        raise TypeError(f"op_sites_list must be a LIST, not a {type(op_sites_list)}")
    if not isinstance(lvals, list):
        raise TypeError(f"lvals should be a list, not a {type(lvals)}")
    elif not all(np.isscalar(dim) and isinstance(dim, int) for dim in lvals):
        raise TypeError("All items of lvals must be scalar integers.")
    if not isinstance(has_obc, bool):
        raise TypeError(f"has_obc should be a BOOL, not a {type(has_obc)}")
    # DEFINE an IDENTITY operator
    ID = identity(op_list[0].shape[0])
    # GENERATE empty dictionary and list to be filled with operators and names
    ops = {}
    ops_list = []
    # Run over all the lattice sites
    for ii in range(prod(lvals)):
        if ii == op_sites_list[0]:
            op = op_list[0]
            op_name = "op1"
        elif ii == op_sites_list[1]:
            op = op_list[1]
            op_name = "op2"
        elif ii == op_sites_list[2]:
            op = op_list[2]
            op_name = "op3"
        else:
            op = ID
            op_name = "ID"
        # GET THE LABEL OF THE SITE
        coords = zig_zag(lvals, ii)
        all_sites_equal = False if site_basis is not None else True
        basis_label = get_site_label(
            coords, lvals, has_obc, all_sites_equal=all_sites_equal
        )
        # PROJECT THE OPERATOR ON THE CORRESPONDING SITE BASIS
        op = apply_basis_projection(op, basis_label, site_basis)
        op_name = f"{op_name}_{basis_label}" if len(basis_label) > 0 else op_name
        # UPDATE THE LIST OF OPERATORS
        ops_list.append(op_name)
        ops[op_name] = op
    return qmb_operator(ops, ops_list)


def four_body_op(
    op_list,
    op_sites_list,
    lvals,
    has_obc,
    staggered_basis=False,
    site_basis=None,
    get_real=False,
):
    """
    This function compute the single plaquette operator term on the lattice with 4 operators
    acting on 4 specific lattice sites (the rest is occupied by identities).

    Args:
        op_list (list of 4 scipy.sparse.matrices): list of the 4 operators involved in the Plaquette Term

        op_sites_list (list of 4 int): list of the positions of two operators in the 1D chain ordering 2d lattice sites

        lvals (list): Dimensions (# of sites) of a d-dimensional lattice

        has_obc (bool): It specifies the type of boundary conditions. If False, the topology is a thorus

        staggered_basis (bool, optional): Whether the lattice has staggered basis. Defaults to False.

        site_basis (dict, optional): Dictionary of Basis Projectors (sparse matrices) for lattice sites
            (corners, borders, lattice core, even/odd sites). Defaults to None.

        get_real (bool, optional): If true, it yields the real part of the operator. Defaults to False.

    Raises:
        TypeError: If the input arguments are of incorrect types or formats.

    Returns:
        scipy.sparse.matrix: QMB Hamiltonian in sparse form
    """
    # CHECK ON TYPES
    if not isinstance(op_list, list):
        raise TypeError(f"op_list must be a LIST, not a {type(op_list)}")
    if not isinstance(op_sites_list, list):
        raise TypeError(f"op_sites_list must be a LIST, not a {type(op_sites_list)}")
    if not isinstance(lvals, list):
        raise TypeError(f"lvals should be a list, not a {type(lvals)}")
    else:
        for ii, ll in enumerate(lvals):
            if not isinstance(ll, int):
                raise TypeError(f"lvals[{ii}] should be INTEGER, not {type(ll)}")
    if not isinstance(has_obc, bool):
        raise TypeError(f"has_obc should be a BOOL, not a {type(has_obc)}")
    if not isinstance(staggered_basis, bool):
        raise TypeError(
            f"staggered_basis should be a BOOL, not a {type(staggered_basis)}"
        )
    # DEFINE an IDENTITY operator
    ID = identity(op_list[0].shape[0])
    # GENERATE empty dictionary and list to be filled with operators and names
    ops = {}
    ops_list = []
    # Run over all the lattice sites
    for ii in range(prod(lvals)):
        if ii == op_sites_list[0]:
            op = op_list[0]
            op_name = "op1"
        elif ii == op_sites_list[1]:
            op = op_list[1]
            op_name = "op2"
        elif ii == op_sites_list[2]:
            op = op_list[2]
            op_name = "op3"
        elif ii == op_sites_list[3]:
            op = op_list[3]
            op_name = "op4"
        else:
            op = ID
            op_name = "ID"
        # GET THE LABEL OF THE SITE
        coords = zig_zag(lvals, ii)
        all_sites_equal = False if site_basis is not None else True
        basis_label = get_site_label(
            coords, lvals, has_obc, staggered_basis, all_sites_equal
        )
        # PROJECT THE OPERATOR ON THE CORRESPONDING SITE BASIS
        op = apply_basis_projection(op, basis_label, site_basis)
        op_name = f"{op_name}_{basis_label}" if len(basis_label) > 0 else op_name
        # UPDATE THE LIST OF OPERATORS
        ops_list.append(op_name)
        ops[op_name] = op
    return qmb_operator(ops, ops_list, get_real)


def get_close_sites_along_direction(coords, lvals, axis, has_obc):
    if not isinstance(lvals, list):
        raise TypeError(f"lvals should be a list, not a {type(lvals)}")
    if not isinstance(axis, str):
        raise TypeError(f"axis should be a STRING, not a {type(axis)}")
    if not isinstance(has_obc, bool):
        raise TypeError(f"has_obc must be a BOOL, not a {type(has_obc)}")
    dimensions = "xyz"[: len(lvals)]
    coords1 = list(coords)
    i1 = inverse_zig_zag(lvals, coords1)
    coords2 = deepcopy(coords1)
    # Check if the site admits a neighbor along the direction axis
    # Look at the specific index of the axis
    indx = dimensions.index(axis)
    # If along that axis, there is space for a twobody term:
    if coords1[indx] < lvals[indx] - 1:
        coords2[indx] += 1
        i2 = inverse_zig_zag(lvals, coords2)
        sites_list = [i1, i2]
        coords_list = [tuple(coords1), tuple(coords2)]
    else:
        # PERIODIC BOUNDARY CONDITIONS
        if not has_obc:
            coords2[indx] = 0
            i2 = inverse_zig_zag(lvals, coords2)
            sites_list = [i1, i2]
            coords_list = [tuple(coords1), tuple(coords2)]
        else:
            sites_list = None
            coords_list = None
    return coords_list, sites_list


"""
# Refactoring local_op
def local_op(
    operator, op_1D_site, lvals, has_obc, staggered_basis=False, site_basis=None
):
    validate_parameters(operator=operator, lvals=lvals, has_obc=has_obc)
    ops_list = construct_operator_list(
        [operator], [op_1D_site], lvals, has_obc, staggered_basis, site_basis
    )
    return qmb_operator(ops_list)


# Refactoring two_body_op
def two_body_op(
    op_list, op_sites_list, lvals, has_obc, staggered_basis=False, site_basis=None
):
    validate_parameters(lvals=lvals, has_obc=has_obc)
    ops_list = construct_operator_list(
        op_list, op_sites_list, lvals, has_obc, staggered_basis, site_basis
    )
    return qmb_operator(ops_list)


# Assuming a similar structure for four_body_op
def four_body_op(
    op_list, op_sites_list, lvals, has_obc, staggered_basis=False, site_basis=None
):
    validate_parameters(lvals=lvals, has_obc=has_obc)
    ops_list = construct_operator_list(
        op_list, op_sites_list, lvals, has_obc, staggered_basis, site_basis
    )
    return qmb_operator(ops_list)


# Note: The functions zig_zag, get_site_label, and qmb_operator are assumed to be defined elsewhere in your code.

# This refactored code should be more modular, easier to maintain, and potentially more efficient.


"""
