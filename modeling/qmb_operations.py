import numpy as np
from scipy.sparse import csr_matrix, identity, isspmatrix, kron
from tools import zig_zag
from simsio import logger

__all__ = [
    "local_op",
    "two_body_op",
    "four_body_op",
    "qmb_operator",
    "lattice_base_configs",
]


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
    if not isinstance(lvals, list):
        raise TypeError(f"lvals should be a list, not a {type(lvals)}")
    else:
        for ii, ll in enumerate(lvals):
            if not isinstance(ll, int):
                raise TypeError(f"lvals[{ii}] should be INTEGER, not {type(ll)}")
    if not isinstance(has_obc, bool):
        raise TypeError(f"has_obc should be a BOOL, not a {type(has_obc)}")
    if not isinstance(staggered, bool):
        raise TypeError(f"staggered should be a BOOL, not a {type(staggered)}")
    """
    This function associate the basis to each lattice site
    and the corresponding dimension.

    Args:
        base (dict of sparse matrices): dict with the proper hilbert basis 
        of a given LGT for each lattice site

        lvals (list of ints): lattice dimensions

        has_obc (bool, optional): true for OBC, false for PBC

        staggered (bool, optional): if True, a staggered basis is required. Defaults to False.

    Returns:
        (np.array((lvals)),np.array((lvals))): the 2D array with the labels of the site
        and the 2D with the corresponding site-basis dimensions
    """
    lattice_base = np.zeros(tuple(lvals), dtype=object)
    loc_dims = np.zeros(tuple(lvals), dtype=int)
    for x in range(lvals[0]):
        for y in range(lvals[1]):
            # PROVIDE A LABEL TO THE LATTICE SITE
            label = get_site_label(
                x, y, lvals, has_obc, staggered, all_sites_equal=False
            )
            lattice_base[x, y] = label
            loc_dims[x, y] = base[label].shape[1]
    return lattice_base, loc_dims


def get_site_label(x, y, lvals, has_obc, staggered=False, all_sites_equal=True):
    """
    This function associate a label to each lattice site according
    to the presence of a staggered basis, the choice of the boundary
    conditions and the position of the site in the lattice.

    Args:
        x (int): x-coordinate of the site

        y (int): y-coordinate of the site

        lvals (list of ints): lattice dimensions

        has_obc (bool, optional): true for OBC, false for PBC

        staggered (bool, optional): if True, a staggered basis is required. Defaults to False.

        all_sites_equal (bool, optional): if False, a different basis can be used for sites
        on borders and corners of the lattice

    Returns:
        (np.array((lvals)),np.array((lvals))): the 2D array with the labels of the site
        and the 2D with the corresponding site-basis dimensions
    """
    if not np.isscalar(x) and not isinstance(x, int):
        raise TypeError(f"x must be SCALAR & INTEGER, not a {type(x)}")
    if not np.isscalar(y) and not isinstance(x, int):
        raise TypeError(f"y must be SCALAR & INTEGER, not a {type(y)}")
    if not isinstance(lvals, list):
        raise TypeError(f"lvals should be a list, not a {type(lvals)}")
    else:
        for ii, ll in enumerate(lvals):
            if not isinstance(ll, int):
                raise TypeError(f"lvals[{ii}] should be INTEGER, not {type(ll)}")
    if not isinstance(has_obc, bool):
        raise TypeError(f"has_obc should be a BOOL, not a {type(has_obc)}")
    if not isinstance(staggered, bool):
        raise TypeError(f"staggered should be a BOOL, not a {type(staggered)}")
    if not isinstance(all_sites_equal, bool):
        raise TypeError(
            f"all_sites_equal should be a BOOL, not a {type(all_sites_equal)}"
        )
    # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
    nx = lvals[0]
    ny = lvals[1]
    # STAGGERED LABEL
    stag = (-1) ** (x + y)
    if (staggered == True) and (stag > 0):
        stag_label = "even"
    elif (staggered == True) and (stag < 0):
        stag_label = "odd"
    else:
        stag_label = ""
    # SITE LABEL
    if not all_sites_equal:
        if has_obc:
            if (x == 0) and (y == 0):
                site_label = "_mx_my"
            elif (x == 0) and (y == ny - 1):
                site_label = "_mx_py"
            elif (x == nx - 1) and (y == 0):
                site_label = "_my_px"
            elif (x == nx - 1) and (y == ny - 1):
                site_label = "_px_py"
            elif x == 0:
                site_label = "_mx"
            elif x == nx - 1:
                site_label = "_px"
            elif y == 0:
                site_label = "_my"
            elif y == ny - 1:
                site_label = "_py"
            else:
                site_label = ""
        else:
            site_label = ""
    else:
        site_label = ""
    return f"{stag_label}{site_label}"


def local_op(
    operator, op_1D_site, lvals, has_obc, staggered_basis=False, site_basis=None
):
    # CHECK ON TYPES
    if not isspmatrix(operator):
        raise TypeError(f"operator should be SPARSE, not a {type(operator)}")
    if not np.isscalar(op_1D_site) and not isinstance(op_1D_site, int):
        raise TypeError(
            f"op_1D_site must be SCALAR & INTEGER, not a {type(op_1D_site)}"
        )
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
    # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
    nx = lvals[0]
    ny = lvals[1]
    n = nx * ny
    # DEFINE an IDENTITY operator
    ID = identity(operator.shape[0])
    # GENERATE empty dictionary and list to be filled with operators and names
    ops = {}
    ops_list = []
    for ii in range(n):
        if ii == op_1D_site:
            op = operator
            op_name = "op"
        else:
            op = ID
            op_name = "ID"
        # GET THE LABEL OF THE SITE
        x, y = zig_zag(nx, ny, ii)
        all_sites_equal = False if site_basis is not None else True
        basis_label = get_site_label(
            x, y, lvals, has_obc, staggered_basis, all_sites_equal
        )
        # PROJECT THE OPERATOR ON THE CORRESPONDING SITE BASIS
        if len(basis_label) > 0:
            op = site_basis[basis_label].transpose() * op * site_basis[basis_label]
            op_name = f"{op_name}_{basis_label}"
        # UPDATE THE LIST OF OPERATORS
        ops_list.append(op_name)
        ops[op_name] = op
    return qmb_operator(ops, ops_list)


def two_body_op(
    op_list, op_sites_list, lvals, has_obc, staggered_basis=False, site_basis=None
):
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
    # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
    nx = lvals[0]
    ny = lvals[1]
    n = nx * ny
    # DEFINE an IDENTITY operator
    ID = identity(op_list[0].shape[0])
    # GENERATE empty dictionary and list to be filled with operators and names
    ops = {}
    ops_list = []
    # Run over all the lattice sites
    for ii in range(n):
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
        x, y = zig_zag(nx, ny, ii)
        all_sites_equal = False if site_basis is not None else True
        basis_label = get_site_label(
            x, y, lvals, has_obc, staggered_basis, all_sites_equal
        )
        # PROJECT THE OPERATOR ON THE CORRESPONDING SITE BASIS
        if len(basis_label) > 0:
            op = site_basis[basis_label].transpose() * op * site_basis[basis_label]
            op_name = f"{op_name}_{basis_label}"
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
    # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
    nx = lvals[0]
    ny = lvals[1]
    n = nx * ny
    # DEFINE an IDENTITY operator
    ID = identity(op_list[0].shape[0])
    # GENERATE empty dictionary and list to be filled with operators and names
    ops = {}
    ops_list = []
    # Run over all the lattice sites
    for ii in range(n):
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
        x, y = zig_zag(nx, ny, ii)
        all_sites_equal = False if site_basis is not None else True
        basis_label = get_site_label(
            x, y, lvals, has_obc, staggered_basis, all_sites_equal
        )
        # PROJECT THE OPERATOR ON THE CORRESPONDING SITE BASIS
        if len(basis_label) > 0:
            op = site_basis[basis_label].transpose() * op * site_basis[basis_label]
            op_name = f"{op_name}_{basis_label}"
        # UPDATE THE LIST OF OPERATORS
        ops_list.append(op_name)
        ops[op_name] = op
    return qmb_operator(ops, ops_list, get_real)
