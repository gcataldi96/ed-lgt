import numpy as np
from scipy.sparse import csr_matrix, identity, isspmatrix_csr, kron
from tools import zig_zag
from simsio import logger

__all__ = ["local_op", "two_body_op", "four_body_op"]


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
    # logger.info("------------------")
    # logger.info(op_list[0])
    for op in op_list[1:]:
        # logger.info(op)
        tmp = kron(tmp, ops[op])
    if add_dagger:
        tmp = csr_matrix(tmp + tmp.conj().transpose())
    if get_real:
        tmp = csr_matrix(tmp + tmp.conj().transpose()) / 2
    elif get_imag:
        tmp = complex(0.0, -0.5) * (csr_matrix(tmp - tmp.conj().transpose()))
    return tmp


def get_surface_label(x, y, lvals, has_obc):
    if not isinstance(has_obc, bool):
        raise TypeError(f"has_obc should be a BOOL, not a {type(has_obc)}")
    # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
    nx = lvals[0]
    ny = lvals[1]
    if has_obc:
        if x == 0:
            if y == 0:
                label = "mx_my"
            elif y == ny - 1:
                label = "mx_py"
            else:
                label = "mx"
        elif x == nx - 1:
            if y == 0:
                label = "my_px"
            elif y == ny - 1:
                label = "px_py"
            else:
                label = "px"
        else:
            if y == 0:
                label = "my"
            elif y == ny - 1:
                label = "py"
            else:
                label = None
    else:
        label = None
    return label


def local_op(
    operator, op_1D_site, lvals, has_obc, staggered_basis=False, site_basis=None
):
    # CHECK ON TYPES
    if not isspmatrix_csr(operator):
        raise TypeError(f"operator should be an CSR_MATRIX, not a {type(operator)}")
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
        x, y = zig_zag(nx, ny, ii)
        # get label of the staggered site
        stag = (-1) ** (x + y)
        site = "even" if stag > 0 else "odd"
        # get label of a site on a lattice border
        border_label = get_surface_label(x, y, lvals, has_obc)
        if ii == op_1D_site:
            op = operator
            op_name = "op"
        else:
            op = ID
            op_name = "ID"
        # project the operator on the corresponding basis
        if site_basis is not None:
            if staggered_basis:
                basis_label = f"{site}_{border_label}"
            else:
                basis_label = border_label
            op = site_basis[basis_label].transpose() * op * site_basis[basis_label]
            op_name = f"{op_name}_{basis_label}"
        # update the list of operators
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
        x, y = zig_zag(nx, ny, ii)
        # get label of the staggered site
        stag = (-1) ** (x + y)
        site = "even" if stag > 0 else "odd"
        # get label of a site on a lattice border
        border_label = get_surface_label(x, y, lvals, has_obc)
        if ii == op_sites_list[0]:
            op = op_list[0]
            op_name = "op1"
        elif ii == op_sites_list[1]:
            op = op_list[1]
            op_name = "op2"
        else:
            op = ID
            op_name = "ID"
        # project the operator on the corresponding basis
        if site_basis is not None:
            if staggered_basis:
                basis_label = f"{site}_{border_label}"
            else:
                basis_label = border_label
            op = site_basis[basis_label].transpose() * op * site_basis[basis_label]
            op_name = f"{op_name}_{basis_label}"
        # update the list of operators
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
        x, y = zig_zag(nx, ny, ii)
        # get label of the staggered site
        stag = (-1) ** (x + y)
        site = "even" if stag > 0 else "odd"
        # get label of a site on a lattice border
        border_label = get_surface_label(x, y, lvals, has_obc)
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
        # project the operator on the corresponding basis
        if site_basis is not None:
            if staggered_basis:
                basis_label = f"{site}_{border_label}"
            else:
                basis_label = border_label
            op = site_basis[basis_label].transpose() * op * site_basis[basis_label]
            op_name = f"{op_name}_{basis_label}"
        # update the list of operators
        ops_list.append(op_name)
        ops[op_name] = op
    return qmb_operator(ops, ops_list, get_real)
