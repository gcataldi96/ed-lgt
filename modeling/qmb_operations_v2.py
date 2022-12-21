import numpy as np
from scipy.sparse import kron
from scipy.sparse import csr_matrix, identity
from scipy.sparse import isspmatrix_csr
from operators import get_SU2_surface_operator
from tools import zig_zag

__all__ = ["local_op", "two_body_op", "four_body_op"]


def get_surface_label(x, y, lvals):
    # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
    nx = lvals[0]
    ny = lvals[1]
    if x == 0:
        if y == 0:
            label = "my_mx"
        elif y == ny - 1:
            label = "py_mx"
        else:
            label = "mx"
    elif x == nx - 1:
        if y == 0:
            label = "my_px"
        elif y == ny - 1:
            label = "py_px"
        else:
            label = "px"
    else:
        if y == 0:
            label = "my"
        elif y == ny - 1:
            label = "py"
        else:
            label = "none"
    return label


def local_op(Operator, Op_1D_site, lvals, same_loc_dim=True):
    # CHECK ON TYPES
    if not isspmatrix_csr(Operator):
        raise TypeError(f"Operator should be an CSR_MATRIX, not a {type(Operator)}")
    if not np.isscalar(Op_1D_site) and not isinstance(Op_1D_site, int):
        raise TypeError(
            f"Op_1D_site must be SCALAR & INTEGER, not a {type(Op_1D_site)}"
        )
    if not isinstance(lvals, list):
        raise TypeError(f"lvals should be a list, not a {type(lvals)}")
    else:
        for ii, ll in enumerate(lvals):
            if not isinstance(ll, int):
                raise TypeError(f"lvals[{ii}] should be INTEGER, not {type(ll)}")
    if not isinstance(same_loc_dim, bool):
        raise TypeError(f"same_loc_dim should be a BOOL, not a {type(same_loc_dim)}")
    # COMPUTE THE TOTAL NUMBER OF LATTICE SITES
    nx = lvals[0]
    ny = lvals[1]
    n = nx * ny
    # GENERATE empty list to be filled with operators
    op_list = []
    for ii in range(n):
        x, y = zig_zag(nx, ny, ii)
        if same_loc_dim:
            if Op_1D_site == ii:
                op_list.append(Operator)
            else:
                op_list.append(identity(Operator.shape[0]))
        else:
            label = get_surface_label(x, y, lvals)
            if Op_1D_site == ii:
                op_list.append(Operator, label)
            else:
                id = get_SU2_surface_operator(identity(Operator.shape[0]), label)
                op_list.append(id)


def two_body_op(Op_list, Op_sites_list, n_sites, add_dagger=False):
    # CHECK ON TYPES
    if not isinstance(Op_list, list):
        raise TypeError(f"Op_list must be a LIST, not a {type(Op_list)}")
    if not isinstance(Op_sites_list, list):
        raise TypeError(f"Op_sites_list must be a LIST, not a {type(Op_sites_list)}")
    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f"n_sites must be SCALAR & INTEGER, not a {type(n_sites)}")
    if not isinstance(add_dagger, bool):
        raise TypeError(f"add_dagger should be a BOOL, not a {type(add_dagger)}")
    # Create a local identity with the same dimensionality of the Operators
    op_dim = Op_list[0].shape[0]
    ID = identity(op_dim)
    # STORE Op_list according to Op_sites_list in ASCENDING ORDER
    Op_NEW_list = [x for _, x in sorted(zip(Op_sites_list, Op_list))]
    # STORE Op_sites_list in ASCENDING ORDER
    Op_sites_list = [x for x, _ in sorted(zip(Op_sites_list, Op_list))]
    # Make a copy of the 1st Operator
    tmp = Op_NEW_list[0]
    for ii in range(Op_sites_list[0] - 1):
        tmp = kron(ID, tmp)
    for ii in range(Op_sites_list[1] - Op_sites_list[0] - 1):
        tmp = kron(tmp, ID)
    tmp = kron(tmp, Op_NEW_list[1])
    for ii in range(n_sites - Op_sites_list[1]):
        tmp = kron(tmp, ID)
    # ADD THE HERMITIAN CONDJUGATE OF THE OPERATOR
    if add_dagger == True:
        tmp = csr_matrix(tmp) + csr_matrix(tmp.conj().transpose())
    return csr_matrix(tmp)


def four_body_op(Op_list, Op_sites_list, n_sites, get_only_part=None):
    # CHECK ON TYPES
    if not isinstance(Op_list, list):
        raise TypeError(f"Op_list must be a LIST, not a {type(Op_list)}")
    if not isinstance(Op_sites_list, list):
        raise TypeError(f"Op_sites_list must be a LIST, not a {type(Op_sites_list)}")
    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f"n_sites must be SCALAR & INTEGER, not a {type(n_sites)}")
    if get_only_part is not None:
        if not isinstance(get_only_part, str):
            raise TypeError(
                f"get_only_part should be a STR, not a {type(get_only_part)}"
            )
    # Create a local identity with the same dimensionality of Operator
    op_dim = Op_list[0].shape[0]
    ID = identity(op_dim)
    # STORE Op_list according to Op_sites_list in ASCENDING ORDER
    Op_NEW_list = [x for _, x in sorted(zip(Op_sites_list, Op_list))]
    # STORE Op_sites_list in ASCENDING ORDER
    Op_sites_list = [x for x, _ in sorted(zip(Op_sites_list, Op_list))]
    # Make a copy of the 1st Operator
    tmp = Op_NEW_list[0]
    for ii in range(Op_sites_list[0] - 1):
        tmp = kron(ID, tmp)
    for ii in range(Op_sites_list[1] - Op_sites_list[0] - 1):
        tmp = kron(tmp, ID)
    tmp = kron(tmp, Op_NEW_list[1])
    for ii in range(Op_sites_list[2] - Op_sites_list[1] - 1):
        tmp = kron(tmp, ID)
    tmp = kron(tmp, Op_NEW_list[2])
    for ii in range(Op_sites_list[3] - Op_sites_list[2] - 1):
        tmp = kron(tmp, ID)
    tmp = kron(tmp, Op_NEW_list[3])
    for ii in range(n_sites - Op_sites_list[3]):
        tmp = kron(tmp, ID)

    return csr_matrix(tmp)


def qmb_operator(op_list, strength, add_dagger=False, get_real=False, get_imag=False):
    tmp = op_list[0]
    for ii in range(len(op_list) - 1):
        tmp = kron(tmp, op_list[ii + 1])
    if add_dagger:
        tmp = csr_matrix(tmp + tmp.conj().transpose())
    if get_real:
        tmp = csr_matrix(tmp + tmp.conj().transpose()) / 2
    elif get_imag:
        tmp = complex(0.0, -0.5) * (csr_matrix(tmp - tmp.conj().transpose()))
    return strength * tmp
