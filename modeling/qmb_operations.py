import numpy as np
from scipy.sparse import kron
from scipy.sparse import csr_matrix, identity
from scipy.sparse import isspmatrix_csr

__all__ = ["local_op", "two_body_op", "four_body_op"]


def local_op(Operator, Op_1D_site, n_sites):
    # CHECK ON TYPES
    if not isspmatrix_csr(Operator):
        raise TypeError(f"Operator should be an CSR_MATRIX, not a {type(Operator)}")
    if not np.isscalar(Op_1D_site) and not isinstance(Op_1D_site, int):
        raise TypeError(
            f"Op_1D_site must be SCALAR & INTEGER, not a {type(Op_1D_site)}"
        )
    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f"n_sites must be SCALAR & INTEGER, not a {type(n_sites)}")

    ID = identity(n_sites)
    tmp = Operator
    for ii in range(Op_1D_site - 1):
        tmp = kron(ID, tmp)
    for ii in range(n_sites - Op_1D_site):
        tmp = kron(tmp, ID)
    return tmp


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

    ID = identity(n_sites)
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
    return tmp


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

    ID = identity(n_sites)
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
    # COMPUTE ONLY THE REAL OR IMAGINARY PART OF THE OPERATOR
    if get_only_part == "REAL":
        tmp = csr_matrix(tmp) + csr_matrix(tmp.conj().transpose())
    elif get_only_part == "IMAG":
        tmp = complex(0.0, -1.0) * (
            csr_matrix(tmp) - csr_matrix(tmp.conj().transpose())
        )
    return tmp
