import numpy as np
from scipy.sparse import csr_matrix, identity, kron
from ed_lgt.tools import validate_parameters
from .lattice_mappings import zig_zag
from .lattice_geometry import get_site_label

__all__ = [
    "qmb_operator",
    "local_op",
    "two_body_op",
    "four_body_op",
    "construct_operator_list",
    "apply_basis_projection",
]


def qmb_operator(ops, op_names_list, add_dagger=False, get_real=False, get_imag=False):
    """
    This function performs the QMB operation of an arbitrary long list
    of operators of arbitrary dimensions.

    Args:
        ops (dict): dictionary storing all the single site operators

        op_names_list (list): list of the names of the operators involved in the qmb operator
        the list is assumed to be stored according to the zig-zag order on the lattice

        strength (scalar): real/complex coefficient applied in front of the operator

        add_dagger (bool, optional): if true, yields the hermitian conjugate. Defaults to False.

        get_real (bool, optional):  if true, yields only the real part. Defaults to False.

        get_imag (bool, optional): if true, yields only the imaginary part. Defaults to False.
    Returns:
        csr_matrix: QMB sparse operator
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
    return tmp


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
    # Validate type of parameters
    validate_parameters(
        op_list=[operator],
        op_sites_list=[op_1D_site],
        lvals=lvals,
        has_obc=has_obc,
        staggered_basis=staggered_basis,
        site_basis=site_basis,
    )
    # Construct the dictionary of operators and their names needed to construct the QMB local term
    ops, op_names_list = construct_operator_list(
        [operator], [op_1D_site], lvals, has_obc, staggered_basis, site_basis
    )
    return qmb_operator(ops, op_names_list)


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
    # Validate type of parameters
    validate_parameters(
        op_list=op_list,
        op_sites_list=op_sites_list,
        lvals=lvals,
        has_obc=has_obc,
        staggered_basis=staggered_basis,
        site_basis=site_basis,
    )
    # Construct the dictionary of operators and their names needed to construct the QMB twobody term
    ops, op_names_list = construct_operator_list(
        op_list, op_sites_list, lvals, has_obc, staggered_basis, site_basis
    )
    return qmb_operator(ops, op_names_list)


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
    # Validate type of parameters
    validate_parameters(
        op_list=op_list,
        op_sites_list=op_sites_list,
        lvals=lvals,
        has_obc=has_obc,
        get_real=get_real,
        staggered_basis=staggered_basis,
        site_basis=site_basis,
    )
    # Construct the dictionary of operators and their names needed to construct the QMB twobody term
    ops, op_names_list = construct_operator_list(
        op_list, op_sites_list, lvals, has_obc, staggered_basis, site_basis
    )
    return qmb_operator(ops, op_names_list, get_real)


def construct_operator_list(
    op_list, op_sites_list, lvals, has_obc, staggered_basis, site_basis
):
    """
    Constructs a dictionary of operators and a list of their names for a quantum many-body lattice.
    Each operator is placed at specified positions on the lattice, and its basis is projected according to the site's characteristics.

    Args:
        operators (list of scipy.sparse matrices): Single-site operators to be placed on the lattice.

        op_sites_list (list of ints): Indices in the lattice where each operator from 'operators' should be placed.

        lvals (list of ints): Dimensions of the lattice, representing the number of sites in each dimension.

        has_obc (bool): Specifies if the lattice has open boundary conditions (True) or periodic boundary conditions (False).

        staggered_basis (bool): Indicates if a staggered basis is used for the lattice.

        site_basis (dict): A dictionary containing the basis projectors for each site, keyed by site labels.

    Returns:
        tuple: A tuple containing a dictionary of operators keyed by their names and a list of operator names.
    """
    # Validate type of parameteres
    validate_parameters(
        op_list=op_list,
        op_sites_list=op_sites_list,
        lvals=lvals,
        has_obc=has_obc,
        staggered_basis=staggered_basis,
        site_basis=site_basis,
    )
    # Define the identity operator
    ID = identity(op_list[0].shape[0])
    # Empty dictionary of operators and list of their names
    ops_dict = {}
    op_names_list = []
    for ii in range(np.prod(lvals)):
        # Assign a name to the operator at position ii
        tmp = 0
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
        basis_label = get_site_label(coords, lvals, has_obc, staggered_basis)
        # Apply projection of the operator on the proper basis for that lattice position (and update its name)
        op, op_name = apply_basis_projection(op, op_name, basis_label, site_basis)
        # Save operator and its name
        ops_dict[op_name] = op
        op_names_list.append(op_name)
    return ops_dict, op_names_list


def apply_basis_projection(op, op_name, basis_label, site_basis):
    """
    Applies basis projection to an operator for a specific lattice site and updates its name.

    Args:
        op (scipy.sparse matrix): The operator to be projected.

        op_name (str): The name of the operator.

        basis_label (str): The label identifying the basis projection applicable to the lattice site.

        site_basis (dict of scipy.sparse matrices): Dictionary containing the basis projectors for each site, keyed by site labels.

    Returns:
        tuple: A tuple containing the projected operator and its updated name.
    """
    # Validate type of parameteres
    validate_parameters(
        op_list=[op],
        op_names_list=[op_name],
        site_basis=site_basis,
    )
    if len(basis_label) > 0 and site_basis is not None:
        # Apply projection of the operator on the proper basis for that lattice position
        op = site_basis[basis_label].transpose() * op * site_basis[basis_label]
        # Update the name of the operator with the label
        op_name = f"{op_name}_{basis_label}"
    return op, op_name
