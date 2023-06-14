# %%
import numpy as np
from itertools import product
from scipy.sparse import csr_matrix, diags, identity, kron

# from simsio import logger
# from modeling import qmb_operator as qmb_op

__all__ = [
    "get_Z2_Hamiltonian_couplings",
    "Z2_dressed_site_operators",
    "Z2_gauge_invariant_states",
]


def qmb_op(ops, op_list, add_dagger=False, get_real=False, get_imag=False):
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


def get_Z2_Hamiltonian_couplings(g, k):
    """
    This function provides the Z2 Hamiltonian coefficients
    starting from the gauge couplings g and K
    Args:
        g (scalar): ELECTRIC FIELD coupling
        k (scalar): MAGNETIC FIELD coupling

    Returns:
        dict: dictionary of Hamiltonian coefficients
    """
    # DICTIONARY WITH MODEL COEFFICIENTS
    coeffs = {
        "g": g,  # ELECTRIC FIELD coupling
        "k": k,  # MAGNETIC FIELD coupling
        "eta": 10 * max(g, k),  # PENALTY
    }
    print(f"LINK SYMMETRY PENALTY {coeffs['eta']}")
    return coeffs


def Z2_border_configs(config):
    """
    This function fixes the value of the electric field on
    lattices with open boundary conditions (has_obc=True).

    Args:
        config (list of ints): configuration of internal rishons in
        the single dressed site basis, ordered as follows:
        [n_matter, n_mx, n_my, n_px, n_py]

    Returns:
        list of strings: list of configs corresponding to a border/corner of the lattice
        with a fixed value of the electric field
    """
    label = []
    if config[1] == 0:
        label.append("mx")
    if config[2] == 0:
        label.append("my")
    if config[3] == 0:
        label.append("px")
    if config[4] == 0:
        label.append("py")
    if (config[1] == 0) and (config[2] == 0):
        label.append("mx_my")
    if (config[1] == 0) and (config[4] == 0):
        label.append("mx_py")
    if (config[2] == 0) and (config[3] == 0):
        label.append("my_px")
    if (config[3] == 0) and (config[4] == 0):
        label.append("px_py")
    return label


def Z2_gauge_invariant_states():
    """
    This function generates the gauge invariant basis of a Z2 LGT
    in a 2D rectangular lattice where gauge and (eventually)
    matter degrees of freedom are merged in a compact-site notation
    by exploiting a rishon-based quantum link model.
    NOTE: the gague invariant basis is different for even
    and odd sites due to the staggered fermion solution
    NOTE: the function provides also a restricted basis for sites
    on the borderd of the lattice where not all the configurations
    are allowed (the external rishons/gauge fields do not contribute)

    Args:
        lattice_dim (int, optional): # of spatial dimensions. Defaults to 2.
        pure (bool): if true corresponds to the theory in absence of matter
    """
    Z2_basis = {}
    Z2_states = {}
    # RUN OVER BORDER & CORNER CONFIGS
    for label in ["core", "mx", "my", "px", "py", "mx_my", "mx_py", "my_px", "px_py"]:
        Z2_states[label] = []
    # RUN OVER GAUGE LINK DEGREES of FREEDOM
    for n_mx, n_my, n_px, n_py in product([0, 1], repeat=4):
        # DEFINE GAUSS LAW
        n_tot = n_mx + n_px + n_my + n_py
        # CHECK GAUSS LAW
        if n_tot % 2 == 0:
            # SAVE THE STATE
            config = [0, n_mx, n_my, n_px, n_py]
            Z2_states["core"].append(config)
            # GET THE CONFIG LABEL
            label = Z2_border_configs(config)
            if label:
                # save the config state in the specific subset for the specif border/corner
                for ll in label:
                    Z2_states[f"{ll}"].append(config)
        # Build the basis as a sparse matrix
        for ll in [
            "",
            "_mx",
            "_my",
            "_px",
            "_py",
            "_mx_my",
            "_mx_py",
            "_my_px",
            "_px_py",
        ]:
            a = 0
    return Z2_basis, Z2_states


# %%
def Z2_inner_site_operators():
    ops = {}
    # Define the MATTER FIELDS OPERATORS
    ops["psi"] = diags(np.array([1], dtype=float), 1, (2, 2))
    ops["psi_dag"] = ops["psi"].transpose()
    ops["P"] = diags(np.array([1, -1], dtype=float), 0, (2, 2))
    ops["N"] = ops["psi_dag"] * ops["psi"]
    ops["ID"] = identity(2)
    ops["gamma"] = ops["psi"] + ops["psi_dag"]
    return ops


def Z2_dressed_site_operators():
    # Get inner site operators (Rishons + Matter fields)
    in_ops = Z2_inner_site_operators()
    # Dictionary for operators
    ops = {}
    ops["C_px,py"] = qmb_op(in_ops, ["ID", "ID", "P", "ID"])
    ops["C_py,mx"] = qmb_op(in_ops, ["ID", "ID", "ID", "P"])
    ops["C_mx,my"] = qmb_op(in_ops, ["P", "ID", "ID", "ID"])
    ops["C_my,px"] = qmb_op(in_ops, ["ID", "P", "ID", "ID"])
    # Rishon Number operators
    for op in ["gamma", "N"]:
        ops[f"{op}_mx"] = qmb_op(in_ops, [op, "ID", "ID", "ID"])
        ops[f"{op}_my"] = qmb_op(in_ops, ["ID", op, "ID", "ID"])
        ops[f"{op}_px"] = qmb_op(in_ops, ["ID", "ID", op, "ID"])
        ops[f"{op}_py"] = qmb_op(in_ops, ["ID", "ID", "ID", op])
    return ops
