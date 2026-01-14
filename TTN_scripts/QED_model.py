import numpy as np
from ed_lgt.operators import QED_dressed_site_operators, QED_gauge_invariant_states
from qtealeaves.operators.tnoperators import TNOperators
from qtealeaves.modeling import (
    QuantumModel,
    LocalTerm,
    TwoBodyTerm2D,
    PlaquetteTerm2D,
)


mapping_func = lambda idx: "even" if idx & 2 == 0 else "odd"


def product_state_preparation(L, local_dim):
    product_state = np.zeros((L, L, local_dim))
    # Start from the staggered bare vacuum
    for xx in range(L):
        for yy in range(L):
            product_state[xx, yy, 9] = 1
    return product_state


def QED_gauge_invariant_ops(spin, pure_theory, lattice_dim):
    in_ops = QED_dressed_site_operators(spin, pure_theory, lattice_dim)
    gauge_basis, _ = QED_gauge_invariant_states(spin, pure_theory, lattice_dim)
    ops = {}
    op_names_list = ["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]
    ops_size = gauge_basis["site"].shape[1]
    for op in in_ops.keys():
        ops[op] = np.array(
            (gauge_basis["site"].T @ in_ops[op] @ gauge_basis["site"]).todense()
        )
    for name in op_names_list:
        ops[f"{name}_dag"] = np.conj(ops[name]).T

    ops["id"] = np.eye(ops_size)
    return ops


class TN_QED_operators(TNOperators):
    def __init__(self, spin=1, pure_theory=True, lattice_dim=2):
        ops = QED_gauge_invariant_ops(spin, pure_theory, lattice_dim)
        super().__init__()
        for key, value in ops.items():
            self[key] = value


def QED_Hamiltonian_couplings(dim=2, g=1.0, m=None, theta=0.0, alpha=10):
    """
    This function provides the QED Hamiltonian coefficients
    starting from the gauge coupling g and the bare mass parameter m

    Args:
        pure_theory (bool): True if the theory does not include matter

        g (scalar): gauge coupling

        m (scalar, optional): bare mass parameter

    Returns:
        dict: dictionary of Hamiltonian coefficients
    """
    if dim == 1:
        E = g / 2
    elif dim == 2:
        E = g / 2
        B = -1 / (2 * g)
    else:
        E = g / 2
        B = -1 / (2 * g)
    # DICTIONARY WITH MODEL COEFFICIENTS
    coeffs = {
        "g": g,
        "E": E,  # ELECTRIC FIELD coupling
        "B": B,  # MAGNETIC FIELD coupling
        "theta": -complex(0, theta * g),  # THETA TERM coupling
        "alpha": alpha,  # COUPLING FOR THE PENALTY TERM
    }
    if m is not None:
        t = 0.5
        coeffs |= {
            "t_x_even": complex(0, t),  # x HOPPING (EVEN SITES)
            "t_x_odd": complex(0, t),  # x HOPPING (ODD SITES)
            "t_y_even": t,  # y HOPPING (EVEN SITES)
            "t_y_odd": t,  # y HOPPING (ODD SITES)
            "t_z_even": t,  # z HOPPING (EVEN SITES)
            "t_z_odd": t,  # z HOPPING (ODD SITES)
            "m_even": m,
            "m_odd": -m,
        }
    return coeffs


def get_QED_model(open_bc=True):
    model_name = lambda params: "QED_L%2.4f" % (params["L"])
    model = QuantumModel(dim=2, lvals="L", name=model_name)
    # ---------------------------------------------------------------------------
    # ELECTRIC ENERGY
    op_name = "E2"
    model += LocalTerm(op_name, strength="E")
    # ---------------------------------------------------------------------------
    # TWO-BODY PENALTY TERM
    op_names_list = ["E_px", "E_mx"]
    model += TwoBodyTerm2D(op_names_list, shift=[1, 0], prefactor=20, has_obc=open_bc)
    op_names_list = ["E_py", "E_my"]
    model += TwoBodyTerm2D(op_names_list, shift=[0, 1], prefactor=20, has_obc=open_bc)
    op_name = "E2"
    model += LocalTerm(op_name, prefactor=10)
    # --------------------------------------------
    # PLAQUETTE TERM: MAGNETIC INTERACTION
    op_names_list = ["C_px,py", "C_my,px", "C_py,mx", "C_mx,my"]
    model += PlaquetteTerm2D(op_names_list, strength="B", prefactor=1, has_obc=open_bc)
    op_names_list = ["C_px,py_dag", "C_my,px_dag", "C_py,mx_dag", "C_mx,my_dag"]
    model += PlaquetteTerm2D(op_names_list, strength="B", prefactor=1, has_obc=open_bc)
    # ---------------------------------------------------------------------------
    ops = TN_QED_operators()
    return model, ops
