import numpy as np
from ed_lgt.operators import SU2_dressed_site_operators, SU2_gauge_invariant_states
from qtealeaves.operators.tnoperators import TNOperators
from qtealeaves.modeling import QuantumModel, LocalTerm, TwoBodyTerm1D


def product_state_preparation(L, local_dim):
    product_state = np.zeros((L, local_dim))
    # Start from the staggered bare vacuum
    for xx in range(L):
        if (xx) % 2 == 0:
            product_state[xx, 0] = 1
        else:
            product_state[xx, 4] = 1
    return product_state


def SU2_gauge_invariant_ops(spin, pure_theory, lattice_dim):
    in_ops = SU2_dressed_site_operators(spin, pure_theory, lattice_dim)
    gauge_basis, _ = SU2_gauge_invariant_states(spin, pure_theory, lattice_dim)
    ops = {}
    ops_size = gauge_basis["site"].shape[1]
    for op in in_ops.keys():
        ops[op] = np.array(
            (gauge_basis["site"].T @ in_ops[op] @ gauge_basis["site"]).todense()
        )
        ops["id"] = np.eye(ops_size)
    return ops


class TN_SU2_operators(TNOperators):
    def __init__(self, spin=0.5, pure_theory=False, lattice_dim=1):
        ops = SU2_gauge_invariant_ops(spin, pure_theory, lattice_dim)
        super().__init__()
        for key, value in ops.items():
            self[key] = value


def SU2_Hamiltonian_couplings(dim=1, g=1.0, m=3.0, theta=0.0):
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
        "theta": -complex(0, theta * g),  # THETA TERM coupling
    }
    if dim > 1:
        coeffs["B"] = B  # MAGNETIC FIELD coupling
    if m is not None:
        t = 0.5
        coeffs |= {
            "t_x_even": complex(0, t),  # x HOPPING (EVEN SITES)
            "m_even": m,
            "m_odd": -m,
        }
    return coeffs


def staggered_mask(site: str, params: dict):
    """Staggered mask function, params it's needed by qtl because it must be a callable"""

    length = params["L"]
    tmp = np.zeros(length, dtype=bool)
    if site == "even":
        for xx in range(length):
            if (xx) % 2 == 0:
                tmp[xx] = True
    elif site == "odd":
        for xx in range(length):
            if (xx) % 2 == 1:
                tmp[xx] = True
    return tmp


def staggered_even_mask(params):
    return staggered_mask("even", params)


def staggered_odd_mask(params):
    return staggered_mask("odd", params)


def get_SU2_model(open_bc=True):
    model_name = lambda params: "SU2_L%2.4f" % (params["L"])
    model = QuantumModel(1, "L", name=model_name)
    # ---------------------------------------------------------------------------
    # ELECTRIC ENERGY
    op_name = "E_square"
    model += LocalTerm(op_name, strength="E", prefactor=+1)
    # -----------------------------------------------------------------------
    # STAGGERED MASS TERM
    op_name = "N_tot"
    model += LocalTerm(op_name, strength="m", prefactor=+1, mask=staggered_even_mask)
    model += LocalTerm(op_name, strength="m", prefactor=-1, mask=staggered_odd_mask)
    # --------------------------------------------------------------------
    #  HOPPING
    op_names_list = [f"Qpx_dag", f"Qmx"]
    model += TwoBodyTerm1D(
        op_names_list,
        shift=1,
        strength=f"t_x_even",
        prefactor=-1,
        has_obc=open_bc,
    )
    op_names_list = [f"Qpx", f"Qmx_dag"]
    model += TwoBodyTerm1D(
        op_names_list,
        shift=1,
        strength=f"t_x_even",
        prefactor=1,
        has_obc=open_bc,
    )
    # ---------------------------------------------------------------------------
    ops = TN_SU2_operators()
    return model, ops
