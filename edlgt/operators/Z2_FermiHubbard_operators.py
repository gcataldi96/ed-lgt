import numpy as np
from scipy.sparse import csr_matrix, kron, identity as ID
from itertools import product
from edlgt.modeling import (
    qmb_operator as qmb_op,
    get_lattice_borders_labels,
    LGT_border_configs,
)
from .bose_fermi_operators import fermi_operators
from .Zn_operators import Zn_rishon_operators
from .SU2_operators import SU2_generators

__all__ = [
    "Z2_FermiHubbard_gauge_invariant_states",
    "Z2_FermiHubbard_dressed_site_operators",
]


def Z2_FermiHubbard_gauge_invariant_states(lattice_dim):
    if not np.isscalar(lattice_dim) or not isinstance(lattice_dim, int):
        raise TypeError(
            f"lattice_dim must be SCALAR & INTEGER, not {type(lattice_dim)}"
        )
    # List of borders/corners of the lattice
    borders = get_lattice_borders_labels(lattice_dim)
    # List of configurations for each element of the dressed site
    single_rishon_configs = np.arange(2)
    matter_config = np.array([0, 1, 1, 2])
    dressed_site_config_list = [single_rishon_configs for i in range(2 * lattice_dim)]
    dressed_site_config_list.insert(0, matter_config)
    core_labels = ["site"]
    # Define useful quantities
    gauge_states = {}
    row = {}
    col_counter = {}
    for ii, main_label in enumerate(core_labels):
        row_counter = -1
        gauge_states[main_label] = []
        row[main_label] = []
        col_counter[main_label] = -1
        for label in borders:
            gauge_states[f"{main_label}_{label}"] = []
            row[f"{main_label}_{label}"] = []
            col_counter[f"{main_label}_{label}"] = -1
        # Look at all the possible configurations of gauge links and matter fields
        for config in product(*dressed_site_config_list):
            # Update row counter
            row_counter += 1
            # Check Gauss Law
            if sum(config) % 2 == 0:
                # FIX row and col of the site basis
                row[main_label].append(row_counter)
                col_counter[main_label] += 1
                # Save the gauge invariant state
                gauge_states[main_label].append(config)
                # Get the config labels
                label = LGT_border_configs(config, 0, pure_theory=False)
                if label:
                    # save the config state also in the specific subset for the specif border
                    for ll in label:
                        gauge_states[f"{main_label}_{ll}"].append(config)
                        row[f"{main_label}_{ll}"].append(row_counter)
                        col_counter[f"{main_label}_{ll}"] += 1
    # Build the basis as a sparse matrix
    gauge_basis = {}
    for name in list(gauge_states.keys()):
        data = np.ones(col_counter[name] + 1, dtype=float)
        x = np.asarray(row[name])
        y = np.arange(col_counter[name] + 1)
        gauge_basis[name] = csr_matrix(
            (data, (x, y)), shape=(row_counter + 1, col_counter[name] + 1)
        )
        # Save the gauge states as a np.array
        gauge_states[name] = np.asarray(gauge_states[name])
    return gauge_basis, gauge_states


def Z2_FermiHubbard_dressed_site_operators(lattice_dim=2):
    """
    This function generates the dressed-site operators of the
    Hubbard Hamiltonian coupled with a Z2 gauge field, in any lattice dimension.

    Args:
        lattice_dim: (int) lattice dimensions

    Returns:
        dict: dictionary with all the operators of the QED (pure or full) Hamiltonian
    """
    if not np.isscalar(lattice_dim) and not isinstance(lattice_dim, int):
        raise TypeError(
            f"lattice_dim must be SCALAR & INTEGER, not {type(lattice_dim)}"
        )
    # Get the Rishon operators according to the chosen n representation s
    in_ops = Zn_rishon_operators(2, False)
    in_ops.update(fermi_operators(has_spin=True))
    in_ops.update(SU2_generators(spin=1 / 2, matter=True))
    in_ops["N_up_half"] = in_ops["N_up"] - 0.5 * ID(4)
    in_ops["N_down_half"] = in_ops["N_down"] - 0.5 * ID(4)
    in_ops["N_pair_half"] = in_ops["N_up_half"] * in_ops["N_down_half"]
    # Dictionary for operators
    ops = {}
    # --------------------------------------------------------------------------------
    # Rishon Number operators
    for op in ["n", "P"]:
        if lattice_dim == 1:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "IDz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["IDz", op])
        elif lattice_dim == 2:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "IDz", "IDz", "IDz"])
            ops[f"{op}_my"] = qmb_op(in_ops, ["IDz", op, "IDz", "IDz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["IDz", "IDz", op, "IDz"])
            ops[f"{op}_py"] = qmb_op(in_ops, ["IDz", "IDz", "IDz", op])
        elif lattice_dim == 3:
            ops[f"{op}_mx"] = qmb_op(in_ops, [op, "IDz", "IDz", "IDz", "IDz", "IDz"])
            ops[f"{op}_my"] = qmb_op(in_ops, ["IDz", op, "IDz", "IDz", "IDz", "IDz"])
            ops[f"{op}_mz"] = qmb_op(in_ops, ["IDz", "IDz", op, "IDz", "IDz", "IDz"])
            ops[f"{op}_py"] = qmb_op(in_ops, ["IDz", "IDz", "IDz", op, "IDz", "IDz"])
            ops[f"{op}_px"] = qmb_op(in_ops, ["IDz", "IDz", "IDz", "IDz", op, "IDz"])
            ops[f"{op}_pz"] = qmb_op(in_ops, ["IDz", "IDz", "IDz", "IDz", "IDz", op])
    for op in ops.keys():
        ops[op] = kron(in_ops["ID_psi"], ops[op])
    # --------------------------------------------------------------------------------
    # Electric field operator
    ops["E"] = 0
    for s in "mp":
        for d in "xyz"[:lattice_dim]:
            ops["E"] += 0.5 * ops[f"P_{s}{d}"]
    # --------------------------------------------------------------------------------
    # Hopping operators
    for s in ["up", "down"]:
        if lattice_dim == 1:
            ops[f"Q{s}_mx_dag"] = qmb_op(in_ops, [f"psi_{s}_dag_P", "Zm", "IDz"])
            ops[f"Q{s}_px_dag"] = qmb_op(in_ops, [f"psi_{s}_dag_P", "P", "Zp"])
        elif lattice_dim == 2:
            ops[f"Q{s}_mx_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag_P", "Zm", "IDz", "IDz", "IDz"]
            )
            ops[f"Q{s}_my_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag_P", "P", "Zm", "IDz", "IDz"]
            )
            ops[f"Q{s}_px_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag_P", "P", "P", "Zp", "IDz"]
            )
            ops[f"Q{s}_py_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag_P", "P", "P", "P", "Zp"]
            )
        elif lattice_dim == 3:
            ops[f"Q{s}_mx_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag_P", "Zm", "IDz", "IDz", "IDz", "IDz", "IDz"]
            )
            ops[f"Q{s}_my_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag_P", "P", "Zm", "IDz", "IDz", "IDz", "IDz"]
            )
            ops[f"Q{s}_mz_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag_P", "P", "P", "Zm", "IDz", "IDz", "IDz", "IDz"]
            )
            ops[f"Q{s}_px_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag_P", "P", "P", "P", "Zp", "IDz", "IDz"]
            )
            ops[f"Q{s}_py_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag_P", "P", "P", "P", "P", "Zp", "IDz"]
            )
            ops[f"Q{s}_pz_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag_P", "P", "P", "P", "P", "P", "Zp"]
            )
    # Add dagger operators
    Qs = {}
    for op in ops:
        dag_op = op.replace("_dag", "")
        Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
    ops.update(Qs)
    # --------------------------------------------------------------------------------
    # Psi NUMBER OPERATORS
    for label in [
        "N_up",
        "N_down",
        "N_tot",
        "N_single",
        "N_pair",
        "N_pair_half",
        "Sz_psi",
        "S2_psi",
    ]:
        ops[label] = qmb_op(in_ops, [label] + ["IDz" for i in range(2 * lattice_dim)])
    # --------------------------------------------------------------------------------
    if lattice_dim == 2:
        # LOCAL OPERATOR WITH THE SUM OF RISHON NUMBERS ALONG EACH LINK
        ops["n_total"] = 0
        for s in "mp":
            for d in "xyz"[:lattice_dim]:
                ops["n_total"] += ops[f"n_{s}{d}"]

        # Sigma X Cross Operator
        ops["X_Cross"] = qmb_op(in_ops, ["ID_psi", "P", "P", "P", "P"])
        # ----------------------------------------------------------------------------
        # Corner Operators
        ops["C_pxpy"] = qmb_op(in_ops, ["ID_psi", "IDz", "IDz", "Zp_P", "Zp"])
        ops["C_pymx"] = qmb_op(in_ops, ["ID_psi", "P_Zm_dag", "P", "P", "Zp"])
        ops["C_mxmy"] = qmb_op(in_ops, ["ID_psi", "Zm_P", "Zm", "IDz", "IDz"])
        ops["C_mypx"] = qmb_op(in_ops, ["ID_psi", "IDz", "Zm_P", "Zp_dag", "IDz"])
        # ----------------------------------------------------------------------------
        # Topological Operator along axis
        ops["Sz_mypy"] = qmb_op(in_ops, ["ID_psi", "IDz", "Zm_dag_P", "P", "Zp"])
        ops["Sz_mxpx"] = qmb_op(in_ops, ["ID_psi", "Zm_dag_P", "P", "Zp", "IDz"])
        ops["Sx_pymy"] = qmb_op(in_ops, ["ID_psi", "IDz", "P", "IDz", "P"])
        ops["Sx_pxmx"] = qmb_op(in_ops, ["ID_psi", "P", "IDz", "P", "IDz"])
    return ops
