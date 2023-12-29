# %%
import numpy as np
from scipy.sparse import csr_matrix, kron
from itertools import product
from math import prod
from ed_lgt.operators import (
    Zn_rishon_operators,
    fermi_operators,
)

from ed_lgt.modeling import Ground_State, LocalTerm, TwoBodyTerm
from ed_lgt.modeling import (
    qmb_operator as qmb_op,
    get_lattice_borders_labels,
    LGT_border_configs,
    check_link_symmetry,
    entanglement_entropy,
    get_reduced_density_matrix,
    diagonalize_density_matrix,
    get_state_configurations,
    truncation,
    lattice_base_configs,
)
from ed_lgt.tools import check_hermitian


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
    in_ops |= fermi_operators(has_spin=True)
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
    # Hopping operators
    for s in ["up", "down"]:
        if lattice_dim == 1:
            ops[f"Q{s}_mx_dag"] = qmb_op(in_ops, [f"psi_{s}_dag", "Zm", "IDz"])
            ops[f"Q{s}_px_dag"] = qmb_op(in_ops, [f"psi_{s}_dag", "P", "Zp"])
        elif lattice_dim == 2:
            ops[f"Q{s}_mx_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag", "Zm", "IDz", "IDz", "IDz"]
            )
            ops[f"Q{s}_my_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag", "P", "Zm", "IDz", "IDz"]
            )
            ops[f"Q{s}_px_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag", "P", "P", "Zp", "IDz"]
            )
            ops[f"Q{s}_py_dag"] = qmb_op(in_ops, [f"psi_{s}_dag", "P", "P", "P", "Zp"])
        elif lattice_dim == 3:
            ops[f"Q{s}_mx_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag", "Zm", "IDz", "IDz", "IDz", "IDz", "IDz"]
            )
            ops[f"Q{s}_my_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag", "P", "Zm", "IDz", "IDz", "IDz", "IDz"]
            )
            ops[f"Q{s}_mz_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag", "P", "P", "Zm", "IDz", "IDz", "IDz", "IDz"]
            )
            ops[f"Q{s}_px_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag", "P", "P", "P", "Zp", "IDz", "IDz"]
            )
            ops[f"Q{s}_py_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag", "P", "P", "P", "P", "Zp", "IDz"]
            )
            ops[f"Q{s}_pz_dag"] = qmb_op(
                in_ops, [f"psi_{s}_dag", "P", "P", "P", "P", "P", "Zp"]
            )
    # Add dagger operators
    Qs = {}
    for op in ops:
        dag_op = op.replace("_dag", "")
        Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
    ops |= Qs
    # --------------------------------------------------------------------------------
    # Psi NUMBER OPERATORS
    for label in ["up", "down", "tot", "single", "pair"]:
        ops[f"N_{label}"] = qmb_op(
            in_ops, [f"N_{label}"] + ["IDz" for i in range(2 * lattice_dim)]
        )
    return ops


# N eigenvalues
n_eigs = 1
# LATTICE DIMENSIONS
lvals = [2, 2]
dim = len(lvals)
directions = "xyz"[:dim]
# TOTAL NUMBER OF LATTICE SITES & particles
n_sites = prod(lvals)
# BOUNDARY CONDITIONS
has_obc = True
# ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
in_ops = Zn_rishon_operators(2, False)
ops = Z2_FermiHubbard_dressed_site_operators(lattice_dim=dim)
# ACQUIRE SU2 BASIS and GAUGE INVARIANT STATES
M, states = Z2_FermiHubbard_gauge_invariant_states(lattice_dim=dim)
# ACQUIRE LOCAL DIMENSION OF EVERY SINGLE SITE
lattice_base, loc_dims = lattice_base_configs(M, lvals, has_obc, staggered=False)
loc_dims = loc_dims.transpose().reshape(n_sites)
lattice_base = lattice_base.transpose().reshape(n_sites)
print("local dimensions:", loc_dims)
# OBTAIN OPERATORS FOR TTN SIMULATIONS
TTN_ops = {}
for op in ops.keys():
    TTN_ops[op] = M["site"].transpose() * ops[op] * M["site"]
# Hamiltonian Couplings
coeffs = {"t": -1, "V": 3, "eta": 100}
# CONSTRUCT THE HAMILTONIAN
H = 0
h_terms = {}
# -------------------------------------------------------------------------------
# LINK PENALTIES & Border penalties
for d in directions:
    op_name_list = [f"n_p{d}", f"n_m{d}"]
    op_list = [ops[op] for op in op_name_list]
    # Define the Hamiltonian term
    h_terms[f"W{d}"] = TwoBodyTerm(
        axis=d,
        op_list=op_list,
        op_name_list=op_name_list,
        lvals=lvals,
        has_obc=has_obc,
        site_basis=M,
    )
    H += h_terms[f"W{d}"].get_Hamiltonian(strength=-2 * coeffs["eta"], add_dagger=False)
    # SINGLE SITE OPERATORS needed for the LINK SYMMETRY/OBC PENALTIES
    for s in "mp":
        op_name = f"n_{s}{d}"
        h_terms[op_name] = LocalTerm(
            ops[op_name],
            op_name,
            lvals=lvals,
            has_obc=has_obc,
            site_basis=M,
        )
        H += h_terms[op_name].get_Hamiltonian(strength=coeffs["eta"])
# -------------------------------------------------------------------------------
# COULOMB POTENTIAL
h_terms["V"] = LocalTerm(
    ops["N_pair"],
    "N_pair",
    lvals=lvals,
    has_obc=has_obc,
    site_basis=M,
)
H += h_terms["V"].get_Hamiltonian(strength=coeffs["V"])
# -------------------------------------------------------------------------------
# HOPPING
for d in directions:
    for s in ["up", "down"]:
        # Define the list of the 2 non trivial operators
        op_name_list = [f"Q{s}_p{d}_dag", f"Q{s}_m{d}"]
        op_list = [ops[op] for op in op_name_list]
        # Define the Hamiltonian term
        h_terms[f"{d}_hop_{s}"] = TwoBodyTerm(
            d,
            op_list,
            op_name_list,
            lvals=lvals,
            has_obc=has_obc,
            site_basis=M,
        )
        H += h_terms[f"{d}_hop_{s}"].get_Hamiltonian(
            strength=coeffs["t"], add_dagger=True
        )
# ===========================================================================
# CHECK THAT THE HAMILTONIAN IS HERMITIAN
check_hermitian(H)
# DIAGONALIZE THE HAMILTONIAN
GS = Ground_State(H, n_eigs)
# Dictionary for results
res = {}
res["energy"] = GS.Nenergies
# ===========================================================================
# DEFINE THE OBSERVABLE LIST
res["entropy"] = []
res["rho_eigvals"] = []
obs_list = [f"n_{s}{d}" for s in "mp" for d in directions]
obs_list += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
for obs in obs_list:
    h_terms[obs] = LocalTerm(ops[obs], obs, lvals, has_obc, site_basis=M)
    res[obs] = []
# ===========================================================================
for ii in range(n_eigs):
    print("====================================================")
    print(f"{ii} ENERGY: {format(res['energy'][ii], '.9f')}")
    # ENTROPY of a BIPARTITION
    # res["entropy"].append(
    #    entanglement_entropy(GS.Npsi[:, ii], loc_dims, lvals, lvals[0])
    # )
    # COMPUTE THE REDUCED DENSITY MATRIX
    rho = get_reduced_density_matrix(GS.Npsi[:, ii], loc_dims, lvals, 0)
    eigvals, _ = diagonalize_density_matrix(rho)
    res["rho_eigvals"].append(eigvals)
    print(f"DM eigvals {res['rho_eigvals']}")
    if has_obc:
        # GET STATE CONFIGURATIONS
        get_state_configurations(truncation(GS.Npsi[:, ii], 1e-10), loc_dims, lvals)
    # ===========================================================================
    # MEASURE OBSERVABLES:
    # ===========================================================================
    for obs in obs_list:
        h_terms[obs].get_expval(GS.Npsi[:, ii])
        res[obs].append(h_terms[obs].avg)
    # CHECK LINK SYMMETRIES
    for ax in directions:
        check_link_symmetry(
            ax, h_terms[f"n_p{ax}"], h_terms[f"n_m{ax}"], value=0, sign=-1
        )
print(f"Energies {res['energy']}")
if n_eigs > 1:
    res["DeltaE"] = np.abs(res["energy"][0] - res["energy"][1])

# %%
