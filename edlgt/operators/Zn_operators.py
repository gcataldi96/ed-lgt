# %%
import numpy as np
from numpy.linalg import eig, eigh
from itertools import product, combinations
from scipy.sparse import csr_matrix, diags, identity, kron
from scipy.sparse.linalg import norm
from copy import deepcopy
from .bose_fermi_operators import fermi_operators as Zn_matter_operators
from edlgt.tools import anti_commutator as anti_comm
from edlgt.tools import check_commutator as check_comm
from edlgt.modeling import qmb_operator as qmb_op
from edlgt.modeling import get_lattice_borders_labels


__all__ = [
    "Zn_rishon_operators",
    "Zn_corner_magnetic_basis",
    "Zn_dressed_site_operators",
    "Zn_gauge_invariant_states",
    "Zn_gauge_invariant_ops",
    "get_lambda_subspace",
]


def check_Zn_Gauss_law_magnetic_basis(L, cutoff):
    """
    This function constructs the Gauss Law operator of a Z_{L} in the Magnetic Basis and look at how it mixes the states within a truncation of the Magnetic Basis itself (from a n-representation L to a n-representation cutoff).

    Args:
        L (scalar, int): Initial value of the n-representation of the Electric Basis of the Z_{L} group

        cutoff (scalar, int): value of the n-representation at which we perform the trunctation of the Magnetic Basis
    """
    # Dimension of the gauge link Hilbert space (in the Electric Basis)
    link_size = 2 * L + 1
    # Dimension of the Gauss Law operator acting on a 2D vertex (with 4 attached gauge links)
    vertex_size = link_size**4
    # Dimension of the gauge link Hilbert space in the Truncated Magnetic Basis
    cutoff_size = (2 * cutoff + 1) ** 4
    offset_size = vertex_size - cutoff_size
    # Generate the Gauge Link Operators (U, E) of the Z_{L} theory in the Electric Basis
    ops = Zn_rishon_operators(L)
    # Generate the Fourier Transform that diagonalizes U
    ops["F"] = change_of_basis(L)
    ops["F_dagger"] = ops["F"].conjugate().transpose()
    # Transform E, U in the Magnetic Basis
    ops["U_tilde"] = ops["F"] * (ops["U"] * ops["F_dagger"])
    ops["E_tilde"] = ops["F"] * (ops["E"] * ops["F_dagger"])
    # Check that U_tilde is diagonal
    # print(ops["U_tilde"])
    # Construct the Electric field operators (in the Electric Basis & Magnetic Basis) along each link of a 2D lattice vertex (mx,my,px,pz)
    for op in ["E", "E_tilde"]:
        ops[f"{op}_mx"] = qmb_op(ops, [op, "IDz", "IDz", "IDz"])
        ops[f"{op}_my"] = qmb_op(ops, ["IDz", op, "IDz", "IDz"])
        ops[f"{op}_px"] = qmb_op(ops, ["IDz", "IDz", op, "IDz"])
        ops[f"{op}_py"] = qmb_op(ops, ["IDz", "IDz", "IDz", op])
    # Gauss Law Operator in the Electric Basis: it is diagonal: the zero entries of the diagonal corresponds to Gauge invariante states. Non-zero diagonal entries are the NON-Gauge Invariant configurations
    ops["G"] = 0
    for d in ["x", "y"]:
        for i, s in enumerate(["p", "m"]):
            ops["G"] += (-1) ** (i) * ops[f"E_{s}{d}"]
    # Isolate the only gauge invariant entries (to be put = 1) and put the rest = 0
    zero_entries = np.where(ops["G"].diagonal() == 0, 1, 0)
    # print(zero_entries)
    ops["G"] = csr_matrix(diags(zero_entries, 0))
    # Express the Gauss Law Operator in the Magnetic Basis, by applying (on the left and on the right) 4 Fourier Transforms, one per each gauge link
    ops["G_mag"] = qmb_op(ops, ["F", "F", "F", "F"]) * (
        ops["G"] * (qmb_op(ops, ["F", "F", "F", "F"]).conj().transpose())
    )
    # Compute the basis P with which we want to reorder states
    labels = np.zeros(vertex_size, dtype=bool)
    for i, config in enumerate(product(range(-L, L + 1), repeat=4)):
        if (np.abs(config) < cutoff + 1).all():
            # print(i, config)
            labels[i] = True
    # Reorder the states of the Magnetic Basis performing a change of basis P^{*}G*P
    reordered_indices = np.concatenate((np.where(labels)[0], np.where(~labels)[0]))
    # P = csr_matrix(np.eye(vertex_size)[:, reordered_indices])
    P = csr_matrix(
        (np.ones(vertex_size), (reordered_indices, np.arange(vertex_size))),
        shape=(vertex_size, vertex_size),
    )
    ops["G_reordered"] = P.transpose() * ops["G_mag"] * P
    # look at the selected parts
    A_size = np.arange(cutoff_size)
    B_size = np.arange(cutoff_size, vertex_size, 1)
    A = ops["G_reordered"][A_size, :][:, A_size]
    off_diag = ops["G_reordered"][A_size, :][:, B_size]
    norm_A = norm(A, "fro") / (cutoff_size**2)
    norm_off = norm(off_diag, "fro") / (offset_size**2)
    print(norm_off / norm_A)


def change_of_basis(N):
    prefactor = 1 / np.sqrt(2 * np.pi)
    basis_size = int(2 * N + 1)
    F = np.zeros((basis_size, basis_size), dtype=complex)
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            F[i, j] = prefactor * np.exp(complex(0, 2) * i * j * np.pi / basis_size)
    return csr_matrix(F)


def Zn_rishon_operators(n, pure_theory):
    """
    This function constructs the Operators (E,U) of a Zn Lattice Gauge Theory
    in the Electric Basis (where E is diagonal) and provides the corresponding
    Rishon Operators that are suitable for a dressed site description.

    Args:
        n (int): dimension of the gauge link Hilbert space

        pure_theory (bool): if True, it only provides gauge link operators.
            If False, it also provides matter field operators and requires n to be even.

    Returns:
        dict: dictionary with Zn operators
    """
    if not all([np.isscalar(n), isinstance(n, int)]):
        raise TypeError(f"n must be SCALAR & INTEGER, not {type(n)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    # Size of the
    size = n
    if not pure_theory:
        if size % 2 != 0:
            raise ValueError(
                f"The dressed site theory with Matter works only for Z_n with n even"
            )
    shape = (size, size)
    # PARALLEL TRANSPORTER
    U_diag = [np.ones(size - 1), np.ones(1)]
    ops = {}
    ops["U"] = diags(U_diag, [-1, size - 1], shape)
    # IDENTITY OPERATOR
    ops["IDz"] = identity(size)
    # ELECTRIC FIELD OPERATORS
    ops["n"] = diags(np.arange(size), 0, shape)
    ops["E"] = ops["n"] - 0.5 * (size - 1) * identity(size)
    ops["E_square"] = ops["E"] ** 2
    # RISHON OPERATORS
    if pure_theory:
        # PARITY OPERATOR
        ops["P"] = identity(size)
    else:
        # PARITY OPERATOR
        ops["P"] = diags([(-1) ** i for i in range(size)], 0, shape)
    # RISHON OPERATORS
    ops["Zp"] = ops["U"]
    ops["Zm"] = ops["P"] * ops["U"]
    for s in "pm":
        ops[f"Z{s}_dag"] = ops[f"Z{s}"].transpose()
    # Useful operators for Corners
    ops["Zm_P"] = ops["Zm"] * ops["P"]
    ops["Zp_P"] = ops["Zp"] * ops["P"]
    ops["P_Zm_dag"] = ops["P"] * ops["Zm_dag"]
    ops["P_Zp_dag"] = ops["P"] * ops["Zp_dag"]
    ops["Zm_dag_P"] = ops["Zm_dag"] * ops["P"]
    ops["Zp_dag_P"] = ops["Zp_dag"] * ops["P"]
    if not pure_theory:
        # PERFORM CHECKS
        for s1, s2 in zip("pm", "mp"):
            # CHECK RISHON MODES TO BEHAVE LIKE FERMIONS
            # anticommute with parity
            a = anti_comm(ops[f"Z{s1}"], ops["P"])
            if norm(a) > 1e-15:
                print(a.todense())
                raise ValueError(f"Z{s1} must anticommute with Parity")
            b = anti_comm(ops[f"Z{s1}"], ops[f"Z{s2}_dag"])
            if norm(b) > 1e-15:
                print(b.todense())
                raise ValueError(f"Z{s1} and Z{s2}_dag must anticommute")
    return ops


def Zn_dressed_site_operators(n, pure_theory=False):
    """
    This function generates the dressed-site operators of the 2D Zn Hamiltonian
    (pure or with matter fields) for any possible value n of Zn
    (the larger n the larger the gauge link Hilbert space) in the Electric Basis

    Args:
        n (int): dimension of the gauge link Hilbert space

        pure_theory (bool, optional): If true, the dressed site includes matter fields. Defaults to False.

    Returns:
        dict: dictionary with all the operators of the QED (pure or full) Hamiltonian
    """
    if not np.isscalar(n) and not isinstance(n, int):
        raise TypeError(f"n must be SCALAR & INTEGER, not {type(n)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    # Get the Rishon operators according to the chosen n representation s
    in_ops = Zn_rishon_operators(n, pure_theory)
    # Dictionary for operators
    ops = {}
    # Electric Operators
    for op in ["n", "E", "E_square"]:
        ops[f"{op}_mx"] = qmb_op(in_ops, [op, "IDz", "IDz", "IDz"])
        ops[f"{op}_my"] = qmb_op(in_ops, ["IDz", op, "IDz", "IDz"])
        ops[f"{op}_px"] = qmb_op(in_ops, ["IDz", "IDz", op, "IDz"])
        ops[f"{op}_py"] = qmb_op(in_ops, ["IDz", "IDz", "IDz", op])
    # Corner Operators: in this case the rishons are bosons: no need of parities
    ops["C_px,py"] = qmb_op(in_ops, ["IDz", "IDz", "Zm_P", "Zp_dag"])  # -1
    ops["C_py,mx"] = qmb_op(in_ops, ["P_Zp_dag", "P", "P", "Zm"])
    ops["C_mx,my"] = qmb_op(in_ops, ["Zm_P", "Zp_dag", "IDz", "IDz"])
    ops["C_my,px"] = qmb_op(in_ops, ["IDz", "Zm_P", "Zp_dag", "IDz"])
    if not pure_theory:
        # Acquire also matter field operators
        in_ops |= Zn_matter_operators(has_spin=False)
        if not pure_theory:
            # Update Electric and Corner operators
            for op in ops.keys():
                ops[op] = kron(in_ops["ID_psi"], ops[op])
        # Hopping operators
        ops["Q_mx_dag"] = qmb_op(in_ops, ["psi_dag", "Zm", "IDz", "IDz", "IDz"])
        ops["Q_my_dag"] = qmb_op(in_ops, ["psi_dag", "P", "Zm", "IDz", "IDz"])
        ops["Q_px_dag"] = qmb_op(in_ops, ["psi_dag", "P", "P", "Zp", "IDz"])
        ops["Q_py_dag"] = qmb_op(in_ops, ["psi_dag", "P", "P", "P", "Zp"])
        # Add dagger operators
        Qs = {}
        for op in ops:
            dag_op = op.replace("_dag", "")
            Qs[dag_op] = csr_matrix(ops[op].conj().transpose())
        ops |= Qs
        # Psi Number operators
        ops["N"] = qmb_op(in_ops, ["N", "IDz", "IDz", "IDz", "IDz"])
    # Sum all the E_square operators with coefficient 1/2
    ops["E_square"] = 0
    for s in ["mx", "my", "px", "py"]:
        ops["E_square"] += 0.5 * ops[f"E_square_{s}"]
    # Check that corner operators commute
    corner_list = ["C_mx,my", "C_py,mx", "C_my,px", "C_px,py"]
    for C1, C2 in combinations(corner_list, 2):
        check_comm(ops[C1], ops[C2])
    return ops


def Zn_gauge_invariant_states(n, pure_theory, lattice_dim):
    """
    This function generates the gauge invariant basis of a Zn LGT
    in a d-dimensional lattice where gauge (and matter) degrees of
    freedom are merged in a compact-site notation by exploiting
    a rishon-based quantum link model.

    NOTE: In presence of matter, the gague invariant basis is different for even
    and odd sites due to the staggered fermion solution

    NOTE: The function provides also a restricted basis for sites
    on the borderd of the lattice where not all the configurations
    are allowed (the external rishons/gauge fields do not contribute)

    Args:
        n (scalar, int): size of the Zn Gauge field, corresponding to a gauge Hilbert space of dimension n

        pure_theory (bool,optional): if True, the theory does not involve matter fields

        lattice_dim (int, optional): number of spatial dimensions. Defaults to 2.

    Returns:
        (dict, dict): dictionaries with the basis and the states
    """
    if not all([np.isscalar(n), isinstance(n, int)]):
        raise TypeError(f"n must be SCALAR & INTEGER, not {type(n)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    if not np.isscalar(lattice_dim) or not isinstance(lattice_dim, int):
        raise TypeError(
            f"lattice_dim must be SCALAR & INTEGER, not {type(lattice_dim)}"
        )
    rishon_size = n
    single_rishon_configs = np.arange(rishon_size)
    # List of borders/corners of the lattice
    borders = get_lattice_borders_labels(lattice_dim)
    # List of configurations for each element of the dressed site
    dressed_site_config_list = [single_rishon_configs for i in range(2 * lattice_dim)]
    # Distinction between pure and full theory
    if pure_theory:
        core_labels = ["site"]
        parity = [1]
    else:
        core_labels = ["even", "odd"]
        parity = [1, -1]
        dressed_site_config_list.insert(0, np.arange(2))
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
            # Define Gauss Law
            left = sum(config)
            right = lattice_dim * (rishon_size - 1) + 0.5 * (1 - parity[ii])
            # Check Gauss Law
            if (left - right) % n == 0:
                # FIX row and col of the site basis
                row[main_label].append(row_counter)
                col_counter[main_label] += 1
                # Save the gauge invariant state
                gauge_states[main_label].append(config)
                # Get the config labels
                label = Zn_border_configs(config, n, pure_theory)
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


def Zn_gauge_invariant_ops(n, pure_theory, lattice_dim):
    in_ops = Zn_dressed_site_operators(n, pure_theory)
    basis, states = Zn_gauge_invariant_states(n, pure_theory, lattice_dim)
    E_ops = {}
    label = "site" if pure_theory else "even"
    for op in in_ops.keys():
        E_ops[op] = basis[label].transpose() * in_ops[op] * basis[label]
    return E_ops


def Zn_border_configs(config, n, pure_theory=False):
    """
    This function fixes the value of the electric field on
    lattices with open boundary conditions (has_obc=True).

    For integer spin representation, the offset of E is naturally
    the central value assumed by the rishon number.

    For semi-integer spin representation, there is some freedom
    in the choice of the offset one possible solution is the one
    corresponding to the first negative value of the electric field

    Args:
        config (list of ints): configuration of internal rishons in
        the single dressed site basis, ordered as follows:
        [n_matter, n_mx, n_my, n_px, n_py]

        spin (int): chosen spin representation for U(1)

        pure_theory (bool): True if the theory does not include matter

    Returns:
        list of strings: list of configs corresponding to a border/corner of the lattice
        with a fixed value of the electric field
    """
    if not isinstance(config, list) and not isinstance(config, tuple):
        raise TypeError(f"config should be a LIST, not a {type(config)}")
    if not np.isscalar(n) and not isinstance(n, int):
        raise TypeError(f"n must be SCALAR & INTEGER, not {type(n)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    if (n % 2) == 0:
        off_set = {"p": 0, "m": 0}
    else:
        off_set = {"p": 0, "m": 0}
    label = []
    if not pure_theory:
        config = config[1:]
    if config[0] == off_set["m"]:
        label.append("mx")
    if config[1] == off_set["m"]:
        label.append("my")
    if config[2] == off_set["p"]:
        label.append("px")
    if config[3] == off_set["p"]:
        label.append("py")
    if (config[0] == off_set["m"]) and (config[1] == off_set["m"]):
        label.append("mx_my")
    if (config[0] == off_set["m"]) and (config[3] == off_set["p"]):
        label.append("mx_py")
    if (config[1] == off_set["m"]) and (config[2] == off_set["p"]):
        label.append("px_my")
    if (config[2] == off_set["p"]) and (config[3] == off_set["p"]):
        label.append("px_py")
    return label


def get_lambda_subspace(vals, vecs, atol=1e-10):
    subspaces_vals = []
    subspaces_vecs = []
    for ii, llambda in enumerate(vals):
        if ii == 0:
            tmp = 0
            subspaces_vals.append(llambda)
            subspaces_vecs.append([vecs[:, ii]])
        else:
            if np.abs(llambda - subspaces_vals[tmp]) < atol:
                subspaces_vecs[tmp].append(vecs[:, ii])
            else:
                tmp += 1
                subspaces_vals.append(llambda)
                subspaces_vecs.append([vecs[:, ii]])
        print(f"i={ii}", format(llambda, ".3f"), len(subspaces_vals))
    return subspaces_vals, subspaces_vecs


def Zn_corner_magnetic_basis(n, pure_theory):
    # Obtain the gauge invariant operators
    ops = Zn_gauge_invariant_ops(n, pure_theory, lattice_dim=2)
    # ACQUIRE BASIS AND GAUGE INVARIANT STATES FOR ANY POSSIBLE TYPE OF LATTICE
    M, _ = Zn_gauge_invariant_states(n, pure_theory, lattice_dim=2)
    # DEFINE OBSERVABLES for MAGNETIC BASIS
    dim_basis = M["site"].shape[1]
    magnetic_basis = {
        "config": np.zeros((dim_basis, 4)),
        "basis": np.zeros((dim_basis, dim_basis)),
    }
    corner_names = ["C_mx,my", "C_px,py", "C_py,mx", "C_my,px"]
    c_name_0 = corner_names[0]
    c_name_1 = corner_names[1]
    c_name_2 = corner_names[2]
    c_name_3 = corner_names[3]

    corners = {}
    for ii, name in enumerate(corner_names):
        corners[name] = {
            "vals": [],
            "vecs": [],
            "s_vals": [],
            "s_vecs": [],
            "secs0": [],
            "secs1": [],
            "secs2": [],
        }
    np.set_printoptions(precision=3, suppress=True)
    # Start from diagonalizing the 1st corner operator
    vals, vecs = eig(ops[c_name_0].toarray())
    corners[c_name_0]["vals"] = vals
    corners[c_name_0]["vecs"] = vecs
    # Register the first set of vals
    magnetic_basis["config"][:, 0] = vals
    # Look at the subeigenspaces of each eigenvalue
    s_vals, s_vecs = get_lambda_subspace(vals, vecs)
    corners[c_name_0]["s_vals"] = deepcopy(s_vals)
    corners[c_name_0]["s_vecs"] = deepcopy(s_vecs)
    # Project the 2nd corner on each sector
    for i0, s0 in enumerate(corners[c_name_0]["s_vecs"]):
        print("=============================================")
        print(f"{i0} {c_name_0}={corners[c_name_0]['s_vals'][i0]}")
        small_dim0 = len(s0)
        large_dim0 = dim_basis
        # Create the projector on the subspace
        c = csr_matrix(np.concatenate(s0).reshape((small_dim0, large_dim0)))
        if small_dim0 == 1:
            for c_name in corner_names[1:]:
                vec = ops[c_name] * c.transpose()
                val = np.dot(vec, c.transpose())
                corners[c_name]["vals"].append(val)
            continue
        else:
            # Project the other corners on this subspace
            for c_name in corner_names[1:]:
                corners[c_name]["secs0"].append(c * ops[c_name] * c.transpose())
                # corners[c_name]["dim_secs"].append(small_dim0)
            # ------------------------------------------------------------------
            # Focus on the 2nd corner and diagonalize it
            sec = corners[c_name_1]["secs0"][-1]
            vals, vecs = eigh(sec.toarray())
            corners[c_name_1]["vals"].append(vals)
            # For each diagonalization, obtain subsectors
            s_vals, s_vecs = get_lambda_subspace(vals, vecs)
            # Register sector eigvals & eigvecs
            corners[c_name_1]["s_vals"].append(s_vals)
            corners[c_name_1]["s_vecs"].append(s_vecs)
        # ------------------------------------------------------------------
        # Focus on the 3rd corner and diagonalize it
        for i1, s1 in enumerate(corners[c_name_1]["s_vecs"][-1]):
            small_dim1 = len(s1)
            large_dim1 = small_dim0
            print("------------------------------")
            print(f"{c_name_1}={corners[c_name_1]['s_vals'][-1][i1]}")
            print(small_dim1, large_dim1)
            # Create the projector on the subspace
            c = csr_matrix(np.concatenate(s1).reshape((small_dim1, large_dim1)))
            if small_dim1 == 1:
                for c_name in corner_names[2:]:
                    vec = corners[c_name]["secs0"][i0] * c.transpose()
                    val = np.dot(vec, c.transpose())
                    corners[c_name]["vals"].append(val)
                continue
            else:
                # Project the other corners on this subspace
                for c_name in corner_names[2:]:
                    corners[c_name]["secs1"].append(
                        c * corners[c_name]["secs0"][i0] * c.transpose()
                    )
                # DIAGONALIZE THE 3RD corner
                sec = corners[c_name_2]["secs1"][-1]
                vals, vecs = eigh(sec.toarray())
                corners[c_name_2]["vals"].append(vals)
                # For each diagonalization, obtain subsectors
                s_vals, s_vecs = get_lambda_subspace(vals, vecs)
                # Register sector eigvals & eigvecs
                corners[c_name_2]["s_vals"].append(s_vals)
                corners[c_name_2]["s_vecs"].append(s_vecs)
            # --------------------------------------------------------------
            # Focus on the 4th corner and diagonalize it
            for i2, s2 in enumerate(corners[c_name_2]["s_vecs"][-1]):
                small_dim2 = len(s2)
                large_dim2 = small_dim1
                print("#################")
                print(f"{c_name_2}={corners[c_name_2]['s_vals'][-1][i2]}")
                print(small_dim2, large_dim2)
                # Create the projector on the subspace
                c = csr_matrix(np.concatenate(s2).reshape((small_dim2, large_dim2)))
                print(c.shape, corners[c_name]["secs1"][-1].shape)
                # Project the other corners on this subspace
                for c_name in corner_names[3:]:
                    corners[c_name]["secs2"].append(
                        c * corners[c_name]["secs1"][-1] * c.transpose()
                    )
                # DIAGONALIZE THE 3RD corner
                sec = corners[c_name_3]["secs2"][-1]
                vals, vecs = eigh(sec.toarray())
                corners[c_name_3]["vals"].append(vals)
                # For each diagonalization, obtain subsectors
                s_vals, s_vecs = get_lambda_subspace(vals, vecs)
                # Register sector eigvals & eigvecs
                corners[c_name_3]["s_vals"].append(s_vals)
                corners[c_name_3]["s_vecs"].append(s_vecs)

    # Register the eigenvalues of the 2nd adn 3rd corner
    corners[c_name_1]["vals"] = np.concatenate(corners[c_name_1]["vals"])
    magnetic_basis["config"][:, 1] = corners[c_name_1]["vals"]
    corners[c_name_2]["vals"] = np.concatenate(corners[c_name_2]["vals"])
    magnetic_basis["config"][:, 2] = corners[c_name_2]["vals"]
    corners[c_name_3]["vals"] = np.concatenate(corners[c_name_3]["vals"])
    magnetic_basis["config"][:, 3] = corners[c_name_3]["vals"]
    return magnetic_basis


# %%
