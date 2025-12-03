# %%
import numpy as np
from ed_lgt.operators import SU2_dressed_site_operators, SU2_gauge_invariant_states
from ed_lgt.models import SU2_Model
from ed_lgt.tools import pauli_operators
import itertools
import logging

logger = logging.getLogger(__name__)


def generate_string_ops(magic_keys, N):
    return [list(p) for p in itertools.product(magic_keys, repeat=N)]


def _is_forbidden_pair(x, y):
    return (
        (x == 0 and y in [1, 3, 5])
        or (x == 1 and y in [0, 2, 4])
        or (x == 2 and y in [0, 2, 4])
        or (x == 3 and y in [1, 3, 5])
        or (x == 4 and y in [1, 3, 5])
        or (x == 5 and y in [0, 2, 4])
    )


def decode_three_site_entries(rc_list, d_loc, data, has_obc=True):
    """
    rc_list: iterable of (row, col) ints for a 3-site block (size d_loc^3).
    d_loc  : local dimension per site.

    returns: list of (r1, r2, r3, c1, c2, c3, value) with
             row = r1*d_loc^2 + r2*d_loc + r3
             col = c1*d_loc^2 + c2*d_loc + c3
    """
    out = []
    d2 = d_loc * d_loc
    left_obc_forbidden = [1, 3, 5]
    right_obc_forbidden = [1, 2, 5]
    for ii, (row, col) in enumerate(rc_list):
        # --- decode row index into (r1, r2, r3)
        r1 = row // d2
        rem_r = row % d2
        r2 = rem_r // d_loc
        r3 = rem_r % d_loc

        # --- decode col index into (c1, c2, c3)
        c1 = col // d2
        rem_c = col % d2
        c2 = rem_c // d_loc
        c3 = rem_c % d_loc

        # --- apply SAME rules to every nearest-neighbor pair
        if (
            _is_forbidden_pair(r1, r2)
            or _is_forbidden_pair(r2, r3)
            or _is_forbidden_pair(c1, c2)
            or _is_forbidden_pair(c2, c3)
        ):
            continue
        if has_obc:
            if (
                (r1 in left_obc_forbidden)
                or (c1 in left_obc_forbidden)
                or (r3 in right_obc_forbidden)
                or (c3 in right_obc_forbidden)
            ):
                continue
        else:
            if _is_forbidden_pair(r3, r1) or _is_forbidden_pair(c3, c1):
                continue
        out.append((int(r1), int(r2), int(r3), int(c1), int(c2), int(c3), data[ii]))

    if len(out) == 0:
        logger.info("")
        logger.info("NULL OPERATOR")
        logger.info("")

    return out


def decode_two_site_entries(rc_list, d_loc, data, has_obc=True):
    """
    rc_list: iterable of (row, col) ints for a 2-site block (size d_loc^2).
    d_loc  : local dimension per site.

    returns: list of (r1, r2, c1, c2) with
             row = r1*d_loc + r2,  col = c1*d_loc + c2
    """
    left_obc_forbidden = [1, 3, 5]
    right_obc_forbidden = [1, 2, 5]
    out = []
    for ii, (row, col) in enumerate(rc_list):
        r1, r2 = divmod(row, d_loc)
        c1, c2 = divmod(col, d_loc)
        if _is_forbidden_pair(r1, r2) or _is_forbidden_pair(c1, c2):
            continue
        if has_obc:
            if (
                (r1 in left_obc_forbidden)
                or (c1 in left_obc_forbidden)
                or (r2 in right_obc_forbidden)
                or (c2 in right_obc_forbidden)
            ):
                continue
        else:
            if _is_forbidden_pair(r2, r1) or _is_forbidden_pair(c2, c1):
                continue
        out.append((int(r1), int(r2), int(c1), int(c2), data[ii]))
    if len(out) == 0:
        logger.info("")
        logger.info("NULL OPERATOR")
        logger.info("")
    return out


from ed_lgt.modeling import qmb_operator as qmb_op

# %%
N = 2
had_obc = False
pauli_ops, pauli_keys = pauli_operators(d=6, setType="full")
strings_ops_list = generate_string_ops(pauli_keys, N)
pauli_string_ops = {}
cnt = 0
for pauli_string in strings_ops_list:
    string_name = "_".join(pauli_string)
    pauli_string_ops[string_name] = qmb_op(pauli_ops, pauli_string)
    logger.info(f"**************** {string_name} ********************")
    coo = pauli_string_ops[string_name].tocoo()
    rows, cols, data = coo.row, coo.col, coo.data
    if N == 2:
        P_lista = decode_two_site_entries(zip(rows, cols), 6, data, had_obc)
    else:
        P_lista = decode_three_site_entries(zip(rows, cols), 6, data, had_obc)
    if len(P_lista) > 0:
        logger.info(f"non-zero entries {len(P_lista)}/{len(data)}")
        for ii in range(len(P_lista)):
            logger.info(f"{P_lista[ii][:-1]} : {P_lista[ii][-1]}")
    else:
        cnt += 1
        logger.info(f"non-zero entries {len(P_lista)}/{len(data)}")
ratio = cnt / len(strings_ops_list)
logger.info(f"NULL STRINGS {cnt}/{len(strings_ops_list)} RATIO {ratio}")
# %%
from scipy.sparse import csr_matrix, identity, kron

Cdata = [1, 1, 1, 1, 1, -1]
Crows = [0, 4, 5, 1, 2, 3]
Ccols = [4, 0, 1, 5, 2, 3]
C = csr_matrix((Cdata, (Crows, Ccols)), shape=(6, 6))
ops["C"] = C
ops["Cdag"] = C.transpose().conj()
CC = qmb_op(ops, ["C", "C"])
CCdag = qmb_op(ops, ["Cdag", "Cdag"])

test = CC @ P @ CCdag
testcoo = test.tocoo()
trows, tcols, tdata = testcoo.row, testcoo.col, testcoo.data

# %%
logger.info("-- MASS TERM -----")
M1 = csr_matrix(ops["N_tot"])
ID = csr_matrix(np.eye(6))
massa = kron(M1, ID) - kron(ID, M1)
coo = massa.tocoo()
rows, cols, data = coo.row, coo.col, coo.data
P_lista = decode_two_site_entries(zip(rows, cols), d_loc=6, data=data)
for ii in range(len(P_lista)):
    logger.info(f"{P_lista[ii][:-1]} : {P_lista[ii][-1]}")

# %%
model_par = {
    "lvals": [2],
    "has_obc": [False],
    "spin": 0.5,
    "pure_theory": False,
    "background": 0,
    "sectors": [2],
    "ham_format": "sparse",
}
# -------------------------------------------------------------------------------
# Initialize the model
model = SU2_Model(**model_par)
# Save parameters
model.default_params()
# Build Hamiltonian
m = 1
g = 1
model.build_Hamiltonian(g, m)
model.diagonalize_Hamiltonian(1, model.ham_format)
# -------------------------------------------------------------------------------
# LIST OF LOCAL OBSERVABLES
local_obs = [f"T2_p{d}" for d in model.directions]
if not model.pure_theory:
    local_obs += [f"N_{label}" for label in ["tot", "single", "pair", "zero"]]
# LIST OF TWOBODY CORRELATORS
twobody_obs = []
twobody_axes = []
# DEFINE OBSERVABLES
model.get_observables(local_obs, twobody_obs, twobody_axes=twobody_axes)
# QUENCH STATE FOR OVERLAP
if model_par["observables"]["get_overlap"]:
    name = model_par["hamiltonian"]["state"]
    config = model.overlap_QMB_state(name)
    logger.info(f"config {config}")
    in_state = model.get_qmb_state_from_configs([config])
    sim.res["overlap"] = np.zeros(model.H.n_eigs, dtype=float)
# -------------------------------------------------------------------------------
# ENTROPY
# DEFINE THE PARTITION FOR THE ENTANGLEMENT ENTROPY
partition_indices = model_par["observables"]["entropy_partition"]
if model_par["observables"]["get_entropy"] or model_par["observables"]["get_RDM"]:
    model._get_partition(partition_indices)
sim.res["entropy"] = np.zeros(model.H.n_eigs, dtype=float)
# INVERSION SYMMETRY
if model_par.get("inversion", None) is not None:
    apply_parity = model_par["inversion"]["get_inversion_sym"]
    wrt_site = model_par["inversion"]["wrt_site"]
else:
    apply_parity = False
if apply_parity:
    model.get_parity_inversion_operator(wrt_site)
# -------------------------------------------------------------------------------
for ii in range(model.H.n_eigs):
    model.H.print_energy(ii)
    if apply_parity:
        if model.momentum_basis is None:
            psi = model.H.Npsi[ii].psi
        else:
            # Project the State from the momentum sector to the coordinate one
            Pk = model._basis_Pk_as_csr()
            psi = Pk @ model.H.Npsi[ii].psi
        psiP = model.parityOP @ psi
        logger.info(f"<psi{ii}|P|psi{ii}> {np.real(np.vdot(psi,psiP))}")
    if model.momentum_basis is None:
        # -------------------------------------------------------------------------------
        # REDUCED DENSITY MATRIX
        if model_par["observables"]["get_RDM"]:
            # Get the reduced density matrix of a partition in the ground state
            RDM = model.H.Npsi[ii].reduced_density_matrix(
                partition_indices, model._partition_cache
            )
            logger.info(f"RDM shape {RDM.shape}")
            rho_eigvals, rho_eigvecs = diagonalize_density_matrix(RDM)
            # Sort eigenvalues and eigenvectors in descending order.
            # Note: np.argsort sorts in ascending order; we reverse to get descending order.
            sorted_indices = np.argsort(rho_eigvals)[::-1]
            rho_eigvals = rho_eigvals[sorted_indices]
            logger.info(f"eigvals {rho_eigvals}")
            sim.res["eigvals"] = rho_eigvals
            rho_eigvecs = rho_eigvecs[:, sorted_indices]
        # -----------------------------------------------------------------------
        # ENTROPY
        if model_par["observables"]["get_entropy"]:
            sim.res["entropy"][ii] = model.H.Npsi[ii].entanglement_entropy(
                partition_indices, model._partition_cache
            )
        # -----------------------------------------------------------------------
        # STATE CONFIGURATIONS
        if model_par["observables"]["get_state_configs"]:
            model.H.Npsi[ii].get_state_configurations(1e-3, model.sector_configs)
    # -----------------------------------------------------------------------
    # MEASURE OBSERVABLES
    if model_par["observables"]["measure_obs"]:
        model.measure_observables(ii)
        sim.res["E2"][ii] = model.link_avg(obs_name="T2")
        if not model.pure_theory:
            sim.res["N_single"][ii] = model.stag_avg(model.res["N_single"])
            sim.res["N_pair"][ii] += 0.5 * model.stag_avg(model.res["N_pair"], "even")
            sim.res["N_pair"][ii] += 0.5 * model.stag_avg(model.res["N_zero"], "odd")
            sim.res["N_zero"][ii] += 0.5 * model.stag_avg(model.res["N_zero"], "even")
            sim.res["N_zero"][ii] += 0.5 * model.stag_avg(model.res["N_pair"], "odd")
            sim.res["N_tot"][ii] = sim.res["N_single"][ii] + 2 * sim.res["N_pair"][ii]
    # ---------------------------------------------------------------------------
    # OVERLAPS with the INITIAL STATE
    if model_par["observables"]["get_overlap"]:
        sim.res["overlap"][ii] = model.measure_fidelity(in_state, ii, print_value=True)
