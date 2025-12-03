# %%
import numpy as np
from sympy import S
from scipy.sparse import kron, csr_matrix
from ed_lgt.operators import (
    SU2_dressed_site_operators,
    Z2_FermiHubbard_gauge_invariant_states,
    Z2_FermiHubbard_dressed_site_operators,
    SU2_gauge_invariant_states,
    fermi_operators,
    SU2_generators,
    get_SU2_singlets,
    couple_two_spins,
    add_new_spin,
    group_sorted_spin_configs,
)
from ed_lgt.tools import check_hermitian

import logging
from numba import njit
from ed_lgt.tools import get_time


logger = logging.getLogger(__name__)


@get_time
@njit
def make_factorial_array_numba(spin_list):
    """
    Build an array fact where fact[n] == n! for n = 0..N_max,
    with N_max = floor(3/2 * sum(2*s_i)) + 1 in twice-units.

    Parameters
    ----------
    spin_list : 1D array of float64
        The spins on one site (e.g. [s_max, s_max, ..., matter_j, bg_j]).

    Returns
    -------
    fact : 1D array of float64
        fact[n] = n! up to the maximal needed n.
    """
    # 1) Compute j2_max = sum of 2*s over all spins
    j2_acc = 0.0
    for s in spin_list:
        j2_acc += 2.0 * s
    # cast to integer (should be exactly integer in well-formed input)
    j2_max = int(j2_acc + 1e-8)

    # 2) Worst-case factorial index: (3/2) j2_max + 1
    N_max = (3 * j2_max) // 2 + 1

    # 3) Allocate and fill the factorial table
    fact = np.empty(N_max + 1, dtype=np.float64)
    fact[0] = 1.0
    for n in range(1, N_max + 1):
        fact[n] = fact[n - 1] * n
    return fact


spin_list = np.array([1 / 2, 1 / 2, 1 / 2, 1 / 2], dtype=np.float64)
a = make_factorial_array_numba(spin_list)

# %%
gauge_basis, gauge_states = SU2_gauge_invariant_states(
    0.5, False, lattice_dim=1, background=0
)
# %%
from ed_lgt.operators import SU2_dressed_site_operators, SU2_gauge_invariant_states


def SU2_gauge_invariant_ops(spin, pure_theory, lattice_dim):
    in_ops = SU2_dressed_site_operators(spin, pure_theory, lattice_dim)
    gauge_basis, _ = SU2_gauge_invariant_states(spin, pure_theory, lattice_dim)
    ops = {}
    for op in in_ops.keys():
        ops[op] = gauge_basis["site"].transpose() @ in_ops[op] @ gauge_basis["site"]
    return ops


ops = SU2_gauge_invariant_ops(spin=1 / 2, pure_theory=False, lattice_dim=1)
from ed_lgt.modeling import qmb_operator as qmb_op


def decode_two_site_entries(rc_list, d_loc, data):
    """
    rc_list: iterable of (row, col) ints for a 2-site block (size d_loc^2).
    d_loc  : local dimension per site.

    returns: list of (r1, r2, c1, c2) with
             row = r1*d_loc + r2,  col = c1*d_loc + c2
    """
    out = []
    for ii, (row, col) in enumerate(rc_list):
        r1, r2 = divmod(row, d_loc)
        c1, c2 = divmod(col, d_loc)
        if np.any(
            [
                r1 == 0 and r2 in [1, 3, 5],
                (r1 == 1 and r2 in [0, 2, 4]),
                (r1 == 2 and r2 in [0, 2, 4]),
                (r1 == 3 and r2 in [1, 3, 5]),
                (r1 == 4 and r2 in [1, 3, 5]),
                (r1 == 5 and r2 in [0, 2, 4]),
                (c1 == 0 and c2 in [1, 3, 5]),
                (c1 == 1 and c2 in [0, 2, 4]),
                (c1 == 2 and c2 in [0, 2, 4]),
                (c1 == 3 and c2 in [1, 3, 5]),
                (c1 == 4 and c2 in [1, 3, 5]),
                (c1 == 5 and c2 in [0, 2, 4]),
            ]
        ):
            continue
        out.append((int(r1), int(r2), int(c1), int(c2), data[ii]))
    return out


# %%
logger.info("-- HOPPING -----")
P = -complex(0, 1) * qmb_op(ops, ["Qpx_dag", "Qmx"])
P += complex(0, 1) * qmb_op(ops, ["Qpx", "Qmx_dag"])
coo = P.tocoo()
rows, cols, data = coo.row, coo.col, coo.data
P_lista = decode_two_site_entries(zip(rows, cols), d_loc=6, data=data)
for ii in range(len(P_lista)):
    logger.info(f"{P_lista[ii][:-1]} : {P_lista[ii][-1]}")
check_hermitian(P)
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
logger.info("-- CASIMIR -----")
E = csr_matrix(ops["E_square"])
casimir = kron(E, ID) + kron(ID, E)
coo = casimir.tocoo()
rows, cols, data = coo.row, coo.col, coo.data
P_lista = decode_two_site_entries(zip(rows, cols), d_loc=6, data=data)
for ii in range(len(P_lista)):
    logger.info(f"{P_lista[ii][:-1]} : {P_lista[ii][-1]}")
logger.info("-------")
# %%
check_hermitian(1j * (CC @ P - P @ CC))
# %%

CPC_lista = decode_two_site_entries(zip(trows, tcols), d_loc=6, data=tdata)
for ii in range(len(CPC_lista)):
    logger.info(f"{CPC_lista[ii][:-1]} : {CPC_lista[ii][-1]-P_lista[ii][-1]}")

# %%

# %%
for ii, singlet in enumerate(gauge_states["site"]):
    logger.info(f" {ii} ")
    singlet.display_singlets()
    logger.info(f" ")

# %%
for ii, singlet in enumerate(gauge_states["site_mx_my"]):
    logger.info(f" {ii} ")
    singlet.display_singlets()
    logger.info(f" ")
# %%
for ii, singlet in enumerate(gauge_states["site_px_py"]):
    logger.info(f" {ii} ")
    singlet.display_singlets()
    logger.info(f" ")
# %%
for ii, singlet in enumerate(gauge_states["site_mx_py"]):
    logger.info(f" {ii} ")
    singlet.display_singlets()
    logger.info(f" ")
# %%
for ii, singlet in enumerate(gauge_states["site_px_my"]):
    logger.info(f" {ii} ")
    singlet.display_singlets()
    logger.info(f" ")
# %%
for ii, singlet in enumerate(gauge_states["site_my"]):
    logger.info(f" {ii} ")
    singlet.display_singlets()
    logger.info(f" ")
# %%
for ii, singlet in enumerate(gauge_states["site_py"]):
    logger.info(f" {ii} ")
    singlet.display_singlets()
    logger.info(f" ")
# %%
from ed_lgt.operators import QED_dressed_site_operators, QED_gauge_invariant_states


def QED_gauge_invariant_ops(spin, pure_theory, lattice_dim, get_only_bulk):
    in_ops = QED_dressed_site_operators(spin, pure_theory, lattice_dim)
    gauge_basis, _ = QED_gauge_invariant_states(
        spin, pure_theory, lattice_dim, get_only_bulk
    )
    ops = {}
    if pure_theory:
        labels = ["site"]
    else:
        labels = ["even", "odd"]
    for label in labels:
        for op in in_ops.keys():
            ops[(label, op)] = (
                gauge_basis[label].transpose() @ in_ops[op] @ gauge_basis[label]
            )
    return ops


# %%
lattice_dim = 2
spin = 1
pure_theory = False
get_only_bulk = True
in_ops = QED_dressed_site_operators(
    spin=spin, pure_theory=pure_theory, lattice_dim=lattice_dim
)
# %%
s, b = QED_gauge_invariant_states(
    spin=spin,
    pure_theory=pure_theory,
    lattice_dim=lattice_dim,
    get_only_bulk=get_only_bulk,
)
ops = QED_gauge_invariant_ops(
    spin=spin,
    pure_theory=pure_theory,
    lattice_dim=lattice_dim,
    get_only_bulk=get_only_bulk,
)


# %%
def print_semilinks(ops):
    # ops is a dict mapping each direction to a diagonal matrix; we'll just
    # extract the n-th diagonal element as before.  Here we hard-code n=0…6.
    SIZE = 5
    CENTER = SIZE // 2  # 2
    indices = [107, 33, 89, 104, 33, 107, 59, 28]
    directions = {
        "px": (+1, 0),
        "mx": (-1, 0),
        "py": (+1, +1),
        "my": (-1, -1),
        "pz": (0, +1),
        "mz": (0, -1),
    }

    for n in indices:
        # 1) read & stringify each diagonal element
        vals = {d: str(int(ops[f"E_{d}"].toarray()[n, n])) for d in directions}

        # 2) figure how wide the widest string is
        wid = max(len(s) for s in vals.values())

        # 3) center each into width=wid
        centered = {d: vals[d].center(wid) for d in vals}
        origin = "o".center(wid)
        empty = " " * wid

        # 4) build blank 5×5
        grid = [[empty for _ in range(SIZE)] for _ in range(SIZE)]

        # 5) place each direction
        for d, (dx, dy) in directions.items():
            r = CENTER - dy
            c = CENTER + dx
            grid[r][c] = centered[d]

        # place the origin
        grid[CENTER][CENTER] = origin

        # 6) print
        logger.info(f"--- state n = {n} ---")
        for row in grid:
            logger.info(" ".join(row))
        logger.info()


print_semilinks(ops)
# %%
indices = np.arange(35)
for n in indices:
    # read and stringify each diagonal element
    vals = {
        d: str(int(ops["odd", f"E_{d}"].toarray()[n, n]))
        for d in ("px", "py", "mx", "my")
    }

    # figure out how wide the widest string is
    wid = max(len(s) for s in vals.values())

    # center each string in a field of width=wid
    centered = {d: vals[d].center(wid) for d in vals}

    # how many spaces to indent so that 'o' (in line 3) lines up under the vertical bars
    indent = len(centered["mx"]) + 1
    spacer = " " * indent

    # a centered vertical bar in a field of width=wid
    bar = "|".center(wid)

    logger.info(f"--- state n = {n} ---")
    logger.info(spacer + centered["py"])
    logger.info(spacer + bar)
    logger.info(f"{centered['mx']}=o= {centered['px']}")
    logger.info(spacer + bar)
    logger.info(spacer + centered["my"])
    logger.info()


# %%


def SU2_gauge_invariant_ops(spin, pure_theory, lattice_dim, background):
    in_ops = SU2_dressed_site_operators(spin, pure_theory, lattice_dim, background)
    gauge_basis, _ = SU2_gauge_invariant_states(
        spin, pure_theory, lattice_dim, background
    )
    ops = {}
    label = "site_px,py"
    for op in in_ops.keys():
        ops[op] = gauge_basis[label].transpose() @ in_ops[op] @ gauge_basis[label]
    return ops


# %%


def Z2Hubbard_gauge_invariant_ops(lattice_dim):
    in_ops = Z2_FermiHubbard_dressed_site_operators(lattice_dim)
    gauge_basis, _ = Z2_FermiHubbard_gauge_invariant_states(lattice_dim)
    ops = {}
    label = "site_my"
    for op in in_ops.keys():
        ops[op] = gauge_basis[label].transpose() @ in_ops[op] @ gauge_basis[label]
    return ops


ops = Z2Hubbard_gauge_invariant_ops(lattice_dim=2)
for op in ops.keys():
    logger.info(op)
    logger.info(ops[op])

# %%
# With the simplified version of SU2
in_ops = SU2_rishon_operators(spin=1 / 2)

U1 = {
    "rr": kron(in_ops["Zr"] @ in_ops["P"], in_ops["Zr_dag"]),
    "rg": kron(in_ops["Zr"] @ in_ops["P"], in_ops["Zg_dag"]),
    "gr": kron(in_ops["Zg"] @ in_ops["P"], in_ops["Zr_dag"]),
    "gg": kron(in_ops["Zg"] @ in_ops["P"], in_ops["Zg_dag"]),
}

rows = [0, 4, 5, 7, 8]
for op in U1.keys():
    U1[op] = csr_matrix(U1[op].toarray()[rows, :][:, rows])

for op in U1.keys():
    logger.info(f"U1----- {op} ---------------")
    logger.info(U1[op].toarray())
    logger.info("--------------------------")

# %%
for op in in_ops.keys():
    logger.info(f"------- {op} ---------------")
    logger.info(in_ops[op].toarray())
    logger.info("--------------------------")
# %%
# With the generalized version of SU2
in_ops2 = SU2_gen_rishon_operators(spin=1 / 2)

U2 = {}
for c1, c2 in ["rr", "rg", "gr", "gg"]:
    U2[f"{c1}{c2}"] = np.sqrt(2) * (
        kron(in_ops2[f"ZA_{c1}"] @ in_ops2["P"], in_ops2[f"ZB_{c2}_dag"])
        + kron(in_ops2[f"ZB_{c1}"] @ in_ops2["P"], in_ops2[f"ZA_{c2}_dag"])
    )

U3 = {
    "rr": np.sqrt(2)
    * (
        kron(in_ops2["Zg_dag"] @ in_ops2["P"], in_ops2["Zr_dag"])
        + kron(in_ops2["Zr"] @ in_ops2["P"], in_ops2["Zg"])
    ),
    "rg": np.sqrt(2)
    * (
        kron(in_ops2["Zg_dag"] @ in_ops2["P"], in_ops2["Zg_dag"])
        - kron(in_ops2["Zr"] @ in_ops2["P"], in_ops2["Zr"])
    ),
    "gr": np.sqrt(2)
    * (
        -kron(in_ops2["Zr_dag"] @ in_ops2["P"], in_ops2["Zr_dag"])
        + kron(in_ops2["Zg"] @ in_ops2["P"], in_ops2["Zg"])
    ),
    "gg": np.sqrt(2)
    * (
        -kron(in_ops2["Zr_dag"] @ in_ops2["P"], in_ops2["Zg_dag"])
        - kron(in_ops2["Zg"] @ in_ops2["P"], in_ops2["Zr"])
    ),
}

rows = [0, 4, 5, 7, 8]
for op in U2.keys():
    U2[op] = csr_matrix(U2[op].toarray()[rows, :][:, rows])
    U3[op] = csr_matrix(U3[op].toarray()[rows, :][:, rows])

for op in U2.keys():
    logger.info(f"U2----- {op} ---------------")
    logger.info(U2[op].toarray())
    logger.info("--------------------------")
# %%
for op in U3.keys():
    logger.info(f"U3----- {op} ---------------")
    logger.info(U3[op].toarray())
    logger.info("--------------------------")
# %%
for op in ["ZA_r", "ZA_g", "ZB_r", "ZB_g"]:
    logger.info(f"------- {op} ---------------")
    logger.info(np.sqrt(np.sqrt(2)) * in_ops2[op].toarray())
    logger.info("--------------------------")
# %%
# Matter operators
f = fermi_operators(has_spin=True, colors=True)
f |= SU2_generators(spin=1 / 2, matter=True)

for op in f.keys():
    logger.info("------", op, "--------")
    logger.info(f[op].toarray())
# %%
spin_list = [S(1) / 2, S(1) / 2, S(1) / 2, S(1) / 2]
states1 = couple_two_spins(S(1) / 2, S(1) / 2, get_singlet=False)
states2 = add_new_spin(states1, S(1) / 2, get_singlet=False)
states3 = add_new_spin(states2, S(1) / 2, get_singlet=True)
sorted_states = group_sorted_spin_configs(
    states3,
    spin_list,
    pure_theory=True,
    psi_vacuum=None,
)
for s in sorted_states:
    s.display_singlets()

singlets = get_SU2_singlets(spin_list, pure_theory=True, psi_vacuum=None)

# %%
for ii, s in enumerate(gauge_states["site"]):
    logger.info(f"{ii}")
    s.display_singlets()
logger.info("TTTTTTTTTTTTTTTTTTTTTTTTTT")
logger.info("")
for ii, s in enumerate(gauge_states["site_mx"]):
    logger.info(f"{ii}")
    s.display_singlets()
logger.info("TTTTTTTTTTTTTTTTTTTTTTTTTT")
logger.info("")
for ii, s in enumerate(gauge_states["site_px"]):
    logger.info(f"{ii}")
    s.display_singlets()
# %%
in_ops = SU2_dressed_site_operators(spin=1 / 2, pure_theory=False, lattice_dim=1)
# %%
ops = SU2_gauge_invariant_ops(
    spin=1 / 2, pure_theory=True, lattice_dim=2, background=True
)

logger.info(ops["E_square"].shape)

# logger.info(8 * ops["E_square"] / 3)
# %%
for s in gauge_states["site"]:
    s.display_singlets()
# %%
logger.info(gauge_basis["site"])

# %%
# %%
import numpy as np
from ed_lgt.symmetries import (
    get_state_configs,
    momentum_basis_k0,
)

loc_dims = np.array([2 for _ in range(18)], dtype=np.uint8)
state_configs = get_state_configs(loc_dims)
MB = momentum_basis_k0(state_configs)
