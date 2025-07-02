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
    SU2_check_gauss_law,
    SU2_generators,
    get_SU2_singlets,
    couple_two_spins,
    add_new_spin,
    group_sorted_spin_configs,
    SU2_singlet_canonical_vector,
    SU2_Rishon,
    SU2_Rishon_gen,
    m_values,
)

import logging

logger = logging.getLogger(__name__)
# %%
from ed_lgt.operators import QED_dressed_site_operators, QED_gauge_invariant_states


def QED_gauge_invariant_ops(spin, pure_theory, lattice_dim):
    in_ops = QED_dressed_site_operators(spin, pure_theory, lattice_dim)
    gauge_basis, _ = QED_gauge_invariant_states(spin, pure_theory, lattice_dim)
    ops = {}
    label = "site"
    for op in in_ops.keys():
        ops[op] = gauge_basis[label].transpose() @ in_ops[op] @ gauge_basis[label]
    return ops


lattice_dim = 3
spin = 1
pure_theory = True
in_ops = QED_dressed_site_operators(
    spin=spin, pure_theory=pure_theory, lattice_dim=lattice_dim
)
ops = QED_gauge_invariant_ops(
    spin=spin, pure_theory=pure_theory, lattice_dim=lattice_dim
)
s, b = QED_gauge_invariant_states(
    spin=spin, pure_theory=pure_theory, lattice_dim=lattice_dim
)


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
        print(f"--- state n = {n} ---")
        for row in grid:
            print(" ".join(row))
        print()


print_semilinks(ops)
# %%
indices = [41, 63, 38, 26]
for n in indices:
    # read and stringify each diagonal element
    vals = {
        d: str(int(ops[f"E_{d}"].toarray()[n, n])) for d in ("px", "py", "mx", "my")
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

    print(f"--- state n = {n} ---")
    print(spacer + centered["py"])
    print(spacer + bar)
    print(f"{centered['mx']}=o= {centered['px']}")
    print(spacer + bar)
    print(spacer + centered["my"])
    print()


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
    print(op)
    print(ops[op])

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
    print(f"U1----- {op} ---------------")
    print(U1[op].toarray())
    print("--------------------------")

# %%
for op in in_ops.keys():
    print(f"------- {op} ---------------")
    print(in_ops[op].toarray())
    print("--------------------------")
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
    print(f"U2----- {op} ---------------")
    print(U2[op].toarray())
    print("--------------------------")
# %%
for op in U3.keys():
    print(f"U3----- {op} ---------------")
    print(U3[op].toarray())
    print("--------------------------")
# %%
for op in ["ZA_r", "ZA_g", "ZB_r", "ZB_g"]:
    print(f"------- {op} ---------------")
    print(np.sqrt(np.sqrt(2)) * in_ops2[op].toarray())
    print("--------------------------")
# %%
# Matter operators
f = fermi_operators(has_spin=True, colors=True)
f |= SU2_generators(spin=1 / 2, matter=True)

for op in f.keys():
    print("------", op, "--------")
    print(f[op].toarray())
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
gauge_basis, gauge_states = SU2_gauge_invariant_states(
    0.5, False, lattice_dim=2, background=False
)
for ii, singlet in enumerate(gauge_states["site_my"]):
    logger.info(f" {ii} ")
    singlet.display_singlets()
    logger.info(f" ")
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

print(ops["E_square"].shape)

# print(8 * ops["E_square"] / 3)
# %%
for s in gauge_states["site"]:
    s.display_singlets()
# %%
print(gauge_basis["site"])

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
