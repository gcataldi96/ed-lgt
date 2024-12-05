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
    SU2_rishon_operators,
    SU2_gen_rishon_operators,
    get_SU2_singlets,
    couple_two_spins,
    add_new_spin,
    group_sorted_spin_configs,
    SU2_singlet_canonical_vector,
    SU2_Rishon,
    SU2_Rishon_gen,
    SU2_operators_gen,
    m_values,
)

import logging

logger = logging.getLogger(__name__)


def SU2_gauge_invariant_ops(spin, pure_theory, lattice_dim, background):
    in_ops = SU2_dressed_site_operators(spin, pure_theory, lattice_dim, background)
    gauge_basis, _ = SU2_gauge_invariant_states(
        spin, pure_theory, lattice_dim, background
    )
    ops = {}
    label = "site"
    for op in in_ops.keys():
        ops[op] = gauge_basis[label].transpose() @ in_ops[op] @ gauge_basis[label]
    return ops


def Z2Hubbard_gauge_invariant_ops(lattice_dim):
    in_ops = Z2_FermiHubbard_dressed_site_operators(lattice_dim)
    gauge_basis, _ = Z2_FermiHubbard_gauge_invariant_states(lattice_dim)
    ops = {}
    label = "site"
    for op in in_ops.keys():
        ops[op] = gauge_basis[label].transpose() @ in_ops[op] @ gauge_basis[label]
    return ops


# %%
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
gauge_basis, gauge_states = SU2_gauge_invariant_states(3 / 2, False, lattice_dim=1)
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
    spin=1 / 2, pure_theory=False, lattice_dim=1, background=False
)
for op in ops.keys():
    print(op + "-----")
    print(ops[op])

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
