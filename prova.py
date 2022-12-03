# %%
import numpy as np
from scipy.sparse import identity
from operators import get_su2_operators
from modeling import Pure_State, LocalTerm2D, TwoBodyTerm2D, PlaquetteTerm2D
from tools import check_matrix, get_energy_density


def entanglement_entropy(psi, loc_dim, n_sites, partition_size):
    if not isinstance(psi, np.ndarray):
        raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
    if not np.isscalar(loc_dim) and not isinstance(loc_dim, int):
        raise TypeError(f"loc_dim must be an SCALAR & INTEGER, not a {type(loc_dim)}")
    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f"n_sites must be an SCALAR & INTEGER, not a {type(n_sites)}")
    if not np.isscalar(partition_size) and not isinstance(partition_size, int):
        raise TypeError(
            f"partition_size must be an SCALAR & INTEGER, not a {type(partition_size)}"
        )
    # COMPUTE THE ENTANGLEMENT ENTROPY OF A SPECIFIC SUBSYSTEM
    tmp = psi.reshape(
        (loc_dim**partition_size, loc_dim ** (n_sites - partition_size))
    )
    S, V, D = np.linalg.svd(tmp)
    tmp = np.array([-(llambda**2) * np.log2(llambda**2) for llambda in V])
    return np.sum(tmp)


def get_Hamiltonian_couplings(pure_theory, g, m=None):
    E = 3 * (g**2) / 16  # ELECTRIC FIELD
    B = -4 / (g**2)  # MAGNETIC FIELD
    # DICTIONARY WITH MODEL COEFFICIENTS
    coeffs = {
        "g": g,
        "E": E,  # ELECTRIC FIELD coupling
        "B": B,  # MAGNETIC FIELD coupling
    }
    if pure_theory:
        coeffs["eta"] = -20 * max(E, np.abs(B))
    if not pure_theory:
        coeffs["eta"] = -20 * max(E, np.abs(B), m)
        coeffs |= {
            "m": m,
            "tx": -0.5j,  # HORIZONTAL HOPPING
            "tx_dag": 0.5j,  # HORIZONTAL HOPPING DAGGER
            "ty_even": -0.5,  # VERTICAL HOPPING (EVEN SITES)
            "ty_odd": 0.5,  # VERTICAL HOPPING (ODD SITES)
            "m_odd": -m,  # EFFECTIVE MASS for ODD SITES
            "m_even": m,  # EFFECTIVE MASS for EVEN SITES
        }
    print(f"PENALTY {coeffs['eta']}")
    return coeffs


def border_mask(lvals, border):
    """
    Defines the masks for all four sides: top, bottom, left,
    and right as well as the four corners.
    NOTE Rows and Columns of the mask array corresponds to (x,y) coordinates!
    """
    lx = lvals[0]
    ly = lvals[1]
    mask = np.zeros((lx, ly), dtype=bool)
    if border == "my":
        mask[:, 0] = True
    elif border == "py":
        mask[:, -1] = True
    elif border == "mx":
        mask[0, :] = True
    elif border == "px":
        mask[-1, :] = True
    return mask


def staggered_mask(lvals, site):
    lx = lvals[0]
    ly = lvals[1]
    mask = np.zeros((lx, ly), dtype=bool)
    for ii in range(lx):
        for jj in range(ly):
            stag = (-1) ** (ii + jj)
            if site == "even":
                if stag > 0:
                    mask[ii, jj] = True
            elif site == "odd":
                if stag < 0:
                    mask[ii, jj] = True
    return mask


# %%
# PARAMETERS
pure_theory = True
if pure_theory:
    loc_dim = 9
else:
    loc_dim = 30
    DeltaN = 0
dim = 2
directions = "xyz"[:dim]
has_obc = False
lvals = [2, 2]
n_sites = lvals[0] * lvals[1]
# Define a Dictionary for the results
res = {}
# ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
ops = get_su2_operators(pure_theory)
ham_terms = {}
coeffs = get_Hamiltonian_couplings(pure_theory, g=0.1, m=1)
H = 0
# %%
# BORDER PENALTIES
if has_obc:
    for d in directions:
        for s in "mp":
            print(f"P_{s}{d}")
            print(border_mask(lvals, f"{s}{d}"))
            ham_terms[f"P_{s}{d}"] = LocalTerm2D(ops[f"P_{s}{d}"], f"P_{s}{d}")
            H += ham_terms[f"P_{s}{d}"].get_Hamiltonian(
                lvals, strength=coeffs["eta"], mask=border_mask(lvals, f"{s}{d}")
            )
# LINK PENALTIES
axes = ["x", "y"]
for i, d in enumerate(directions):
    op_list = [ops[f"W_{s}{d}"] for s in "pm"]
    op_name_list = [f"W_{s}{d}" for s in "pm"]
    ham_terms[f"W_{axes[i]}_link"] = TwoBodyTerm2D(axes[i], op_list, op_name_list)
    H += ham_terms[f"W_{axes[i]}_link"].get_Hamiltonian(
        lvals, strength=coeffs["eta"], has_obc=has_obc, add_dagger=False
    )
# %%
# ELECTRIC ENERGY
ham_terms["gamma"] = LocalTerm2D(ops["gamma"], "gamma")
H += ham_terms["gamma"].get_Hamiltonian(lvals, strength=coeffs["E"])

# %%
# MAGNETIC ENERGY
op_name_list = ["C_py_px", "C_py_mx", "C_my_px", "C_my_mx"]
op_list = [ops[op] for op in op_name_list]
ham_terms["plaq"] = PlaquetteTerm2D(op_list, op_name_list)
H += ham_terms["plaq"].get_Hamiltonian(
    lvals, strength=coeffs["B"], has_obc=has_obc, add_dagger=True
)
# %%
if not pure_theory:
    # STAGGERED MASS TERM
    ham_terms["mass_op"] = LocalTerm2D(ops["mass_op"], "mass_op")
    for site in ["even", "odd"]:
        H += ham_terms["mass_op"].get_Hamiltonian(
            lvals, strength=coeffs[f"m_{site}"], mask=staggered_mask(lvals, site)
        )

    # HOPPING ACTIVITY along x AXIS
    op_name_list = ["Q_px_dag", "Q_mx"]
    op_list = [ops[op] for op in op_name_list]
    ham_terms["x_hopping"] = TwoBodyTerm2D("x", op_list, op_name_list)
    H += ham_terms["x_hopping"].get_Hamiltonian(
        lvals, strength=coeffs["tx"], has_obc=has_obc, add_dagger=True
    )

    # HOPPING ACTIVITY along y AXIS
    op_name_list = ["Q_py_dag", "Q_my"]
    op_list = [ops[op] for op in op_name_list]
    ham_terms["y_hopping"] = TwoBodyTerm2D("y", op_list, op_name_list)
    for site in ["even", "odd"]:
        H += ham_terms["y_hopping"].get_Hamiltonian(
            lvals,
            strength=coeffs[f"ty_{site}"],
            has_obc=has_obc,
            add_dagger=True,
            mask=staggered_mask(lvals, site),
        )

    if DeltaN != 0:
        # SELECT THE SYMMETRY SECTOR with N PARTICLES
        tot_hilb_space = loc_dim ** (lvals[0] * lvals[1])
        ham_terms["fix_N"] = LocalTerm2D(ops["n_tot"], "n_tot")
        H += (
            -coeffs["eta"]
            * (
                ham_terms["fix_N"].get_Hamiltonian(lvals, strength=1)
                - DeltaN * identity(tot_hilb_space)
            )
            * (
                ham_terms["fix_N"].get_Hamiltonian(lvals, strength=1)
                - DeltaN * identity(tot_hilb_space)
            )
        )
# %%
# CHECK THAT THE HAMILTONIAN IS HERMITIAN
check_matrix(H, H.getH())
# DIAGONALIZE THE HAMILTONIAN
psi = Pure_State()
psi.ground_state(H)
# RESCALE ENERGY
res["energy"] = get_energy_density(
    psi.GSenergy[0],
    lvals,
    penalty=coeffs["eta"],
    border_penalty=True,
    link_penalty=True,
    plaquette_penalty=False,
    PBC=not has_obc,
)
# %%
# CHECK BORDER PENALTIES
if has_obc:
    for d in directions:
        for s in "mp":
            ham_terms[f"P_{s}{d}"].get_loc_expval(psi.GSpsi, lvals)
            ham_terms[f"P_{s}{d}"].check_on_borders(border=f"{s}{d}", value=1)
# CHECK LINK PENALTIES
axes = ["x", "y"]
for i, d in enumerate(directions):
    op_list = [ops[f"W_{s}{d}"] for s in "pm"]
    op_name_list = [f"W_{s}{d}" for s in "pm"]
    ham_terms[f"W_{axes[i]}_link"].get_expval(psi.GSpsi, lvals, has_obc=has_obc)
    ham_terms[f"W_{axes[i]}_link"].check_link_symm(value=1, has_obc=has_obc)
# %%
# COMPUTE GAUGE OBSERVABLES
res["gamma"] = ham_terms["gamma"].get_loc_expval(psi.GSpsi, lvals)
# %%
res["plaq"] = ham_terms["plaq"].get_plaq_expval(
    psi.GSpsi, lvals, has_obc=has_obc, get_imag=False
)
# %%
# COMPUTE MATTER OBSERVABLES
if not pure_theory:
    local_obs = ["n_single", "n_pair", "n_tot"]
    for obs in local_obs:
        ham_terms[obs] = LocalTerm2D(ops[obs], obs)
        res[f"{obs}_even"], res[f"{obs}_odd"] = ham_terms[obs].get_loc_expval(
            psi.GSpsi, lvals, staggered=True
        )
# %%
# COMPUTE ENTROPY
res["entropy"] = entanglement_entropy(
    psi=psi.GSpsi, loc_dim=loc_dim, n_sites=n_sites, partition_size=int(n_sites / 2)
)
# LOOK AT THE STATE
print("----------------------------------------------------")
print(f" ENERGY:   {res['energy']}")
print(f" ENTROPY:  {res['entropy']}")
print(f" ELECTRIC: {res['gamma']}")
print(f" MAGNETIC: {res['plaq']}")
print("----------------------------------------------------")

# %%
