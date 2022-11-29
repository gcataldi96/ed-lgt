# %%
import numpy as np
from simsio import logger
from scipy.sparse import identity
from operators import get_su2_operators
from modeling import Pure_State, LocalTerm2D, TwoBodyTerm2D, PlaquetteTerm2D
from tools import check_matrix


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
    if border == "mx":
        mask[0, :] = True
    elif border == "px":
        mask[-1, :] = True
    if border == "my":
        mask[:, 0] = True
    elif border == "py":
        mask[:, -1] = True
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
dim = 2
directions = "xyz"[:dim]
has_obc = True
lvals = [2, 2]

# %%
# ACQUIRE OPERATORS AS CSR MATRICES IN A DICTIONARY
ops = get_su2_operators(pure_theory)

ham_terms = {}
coeffs = get_Hamiltonian_couplings(pure_theory, g=1, m=1)
H = 0
# %%
# BORDER PENALTIES
if has_obc:
    for d in directions:
        for s in "mp":
            ham_terms[f"P_{s}{d}"] = LocalTerm2D(ops[f"P_{s}{d}"], f"P_{s}{d}")
            mask=border_mask(lvals, f"{s}{d}")
            print(mask)
            H += ham_terms[f"P_{s}{d}"].get_Hamiltonian(
                lvals, strength=coeffs["eta"], mask=mask
            )
# %%
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
op_name_list = ["C_py_px", "C_my_px", "C_py_mx", "C_my_mx"]
op_list = [ops[op] for op in op_name_list]
ham_terms["plaq"] = PlaquetteTerm2D(op_list, op_name_list)
H += ham_terms["plaq"].get_Hamiltonian(
    lvals, strength=coeffs["B"], has_obc=has_obc, add_dagger=True
)
# %%
if not pure_theory:
    # STAGGERED MASS TERM
    ham_terms["mass_op"] += LocalTerm2D(ops["mass_op"], "mass_op")
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
    ham_terms["y_hopping"] = TwoBodyTerm2D("x", op_list, op_name_list)
    for site in ["even", "odd"]:
        H += ham_terms["y_hopping"].get_Hamiltonian(
            lvals,
            strength=coeffs[f"ty_{site}"],
            has_obc=has_obc,
            add_dagger=True,
            mask=staggered_mask(lvals, site),
        )

# %%
# CHECK THAT THE HAMILTONIAN IS HERMITIAN
check_matrix(H, H.getH())
# %%
# DIAGONALIZE THE HAMILTONIAN
psi = Pure_State()
psi.ground_state(H)

psi  # %%
# COMPUTE THE OBSERVABLES

if pure_theory:
    local_obs = ["gamma"]
else:
    local_obs = ["gamma", "mass_op", "n_single", "n_pair", "n_tot"]

twobody_obs = ["W_x_link", "W_y_link"]
# %%
# COMPUTE ENTROPY


# %%
# LOOK AT THE STATE

# %%
# SAVE THE STATE
