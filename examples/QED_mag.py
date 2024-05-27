# %%
import numpy as np
from ed_lgt.operators import Zn_rishon_operators, QED_Hamiltonian_couplings
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, QMB_hamiltonian
from scipy.linalg import eig
from scipy.sparse import csr_matrix
from ed_lgt.modeling import truncation
import logging

logger = logging.getLogger(__name__)


def diagonalize_magnetic_basis(N, threshold):
    N = int(N)
    ops = Zn_rishon_operators(N, True)
    # Diagonalize the Parallel transporter
    E, F = eig(a=ops["U"].toarray())
    # Sort eigenvalues by the real part
    sorted_indices = np.argsort(E.real)
    E = E[sorted_indices]
    F = F[:, sorted_indices]
    # Restict the new basis only to the angles with largest cosine
    E, F = E[-threshold:], F[:, -threshold:]
    mag_ops = {
        "phi": truncation(F.conj().transpose() @ ops["E"] @ F, 1e-15),
        "T": np.diag(2 * np.real(E)),
    }
    mag_ops["cos_phi"] = csr_matrix(np.cos((2 * np.pi / N) * mag_ops["phi"]))
    mag_ops["sin_phi"] = csr_matrix(np.sin((2 * np.pi / N) * mag_ops["phi"]))
    return mag_ops


gs = []
plaq = []
gap = []
par = {
    "L": 7,  # Truncation of the Magnetic Basis
    "pure_theory": True,
    "lvals": [3, 3],
    "g": 0.1,
    "n_eigs": 2,
}

N_list = np.arange(11, 201, 10, dtype=int)
for N in N_list:
    ops = diagonalize_magnetic_basis(N, par["L"])
    # In the dual lattice, the sites correspond to the number of plaquettes
    dual_lvals = [ll - 1 for ll in par["lvals"]]
    n_sites = np.prod(dual_lvals)
    lattice_dim = len(par["lvals"])
    directions = "xyz"[:lattice_dim]
    # The magnetic basis only works in OBC
    has_obc = [True for _ in range(lattice_dim)]
    # ACQUIRE HAMILTONIAN COEFFICIENTS
    coeffs = QED_Hamiltonian_couplings(
        lattice_dim, par["pure_theory"], par["g"], m=0.1, magnetic_basis=True
    )
    # ACQUIRE LOCAL DIMENSION OF EVERY SINGLE SITE
    loc_dims = np.array([par["L"] for _ in range(n_sites)])
    logger.info(f"local dimensions: {loc_dims}")
    # ===============================================================================
    # CONSTRUCT THE HAMILTONIAN
    H = QMB_hamiltonian(0, dual_lvals)
    h_terms = {}
    # -------------------------------------------------------------------------------
    # ELECTRIC TWO BODY OPERATORS
    for d in directions:
        for func in ["cos", "sin"]:
            op_names_list = [f"{func}_phi", f"{func}_phi"]
            op_list = [ops[op] for op in op_names_list]
            # Define the Hamiltonian term
            h_terms[f"E_{func}"] = TwoBodyTerm(
                axis=d,
                op_list=op_list,
                op_names_list=op_names_list,
                lvals=dual_lvals,
                has_obc=has_obc,
            )
            H.Ham += h_terms[f"E_{func}"].get_Hamiltonian(strength=coeffs["E"])
    # -------------------------------------------------------------------------------
    # PLAQUETTE LOCAL OPERATOR
    op_name = "T"
    h_terms["T"] = LocalTerm(ops[op_name], op_name, lvals=dual_lvals, has_obc=has_obc)
    H.Ham += h_terms["T"].get_Hamiltonian(strength=coeffs["B"])
    # DIAGONALIZE THE HAMILTONIAN
    H.diagonalize(par["n_eigs"], "dense", loc_dims)
    h_terms["T"].get_expval(H.Npsi[0])
    plaq.append(h_terms["T"].avg)
    # Dictionary for results
    res = {}
    gs.append(H.GSenergy)
    res["energy"] = H.Nenergies
    gap.append(H.Nenergies[1] - H.Nenergies[0])

# %%
from matplotlib import pyplot as plt

textwidth_pt = 510.0
textwidth_in = textwidth_pt / 72.27


@plt.FuncFormatter
def fake_log(x, pos):
    "The two args are the value and tick position"
    return r"$10^{%d}$" % (x)


fig, ax = plt.subplots(
    1,
    1,
    sharex=True,
    sharey="row",
    figsize=(textwidth_in, textwidth_in),
    constrained_layout=True,
)

ax.plot(N_list, gs, "o-")
# %%
