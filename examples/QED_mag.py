# %%
import numpy as np
from ed_lgt.operators import Zn_rishon_operators, QED_Hamiltonian_couplings
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, QMB_hamiltonian
from scipy.linalg import eig
from ed_lgt.modeling import truncation
import logging

logger = logging.getLogger(__name__)


def diagonalize_magnetic_basis(N, threshold):
    ops = Zn_rishon_operators(N, True)
    # Diagonalize the Parallel transporter
    E, F = eig(a=ops["U"].toarray())
    # Restict the new basis only to the angles with largest cosine
    E, F = E[-threshold:], F[:, -threshold:]
    mag_ops = {
        "phi": truncation(F.conj().transpose() @ ops["E"] @ F, 1e-15),
        "T": np.real(np.diag(2 * E)),
    }
    mag_ops["cos_phi"] = np.cos((2 * np.pi / N) * mag_ops["cos_phi"])
    mag_ops["sin_phi"] = np.sin((2 * np.pi / N) * mag_ops["cos_phi"])
    return mag_ops


par = {
    "N": 20,  # Size of the Electric Basis
    "L": 5,  # Truncation of the Magnetic Basis
    "pure_theory": True,
    "lvals": [3, 3],
    "g": 0.1,
    "n_eigs": 1,
}
ops = diagonalize_magnetic_basis(par["N"], par["L"])
# In the dual lattice, the sites correspond to the number of plaquettes
dual_lvals = [ll - 1 for ll in par["lvals"]]
n_sites = np.prod(dual_lvals)
lattice_dim = len(par["lvals"])
directions = "xyz"[:lattice_dim]
# The magnetic basis only works in OBC
has_obc = [True for _ in range(lattice_dim)]
# ACQUIRE HAMILTONIAN COEFFICIENTS
coeffs = QED_Hamiltonian_couplings(par["pure_theory"], par["g"])
# ACQUIRE LOCAL DIMENSION OF EVERY SINGLE SITE
loc_dims = np.array([par["L"] for _ in range(n_sites)])
logger.info(f"local dimensions: {loc_dims}")
# ===============================================================================
# CONSTRUCT THE HAMILTONIAN
H = QMB_hamiltonian(0, par["lvals"], loc_dims)
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
h_terms[op_name] = LocalTerm(ops[op_name], op_name, lvals=dual_lvals, has_obc=has_obc)
H.Ham += h_terms[op_name].get_Hamiltonian(strength=coeffs["B"])
# DIAGONALIZE THE HAMILTONIAN
H.diagonalize(par["n_eigs"])
# Dictionary for results
res = {}
res["energy"] = H.Nenergies
logger.info(H.Nenergies)
