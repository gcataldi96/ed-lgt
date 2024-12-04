# %%
import numpy as np
from math import prod
from ed_lgt.models import phi4_model

from mean_field import sim
from time import time
import logging

logger = logging.getLogger(__name__)

# N eigenvalues
n_eigs = 1
# LATTICE GEOMETRY
lvals = [2]
dim = len(lvals)
# directions = "xyz"[:dim]
n_sites = prod(lvals)
has_obc = [False]
d_loc = 10
loc_dims = np.array([d_loc for _ in range(n_sites)])
# parameters
par = {
    "lvals": lvals,
    "has_obc": has_obc,
    "n_max": d_loc - 1,
}

# ACQUIRE HAMILTONIAN COEFFICIENTS
coeffs = {"mu2": -0.2, "lambda": 0.6}

start = time()
# CONSTRUCT THE HAMILTONIAN
model = phi4_model.Phi4Model(**par)
model.build_Hamiltonian_bulk(coeffs=coeffs)
simulation = sim(model.H.Ham, par, error_mean=1e-10, error_dec=1e-15)

diag = model.H.diagonalize(n_eigs=n_eigs, format="sparse", loc_dims=loc_dims)
res = {}
res["energy"] = model.H.Nenergies

print(res["energy"] / lvals[0])

# ---------------------------------------------------------------------------
# NEAREST NEIGHBOR INTERACTION

# EXTERNAL MAGNETIC FIELD

# ===========================================================================
# DIAGONALIZE THE HAMILTONIAN


# # Dictionary for results
# res = {}
# res["energy"] = H.Nenergies
# ===========================================================================
