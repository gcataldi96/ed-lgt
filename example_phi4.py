# %%
import numpy as np
from math import prod

# from ed_lgt.modeling import abelian_sector_indices
from ed_lgt.models import phi4_model
from ed_lgt.modeling import LocalTerm, TwoBodyTerm

from time import time
import logging

logger = logging.getLogger(__name__)

# N eigenvalues
n_eigs = 1
# LATTICE GEOMETRY
lvals = [4]
dim = len(lvals)
# directions = "xyz"[:dim]
n_sites = prod(lvals)
has_obc = [False]
d_loc = 10
loc_dims = np.array([d_loc for _ in range(n_sites)])
# parameters
par = {"lvals": lvals, "has_obc": has_obc, "n_max": d_loc - 1}


# HAMILTONIAN COEFFICIENTS
coeffs = {"mu2": -0.2, "lambda": 0.6}
# ACQUIRE OPERATORS

start = time()
# CONSTRUCT THE HAMILTONIAN
model = phi4_model.Phi4Model(**par)
model.build_Hamiltonian(coeffs=coeffs)

model.H.diagonalize(n_eigs=n_eigs, format="sparse", loc_dims=loc_dims)
# diagonalize_Hamiltonian


res = {}
res["energy"] = model.H.Nenergies

print("Eigenvalues computed:", model.H.Nenergies)
print("Number of eigenvalus:", model.H.n_eigs)

print(res["energy"] / lvals[0])

# observable
loc_obs = ["phi"]
model.get_observables(loc_obs)  # how can I check if the set of observbles is not zero?

print(model.get_observables())

for ii in range(model.H.n_eigs):
    # MEASURE OBSERVABLES
    model.measure_observables(ii)
