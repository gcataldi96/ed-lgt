# %%
import numpy as np
from math import prod
import os
import json

from ed_lgt.models import phi4_model
from ed_lgt.algorithms.mean_field import mean_field
from time import time
import logging

logger = logging.getLogger(__name__)

from test_model import phi4_model_test

############# Simulation
def red_densities(state, n_side_mf, d_loc):
    """
    Arguments:
    state: state of mf calculation
    n_side_mf: 2, 3 .. mf
    d_loc: local dim

    Return:
    Reduces densites and mean density
    TODO: Generalize this n-side mean field.
    (reshaping is already fine, work on tracing)
    """

    state_r = state.reshape(-1, 1)
    rho = np.dot(state_r, state_r.T)
    rho_r = rho.reshape(2 * n_side_mf * [d_loc])

    t = [2 * i for i in range(n_side_mf)] + [2 * i + 1 for i in range(n_side_mf)]
    rho_r = rho_r.transpose(t)

    rho1 = np.trace(rho_r, axis1=2, axis2=3)
    rho2 = np.trace(rho_r, axis1=0, axis2=1)
    rho_m = (1 / 2) * (rho1 + rho2)

    return [rho1, rho2, rho_m]


# N eigenvalues
n_eigs = 1
# LATTICE GEOMETRY
lvals = [4]
dim = len(lvals)
# directions = "xyz"[:dim]
n_sites = prod(lvals)
has_obc = [False]
d_loc = 24
loc_dims = np.array([d_loc for _ in range(n_sites)])
# parameters
par = {"lvals": lvals, "has_obc": has_obc, "n_max": d_loc - 1}
par_m = {"d_loc": d_loc, "n_side_mf": 2}

# ACQUIRE HAMILTONIAN COEFFICIENTS
coeffs = {"mu2": -0.2, "lambda": 0.6}

start = time()
# CONSTRUCT THE HAMILTONIAN
model = phi4_model.Phi4Model(**par)
model.build_Hamiltonian_bulk(coeffs=coeffs)

#try other hamiltonian
#H=phi4_model_test(d_loc,coeffs)
#simulation = mean_field([H], par, mf_error=1e-12, decomp_error=1e-12)

simulation = mean_field([model.H.Ham.toarray()], par, mf_error=1e-12, decomp_error=1e-12)
simulation.sim(par_m)

# observable
res = simulation.get_result()
rhos = red_densities(res["state"], par_m["n_side_mf"], d_loc)

#of rho1, double check precission
eigval, eigvec = np.linalg.eigh(rhos[0])

#just save rho1
np.savetxt("rho1.txt", rhos[0], delimiter=" ")

# print to dict
dir_path = "mf_data"
os.makedirs(dir_path, exist_ok=True)

# turn np.array into list
res["state"] = res["state"].tolist()
res["eigval"] = eigval.tolist()

name = (
    dir_path
    + "/d_loc"
    + str(d_loc)
    + "mu2"
    + str(coeffs["mu2"])
    + "lambda_"
    + str(coeffs["lambda"])
    + ".json"
)
with open(name, "w") as json_file:
    json.dump(res, json_file, indent=4)

#sorting according size of eigval
idx = eigval.argsort()[::-1]
eigenValues = eigval[idx]
eigenVectors = eigvec[:, idx]

np.savetxt("rho1_eigVec.txt", eigenVectors, delimiter=" ")
np.savetxt("rho1_eigVal.txt", eigenValues, delimiter=" ")


#mean rho
eigval, eigvec = np.linalg.eigh(rhos[0])

idx = eigval.argsort()[::-1]
eigenValues = eigval[idx]
eigenVectors = eigvec[:, idx]

np.savetxt("rhomean_eigVec.txt", eigenVectors, delimiter=" ")
np.savetxt("rhomean_eigVal.txt", eigenValues, delimiter=" ")