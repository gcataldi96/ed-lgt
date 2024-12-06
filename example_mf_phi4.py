# %%
import numpy as np
from math import prod
from ed_lgt.models import phi4_model

from ed_lgt.algorithms.mean_field import mean_field
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
d_loc = 14
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
simulation = mean_field([model.H.Ham], par, mf_error=1e-12, decomp_error=1e-12)
simulation.sim(par_m)

# observable
res = simulation.get_result()
state = res["state"]


def red_densities(state, n_side_mf, d_loc):
    """
    Arguments:
    state: state of mf calculation
    n_side_mf: 2, 3 .. mf
    d_loc: local dim

    Return:
    Reduces densites and mean density
    TODO: Generalize this
    """

    state_r = state.reshape(-1, 1)
    rho = np.dot(state_r, state_r.T)

    rho_r = rho.reshape(2 * n_side_mf * [d_loc])
    rho_r = rho_r.transpose(0, 2, 1, 3)

    rho1 = np.trace(rho_r, axis1=2, axis2=3)
    rho2 = np.trace(rho_r, axis1=0, axis2=1)
    rho_m = (1 / 2) * (rho1 + rho2)

    return [rho1, rho2, rho_m]


red_densities(state, par_m["n_side_mf"], d_loc)


# ===========================================================================
# DIAGONALIZE THE HAMILTONIAN

diag = model.H.diagonalize(n_eigs=n_eigs, format="sparse", loc_dims=loc_dims)
res = {}
res["energy"] = model.H.Nenergies

print(res["energy"] / lvals[0])
# ===========================================================================
