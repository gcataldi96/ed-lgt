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
    """

    state_r = state.reshape(-1, 1)
    rho = np.dot(state_r, state_r.conj().T)
    rho_r = rho.reshape(2 * n_side_mf * [d_loc])

    t = [2 * i for i in range(n_side_mf)] + [2 * i + 1 for i in range(n_side_mf)]
    #rho_r = rho_r.transpose(t)
    error=1e-15

    if len(t)==4:
        rho1 = np.trace(rho_r, axis1=1, axis2=3)
        rho2 = np.trace(rho_r, axis1=0, axis2=2)
        rho_m = (1 / 2) * (rho1 + rho2)
        rhos=[rho1, rho2,rho_m]

        assert all(np.isclose(np.trace(x), 1, rtol=error, atol=error)  for x in rhos)

        return rhos
    elif len(t)==8:
        
        rho1 = np.einsum('ijklmjkl->im', rho_r)  
        rho2 = np.einsum('ijklimkl->jm', rho_r)  
        rho3 = np.einsum('ijklijml->km', rho_r)  
        rho4 = np.einsum('ijklijkm->lm', rho_r)
        rho_m=(1/4)*(rho1+rho2+rho3+rho4)
        rhos=[rho1,rho2,rho3,rho4,rho_m]

        assert all(np.isclose(np.trace(x), 1, rtol=error, atol=error) for x in rhos)

        return rhos

def H_effective(H,rho):
    Pi=np.kron(rho,rho)
    return np.matmul(Pi.conj().T,np.matmul(H,Pi))


def diag_print(rho,path,nameVec,nameVal):
    eigval, eigvec = np.linalg.eigh(rho)
    idx = eigval.argsort()[::-1]
    eigenValues = eigval[idx]
    eigenVectors = eigvec[:, idx]
    np.savetxt(path+nameVec, eigenVectors, delimiter=" ")
    np.savetxt(path+nameVal, eigenValues, delimiter=" ")


# N eigenvalues
n_eigs = 1
# LATTICE GEOMETRY
lvals = [2]
dim = len(lvals)
# directions = "xyz"[:dim]
n_sites = prod(lvals)
has_obc = [False]
d_loc =8
d_red=5
loc_dims = np.array([d_red for _ in range(n_sites)])
mf_sites=2
point=70
# parameters
par = {"lvals": lvals, "has_obc": has_obc, "n_max": d_loc - 1,"reduction":False, "d_red":False, "map":False}
par_m = {"d_loc": d_loc,"n_sites":mf_sites}

# ACQUIRE HAMILTONIAN COEFFICIENTS
coeffs = {"mu2": -0.2, "lambda": 0.6}

start = time()
# CONSTRUCT THE HAMILTONIAN
model = phi4_model.Phi4Model(**par)
model.build_Hamiltonian_bulk(coeffs=coeffs)

#load reduced density
#rho=np.loadtxt("rho1.txt",dtype=complex)
#eigval, eigvec = np.linalg.eigh(rho)

#idx = eigval.argsort()[::-1]
#eigenValues = eigval[idx]
#eigenVectors = eigvec[:, idx]

#Pi=eigenVectors[:,10]
#project 

if mf_sites==2:
    H_2site=model.H.Ham.toarray()
    H_bulk=model.H.Ham.toarray()
    eigval, eigvec = np.linalg.eigh(H_bulk)

elif mf_sites==4:
    H_2site=model.H.Ham.toarray()
    par["lvals"]=[4]
    model_bulk = phi4_model.Phi4Model(**par)
    model_bulk.build_Hamiltonian_bulk(coeffs=coeffs)
    H_bulk=model_bulk.H.Ham.toarray()

simulation = mean_field([H_2site],H_bulk ,par, mf_error=1e-12, decomp_error=1e-12)
simulation.sim(par_m)

# observable
res = simulation.get_result()
rhos = red_densities(res["state"], par_m["n_sites"], d_loc)
res["state"]=res["state"].tolist()

name = (
    "d_loc"
    + str(d_loc)
    + "mu2"
    + str(coeffs["mu2"])
    + "lambda_"
    + str(coeffs["lambda"])
    + ".json"
)

if mf_sites==2:
    dir_path = "mf_data/2_site/"+"lambda_"+str(coeffs["lambda"])+"/point_"+str(point)+"/"
    os.makedirs(dir_path, exist_ok=True)

    with open(dir_path+"/"+name, "w") as json_file:
        json.dump(res, json_file, indent=4)

    #rho's
    for i,y in enumerate(rhos):
        if i==len(rhos)-1:
            np.savetxt(dir_path+name[:-5]+"rho_mean.txt", rhos[2], delimiter=" ")
        else: 
            np.savetxt(dir_path+name[:-5]+"rho"+str(i+1)+".txt", y, delimiter=" ")

    #rho's diag
    diag_print(rhos[0],dir_path+"/",nameVec=name[:-5]+"rho1_eigVec.txt",nameVal=name[:-5]+"rho1_eigVal.txt")
    diag_print(rhos[1],dir_path+"/",nameVec=name[:-5]+"rho2_eigVec.txt",nameVal=name[:-5]+"rho2_eigVal.txt")
    diag_print(rhos[2],dir_path+"/",nameVec=name[:-5]+"rhomean_eigVec.txt",nameVal=name[:-5]+"rhomean_eigVal.txt")
elif mf_sites==4:
    dir_path = "mf_data/4_site/"+"lambda_"+str(coeffs["lambda"])+"/point_"+str(point)+"/"
    os.makedirs(dir_path, exist_ok=True)

    with open(dir_path+"/"+name, "w") as json_file:
        json.dump(res, json_file, indent=4)

    #rho's
    np.savetxt(dir_path+"/"+"rho1.txt", rhos[0], delimiter=" ")
    np.savetxt(dir_path+"/"+"rho2.txt", rhos[1], delimiter=" ")
    np.savetxt(dir_path+"/"+"rho3.txt", rhos[2], delimiter=" ")
    np.savetxt(dir_path+"/"+"rho4.txt", rhos[3], delimiter=" ")
    np.savetxt(dir_path+"/"+"rho_mean.txt", rhos[2], delimiter=" ")

    #rho's diag
    diag_print(rhos[0],dir_path+"/",nameVec="rho1_eigVec.txt",nameVal="rho1_eigVal.txt")
    diag_print(rhos[1],dir_path+"/",nameVec="rho2_eigVec.txt",nameVal="rho2_eigVal.txt")
    diag_print(rhos[2],dir_path+"/",nameVec="rho3_eigVec.txt",nameVal="rho3_eigVal.txt")
    diag_print(rhos[3],dir_path+"/",nameVec="rho4_eigVec.txt",nameVal="rho4_eigVal.txt")
    diag_print(rhos[4],dir_path+"/",nameVec="rhomean_eigVec.txt",nameVal="rhomean_eigVal.txt")


