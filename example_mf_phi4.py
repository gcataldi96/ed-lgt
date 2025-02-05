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


def red_densities_spec(state, n_side_mf, d_loc):
    """
    Arguments:
    state: state of mf calculation
    n_side_mf: 2, 3 .. mf
    d_loc: local dim

    Return:
    Reduces densites and mean density
    TODO: Generalize this n-side mean field.
    """
    rho_r=state.reshape(2 * n_side_mf * [d_loc])

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
d_loc =44
d_red=8
loc_dims = np.array([d_red for _ in range(n_sites)])
mf_sites=2
point=100
# parameters
#set also reduction
Pi=np.eye(d_loc,d_red)
path="mf_data/2_site/lambda_0.6/point_100/"

if True:    
    # Pi = np.loadtxt(path+"rhomean_eigVec.txt")
    Pi=np.identity(3)

par = {"lvals": lvals, "has_obc": has_obc, "n_max": d_loc - 1,"reduction":False, "d_red":d_red, "map":Pi[:,:d_red]}
par_m = {"d_loc": d_loc,"n_sites":mf_sites}

# ACQUIRE HAMILTONIAN COEFFICIENTS
coeffs = {"mu2": -2, "lambda": 0.6}

start = time()
# CONSTRUCT THE HAMILTONIAN
model = phi4_model.Phi4Model(**par)
model.build_Hamiltonian_bulk(coeffs=coeffs)

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
rhos = red_densities(res["state"], par_m["n_sites"], int(np.sqrt(H_2site.shape[0])))
res["state"]=res["state"].tolist()
states=np.array(res["states"])


rho_spec_m=0


for col in [states.T[:,0],states.T[:,1]]:  
    state_r = col.reshape(-1, 1)
    rho_spec_m+=(1/2)*np.dot(state_r, state_r.conj().T)

rhos_spec=red_densities_spec(rho_spec_m, par_m["n_sites"], int(np.sqrt(H_2site.shape[0])))

if par["reduction"]:
    dir_path = "mf_data/"+str(mf_sites)+"_site/"+"reduced/"+"lambda_"+str(coeffs["lambda"])+"/point_"+str(point)+"/"
    os.makedirs(dir_path, exist_ok=True)
    name = (
        "d_loc"
        + str(d_loc)
        +"d_red"
        +str(d_red)
        + "mu2"
        + str(coeffs["mu2"])
        + "lambda_"
        + str(coeffs["lambda"])
        + ".json"
    )
else:
    name = (
    "d_loc"
    + str(d_loc)
    + "mu2"
    + str(coeffs["mu2"])
    + "lambda_"
    + str(coeffs["lambda"])
    + ".json"
    )
    dir_path = "mf_data/"+str(mf_sites)+"_site/"+"lambda_"+str(coeffs["lambda"])+"/point_"+str(point)+"/"
    os.makedirs(dir_path, exist_ok=True)

with open(dir_path+"/"+name, "w") as json_file:
    json.dump(res, json_file, indent=4)

#rho's & rho's diag
for i,y in enumerate(rhos):
    if i==len(rhos)-1:
        np.savetxt(dir_path+name[:-5]+"rho_mean.txt", y, delimiter=" ")
        diag_print(y,dir_path+"/",nameVec=name[:-5]+"rhomean_eigVec.txt",nameVal=name[:-5]+"rhomean_eigVal.txt")
    else: 
        np.savetxt(dir_path+name[:-5]+"rho"+str(i)+".txt", y, delimiter=" ")
        diag_print(y,dir_path+"/",nameVec=name[:-5]+"rho"+str(i)+"_eigVec.txt",nameVal=name[:-5]+"rho"+str(i)+"_eigVal.txt")



#same with spectrum rhos 
#rho's & rho's diag
for i,y in enumerate(rhos_spec):
    if i==len(rhos_spec)-1:
        np.savetxt(dir_path+name[:-5]+"rho_spec2_mean.txt", y, delimiter=" ")
        diag_print(y,dir_path+"/",nameVec=name[:-5]+"rhospec2_mean_eigVec.txt",nameVal=name[:-5]+"rhomean_eigVal.txt")
    else: 
        np.savetxt(dir_path+name[:-5]+"rhospec2"+str(i)+".txt", y, delimiter=" ")
        diag_print(y,dir_path+"/",nameVec=name[:-5]+"rhospec2"+str(i)+"_eigVec.txt",nameVal=name[:-5]+"rhospec2"+str(i)+"_eigVal.txt")