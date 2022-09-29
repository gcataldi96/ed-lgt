# %%
import numpy as np
from simsio import *
import pickle


def save_dictionary(dict, filename):
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(dict, outp, pickle.HIGHEST_PROTOCOL)
    outp.close


# %%
# ACQUIRE SIMULATION RESULTS
config_filename = "MG_grid_5eigs"

match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["mass", "gSU2"])

# %%
def entanglement_entropy(psi, loc_dim, partition):
    # COMPUTE THE ENTANGLEMENT ENTROPY OF A SPECIFIC SUBSYSTEM
    tmp = psi.reshape((loc_dim**partition, loc_dim**partition))
    S, V, D = np.linalg.svd(tmp)
    tmp = np.array(
        [-np.abs(llambda**2) * np.log(np.abs(llambda**2)) for llambda in V]
    )
    return np.sum(tmp)


Ent_entropy = np.zeros((vals["mass"].shape[0], vals["gSU2"].shape[0]))
for ii in range(vals["mass"].shape[0]):
    for jj in range(vals["gSU2"].shape[0]):
        psi = extract_dict(ugrid[ii][jj], key="res", glob="psi")
        Ent_entropy[ii][jj] = entanglement_entropy(psi, loc_dim=30, partition=2)


# %%
with open("Results/Simulation_Dicts/ED_2x2_MG_5eigs.pkl", "rb") as dict:
    ED_data = pickle.load(dict)
ED_data["entropy"] = Ent_entropy
# %%
energy = np.zeros((30, 30, 5))
for ii in range(30):
    for jj in range(30):
        energy[ii][jj] = extract_dict(ugrid[ii][jj], key="res", glob="energy")

ED_data["energy"] = energy

# %%
fidelity = np.zeros((30, 29))
for ii in range(vals["mass"].shape[0]):
    for jj in range(vals["gSU2"].shape[0] - 1):
        fidelity[ii][jj] = np.abs(
            np.real(
                np.dot(
                    np.conjugate(extract_dict(ugrid[ii][jj], key="res", glob="psi")),
                    extract_dict(ugrid[ii][jj + 1], key="res", glob="psi"),
                )
            )
        )

# %%
res = {}
res["energy"] = np.vectorize(extract_dict)(ugrid, key="res", glob="energy")
res["gamma"] = np.vectorize(extract_dict)(ugrid, key="res", glob="gamma")
res["plaq"] = np.vectorize(extract_dict)(ugrid, key="res", glob="plaq")
res["n_single_EVEN"] = np.vectorize(extract_dict)(
    ugrid, key="res", glob="n_single_EVEN"
)
res["n_single_ODD"] = np.vectorize(extract_dict)(ugrid, key="res", glob="n_single_ODD")
res["n_pair_EVEN"] = np.vectorize(extract_dict)(ugrid, key="res", glob="n_pair_EVEN")
res["n_pair_ODD"] = np.vectorize(extract_dict)(ugrid, key="res", glob="n_pair_ODD")

res["n_tot_EVEN"] = np.vectorize(extract_dict)(ugrid, key="res", glob="n_tot_EVEN")
res["n_tot_ODD"] = np.vectorize(extract_dict)(ugrid, key="res", glob="n_tot_ODD")

res["params"] = vals
# %%
save_dictionary(res, "ED_2x2_MG_5eigs.pkl")
