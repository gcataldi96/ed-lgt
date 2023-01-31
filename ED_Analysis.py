# %%
import numpy as np
from simsio import *
import pickle
from matplotlib import pyplot as plt

"""
To extract simulations use
    op1) energy[ii][jj] = extract_dict(ugrid[ii][jj], key="res", glob="energy")
    op2) energy[ii][jj] = get_sim(ugrid[ii][jj]).res["energy"])
To acquire the psi file
    sim= get_sim(ugrid[ii][jj])
    sim.link("psi")
    psi= sim.load("psi", cache=True)
"""


def save_dictionary(dict, filename):
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(dict, outp, pickle.HIGHEST_PROTOCOL)
    outp.close


def load_dictionary(filename):
    with open(filename, "rb") as outp:
        return pickle.load(outp)


def get_obs_list(pure, has_obc):
    obs_list = [
        "energy",
        "entropy",
        "gamma",
        "plaq",
        "delta_gamma",
        "delta_plaq",
    ]
    if not pure:
        obs_list += [
            "n_single_even",
            "n_single_odd",
            "n_pair_even",
            "n_pair_odd",
            "n_tot_even",
            "n_tot_odd",
            "delta_n_single_even",
            "delta_n_single_odd",
            "delta_n_pair_even",
            "delta_n_pair_odd",
            "delta_n_tot_even",
            "delta_n_tot_odd",
        ]
    if not has_obc:
        obs_list += ["py_sector", "px_sector"]
    return obs_list


# %%
# Pure Topology
config_filename = "pure_topology"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])

obs_list = get_obs_list(pure=True, has_obc=False)

res = {}
res["g"] = vals["g"]
for kk, obs in enumerate(["energy", "py_sector", "px_sector"]):
    res[obs] = np.zeros((vals["g"].shape[0], 7))
    for ii, g in enumerate(vals["g"]):
        for n in range(7):
            res[obs][ii][n] = get_sim(ugrid[ii]).res[obs][n]

save_dictionary(res, "dict_simulations/pure_topology.pkl")


# %%
# Full Topology PBC
config_filename = "full_topology3"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])

obs_list = get_obs_list(pure=False, has_obc=False)

res = {}
res["g"] = vals["g"]
res["m"] = vals["m"]
for kk, obs in enumerate(obs_list):
    res[obs] = np.zeros((vals["g"].shape[0], vals["m"].shape[0]))
    for ii, g in enumerate(vals["g"]):
        for jj, m in enumerate(vals["m"]):
            res[obs][ii][jj] = get_sim(ugrid[ii][jj]).res[obs]

save_dictionary(res, "dict_simulations/full_topology2.pkl")

# %%
for kk, obs in enumerate(obs_list[:10]):
    fig = plt.figure()
    for jj, m in enumerate(vals["m"]):
        plt.plot(vals["g"], res[obs][:, jj], "-o")
        plt.xscale("log")
        if kk > 3:
            plt.yscale("log")
        plt.ylabel(obs)

fig = plt.figure()
for jj, m in enumerate(vals["m"]):
    plt.plot(vals["m"], res["energy"][:, jj], "-o")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("py_sector")

fig = plt.figure()
for ii, g in enumerate(vals["g"]):
    plt.plot(vals["m"], 1 - res["py_sector"][ii, :], "-o")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("py_sector")

# %%
# Full Topology PBC
config_filename = "full_low_m_behavior"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])

obs_list = get_obs_list(pure=False, has_obc=True)

res = {}
res["g"] = vals["g"]
res["m"] = vals["m"]
for kk, obs in enumerate(obs_list):
    res[obs] = np.zeros((vals["g"].shape[0], vals["m"].shape[0]))
    for ii, g in enumerate(vals["g"]):
        for jj, m in enumerate(vals["m"]):
            res[obs][ii][jj] = get_sim(ugrid[ii][jj]).res[obs]

save_dictionary(res, "dict_simulations/full_lowmass_behavior.pkl")
# %%
for kk, obs in enumerate(obs_list):
    fig = plt.figure()
    for jj, m in enumerate(vals["m"]):
        plt.plot(vals["g"], res[obs][:, jj], "-o", label=rf"$m={m}$")
        plt.xscale("log")
        plt.ylabel(obs)
    plt.legend()

# %%
