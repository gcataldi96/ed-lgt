# %%
import numpy as np
from simsio import *
from matplotlib import pyplot as plt
from tools import save_dictionary, get_charge, get_density, structure_factor

"""
To extract simulations use
    op1) energy[ii][jj] = extract_dict(ugrid[ii][jj], key="res", glob="energy")
    op2) energy[ii][jj] = get_sim(ugrid[ii][jj]).res["energy"])
To acquire the psi file
    sim= get_sim(ugrid[ii][jj])
    sim.link("psi")
    psi= sim.load("psi", cache=True)
"""


@plt.FuncFormatter
def fake_log(x, pos):
    "The two args are the value and tick position"
    return r"$10^{%d}$" % (x)


def get_obs_list(model, pure=None, has_obc=True):
    obs_list = [
        "energy",
        "entropy",
        "E_square",
        "plaq",
    ]
    if model == "SU2":
        obs_list += ["delta_E_square", "delta_plaq"]
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
    else:
        obs_list += ["N"]
    return obs_list


# %%
# ========================================================================
# SU(2) SIMULATIONS
# ========================================================================
# Pure Topology
config_filename = "SU2/pure/topology"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = get_obs_list(model="SU2", pure=True, has_obc=True)
res = {"g": vals["g"]}

for obs in ["energy", "py_sector", "px_sector"]:
    res[obs] = np.zeros((vals["g"].shape[0], 5))
    for ii in range(len(res["g"])):
        for n in range(5):
            res[obs][ii][n] = get_sim(ugrid[ii]).res[obs][n]
fig = plt.figure()
for n in range(1, 5):
    plt.plot(
        vals["g"],
        res["energy"][:, n] - res["energy"][:, 0],
        "-o",
        label=f"{format(res['px_sector'][0, n],'.5f')}",
    )
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.ylabel("energy")
save_dictionary(res, "saved_dicts/SU2_pure_topology.pkl")
# %%
# Full Topology 1
config_filename = "SU2/full/topology1"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
obs_list = get_obs_list(model="SU2", pure=False, has_obc=True)
res = {"g": vals["g"], "m": vals["m"]}
for obs in ["energy", "py_sector", "px_sector"]:
    res[obs] = np.zeros((res["g"].shape[0], res["m"].shape[0]))
    for ii, g in enumerate(res["g"]):
        for jj, m in enumerate(res["m"]):
            res[obs][ii][jj] = get_sim(ugrid[ii][jj]).res[obs]
fig = plt.figure()
for jj, m in enumerate(res["m"]):
    plt.plot(vals["g"], 1 - res["py_sector"][:, jj], "-o", label=f"m={format(m,'.3f')}")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.ylabel("1-py_sector")
save_dictionary(res, "saved_dicts/SU2_full_topology1.pkl")
# %%
# Full Topology 2
config_filename = "SU2/full/topology2"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
obs_list = get_obs_list(model="SU2", pure=False, has_obc=True)
res = {"g": vals["g"], "m": vals["m"]}
for obs in ["energy", "py_sector", "px_sector"]:
    res[obs] = np.zeros((res["g"].shape[0], res["m"].shape[0]))
    for ii, g in enumerate(res["g"]):
        for jj, m in enumerate(res["m"]):
            res[obs][ii][jj] = get_sim(ugrid[ii][jj]).res[obs]
fig = plt.figure()
for ii, g in enumerate(res["g"]):
    plt.plot(vals["m"], 1 - res["py_sector"][ii, :], "-o", label=f"g={format(g,'.3f')}")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.ylabel("1-py_sector")
save_dictionary(res, "saved_dicts/SU2_full_topology2.pkl")
# %%
# SU2 PHASE DIAGRAM
config_filename = "SU2/full/phase_diagram"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
obs_list = get_obs_list(model="SU2", pure=False, has_obc=False)
res = {"g": vals["g"], "m": vals["m"]}
for obs in obs_list:
    res[obs] = np.zeros((res["g"].shape[0], res["m"].shape[0]))
    for ii, g in enumerate(res["g"]):
        for jj, m in enumerate(res["m"]):
            res[obs][ii][jj] = get_sim(ugrid[ii][jj]).res[obs]

fig, axs = plt.subplots(
    2,
    1,
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
obs = ["E_square", "n_tot_even"]
for ii, ax in enumerate(axs.flat):
    # IMSHOW
    img = ax.imshow(
        np.transpose(res[obs[ii]]),
        origin="lower",
        cmap="magma",
        extent=[-2, 2, -3, 1],
    )
    ax.set_ylabel(r"m")
    axs[1].set_xlabel(r"g2")
    ax.set(xticks=[-2, -1, 0, 1, 2], yticks=[-3, -2, -1, 0, 1])
    ax.xaxis.set_major_formatter(fake_log)
    ax.yaxis.set_major_formatter(fake_log)

    cb = fig.colorbar(
        img,
        ax=ax,
        aspect=20,
        location="right",
        orientation="vertical",
        pad=0.01,
        label=obs[ii],
    )
save_dictionary(res, "saved_dicts/SU2_full_phase_diagram.pkl")
# %%
# SU(2) SuperConductingOrderParameter
config_filename = "SU2/full/SCOP"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = get_obs_list(model="SU2", pure=False, has_obc=True)
res = {"g": vals["g"]}
for obs in obs_list:
    res[obs] = np.zeros(res["g"].shape[0])
    for ii, g in enumerate(res["g"]):
        res[obs][ii] = get_sim(ugrid[ii]).res[obs]
# SCOP MEASURES
res["SCOP"] = []
for ii, g in enumerate(res["g"]):
    res["SCOP"].append(get_sim(ugrid[ii]).res["SCOP"])

# SCOP = structure_factor(res["SCOP"][0], [2, 2])

# %%
# QED U COMPARISON
config_filename = "QED/U_comparison"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["U", "spin"])
obs_list = get_obs_list(model="SU2", pure=False, has_obc=False)
res = {"U": vals["U"], "spin": vals["spin"]}
res["DeltaE"] = np.zeros((res["U"].shape[0], res["spin"].shape[0]))
for ii, g in enumerate(res["U"]):
    for jj, m in enumerate(res["spin"]):
        res["DeltaE"][ii][jj] = get_sim(ugrid[ii][jj]).res["DeltaE"]
        res["DeltaE"][ii][jj] /= np.abs(get_sim(ugrid[ii][jj]).res["energy"][0])
fig = plt.figure()
plt.ylabel(r"|Delta E|/|E0|")
plt.xlabel(r"s")
plt.yscale("log")
plt.grid()
for ii, U in enumerate(res["U"]):
    plt.plot(res["spin"], res["DeltaE"][ii][:], "-o", label=f"U={U}")
plt.legend()
save_dictionary(res, "saved_dicts/QED_U_comparison.pkl")
# %%
# QED ENTANGLEMENT
config_filename = "QED/entanglement"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["spin", "g"])
obs_list = get_obs_list(model="SU2", pure=False, has_obc=False)
res = {"g": vals["g"], "spin": vals["spin"]}
res["entropy"] = np.zeros((res["spin"].shape[0], res["g"].shape[0]))
for ii, s in enumerate(res["spin"]):
    for jj, g in enumerate(res["g"]):
        res["entropy"][ii][jj] = get_sim(ugrid[ii][jj]).res["entropy"]
fig = plt.figure()
plt.ylabel(r"Entanglement entropy")
plt.xlabel(r"g")
plt.xscale("log")
plt.grid()
for ii, s in enumerate(res["spin"]):
    plt.plot(res["g"], res["entropy"][ii, :], "-o", label=f"s={s}")
plt.legend()
save_dictionary(res, "saved_dicts/QED_entanglement.pkl")
# %%
# QED singular values
config_filename = "QED/DM_scaling_PBC"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = get_obs_list(model="SU2", pure=False, has_obc=False)
res = {"g": vals["g"]}
res["rho0"] = []
res["rho1"] = []
for ii, g in enumerate(res["g"]):
    res["rho0"].append(get_sim(ugrid[ii]).res["rho_eigvals"][0][::-1])
    res["rho1"].append(get_sim(ugrid[ii]).res["rho_eigvals"][1][::-1])
fig = plt.figure()
plt.ylabel(r"Value")
plt.yscale("log")
plt.xlabel(r"Singular Values")
plt.grid()
for ii, g in enumerate(res["g"]):
    plt.plot(np.arange(35), res["rho0"][ii], "-", label=f"g={format(g,'.3f')}")
plt.legend()
fig = plt.figure()
plt.ylabel(r"Value")
plt.yscale("log")
plt.xlabel(r"Singular Values")
plt.grid()
for ii, g in enumerate(res["g"]):
    plt.plot(np.arange(35), res["rho1"][ii], "-", label=f"g={format(g,'.3f')}")
plt.legend()
save_dictionary(res, "saved_dicts/QED_singular_values.pkl")

# %%
