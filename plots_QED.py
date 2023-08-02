# %%
import numpy as np
from simsio import *
from matplotlib import pyplot as plt
from tools import save_dictionary

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
# QED ENTANGLEMENT SCALING
# ========================================================================
config_filename = "QED/entanglement"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["spin", "g"])
obs_list = get_obs_list(model="QED", pure=False, has_obc=False)
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
# ========================================================================
# QED SINGULAR VALUES
# ========================================================================
config_filename = "QED/DM_scaling_PBC"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = get_obs_list(model="QED", pure=False, has_obc=False)
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
# ========================================================================
# QED Energy Gap convergence with different Parallel Transporters
# ========================================================================
U_definitions = ["spin", "ladder"]
for U in U_definitions:
    config_filename = f"QED/U_{U}"
    match = SimsQuery(group_glob=config_filename)
    ugrid, vals = uids_grid(match.uids, ["g", "spin"])
    res = {"g": vals["g"], "spin": vals["spin"]}
    res_shape = (res["g"].shape[0], res["spin"].shape[0])
    for obs in ["DeltaE", "E0", "E1", "B0", "B1", "DeltaB"]:
        res[obs] = np.zeros(res_shape)
    for ii, g in enumerate(res["g"]):
        for jj, spin in enumerate(res["spin"]):
            res["E0"][ii, jj] = get_sim(ugrid[ii, jj]).res["energy"][0]
            res["E1"][ii, jj] = get_sim(ugrid[ii, jj]).res["energy"][1]
            res["B0"][ii, jj] = get_sim(ugrid[ii, jj]).res["plaq"][0]
            res["B1"][ii, jj] = get_sim(ugrid[ii, jj]).res["plaq"][1]
            res["DeltaE"][ii, jj] = get_sim(ugrid[ii, jj]).res["DeltaE"]
            res["DeltaB"][ii, jj] = np.abs(res["B1"][ii, jj] - res["B0"][ii, jj])

    for ii, g in enumerate(res["g"]):
        beta = 1 / (g**2)
        fig = plt.figure()
        plt.ylabel(r"|Delta E|")
        plt.xlabel(r"s")
        plt.yscale("log")
        plt.grid()
        plt.plot(res["spin"][:], res["DeltaE"][ii, :], "-o", label=f"beta={beta}")
        plt.legend()
    save_dictionary(res, f"saved_dicts/QED_U_{U}.pkl")

# %%
