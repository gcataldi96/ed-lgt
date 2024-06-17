# %%
from simsio import *
from math import prod
from matplotlib import pyplot as plt
from ed_lgt.tools import save_dictionary, load_dictionary


@plt.FuncFormatter
def fake_log(x, pos):
    "The two args are the value and tick position"
    return r"$10^{%d}$" % (x)


# %%
# List of local observables
local_obs = [f"n_{s}{d}" for d in "xy" for s in "mp"]
local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
local_obs += ["X_Cross", "S2"]

BC_list = ["PBCxy"]
lattize_size_list = ["2x2", "3x2"]
for ii, BC in enumerate(BC_list):
    res = {}
    # define the observables arrays
    res["energy"] = np.zeros((len(lattize_size_list), 25), dtype=float)
    for obs in local_obs:
        res[obs] = np.zeros((len(lattize_size_list), 25), dtype=float)
    for jj, size in enumerate(lattize_size_list):
        # look at the simulation
        config_filename = f"Z2FermiHubbard/{BC}/{size}"
        match = SimsQuery(group_glob=config_filename)
        ugrid, vals = uids_grid(match.uids, ["U"])
        lvals = get_sim(ugrid[0]).par["model"]["lvals"]
        for kk, U in enumerate(vals["U"]):
            for obs in local_obs:
                res[obs][jj, kk] = np.mean(get_sim(ugrid[kk]).res[obs])
                res["energy"][jj, kk] = get_sim(ugrid[kk]).res["energies"][0] / (
                    prod(lvals)
                )
    save_dictionary(res, f"{BC}.pkl")
# %%
for obs in ["X_Cross", "N_pair", "N_single", "energy", "S2"]:
    print(obs)
    fig = plt.figure()
    plt.ylabel(rf"{obs}")
    plt.xlabel(r"U")
    plt.xscale("log")
    plt.grid()
    for ii, label in enumerate(lattize_size_list):
        plt.plot(vals["U"], res[obs][ii, :], "-o", label=label)
    plt.legend(loc=(0.05, 0.11))
    # plt.savefig(f"{obs}.pdf")
# %%


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set(xscale="log", yscale="log")
config_filename = f"Z2FermiHubbard/entropy"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["U"])
lvals = get_sim(ugrid[0]).par["model"]["lvals"]
entropies = np.zeros((len(vals["U"]), 3))
for kk, U in enumerate(vals["U"]):
    C = get_sim(ugrid[kk]).res["Sz_Sz"][:]
    entropies[kk, :] = get_sim(ugrid[kk]).res["entropy"]
    ax.plot(C[:, 0], C[:, 1], "-o", label=f"U={U}")
ax.set(xlabel="r=|i-j|", ylabel="<SzSz>")
plt.legend()
# %%
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

sm = cm.ScalarMappable(cmap="plasma", norm=LogNorm())
palette = sm.to_rgba(vals["U"])
fig1, ax1 = plt.subplots()
for kk, U in enumerate(vals["U"]):
    ax1.plot(np.arange(3), entropies[kk, :], "-o", color=palette[kk])
ax1.set(xlabel="A", ylabel="EE")
plt.legend()
# %%
config_filename = "Z2_FermiHubbard/entropy"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["has_obc", "U"])

res = {}
# List of local observables
lvals = get_sim(ugrid[0][0]).par["lvals"]
local_obs = [f"n_{s}{d}" for d in "xyz"[: len(lvals)] for s in "mp"]
local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
local_obs += ["X_Cross"]
for obs in local_obs:
    res[obs] = np.zeros((vals["has_obc"].shape[0], vals["U"].shape[0]))

res["energy"] = np.zeros((vals["has_obc"].shape[0], vals["U"].shape[0]))

for ii, has_obc in enumerate(vals["has_obc"]):
    for jj, U in enumerate(vals["U"]):
        res["energy"][ii, jj] = get_sim(ugrid[ii][jj]).res["energies"][0] / (
            prod(lvals)
        )
        for obs in local_obs:
            res[obs][ii, jj] = np.mean(get_sim(ugrid[ii][jj]).res[obs])

fig = plt.figure()
plt.ylabel(r"X_cross")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["X_Cross"][ii, :], "-o", label=BC_label)
plt.legend()

fig = plt.figure()
plt.ylabel(r"N")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["N_pair"][ii, :], "-o", label=f"pair ({BC_label})")
    plt.plot(vals["U"], res["N_single"][ii, :], "-o", label=f"single ({BC_label})")
plt.legend()

fig = plt.figure()
plt.ylabel(r"Energy Density")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["energy"][ii, :], "-o", label=BC_label)
plt.legend(loc="lower right")

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set(xscale="log", yscale="log")
config_filename = f"Z2FermiHubbard/prova1"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["U"])
lvals = get_sim(ugrid[0]).par["model"]["lvals"]
entropies = []
for kk, U in enumerate(vals["U"]):
    C = get_sim(ugrid[kk]).res["C"]
    entropies.append(get_sim(ugrid[kk]).res["entropy"])
    ax.plot(C[:, 0], C[:, 1], "-o", label=f"U={U}")
ax.set(xlabel="r=|i-j|", ylabel="<SzSz>")
plt.legend()

fig1, ax1 = plt.subplots()
for kk, U in enumerate(vals["U"]):
    ax1.plot(np.arange(4), entropies[kk][:4], "-o", label=f"U={U}")
ax1.set(xlabel="A", ylabel="EE")
plt.legend()
# %%
config_filename = "Z2_FermiHubbard/U_potential"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["has_obc", "U"])

res = {}
# List of local observables
lvals = get_sim(ugrid[0][0]).par["lvals"]
local_obs = [f"n_{s}{d}" for d in "xyz"[: len(lvals)] for s in "mp"]
local_obs += [f"N_{label}" for label in ["up", "down", "tot", "single", "pair"]]
local_obs += ["X_Cross"]
for obs in local_obs:
    res[obs] = np.zeros((vals["has_obc"].shape[0], vals["U"].shape[0]))

res["energy"] = np.zeros((vals["has_obc"].shape[0], vals["U"].shape[0]))

for ii, has_obc in enumerate(vals["has_obc"]):
    for jj, U in enumerate(vals["U"]):
        res["energy"][ii, jj] = get_sim(ugrid[ii][jj]).res["energies"][0] / (
            prod(lvals)
        )
        for obs in local_obs:
            res[obs][ii, jj] = np.mean(get_sim(ugrid[ii][jj]).res[obs])

fig = plt.figure()
plt.ylabel(r"X_cross")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["X_Cross"][ii, :], "-o", label=BC_label)
plt.legend()

fig = plt.figure()
plt.ylabel(r"N")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["N_pair"][ii, :], "-o", label=f"pair ({BC_label})")
    plt.plot(vals["U"], res["N_single"][ii, :], "-o", label=f"single ({BC_label})")
plt.legend()

fig = plt.figure()
plt.ylabel(r"Energy Density")
plt.xlabel(r"U")
plt.xscale("log")
plt.grid()
for ii, has_obc in enumerate(vals["has_obc"]):
    BC_label = "OBC" if has_obc else "PBC"
    plt.plot(vals["U"], res["energy"][ii, :], "-o", label=BC_label)
plt.legend(loc="lower right")

# %%
# ========================================================================
# ISING MODEL 1D ENERGY GAPS
# ========================================================================
config_filename = "Ising/Ising1D"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["lvals", "h"])
res = {
    "th_gap": np.zeros((vals["lvals"].shape[0], vals["h"].shape[0])),
    "true_gap": np.zeros((vals["lvals"].shape[0], vals["h"].shape[0])),
}
for ii, lvals in enumerate(vals["lvals"]):
    for jj, h in enumerate(vals["h"]):
        for obs in res.keys():
            res[obs][ii, jj] = get_sim(ugrid[ii][jj]).res[obs]

res["abs_distance"] = np.zeros((vals["lvals"].shape[0], vals["h"].shape[0]))
res["rel_distance"] = np.zeros((vals["lvals"].shape[0], vals["h"].shape[0]))
for ii, lvals in enumerate(vals["lvals"]):
    for jj, h in enumerate(vals["h"]):
        res["abs_distance"][ii, jj] = np.abs(
            res["true_gap"][ii, jj] - res["th_gap"][ii, jj]
        )
        res["rel_distance"][ii, jj] = (
            res["abs_distance"][ii, jj] / res["true_gap"][ii, jj]
        )
for obs in ["Sz", "Sx"]:
    res[obs] = np.zeros((vals["lvals"].shape[0], vals["h"].shape[0]))
    for ii, lvals in enumerate(vals["lvals"]):
        for jj, h in enumerate(vals["h"]):
            res[obs][ii, jj] = np.mean(get_sim(ugrid[ii][jj]).res[obs])

fig = plt.figure()
plt.ylabel(r"True_Gap")
plt.xlabel(r"h")
plt.xscale("log")
plt.yscale("log")
plt.grid()
for ii, lvals in enumerate(vals["lvals"]):
    plt.plot(vals["h"][:], res["true_gap"][ii, :], "-o", label=f"L={lvals}")
plt.legend()
plt.savefig("True_Gap_log.pdf")


fig = plt.figure()
plt.ylabel(r"Abs difference (true gap - th gap)")
plt.xlabel(r"h")
plt.xscale("log")
plt.grid()
for ii, lvals in enumerate(vals["lvals"]):
    plt.plot(vals["h"], res["abs_distance"][ii, :], "-o", label=f"L={lvals}")
plt.legend()
plt.savefig("abs_diff.pdf")

fig = plt.figure()
plt.ylabel(r"Rel difference (true gap - th gap)/(true gap)")
plt.xlabel(r"h")
plt.xscale("log")
plt.yscale("log")
plt.grid()
for ii, lvals in enumerate(vals["lvals"]):
    plt.plot(vals["h"][4:], res["rel_distance"][ii, 4:], "-o", label=f"L={lvals}")
plt.legend()
plt.savefig("rel_diff.pdf")

fig = plt.figure()
plt.ylabel(r"Sz")
plt.xlabel(r"h")
plt.xscale("log")
plt.grid()
for ii, lvals in enumerate(vals["lvals"]):
    plt.plot(vals["h"], res["Sz"][ii, :], "-o", label=f"L={lvals}")
plt.legend()
plt.savefig("Sz.pdf")


# %%
def LGT_obs_list(model, pure=None, has_obc=True):
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


# ========================================================================
# QED ENTANGLEMENT SCALING
# ========================================================================
config_filename = "QED/entanglement"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["spin", "g"])
obs_list = LGT_obs_list(model="QED", pure=False, has_obc=False)
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
obs_list = LGT_obs_list(model="QED", pure=False, has_obc=False)
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
# ========================================================================
# SU(2) SIMULATIONS PURE FLUCTUATIONS
# ========================================================================
config_filename = "SU2/pure/fluctuations"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = LGT_obs_list(model="SU2", pure=True, has_obc=True)
res = {"g": vals["g"]}
for obs in obs_list:
    res[obs] = []
    for ii in range(len(res["g"])):
        res[obs].append(get_sim(ugrid[ii]).res[obs])
    res[obs] = np.asarray(res[obs])

fig, ax = plt.subplots()
ax.plot(res["g"], res["E_square"], "-o", label=f"E2")
ax.plot(res["g"], res["delta_E_square"], "-o", label=f"Delta")
ax.set(xscale="log")
ax2 = ax.twinx()
ax2.plot(res["g"], -res["plaq"] + max(res["plaq"]), "-^", label=f"B2")
ax2.plot(res["g"], res["delta_plaq"], "-*", label=f"DeltaB")
ax.legend()
ax2.legend()
ax.grid()
save_dictionary(res, "saved_dicts/SU2_pure_fluctuations.pkl")
# %%
# ========================================================================
# SU(2) SIMULATIONS PURE TOPOLOGY
# ========================================================================
config_filename = "SU2/pure/topology"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = LGT_obs_list(model="SU2", pure=True, has_obc=False)
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
# ========================================================================
# SU(2) FULL TOPOLOGY 1
# ========================================================================
config_filename = "SU2/full/topology1"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=True)
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
# ========================================================================
# SU(2) FULL TOPOLOGY 2
# ========================================================================
config_filename = "SU2/full/topology2"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=True)
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
# ========================================================================
# SU(2) PHASE DIAGRAM
# ========================================================================
config_filename = "SU2/full/phase_diagram"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=False)
res = {"g": vals["g"], "m": vals["m"]}
for obs in obs_list:
    res[obs] = np.zeros((res["g"].shape[0], res["m"].shape[0]))
    for ii, g in enumerate(res["g"]):
        for jj, m in enumerate(res["m"]):
            res[obs][ii][jj] = get_sim(ugrid[ii][jj]).res[obs]

fig, axs = plt.subplots(
    3,
    1,
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
obs = ["E_square", "rho", "spin"]
for ii, ax in enumerate(axs.flat):
    # IMSHOW
    img = ax.imshow(
        np.transpose(res[obs[ii]]),
        origin="lower",
        cmap="magma",
        extent=[-2, 2, -3, 1],
    )
    ax.set_ylabel(r"m")
    axs[2].set_xlabel(r"g2")
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
# ========================================================================
# SU(2) FULL THEORY CHARGE vs DENSITY
# ========================================================================
config_filename = "SU2/full/charge_vs_density"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=True)
res = {"g": vals["g"], "m": vals["m"]}
for obs in ["n_tot_even", "n_tot_odd"]:
    res[obs] = np.zeros((res["g"].shape[0], res["m"].shape[0]))
    for ii, g in enumerate(res["g"]):
        for jj, m in enumerate(res["m"]):
            res[obs][ii][jj] = get_sim(ugrid[ii][jj]).res[obs]
fig = plt.figure()
for ii, m in enumerate(res["m"]):
    plt.plot(
        vals["g"],
        2 + res["n_tot_even"][:, ii] - res["n_tot_odd"][:, ii],
        "-o",
        label=f"g={format(m,'.3f')}",
    )
plt.xscale("log")
plt.legend()
plt.ylabel("rho")
save_dictionary(res, "saved_dicts/charge_vs_density.pkl")
# %%
# ========================================================================
# SU(2) FULL THEORY TTN COMPARISON
# ========================================================================
config_filename = "SU2/full/TTN_comparison"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=True)
res = {"g": vals["g"]}
for obs in ["energy", "n_tot_even", "n_tot_odd", "E_square"]:
    res[obs] = np.zeros(res["g"].shape[0])
    for ii, g in enumerate(res["g"]):
        res[obs][ii] = get_sim(ugrid[ii]).res[obs]
fig = plt.figure()
plt.plot(vals["g"], 2 + res["n_tot_even"][:] - res["n_tot_odd"][:], "-o")
plt.xscale("log")
plt.legend()
plt.ylabel("rho")
save_dictionary(res, "saved_dicts/TTN_comparison.pkl")
# %%
# ========================================================================
# SU(2) FULL THEORY ENERGY GAPS
# ========================================================================
config_filename = "SU2/full/energy_gaps"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["DeltaN", "g", "k"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=True)
res = {"DeltaN": vals["DeltaN"], "g": vals["g"], "k": vals["k"]}
res_shape = (res["DeltaN"].shape[0], res["g"].shape[0], res["k"].shape[0])
res["energy"] = np.zeros(res_shape)
res["m"] = np.zeros(res_shape)
for ii, DeltaN in enumerate(res["DeltaN"]):
    for jj, g in enumerate(res["g"]):
        for kk, k in enumerate(res["k"]):
            res["m"][ii, jj, kk] = get_sim(ugrid[ii, jj, kk]).res["m"]
            res["energy"][ii, jj, kk] = get_sim(ugrid[ii, jj, kk]).res["energy"][0]

fig = plt.figure()
for kk, k in enumerate(res["k"]):
    plt.plot(
        vals["g"] ** 2,
        res["energy"][1, :, kk] - res["energy"][0, :, kk],
        "--o",
        label=f"k={k}, TOT",
    )


plt.xscale("log")
plt.yscale("log")
plt.xlabel("g2")
plt.legend()
plt.ylabel("DEltaE")

fig = plt.figure()
for kk, k in enumerate(res["k"]):
    plt.plot(
        vals["g"] ** 2,
        res["energy"][1, :, kk] - res["energy"][0, :, kk] - 0.5 * res["m"][1, :, kk],
        "-^",
        label=f"k={k} RES",
    )
plt.xscale("log")
plt.yscale("log")
plt.xlabel("g2")
plt.legend()
plt.ylabel("DEltaE_res")
save_dictionary(res, "saved_dicts/SU2_energy_gap.pkl")

# %%
# ========================================================================
# SU(2) FULL THEORY ENERGY GAPS
# ========================================================================
config_filename = "SU2/full/energy_gaps"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["DeltaN", "m", "k"])
obs_list = LGT_obs_list(model="SU2", pure=False, has_obc=False)
res = {"DeltaN": vals["DeltaN"], "m": vals["m"], "k": vals["k"]}
res_shape = (res["DeltaN"].shape[0], res["m"].shape[0], res["k"].shape[0])
res["energy"] = np.zeros(res_shape)
res["g"] = np.zeros(res_shape)
for ii, DeltaN in enumerate(res["DeltaN"]):
    for jj, m in enumerate(res["m"]):
        for kk, k in enumerate(res["k"]):
            res["g"][ii, jj, kk] = get_sim(ugrid[ii, jj, kk]).res["g"]
            res["energy"][ii, jj, kk] = get_sim(ugrid[ii, jj, kk]).res["energy"][0]

fig = plt.figure()
for kk, k in enumerate(res["k"]):
    plt.plot(
        res["m"],
        res["energy"][1, :, kk] - res["energy"][0, :, kk],
        "--o",
        label=f"k={k}",
    )
plt.xscale("log")
plt.yscale("log")
plt.xlabel("m")
plt.legend()
plt.ylabel("DEltaE")

fig = plt.figure()
for kk, k in enumerate(res["k"]):
    plt.plot(
        vals["m"],
        res["energy"][1, :, kk] - res["energy"][0, :, kk] - 0.5 * res["m"],
        "-^",
        label=f"k={k} RES",
    )
plt.xscale("log")
plt.yscale("log")
plt.xlabel("g2")
plt.legend()
plt.ylabel("DEltaE_res")
save_dictionary(res, "SU2_energy_gap_new.pkl")
