# %%
from simsio import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from ed_lgt.tools import save_dictionary, load_dictionary, fake_log, get_tline, set_size

textwidth_pt = 510.0
textwidth_in = textwidth_pt / 72.27
columnwidth_pt = 246.0
columnwidth_in = columnwidth_pt / 72.27


# %%
def get_MPO(sim_name):
    config_filename = f"scattering/{sim_name}"
    match = SimsQuery(group_glob=config_filename)
    ugrid, vals = uids_grid(match.uids, ["g", "m"])
    MPO_list = []
    for ii in range(4):
        MPO_list.append(get_sim(ugrid[0, 0]).res[f"MPO[{ii}]"])
    return MPO_list


MPO = get_MPO("get_MPO")
np.savez_compressed("MPO.npz", *MPO)


def get_MPO():
    data = np.load("MPO.npz")
    return [data[f"arr_{i}"] for i in range(len(data.files))]


mpo = get_MPO()
# %%
# ===================================================================
# SCATTERING MATRIX
# ===================================================================
res = {}
config_filename = f"scattering/convolution_N0old"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m"])
np.savez_compressed(
    "wannier_TC_MM.npz",
    matrix=get_sim(ugrid[0][0]).res["k1k2matrix"],
    kvals=get_sim(ugrid[0][0]).res["kvals"],
    gs=get_sim(ugrid[0][0]).res["gs_energy"],
    g=vals["g"][0],
    m=vals["m"][0],
    sector=14,
    TC_symmetry=True,
)

# %%
# ===================================================================
# SCATTERING MATRIX
# ===================================================================
res = {}
config_filename = f"scattering/test_BB"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["momentum_k_vals"])
np.savez_compressed(
    "wannier_TC_14BB.npz",
    psi=get_sim(ugrid[0]).res["psi0"],
    energy=get_sim(ugrid[0]).res["enery"],
    g=get_sim(ugrid[0]).par["g"],
    m=get_sim(ugrid[0]).par["m"],
    sector=14,
    TC_symmetry=True,
)

# %%
# ===================================================================
# BAND STRUCTURE
# ===================================================================
params = {
    "g": [0.1, 1, 3, 5, 10],
    "m": [0.1, 1, 3, 5, 10],
    "sector": [6, 8, 10, 12, 14],
    "momentum_k": [0, 1, 2, 3, 4, 5],
}
res = {}
config_filename = f"scattering/bands"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["g", "m", "sector", "momentum_k"])

res = {
    "energy": np.zeros(
        (
            len(vals["g"]),
            len(vals["m"]),
            len(vals["sector"]),
            len(vals["momentum_k"]),
            10,
        )
    ),
    "E2": np.zeros(
        (
            len(vals["g"]),
            len(vals["m"]),
            len(vals["sector"]),
            len(vals["momentum_k"]),
            10,
        )
    ),
    "N_single": np.zeros(
        (
            len(vals["g"]),
            len(vals["m"]),
            len(vals["sector"]),
            len(vals["momentum_k"]),
            10,
        )
    ),
    "N_pair": np.zeros(
        (
            len(vals["g"]),
            len(vals["m"]),
            len(vals["sector"]),
            len(vals["momentum_k"]),
            10,
        )
    ),
}

for ii, g in enumerate(vals["g"]):
    for jj, m in enumerate(vals["m"]):
        for kk, sector in enumerate(vals["sector"]):
            for ll, k in enumerate(vals["momentum_k"]):
                sim_res = get_sim(ugrid[ii][jj][kk][ll]).res
                res["energy"][ii, jj, kk, ll] = sim_res["energy"]
                res["E2"][ii, jj, kk, ll] = sim_res["E2"]
                res["N_single"][ii, jj, kk, ll] = sim_res["N_single"]
                res["N_pair"][ii, jj, kk, ll] = sim_res["N_pair"]

np.savez_compressed(
    "bands.npz",
    energy=res["energy"],
    E2=res["E2"],
    N_single=res["N_single"],
    N_pair=res["N_pair"],
    g=np.asarray(vals["g"]),
    m=np.asarray(vals["m"]),
    sector=np.asarray(vals["sector"]),
    momentum_k=np.asarray(vals["momentum_k"]),
)
# %%
params = {
    "sector": [16, 18, 20],
    "momentum_k_vals": [0, 1, 2, 3, 4, 5, 6, 7],
}
res = {}
config_filename = f"scattering/casestudy"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["sector", "momentum_k_vals"])

res = {
    "energy": np.zeros((len(vals["sector"]), len(vals["momentum_k_vals"]), 10)),
    "E2": np.zeros(
        (
            len(vals["sector"]),
            len(vals["momentum_k_vals"]),
            10,
        )
    ),
    "N_single": np.zeros(
        (
            len(vals["sector"]),
            len(vals["momentum_k_vals"]),
            10,
        )
    ),
    "N_pair": np.zeros(
        (
            len(vals["sector"]),
            len(vals["momentum_k_vals"]),
            10,
        )
    ),
}

for kk, sector in enumerate(vals["sector"]):
    for ll, k in enumerate(vals["momentum_k_vals"]):
        sim_res = get_sim(ugrid[kk][ll]).res
        res["energy"][kk, ll] = sim_res["energy"]
        res["E2"][kk, ll] = sim_res["E2"]
        res["N_single"][kk, ll] = sim_res["N_single"]
        res["N_pair"][kk, ll] = sim_res["N_pair"]

np.savez_compressed(
    "case.npz",
    energy=res["energy"],
    E2=res["E2"],
    N_single=res["N_single"],
    N_pair=res["N_pair"],
    sector=np.asarray(vals["sector"]),
    momentum_k_vals=np.asarray(vals["momentum_k_vals"]),
)
# %%
res = {}
config_filename = f"scattering/test_TC"
match = SimsQuery(group_glob=config_filename)
ugrid, vals = uids_grid(match.uids, ["momentum_k_vals"])

res = {
    "energy": np.zeros((len(vals["momentum_k_vals"]), 10)),
    "E2": np.zeros((len(vals["momentum_k_vals"]), 10)),
    "N_single": np.zeros((len(vals["momentum_k_vals"]), 10)),
    "N_pair": np.zeros((len(vals["momentum_k_vals"]), 10)),
}

for ll, k in enumerate(vals["momentum_k_vals"]):
    sim_res = get_sim(ugrid[ll]).res
    res["energy"][ll] = sim_res["energy"]
    res["E2"][ll] = sim_res["E2"]
    res["N_single"][ll] = sim_res["N_single"]
    res["N_pair"][ll] = sim_res["N_pair"]
np.savez_compressed(
    "TC.npz",
    energy=res["energy"],
    E2=res["E2"],
    N_single=res["N_single"],
    N_pair=res["N_pair"],
    momentum_k_vals=np.asarray(vals["momentum_k_vals"]),
)
# %%
fig, ax = plt.subplots(
    5,
    5,
    figsize=set_size(2 * textwidth_pt, subplots=(5, 5), height_factor=2),
    constrained_layout=True,
    sharex=True,
)
ax[0, 0].set(xticks=np.arange(0, 6, 1))
for ii in range(5):
    ax[ii, 0].set(ylabel=r"energy E")
    ax[-1, ii].set(xlabel=r"momentum $k$")


for ii, g in enumerate(params["g"]):
    for jj, m in enumerate(params["m"]):
        ax[ii, jj].annotate(
            r"$g\!=\!" + f"{g}, m\!=\!{m}$",
            xy=(0.935, 0.9),
            xycoords="axes fraction",
            fontsize=10,
            horizontalalignment="right",
            verticalalignment="bottom",
            bbox=dict(facecolor="white", edgecolor="black"),
        )
        for kk, sector in enumerate(params["sector"]):
            for ss, eig in enumerate(range(10)):
                ax[ii, jj].plot(
                    np.arange(0, 6, 1),
                    res["energy"][ii, jj, kk, :, ss],
                    "o-",
                    markersize=4,
                    markeredgecolor="darkblue",
                    markerfacecolor="white",
                    markeredgewidth=0.5,
                )

# %%
