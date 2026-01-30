# %%
import numpy as np
from simsio import *
import matplotlib.pyplot as plt
from ed_lgt.tools import localize_Wannier
import logging

logger = logging.getLogger(__name__)


# function to acquire the convolution matrix and k indices from the simulation
def get_simulation_data1(sim_name):
    # ================================================================
    # We load the convolution results
    config_filename = f"scattering/{sim_name}"
    match = SimsQuery(group_glob=config_filename)
    ugrid, _ = uids_grid(match.uids, ["g", "m"])
    # (Nk, Nk) complex128 M_{k1,k2} = <k1|H_0|k2>
    Mk1k2_matrix = get_sim(ugrid[0][0]).res["k1k2matrix"]
    # (Nk,) integer indices of k T^2 vs TC
    k_indices = get_sim(ugrid[0][0]).res["k_indices"]
    gs_energy = -4.580269235030599 - 1.251803175199139e-18j
    return Mk1k2_matrix, k_indices, gs_energy


def get_simulation_data2(sim_name):
    with np.load(sim_name, allow_pickle=False) as z:
        Mk1k2_matrix = z["matrix"]
        k_indices = z["kvals"]
        gs_energy = z["gs"]
    return Mk1k2_matrix, k_indices, gs_energy


# %%
Mk1k2_matrix, k_indices, gs_energy = get_simulation_data2("wannier_TC_MM.npz")
E_best, best_sigma, best_theta = localize_Wannier(
    Mk1k2_matrix, k_indices, gs_energy, center_mode=1
)
Nk = E_best.shape[0]
# %%
Mk1k2_matrix, k_indices, gs_energy = get_simulation_data1("convolution1_N0")
E_best, best_sigma, best_theta = localize_Wannier(
    Mk1k2_matrix, k_indices, gs_energy, center_mode=1
)
Nk = E_best.shape[0]
# %%
fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.plot(range(Nk), E_best, marker="o")
ax.set(xticks=np.arange(Nk))
ax.grid()
# ax.set(yscale="log")
# ax.set(ylim=[1e-4, max(E_best) + 1])

# %%
