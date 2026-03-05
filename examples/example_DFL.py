# %%
import logging
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from edlgt.tools import moving_time_integral_centered
from edlgt.workflows import run_DFL_dynamics, run_DFL_dynamics_sector_by_sector

logger = logging.getLogger(__name__)

params = {
    "model": {
        "lvals": [8],
        "has_obc": [False],
        "spin": 0.5,
        "pure_theory": False,
        "ham_format": "sparse",
    },
    "dynamics": {
        "time_evolution": True,
        "start": 0,
        "stop": 10,
        "delta_n": 0.1,
        "logical_stag_basis": 2,
    },
    "momentum": {
        "get_momentum_basis": False,
        "unit_cell_size": [2],
        "TC_symmetry": False,
    },
    "observables": {
        "measure_obs": False,
        "get_entropy": False,
        "get_PE": True,
        "get_SRE": False,
        "entropy_partition": [0, 1, 2, 3],
        "get_state_configs": False,
        "get_fidelity": False,
    },
    "ensemble": {
        "microcanonical": {"average": False},
        "diagonal": {"average": False},
        "canonical": {"average": False},
    },
    "g_values": np.linspace(0, 10, 11),
    "m": 1,
}
res = run_DFL_dynamics_sector_by_sector(params)
# Example alternate call:
# res = run_DFL_dynamics(params)
# %%
gvals = res["g_values"]
sm = cm.ScalarMappable(cmap="magma")
palette = sm.to_rgba(gvals)
fig, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
for gidx, _ in enumerate(gvals):
    axs.plot(
        res["time"][1:],
        moving_time_integral_centered(res["time"][1:], res["PE"][gidx, 1:], 50),
        "-",
        color=palette[gidx],
        markersize=2,
        markeredgecolor=palette[gidx],
        markerfacecolor="white",
        markeredgewidth=1,
    )
axs.set(
    ylabel=r"Participation entropy $PE_{2}(t)$",
    xlabel=r"time $t$",
    xscale="log",
)
axs.grid(True, which="both", linestyle="-", linewidth=0.4)
cb = fig.colorbar(
    sm, ax=axs, aspect=30, location="right", orientation="vertical", pad=0.02
)
cb.set_label(label=r"$g^{2}$", rotation=0, labelpad=-17, x=-0.2, y=-0.03)
plt.savefig("DFL_PE_N8.pdf")
