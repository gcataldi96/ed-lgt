# %%
import numpy as np
from ed_lgt.models import IsingModel
import logging

logger = logging.getLogger(__name__)
res = {}
par = {
    "model": {"lvals": [6], "has_obc": [False], "momentum_basis": False},
    "symmetries": {"sym_ops": ["Sz"], "sym_sectors": [1], "sym_type": "P"},
    "hamiltonian": {"diagonalize": True, "n_eigs": 2, "format": "sparse"},
    "h": 10,
}

model = IsingModel(**par["model"])
# SYMMETRIES
global_ops = [model.ops[op] for op in par["symmetries"]["sym_ops"]]
global_sectors = par["symmetries"]["sym_sectors"]
global_sym_type = par["symmetries"]["sym_type"]
model.get_abelian_symmetry_sector(
    global_ops=global_ops,
    global_sectors=global_sectors,
    global_sym_type=global_sym_type,
)
# DEFAUL PARAMS
model.default_params()
# -------------------------------------------------------------------------------
# BUILD THE HAMILTONIAN
coeffs = {"J": 1, "h": par["h"]}
model.build_Hamiltonian(coeffs)
# -------------------------------------------------------------------------------
# DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
if par["hamiltonian"]["diagonalize"]:
    model.diagonalize_Hamiltonian(
        n_eigs=par["hamiltonian"]["n_eigs"],
        format=par["hamiltonian"]["format"],
    )
    res["energy"] = model.H.Nenergies
# -------------------------------------------------------------------------------
# LIST OF LOCAL OBSERVABLES
local_obs = ["Sx", "Sz"]
# LIST OF TWOBODY CORRELATORS
twobody_obs = []
# DEFINE OBSERVABLES
model.get_observables(local_obs, twobody_obs)
# -------------------------------------------------------------------------------
res["entropy"] = np.zeros(model.n_eigs, dtype=float)
for obs in local_obs:
    res[obs] = np.zeros((model.n_eigs, model.n_sites), dtype=float)
# -------------------------------------------------------------------------------
for ii in range(model.n_eigs):
    model.H.print_energy(ii)
    if not model.momentum_basis:
        # -----------------------------------------------------------------------
        # ENTROPY
        res["entropy"][ii] = model.H.Npsi[ii].entanglement_entropy(
            list(np.arange(0, int(model.lvals[0] / 2), 1)),
            sector_configs=model.sector_configs,
        )
        # -----------------------------------------------------------------------
        # STATE CONFIGURATIONS
        model.H.Npsi[ii].get_state_configurations(
            threshold=1e-1, sector_configs=model.sector_configs
        )
    # ---------------------------------------------------------------------------
    # MEASURE OBSERVABLES
    model.measure_observables(ii, dynamics=False)
    for obs in local_obs:
        res[obs][ii, :] = model.res[obs]
# ---------------------------------------------------------------------------
# %%
# COMPUTING THE MASS GAP
# Hamiltonian informations
H_info = {
    "ops": [[["Sz"]], [["Sx", "Sx"]]],
    "coeffs": [[coeffs["h"]], [coeffs["J"]]],
}
# Excitation operators
local_ops = []  # [["Sp"]]
twobody_ops = [["Sm", "Sm"]]  # , ["Sp", "Sm"]]  # , ["Sm", "Sm"]]
threebody_ops = []  # [["Sp", "Sz", "Sp"]]
# List with the names of the excitation operators
ops_names = [local_ops, twobody_ops, threebody_ops]
# Build an array with the number of excitations per type of excitations
ex_counts = np.array([len(ex_type) for ex_type in ops_names])
# %%
# Measure the mass gap
model.get_energy_gap(ex_counts, ops_names, H_info)
# %%
