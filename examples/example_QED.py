# %%
import numpy as np
from ed_lgt.operators import QED_Hamiltonian_couplings
from ed_lgt.models import QED_Model
import logging

logger = logging.getLogger(__name__)

par = {
    "model": {
        # LATTICE DIMENSIONS
        "lvals": [10],
        # BOUNDARY CONDITIONS
        "has_obc": [False],
        # GAUGE TRUNCATION
        "spin": 1,
        # PURE or FULL THEORY
        "pure_theory": False,
    },
    "hamiltonian": {
        "diagonalize": True,
        # N EIGENVALUES
        "n_eigs": 1,
        # FORMAT (DENSE, SPARSE)
        "format": "sparse",
    },
    # g COUPLING
    "g": 1,
    # m COUPLING
    "m": 7,
}
res = {}

model = QED_Model(**par["model"])
# -------------------------------------------------------------------------------
# BUILD AND DIAGONALIZE HAMILTONIAN
coeffs = QED_Hamiltonian_couplings(model.dim, model.pure_theory, par["g"], par["m"])
model.build_Hamiltonian(coeffs)
# -------------------------------------------------------------------------------
# DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
if par["hamiltonian"]["diagonalize"]:
    model.diagonalize_Hamiltonian(
        n_eigs=par["hamiltonian"]["n_eigs"],
        format=par["hamiltonian"]["format"],
    )
    res["energy"] = model.H.Nenergies
# ---------------------------------------------------------------------
# LIST OF LOCAL OBSERVABLES
local_obs = [f"E_{s}{d}" for d in model.directions for s in "mp"] + ["E_square"]
if not model.pure_theory:
    local_obs += ["N"]
# LIST OF TWOBODY CORRELATORS
twobody_obs = [[f"P_p{d}", f"P_m{d}"] for d in model.directions]
twobody_axes = [d for d in model.directions]
# LIST OF PLAQUETTE OPERATORS
if model.dim == 2:
    plaquette_obs = [["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]]
else:
    plaquette_obs = []
# DEFINE OBSERVABLES
model.get_observables(local_obs, twobody_obs, plaquette_obs, twobody_axes=twobody_axes)
res["entropy"] = np.zeros(par["hamiltonian"]["n_eigs"], dtype=float)
# -------------------------------------------------------------------------------
for ii in range(model.n_eigs):
    logger.info(f"================== {ii} ===================")
    # -----------------------------------------------------------------------
    # PRINT ENERGY
    model.H.print_energy(ii)
    # -----------------------------------------------------------------------
    # ENTROPY
    res["entropy"][ii] = model.H.Npsi[ii].entanglement_entropy(
        list(np.arange(0, int(model.n_sites / 2), 1)),
        sector_configs=model.sector_configs,
    )
    # -----------------------------------------------------------------------
    # STATE CONFIGURATIONS
    model.H.Npsi[ii].get_state_configurations(
        threshold=1e-1, sector_configs=model.sector_configs
    )
    # MEASURE OBSERVABLES
    model.measure_observables(ii)
    # CHECK LINK SYMMETRIES
    # model.check_symmetries()

# %%
