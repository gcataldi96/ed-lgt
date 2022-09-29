# %%
from typing import OrderedDict
import numpy as np
from tn_py_frontend.simulation_setup import ATTNSimulation, TNOperators
from tn_py_frontend.tnobservables import TNObservables, TNObsLocal, TNObsCorr, TNState2File, TNDistance2Pure
from tn_py_frontend.rydberg_model import QuantumModel, LocalTerm, TwoBodyTerm2D
from Contrib.qed_models import PlaquetteTerm, get_SU2_Full_Operators



"""# Hamiltonian Coefficients
params={
    'BD': 10,                # MAXIMAL BOND DIMENSION
    'L' : 4,                 # LATTICE SIZE
    'g_SU2': 5,              # Gauge Coupling
    'E' : E_SU2,             # ELECTRIC FIELD coupling
    'B' : B_SU2,             # MAGNETIC FIELD coupling
    't_h' : -0.5j,           # HORIZONTAL HOPPING
    't_h_DAG' : 0.5j,        # HORIZONTAL HOPPING DAGGER
    't_v_EVEN' : -0.5,       # VERTICAL HOPPING (EVEN SITES)
    't_v_ODD' : 0.5,         # VERTICAL HOPPING (ODD SITES)
    'm_odd' : -mass,         # EFFECTIVE MASS for ODD SITES
    'm_even' : mass,         # EFFECTIVE MASS for EVEN SITES
    'eta' : penalty,         # PENALTY
}"""

params={
    "dim":2,
    "lvals":4,
    "boundary": False,
    "has_obc": False,
    'g_SU2': g_SU2,          # Gauge Coupling
    'E' : E_SU2,             # ELECTRIC FIELD coupling
    'B' : B_SU2,             # MAGNETIC FIELD coupling
    't_h' : -0.5j,           # HORIZONTAL HOPPING
    't_h_DAG' : 0.5j,        # HORIZONTAL HOPPING DAGGER
    't_v_EVEN' : -0.5,       # VERTICAL HOPPING (EVEN SITES)
    't_v_ODD' : 0.5,         # VERTICAL HOPPING (ODD SITES)
    'm_odd' : -mass,         # EFFECTIVE MASS for ODD SITES
    'm_even' : mass,         # EFFECTIVE MASS for EVEN SITES
    'eta' : penalty          # PENALTY
}
params['SymmetrySectors'] = [params["L"]**2]
params['SymmetryGenerators'] = ['n_TOTAL']
params['SymmetryTypes'] = ['U']

class SU2_LGT_Model(QuantumModel):
    def __init__(self,params):
        if not isinstance(params,dict):
            raise TypeError(f"params should be a dict, not a {type(params)}")
        
        super().__init__(params["dim"],params["lvals"],params["boundary"])

        if self.dim != 2:
            raise NotImplementedError("Only 2D lattices are currently supported.")
        
        self.directions = params.setdefault("directions", "xyz"[: self.dim])
        self.has_obc=params["has_obc"]

        # Border Penalties
        if not self.has_obc:
            for i, d in enumerate(self.directions):
                for j, s in enumerate("mp"):
                    self += LocalTerm(f"P_{d}{s}")
        

        # Link Symmetries
        for i, d in enumerate(self.directions):
            self += TwoBodyTerm2D([f"W_{s}{d}" for s in "pm"],
                    d_inds == i,
                    prefactor=-1,
                    isotropy_xyz=False
                )
        # Hopping

    
    def get_operators(self):
        tn_ops=TNOperators('operators')
        # TODO: to complete
        return tn_ops

    def get_observables(self):
        # TODO: Computing Plaquettes
        tn_obs=TNObservables()
        # Border Penalties
        if not self.has_obc:
            for d in self.directions:
                for s in "mp":
                    tn_obs+=TNObsLocal(f"P_{s}{d}",f"P_{s}{d}")
        # Link Symmetry Correlators
        for d in self.directions:
            tn_obs+=TNObsCorr(f"W_{d}_link",[f"W_{s}{d}" for s in "pm"])
        # Gamma Electric Energy
        tn_obs+=TNObsLocal("gamma","gamma")
        # Number Operators
        tn_obs+=TNObsLocal("n_single","n_single")
        tn_obs+=TNObsLocal("n_pair","n_pair")
        tn_obs+=TNObsLocal("n_tot","n_tot")
        # Density Correlator
        tn_obs+=TNObsCorr('nn_density',['n_tot','n_tot'])
        # S-wave SCOP Correlator
        tn_obs+=TNObsCorr('S_Wave',['Delta_Dagger','Delta'])
        return tn_obs
 



# %%
