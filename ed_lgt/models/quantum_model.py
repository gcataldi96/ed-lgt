import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import expm
from math import prod
from ed_lgt.modeling import (
    LocalTerm,
    TwoBodyTerm,
    PlaquetteTerm,
    NBodyTerm,
    QMB_hamiltonian,
)
from ed_lgt.symmetries import (
    get_symmetry_sector_generators,
    link_abelian_sector,
    global_abelian_sector,
    momentum_basis_k0,
)
from ed_lgt.modeling import get_lattice_link_site_pairs, lattice_base_configs
import logging

logger = logging.getLogger(__name__)


__all__ = ["QuantumModel"]


class QuantumModel:
    def __init__(self, lvals, has_obc, momentum_basis=False):
        # Lattice parameters
        self.lvals = lvals
        self.dim = len(self.lvals)
        self.directions = "xyz"[: self.dim]
        self.n_sites = prod(self.lvals)
        self.has_obc = has_obc
        # Symmetry sector indices
        self.sector_configs = None
        self.sector_indices = None
        # Gauge Basis
        self.gauge_basis = None
        # Staggered Basis
        self.staggered_basis = False
        # Momentum Basis
        self.momentum_basis = momentum_basis
        # Dictionary for results
        self.res = {}

    def default_params(self):
        self.def_params = {
            "lvals": self.lvals,
            "has_obc": self.has_obc,
            "gauge_basis": self.gauge_basis,
            "sector_configs": self.sector_configs,
            "staggered_basis": self.staggered_basis,
        }

    def get_local_site_dimensions(self):
        if self.gauge_basis is not None:
            # Acquire local dimension and lattice label
            lattice_labels, loc_dims = lattice_base_configs(
                self.gauge_basis, self.lvals, self.has_obc, self.staggered_basis
            )
            self.loc_dims = loc_dims.transpose().reshape(self.n_sites)
            self.lattice_labels = lattice_labels.transpose().reshape(self.n_sites)
        else:
            self.lattice_labels = None
        logger.info(f"local dimensions: {self.loc_dims}")

    def get_abelian_symmetry_sector(
        self,
        global_ops=None,
        global_sectors=None,
        global_sym_type="U",
        link_ops=None,
        link_sectors=None,
    ):
        # ================================================================================
        # GLOBAL ABELIAN SYMMETRIES
        if global_ops is not None:
            global_ops = get_symmetry_sector_generators(
                global_ops,
                loc_dims=self.loc_dims,
                action="global",
                gauge_basis=self.gauge_basis,
                lattice_labels=self.lattice_labels,
            )
            self.sector_indices, self.sector_configs = global_abelian_sector(
                loc_dims=self.loc_dims,
                sym_op_diags=global_ops,
                sym_sectors=np.array(global_sectors, dtype=float),
                sym_type=global_sym_type,
                configs=self.sector_configs,
            )
        # ================================================================================
        # ABELIAN Z2 SYMMETRIES
        if link_ops is not None:
            link_ops = get_symmetry_sector_generators(
                link_ops,
                loc_dims=self.loc_dims,
                action="link",
                gauge_basis=self.gauge_basis,
                lattice_labels=self.lattice_labels,
            )
            site_pairs = get_lattice_link_site_pairs(self.lvals, self.has_obc)
            self.sector_indices, self.sector_configs = link_abelian_sector(
                loc_dims=self.loc_dims,
                sym_op_diags=link_ops,
                sym_sectors=link_sectors,
                pair_list=site_pairs,
                configs=self.sector_configs,
            )

    def diagonalize_Hamiltonian(self, n_eigs):
        self.n_eigs = n_eigs
        if isinstance(self.H, QMB_hamiltonian):
            # DIAGONALIZE THE HAMILTONIAN
            self.H.diagonalize(self.n_eigs)
            self.res["energy"] = self.H.Nenergies
            if self.n_eigs > 1:
                self.res["true_gap"] = self.H.Nenergies[1] - self.H.Nenergies[0]

    def momentum_basis_projection(self, logical_unit_size):
        # Project the Hamiltonian onto the momentum sector with k=0
        B = momentum_basis_k0(self.sector_configs, logical_unit_size)
        self.H.Ham = csr_matrix(B).transpose() * self.H.Ham * csr_matrix(B)
        logger.info(f"Momentum basis shape {B.shape}")

    def time_evolution_Hamiltonian(self, initial_state, start, stop, n_steps):
        if isinstance(self.H, QMB_hamiltonian):
            # DIAGONALIZE THE HAMILTONIAN
            self.H.time_evolution(initial_state, start, stop, n_steps)

    def get_thermal_beta(self):
        if isinstance(self.H, QMB_hamiltonian):
            # DIAGONALIZE THE HAMILTONIAN
            return self.H.get_beta()

    def thermal_average(self, local_obs, beta):
        Op = LocalTerm(
            operator=self.ops[local_obs], op_name=local_obs, **self.def_params
        )
        Op_matrix = Op.get_Hamiltonian(1)
        if isinstance(self.H, QMB_hamiltonian):
            Z = self.H.partition_function(beta)
        return np.real(
            csc_matrix(Op_matrix).dot(expm(-beta * csc_matrix(self.H.Ham))).trace()
        ) / (Z * self.n_sites)

    def get_observables(
        self,
        local_obs=[],
        twobody_obs=[],
        plaquette_obs=[],
        nbody_obs=[],
        twobody_axes=None,
    ):
        self.local_obs = local_obs
        self.twobody_obs = twobody_obs
        self.twobody_axes = twobody_axes
        self.plaquette_obs = plaquette_obs
        self.nbody_obs = nbody_obs
        self.obs_list = {}
        # ---------------------------------------------------------------------------
        # LIST OF LOCAL OBSERVABLES
        for obs in local_obs:
            self.obs_list[obs] = LocalTerm(
                operator=self.ops[obs], op_name=obs, **self.def_params
            )
        # ---------------------------------------------------------------------------
        # LIST OF TWOBODY CORRELATORS
        for ii, op_names_list in enumerate(twobody_obs):
            obs = "_".join(op_names_list)
            op_list = [self.ops[op] for op in op_names_list]
            self.obs_list[obs] = TwoBodyTerm(
                axis=twobody_axes[ii] if twobody_axes is not None else "x",
                op_list=op_list,
                op_names_list=op_names_list,
                **self.def_params,
            )
        # ---------------------------------------------------------------------------
        # LIST OF PLAQUETTE CORRELATORS
        for op_names_list in plaquette_obs:
            obs = "_".join(op_names_list)
            op_list = [self.ops[op] for op in op_names_list]
            self.obs_list[obs] = PlaquetteTerm(
                axes=["x", "y"],
                op_list=op_list,
                op_names_list=op_names_list,
                **self.def_params,
            )
        # ---------------------------------------------------------------------------
        # LIST OF NBODY CORRELATORS
        for op_names_list in nbody_obs:
            obs = "_".join(op_names_list)
            op_list = [self.ops[op] for op in op_names_list]
            self.obs_list[obs] = NBodyTerm(
                op_list=op_list, op_names_list=op_names_list, **self.def_params
            )

    def measure_observables(self, state_number, dynamics=False):
        if not dynamics:
            for obs in self.local_obs:
                self.obs_list[obs].get_expval(self.H.Npsi[state_number])
                self.res[obs] = self.obs_list[obs].obs
            for op_names_list in self.twobody_obs:
                obs = "_".join(op_names_list)
                self.obs_list[obs].get_expval(self.H.Npsi[state_number])
                self.res[obs] = self.obs_list[obs].corr
                if self.twobody_axes is not None:
                    self.obs_list[obs].print_nearest_neighbors()
            for op_names_list in self.plaquette_obs:
                obs = "_".join(op_names_list)
                self.obs_list[obs].get_expval(self.H.Npsi[state_number])
                self.res[obs] = self.obs_list[obs].avg
            for op_names_list in self.nbody_obs:
                obs = "_".join(op_names_list)
                self.obs_list[obs].get_expval(self.H.Npsi[state_number])
                self.res[obs] = self.obs_list[obs].corr
        else:
            for obs in self.local_obs:
                self.obs_list[obs].get_expval(self.H.psi_time[state_number])
                self.res[obs] = self.obs_list[obs].obs
            for op_names_list in self.twobody_obs:
                obs = "_".join(op_names_list)
                self.obs_list[obs].get_expval(self.H.psi_time[state_number])
                self.res[obs] = self.obs_list[obs].corr
                if self.twobody_axes is not None:
                    self.obs_list[obs].print_nearest_neighbors()
            for op_names_list in self.plaquette_obs:
                obs = "_".join(op_names_list)
                self.obs_list[obs].get_expval(self.H.psi_time[state_number])
                self.res[obs] = self.obs_list[obs].avg
            for op_names_list in self.nbody_obs:
                obs = "_".join(op_names_list)
                self.obs_list[obs].get_expval(self.H.psi_time[state_number])
                self.res[obs] = self.obs_list[obs].corr
