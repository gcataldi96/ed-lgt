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
    QMB_state,
    get_lattice_link_site_pairs,
    lattice_base_configs,
)
from ed_lgt.symmetries import (
    get_symmetry_sector_generators,
    link_abelian_sector,
    global_abelian_sector,
    momentum_basis_k0,
)
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
        # Initialize the Hamiltonian
        self.H = QMB_hamiltonian(0, self.lvals)

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
        # DIAGONALIZE THE HAMILTONIAN
        self.H.diagonalize(self.n_eigs, self.loc_dims)
        self.res["energy"] = self.H.Nenergies

    def momentum_basis_projection(self, logical_unit_size):
        # Project the Hamiltonian onto the momentum sector with k=0
        self.B = momentum_basis_k0(self.sector_configs, logical_unit_size)
        self.H.Ham = csr_matrix(self.B).transpose() * self.H.Ham * csr_matrix(self.B)
        logger.info(f"Momentum basis shape {self.B.shape}")

    def time_evolution_Hamiltonian(self, initial_state, start, stop, n_steps):
        self.H.time_evolution(initial_state, start, stop, n_steps, self.loc_dims)

    def get_qmb_state_from_config(self, config):
        # Get the corresponding QMB index
        index = np.where((self.sector_configs == config).all(axis=1))[0]
        # INITIALIZE the STATE
        state = np.zeros(len(self.sector_configs), dtype=float)
        state[index] = 1
        if self.momentum_basis:
            # Project the state in the momentum sector
            state = self.B.transpose().dot(state)
        return state

    def measure_overlap(self, state, index, dynamics=False):
        if dynamics:
            return np.abs(state.conj().dot(self.H.psi_time[index].psi)) ** 2
        else:
            return np.abs(state.conj().dot(self.H.Npsi[index].psi)) ** 2

    def get_thermal_beta(self, state, threshold):
        return self.H.get_beta(state, threshold)

    def canonical_average(self, local_obs, beta):
        op_matrix = LocalTerm(
            operator=self.ops[local_obs], op_name=local_obs, **self.def_params
        ).get_Hamiltonian(1)
        # Compute the partition function
        Z = self.H.partition_function(beta)
        # Compute the canonical average
        canonical_average = np.real(
            csc_matrix(op_matrix).dot(expm(-beta * csc_matrix(self.H.Ham))).trace()
        ) / (Z * self.n_sites)
        logger.info(f"Canonical avg: {canonical_average}")
        return canonical_average

    def microcanonical_average(self, local_obs, state):
        op_matrix = LocalTerm(
            operator=self.ops[local_obs], op_name=local_obs, **self.def_params
        ).get_Hamiltonian(1)
        # Get the expectation value of the energy of the reference state
        Eq = QMB_state(state).expectation_value(self.H.Ham)
        logger.info(f"E ref {Eq}")
        E2q = QMB_state(state).expectation_value(self.H.Ham**2)
        # Get the corresponding variance
        DeltaE = np.sqrt(E2q - (Eq**2))
        logger.info(f"delta E {DeltaE}")
        # Initialize a state as the superposition of all the eigenstates within an energy shell
        # of amplitude Delta E around Eq
        psi_thermal = np.zeros(len(self.sector_configs), dtype=np.complex128)
        list_states = []
        for ii in range(self.n_eigs):
            if np.abs(self.H.Nenergies[ii] - Eq) < DeltaE:
                logger.info(f"{ii} {self.H.Nenergies[ii]}")
                list_states.append(ii)
                psi_thermal += self.H.Npsi[ii].psi
        norm = len(list_states)
        psi_thermal /= np.sqrt(norm)
        # Compute the microcanonical average of the local observable
        microcanonical_average = 0
        for ii, state_indx in enumerate(list_states):
            microcanonical_average += self.H.Npsi[state_indx].expectation_value(
                op_matrix
            ) / (norm * self.n_sites)
        logger.info(f"Microcanonical avg: {microcanonical_average}")
        return psi_thermal, microcanonical_average

    def diagonal_average(self, local_obs, state):
        op_matrix = LocalTerm(
            operator=self.ops[local_obs], op_name=local_obs, **self.def_params
        ).get_Hamiltonian(1)
        # Step 1: Project initial state onto each eigenstate to find coefficients
        diagonal_average = 0
        for ii in range(self.n_eigs):
            prob = self.measure_overlap(state, ii, False)
            exp_val = self.H.Npsi[ii].expectation_value(op_matrix) / self.n_sites
            diagonal_average += prob * exp_val
        logger.info(f"Diagonal avg: {diagonal_average}")
        return diagonal_average

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
