import numpy as np
from scipy.linalg import eig
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm
from math import prod
from ed_lgt.tools import stag_avg
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
    get_operators_nbody_term as nbops,
    link_abelian_sector,
    global_abelian_sector,
    momentum_basis_k0_par,
    momentum_basis_k0,
    symmetry_sector_configs,
)
import logging

logger = logging.getLogger(__name__)


__all__ = ["QuantumModel"]


class QuantumModel:
    def __init__(self, lvals, has_obc, momentum_basis=False, logical_unit_size=1):
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
        self.logical_unit_size = int(logical_unit_size)
        # Dictionary for results
        self.res = {}

    def default_params(self):
        if self.momentum_basis:
            if self.has_obc[0]:
                raise ValueError(f"Momentum is not conserved in OBC")
            logger.info(f"Momentum basis")
            self.B = momentum_basis_k0_par(self.sector_configs, self.logical_unit_size)
            logger.info(f"Momentum basis shape {self.B.shape}")
            hamiltonian_size = self.B.shape[1]
        elif self.sector_configs is not None:
            self.B = None
            hamiltonian_size = self.sector_configs.shape[0]
        else:
            hamiltonian_size = np.prod(self.loc_dims)
            self.B = None
        # Define the default parameters as a dictionary
        self.def_params = {
            "lvals": self.lvals,
            "has_obc": self.has_obc,
            "gauge_basis": self.gauge_basis,
            "sector_configs": self.sector_configs,
            "staggered_basis": self.staggered_basis,
            "momentum_basis": self.B,
            "momentum_k": 0,
        }
        # Initialize the Hamiltonian
        self.H = QMB_hamiltonian(self.lvals, size=hamiltonian_size)

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
            pair_list = get_lattice_link_site_pairs(self.lvals, self.has_obc)
        # ================================================================================
        if global_ops is not None and link_ops is not None:
            self.sector_indices, self.sector_configs = symmetry_sector_configs(
                loc_dims=self.loc_dims,
                glob_op_diags=global_ops,
                glob_sectors=np.array(global_sectors, dtype=float),
                sym_type_flag=global_sym_type,
                link_op_diags=link_ops,
                link_sectors=link_sectors,
                pair_list=pair_list,
            )
        elif global_ops is not None:
            self.sector_indices, self.sector_configs = global_abelian_sector(
                loc_dims=self.loc_dims,
                sym_op_diags=global_ops,
                sym_sectors=np.array(global_sectors, dtype=float),
                sym_type=global_sym_type,
            )
        elif link_ops is not None:
            self.sector_indices, self.sector_configs = link_abelian_sector(
                loc_dims=self.loc_dims,
                sym_op_diags=link_ops,
                sym_sectors=link_sectors,
                pair_list=pair_list,
            )

    def diagonalize_Hamiltonian(self, n_eigs, format):
        # DIAGONALIZE THE HAMILTONIAN
        self.H.diagonalize(n_eigs, format, self.loc_dims)
        self.n_eigs = self.H.n_eigs
        self.res["energy"] = self.H.Nenergies

    def time_evolution_Hamiltonian(self, initial_state, time_line):
        self.H.time_evolution(initial_state, time_line, self.loc_dims)

    def momentum_basis_projection(self, operator):
        # Project the Hamiltonian onto the momentum sector with k=0
        if isinstance(operator, str) and operator == "H":
            self.H.Ham = self.B.transpose() * self.H.Ham * self.B
        else:
            return self.B.transpose() * operator * self.B

    def get_qmb_state_from_configs(self, configs):
        # INITIALIZE the STATE
        state = np.zeros(len(self.sector_configs), dtype=float)
        # Get the corresponding QMB index of each config
        for config in configs:
            if not np.any(np.all(self.sector_configs == config, axis=1)):
                logger.info(config)
                raise ValueError(f"config not compatible with the symmetry sector")
            else:
                index = np.where((self.sector_configs == config).all(axis=1))[0]
                state[index] = 1
        # Normalize the state
        state /= np.sqrt(len(configs))
        if self.momentum_basis:
            # Project the state in the momentum sector
            state = self.B.transpose().dot(state)
        return state

    def measure_fidelity(self, state, index, dynamics=False, print_value=False):
        if not isinstance(state, np.ndarray):
            raise TypeError(f"state must be np.array not {type(state)}")
        if len(state) != self.H.Ham.shape[0]:
            raise ValueError(
                f"len(state) must be {self.H.Ham.shape[0]} not {len(state)}"
            )
        # Define the reference state
        ref_psi = self.H.Npsi[index].psi if not dynamics else self.H.psi_time[index].psi
        fidelity = np.abs(state.conj().dot(ref_psi)) ** 2
        if print_value:
            logger.info(f"FIDELITY: {fidelity}")
        return fidelity

    def compute_expH(self, beta):
        return csc_matrix(expm(-beta * self.H.Ham))

    def get_thermal_beta(self, state, threshold):
        return self.H.get_beta(state, threshold)

    def canonical_avg(self, local_obs, beta):
        logger.info("----------------------------------------------------")
        logger.info("CANONICAL ENSEMBLE")
        op_matrix = LocalTerm(
            operator=self.ops[local_obs], op_name=local_obs, **self.def_params
        ).get_Hamiltonian(1)
        # Define the exponential of the Hamiltonian
        expm_matrix = self.compute_expH(beta)
        Z = np.real(expm_matrix.trace())
        canonical_avg = np.real(csc_matrix(op_matrix).dot(expm_matrix).trace()) / (
            Z * self.n_sites
        )
        logger.info(f"Canonical avg: {canonical_avg}")
        logger.info("----------------------------------------------------")
        return canonical_avg

    def microcanonical_avg(self, local_obs, state):
        logger.info("----------------------------------------------------")
        logger.info("MICRO-CANONICAL ENSEMBLE")
        op_matrix = LocalTerm(
            operator=self.ops[local_obs], op_name=local_obs, **self.def_params
        ).get_Hamiltonian(1)
        # Get the expectation value of the energy of the reference state
        Eq = QMB_state(state).expectation_value(self.H.Ham)
        logger.info(f"Energy ref {Eq}")
        E2q = QMB_state(state).expectation_value(self.H.Ham**2)
        # Get the corresponding variance
        DeltaE = np.sqrt(E2q - (Eq**2))
        logger.info(f"delta E {DeltaE}")
        # Check that Eq is contained in the set of eigvals:
        if Eq + DeltaE > max(self.H.Nenergies):
            msg = f"Need more eigvals to cover the Energy shell: {Eq+DeltaE} > {max(self.H.Nenergies)}"
            raise ValueError(msg)
        # Initialize a state as the superposition of all the eigenstates within
        # an energy shell of amplitude Delta E around Eq
        psi_thermal = np.zeros(self.H.Ham.shape[0], dtype=np.complex128)
        list_states = []
        for ii in range(self.n_eigs):
            if np.abs(self.H.Nenergies[ii] - Eq) < DeltaE:
                list_states.append(ii)
                psi_thermal += self.H.Npsi[ii].psi
        norm = len(list_states)
        psi_thermal /= np.sqrt(norm)
        # Compute the microcanonical average of the local observable
        microcanonical_avg = 0
        for ii, state_indx in enumerate(list_states):
            microcanonical_avg += self.H.Npsi[state_indx].expectation_value(op_matrix)
        microcanonical_avg /= norm * self.n_sites
        logger.info(f"Microcanonical avg: {microcanonical_avg}")
        logger.info("----------------------------------------------------")
        return psi_thermal, microcanonical_avg

    def diagonal_avg(self, local_obs, state):
        logger.info("----------------------------------------------------")
        logger.info("DIAGONAL ENSEMBLE")
        # check that the hamiltonian has been already fully diagonalized:
        if self.n_eigs != self.H.Ham.shape[0]:
            msg = f"Need all H eigvals {self.H.Ham.shape[0]}, not only {self.n_eigs}"
            raise ValueError(msg)
        op_matrix = LocalTerm(
            operator=self.ops[local_obs], op_name=local_obs, **self.def_params
        ).get_Hamiltonian(1)
        diagonal_avg = 0
        for ii in range(self.n_eigs):
            prob = self.measure_fidelity(state, ii, False)
            exp_val = self.H.Npsi[ii].expectation_value(op_matrix) / self.n_sites
            diagonal_avg += prob * exp_val
        logger.info(f"Diagonal avg: {diagonal_avg}")
        logger.info("----------------------------------------------------")
        return diagonal_avg

    def get_observables(
        self,
        local_obs=[],
        twobody_obs=[],
        plaquette_obs=[],
        nbody_obs=[],
        nbody_dist=[],
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
        for ii, op_names_list in enumerate(nbody_obs):
            obs = "_".join(op_names_list)
            op_list = [self.ops[op] for op in op_names_list]
            distances = nbody_dist[ii]
            self.obs_list[obs] = NBodyTerm(
                op_list=op_list,
                op_names_list=op_names_list,
                distances=distances,
                **self.def_params,
            )

    def measure_observables(self, index, dynamics=False):
        state = self.H.Npsi[index] if not dynamics else self.H.psi_time[index]
        for obs in self.local_obs:
            self.obs_list[obs].get_expval(state)
            self.res[obs] = self.obs_list[obs].obs
        for op_names_list in self.twobody_obs:
            obs = "_".join(op_names_list)
            self.obs_list[obs].get_expval(state)
            self.res[obs] = self.obs_list[obs].corr
            if self.twobody_axes is not None:
                self.obs_list[obs].print_nearest_neighbors()
        for op_names_list in self.plaquette_obs:
            obs = "_".join(op_names_list)
            self.obs_list[obs].get_expval(state)
            self.res[obs] = self.obs_list[obs].avg
        for op_names_list in self.nbody_obs:
            obs = "_".join(op_names_list)
            self.obs_list[obs].get_expval(state)
            self.res[obs] = self.obs_list[obs].obs

    def microcanonical_avg1(
        self, local_obs: str, state: np.ndarray, special_norm=None, staggered_avg=None
    ):
        logger.info("----------------------------------------------------")
        logger.info("MICRO-CANONICAL ENSEMBLE")
        # Get the expectation value of the energy of the reference state
        Eq = QMB_state(state).expectation_value(self.H.Ham)
        logger.info(f"Energy ref {Eq}")
        E2q = QMB_state(state).expectation_value(self.H.Ham @ self.H.Ham)
        # Get the corresponding variance
        DeltaE = np.sqrt(E2q - (Eq**2))
        logger.info(f"delta E {DeltaE}")
        # Check that Eq is contained in the set of eigvals:
        if Eq + DeltaE > max(self.H.Nenergies):
            msg = f"Need more eigvals for the Energy shell: {Eq+DeltaE} > {max(self.H.Nenergies)}"
            raise ValueError(msg)
        # Initialize a state as the superposition of all the eigenstates within
        # an energy shell of amplitude Delta E around Eq
        psi_thermal = np.zeros(self.H.Ham.shape[0], dtype=np.complex128)
        list_states = []
        for ii in range(self.n_eigs):
            if np.abs(self.H.Nenergies[ii] - Eq) < DeltaE:
                list_states.append(ii)
                psi_thermal += self.H.Npsi[ii].psi
        n_shell_states = len(list_states)
        # Normalize the state as the superposition of all the states in the shell
        psi_thermal /= np.sqrt(n_shell_states)
        # Define the local observable as an operator
        local_op = LocalTerm(self.ops[local_obs], local_obs, **self.def_params)
        local_op.get_expval(QMB_state(psi_thermal))
        if special_norm is not None:
            logger.info(f"norm {special_norm}")
            microcanonical_avg = np.dot(local_op.obs, special_norm) / self.n_sites
        else:
            microcanonical_avg = stag_avg(local_op.obs, staggered_avg)
        logger.info(f"Microcanonical avg: {microcanonical_avg}")
        logger.info("----------------------------------------------------")
        return psi_thermal, microcanonical_avg

    def diagonal_avg1(
        self, local_obs: str, state: np.ndarray, special_norm=None, staggered_avg=None
    ):
        logger.info("----------------------------------------------------")
        logger.info("DIAGONAL ENSEMBLE")
        # check that the hamiltonian has been already fully diagonalized:
        if self.n_eigs != self.H.Ham.shape[0]:
            msg = f"Need all H eigvals {self.H.Ham.shape[0]}, not only {self.n_eigs}"
            raise ValueError(msg)
        # Define the local observable as an operator
        local_op = LocalTerm(self.ops[local_obs], local_obs, **self.def_params)
        # Measure the operator on every eigenstate
        diagonal_avg = 0
        for ii in range(self.n_eigs):
            prob = self.measure_fidelity(state, ii, False)
            local_op.get_expval(self.H.Npsi[ii], print_values=False)
            if special_norm is not None:
                exp_val = np.dot(local_op.obs, special_norm) / self.n_sites
            else:
                exp_val = stag_avg(local_op.obs, staggered_avg)
            diagonal_avg += prob * exp_val
        logger.info(f"Diagonal avg: {diagonal_avg}")
        logger.info("----------------------------------------------------")
        return diagonal_avg
