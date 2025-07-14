import numpy as np
from scipy.sparse import csc_matrix, isspmatrix, csr_matrix
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
    get_neighbor_sites,
    zig_zag,
    staggered_mask,
)
from ed_lgt.symmetries import (
    get_symmetry_sector_generators,
    get_link_sector_configs,
    global_abelian_sector,
    get_momentum_basis,
    symmetry_sector_configs,
    config_to_index_binarysearch,
)
import logging

logger = logging.getLogger(__name__)


__all__ = ["QuantumModel"]


class QuantumModel:
    def __init__(
        self,
        lvals: list[int],
        has_obc: list[bool],
        ham_format="sparse",
        logical_unit_size: int = 1,
        momentum_basis: bool = False,
        momentum_k=0,
        basis_projector: np.ndarray = None,
    ):
        # Lattice parameters
        self.lvals = lvals
        self.dim = len(self.lvals)
        self.directions = "xyz"[: self.dim]
        self.n_sites = prod(self.lvals)
        self.has_obc = has_obc
        # Symmetry sector configurations
        self.sector_configs = None
        # Gauge Basis
        self.gauge_basis = None
        # Staggered Basis
        self.staggered_basis = False
        # Momentum Basis
        self.momentum_basis = momentum_basis
        self.momentum_k = momentum_k
        self.logical_unit_size = int(logical_unit_size)
        # Hamiltonian format
        self.ham_format = ham_format
        # Efficient reduced basis projector
        self.basis_projector = basis_projector
        if basis_projector is not None:
            logger.info(f"Efficient basis projector: {basis_projector.shape}")
        # Dictionary for results
        self.res = {}

    def default_params(self):
        if self.momentum_basis:
            if self.has_obc[0]:
                raise ValueError(f"Momentum is not conserved in OBC")
            self.B = get_momentum_basis(
                self.sector_configs, self.logical_unit_size, self.momentum_k
            )
            logger.info(f"Momentum basis {self.momentum_k} shape {self.B.shape}")
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
            "sector_configs": self.sector_configs,
            "momentum_basis": self.B,
        }
        # Initialize the Hamiltonian
        self.H = QMB_hamiltonian(self.lvals, size=hamiltonian_size)

    def project_operators(self, ops_dict: dict[csr_matrix]):
        """
        Compute the local basis of each site and the corresponding lattice labels.
        Project a dictionary of operators into a gauge-invariant or optimal subspace of each
        lattice site.

        Parameters:
            ops_dict (dict): Dictionary of operators (each one a scipy.sparse.csr_matrix).

        Returns:
            dict: New dictionary of projected operators (np.ndarray) with shape (n_sites, max_loc_dim, max_loc_dim).
            The keys are the same as the input dictionary and contains the effective matrix
            for each site, accounting for the possibility of different local Hilbert spaces among the sites.
        """
        if self.gauge_basis is not None:
            # Acquire local dimension and lattice label
            lattice_labels, loc_dims = lattice_base_configs(
                self.gauge_basis, self.lvals, self.has_obc, self.staggered_basis
            )
            loc_dims = loc_dims.transpose().reshape(self.n_sites)
            self.lattice_labels = lattice_labels.transpose().reshape(self.n_sites)
        else:
            # Local dimension is the same for all sites
            local_dim = ops_dict[list(ops_dict.keys())[0]].shape[0]
            loc_dims = np.array([local_dim] * self.n_sites, dtype=int)
            self.lattice_labels = None
        # Determine effective local dimension
        # NOTE: we assume that the basis projector is the same for all sites
        # This will be eventually generalized
        if self.basis_projector is not None:
            local_dim = self.basis_projector.shape[1]
            self.loc_dims = np.array([local_dim] * self.n_sites, dtype=int)
        else:
            self.loc_dims = loc_dims
        logger.info(f"local dimensions: {self.loc_dims}")
        # Determine the maximum local dimension
        max_loc_dim = max(self.loc_dims)
        # -----------------------------------------------------------------------------
        # Initialize new dictionary with the projected operators
        logger.debug(f"Projecting operators to the effective Hilbert space")
        self.ops = {}
        # Iterate over operators
        for op, operator in ops_dict.items():
            self.ops[op] = np.zeros(
                (self.n_sites, max_loc_dim, max_loc_dim), dtype=ops_dict[op].dtype
            )
            # Run over the sites
            for jj, loc_dim in enumerate(self.loc_dims):
                # For Lattice Gauge Theories where sites have different Hilbert Bases
                if self.gauge_basis is not None:
                    # Get the label of the site
                    site_label = self.lattice_labels[jj]
                    # Get the projected operator
                    eff_op = apply_projection(
                        projector=self.gauge_basis[site_label],
                        operator=operator,
                    ).toarray()
                # For Theories where all the sites have the same Hilber basis
                else:
                    eff_op = operator.toarray()
                # If an extra projector is given (to reduce the local Hilbert space)
                if self.basis_projector is not None:
                    eff_op = apply_projection(
                        projector=self.basis_projector, operator=eff_op
                    )
                # Save it inside the new list of operators
                # NOTE: here we assume all the operators to be real
                self.ops[op][jj, :loc_dim, :loc_dim] = np.real(eff_op)

    def get_abelian_symmetry_sector(
        self,
        global_ops: list[np.ndarray] = None,
        global_sectors: list = None,
        global_sym_type: str = "U",
        link_ops: list[np.ndarray] = None,
        link_sectors: list = None,
        nbody_ops: list[np.ndarray] = None,
        nbody_sectors: list = None,
        nbody_sites_list=None,
        nbody_sym_type: str = None,
    ):
        # ================================================================================
        # GLOBAL ABELIAN SYMMETRIES
        if global_ops is not None:
            logger.debug("Global Symmetry operators")
            global_ops = get_symmetry_sector_generators(global_ops, action="global")
        # ================================================================================
        # ABELIAN Z2 SYMMETRIES
        if link_ops is not None:
            logger.debug("Link Symmetry operators")
            link_ops = get_symmetry_sector_generators(link_ops, action="link")
            pair_list = get_lattice_link_site_pairs(self.lvals, self.has_obc)
        # ================================================================================
        # nBODY ABELIAN SYMMETRIES
        if nbody_ops is not None:
            logger.debug("Nbody Symmetry operators")
            nbody_ops = get_symmetry_sector_generators(nbody_ops, action="nbody")
        # ================================================================================
        if global_ops is not None and link_ops is not None:
            logger.debug("Global & Link symmetry sector")
            self.sector_configs = symmetry_sector_configs(
                loc_dims=self.loc_dims,
                glob_op_diags=global_ops,
                glob_sectors=np.array(global_sectors, dtype=float),
                sym_type_flag=global_sym_type,
                link_op_diags=link_ops,
                link_sectors=link_sectors,
                pair_list=pair_list,
            )
        elif global_ops is not None:
            logger.debug("Global symmetry sector")
            self.sector_configs = global_abelian_sector(
                loc_dims=self.loc_dims,
                sym_op_diags=global_ops,
                sym_sectors=np.array(global_sectors, dtype=float),
                sym_type=global_sym_type,
            )
        elif link_ops is not None:
            if nbody_ops is not None:
                logger.debug("Link & Nbody symmetry sector")
            else:
                logger.debug("Link symmetry sector")
            self.sector_configs = get_link_sector_configs(
                loc_dims=self.loc_dims,
                link_op_diags=link_ops,
                link_sectors=link_sectors,
                pair_list=pair_list,
                nbody_op_diags=nbody_ops,
                nbody_sectors=nbody_sectors,
                nbody_sites_list=nbody_sites_list,
                nbody_sym_type=nbody_sym_type,
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
        state = np.zeros(len(self.sector_configs), dtype=np.complex128)
        # Get the corresponding QMB index of each config
        for config in configs:
            index = config_to_index_binarysearch(config, self.sector_configs)
            if index < 0:
                logger.info(f"{config}")
                raise ValueError(f"config not compatible with the symmetry sector")
            logger.info(f"{config} in state {index}")
            state[index] = complex(1 / np.sqrt(len(configs)), 0)
        if self.momentum_basis:
            # Project the state in the momentum sector
            state = self.B.transpose().dot(state)
        logger.info("----------------------------------------------------")
        return state

    def measure_fidelity(
        self,
        state: np.ndarray,
        index: int,
        dynamics: bool = False,
        print_value: bool = False,
    ):
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
        E2q = QMB_state(state).expectation_value(self.H.Ham @ self.H.Ham)
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
        logger.info("----------------------------------------------------")
        logger.info("BUILDING OBSERVABLES")
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
                axes=[op_names_list[-1][3], op_names_list[-1][6]],
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

    def link_avg(self, obs_px, obs_py):
        avg = 0
        tmp = 0
        for ii in range(prod(self.lvals)):
            # Compute the corresponding coords
            coords = zig_zag(self.lvals, ii)
            # Check if it admits a x twobody term according to the lattice geometry
            _, sites_list = get_neighbor_sites(coords, self.lvals, "x", self.has_obc)
            if sites_list is not None:
                avg += obs_px[ii]
                tmp += 1
            # Check if it admits a y twobody term according to the lattice geometry
            _, sites_list = get_neighbor_sites(coords, self.lvals, "y", self.has_obc)
            if sites_list is not None:
                avg += obs_py[ii]
                tmp += 1
        logger.debug(f"{tmp} effective links")
        return avg / tmp

    def microcanonical_avg1(
        self,
        local_obs_list: list[str],
        state: np.ndarray,
        special_norm=dict,
        staggered_avg=dict,
    ):
        """
        Compute the microcanonical ensemble averages for a set of local observables.

        The microcanonical ensemble is defined as an equal probability incoherent mixture
        of eigenstates whose energies lie within an energy shell [Eq, Eq + ΔE], where Eq
        is the expectation value of energy for the reference state and ΔE is its corresponding
        energy uncertainty. For each observable, the expectation value is computed for every
        eigenstate in the shell and then averaged.

        Args:
            local_obs_list (list of str): List of keys corresponding to local observables.
                For each observable key, self.ops[obs] should give the corresponding operator.
            state (np.ndarray): The reference quantum state (vector) used to compute the energy.
            special_norms (dict): A dictionary mapping observable keys to a special
                normalization array. If provided for an observable, the expectation value is computed
                as a weighted dot product with that norm.
            staggered_avgs (dict): A dictionary mapping observable keys to a staggered
                averaging function (e.g. a function that averages only over even or odd sites).
                If provided for an observable, its measured expectation values will be averaged using
                that function.

        Returns:
            tuple:
                psi_thermal (np.ndarray): The coherent superposition (normalized) of all eigenstates
                    in the energy shell.
                micro_avg (dict): A dictionary mapping each observable key to its microcanonical average.
        """
        logger.info("----------------------------------------------------")
        logger.info("MICRO-CANONICAL ENSEMBLE (MULTI-OBSERVABLE)")

        # Compute the reference energy and its variance for the energy shell.
        Eq = QMB_state(state).expectation_value(self.H.Ham)
        logger.info(f"Energy ref: {Eq}")
        E2q = QMB_state(state).expectation_value(self.H.Ham @ self.H.Ham)
        DeltaE = np.sqrt(E2q - Eq**2)
        logger.info(f"DeltaE: {DeltaE}")

        # Check that the energy shell is covered by our eigenstates.
        if Eq + DeltaE > max(self.H.Nenergies):
            msg = f"Need more eigvals: {Eq+DeltaE} > {max(self.H.Nenergies)}"
            raise ValueError(msg)

        # Initialize the thermal state (for coherent superposition) and the observable accumulators.
        psi_thermal = np.zeros(self.H.Ham.shape[0], dtype=np.complex128)
        local_ops = {
            obs: LocalTerm(self.ops[obs], obs, **self.def_params)
            for obs in local_obs_list
        }
        micro_avg = {f"ME_{obs}": 0.0 for obs in local_obs_list}
        n_shell_states = 0

        # Pre-select eigenstate indices that are within the energy shell.
        shell_mask = np.abs(self.H.Nenergies - Eq) < DeltaE
        shell_indices = np.where(shell_mask)[0]
        n_shell_states = len(shell_indices)
        # Loop over the shell eigenstates.
        for ii in shell_indices:
            if np.abs(self.H.Nenergies[ii] - Eq) < DeltaE:
                psi_thermal += self.H.Npsi[ii].psi
                # For each observable, build the local operator and measure its expectation value.
                for op in local_obs_list:
                    local_ops[op].get_expval(self.H.Npsi[ii], print_values=False)
                    if special_norm[op] is not None:
                        micro_avg[f"ME_{op}"] += (
                            np.dot(local_ops[op].obs, special_norm[op]) / self.n_sites
                        )
                    else:
                        micro_avg[f"ME_{op}"] += self.stag_avg(
                            local_ops[op].obs, staggered_avg[op]
                        )

        # Normalize the coherent superposition and each observable's accumulated value.
        psi_thermal /= np.sqrt(n_shell_states)
        for obs in local_obs_list:
            micro_avg[f"ME_{obs}"] /= n_shell_states
        # Print microcanonical averages
        logger.info(f"Microcanonical averages")
        for obs in local_obs_list:
            logger.info(f"{obs}: {micro_avg[f'ME_{obs}']}")
        logger.info("----------------------------------------------------")
        return psi_thermal, micro_avg

    def diagonal_avg1(
        self,
        local_obs_list: list[str],
        state: np.ndarray,
        special_norms: dict = None,
        staggered_avg: dict = None,
    ):
        """
        Compute the diagonal ensemble averages for a set of local observables.

        In the diagonal ensemble, the full Hamiltonian is diagonalized and the
        expectation value of an observable O is computed as a weighted sum over all eigenstates:

            ⟨O⟩_diag = Σ_i p_i ⟨E_i|O|E_i⟩,

        where the weights p_i are given by the fidelity (overlap squared) between the
        initial state and each eigenstate.

        This function generalizes the computation to a list of local observables, avoiding
        redundant computation of the fidelity weights and allowing for special normalization
        or staggered averaging for each observable.

        Args:
            local_obs_list (list of str): List of keys corresponding to local observables.
                For each key, self.ops[obs] should provide the corresponding operator.
            state (np.ndarray): The reference quantum state (vector) used to compute the energy
                and the fidelity weights.
            special_norms (dict, optional): Dictionary mapping observable keys to a special
                normalization array. If provided for an observable, the expectation value is computed
                as a weighted dot product with that norm.
            staggered_avgs (dict, optional): Dictionary mapping observable keys to a staggered
                averaging function (e.g. one that averages only over even or odd sites).

        Returns:
            dict: A dictionary mapping each observable key to its diagonal ensemble average.
        """
        logger.info("----------------------------------------------------")
        logger.info("DIAGONAL ENSEMBLE (MULTI-OBSERVABLE)")

        # Ensure full diagonalization is available.
        if self.n_eigs != self.H.Ham.shape[0]:
            msg = f"Need all H eigvals {self.H.Ham.shape[0]}, not only {self.n_eigs}"
            raise ValueError(msg)

        # Create a dictionary of LocalTerm objects for each observable.
        local_ops = {
            obs: LocalTerm(self.ops[obs], obs, **self.def_params)
            for obs in local_obs_list
        }

        # Initialize accumulators for the diagonal ensemble average.
        diag_avg = {f"DE_{obs}": 0.0 for obs in local_obs_list}
        # Loop over every eigenstate; full diagonalization is assumed.
        for ii in range(self.n_eigs):
            # Fidelity of eigenstate ii.
            prob = self.measure_fidelity(state, ii, False)
            # For each observable, measure the expectation value in eigenstate ii.
            for obs in local_obs_list:
                local_ops[obs].get_expval(self.H.Npsi[ii], print_values=False)
                if special_norms[obs] is not None:
                    exp_val = (
                        np.dot(local_ops[obs].obs, special_norms[obs]) / self.n_sites
                    )
                elif staggered_avg[obs] is not None:
                    exp_val = self.stag_avg(local_ops[obs].obs, staggered_avg[obs])
                else:
                    # Default: use the average over all sites.
                    exp_val = np.mean(local_ops[obs].obs)
                diag_avg[f"DE_{obs}"] += prob * exp_val

        # Optionally, you could print or log the individual averages.
        logger.info("Diagonal Ensemble Averages:")
        for obs in local_obs_list:
            logger.info(f"{obs}: {diag_avg[f'DE_{obs}']}")
        logger.info("----------------------------------------------------")
        return diag_avg

    def stag_avg(self, arr_flat: np.ndarray, staggered_avg=None):
        """
        Compute the average of arr_flat over either all sites, or only the "even"
        or "odd" checkerboard sites, *using* your custom zig-zag flattening.

        Parameters
        ----------
        arr_flat : 1D np.ndarray, length = prod(lvals)
            Your data, flattened in zig-zag order.
        lvals : tuple of ints, e.g. (Nx, Ny)
            The original lattice shape.
        staggered_avg : None, "even", or "odd"
            Which checkerboard parity to average over.  None => average everything.
        """
        if staggered_avg is None:
            return np.mean(arr_flat)
        # 1) build the true 2D checkerboard mask:
        mask2d = staggered_mask(self.lvals, staggered_avg)
        # 2) allocate a 1D mask of the same length as arr_flat:
        N = arr_flat.size
        mask1d = np.zeros(N, dtype=bool)
        # 3) for each flattened index d, map back to coords, then lookup mask2d:
        for d in range(N):
            coords = zig_zag(self.lvals, d)
            mask1d[d] = mask2d[coords]
        # 4) select and average:
        return np.mean(arr_flat[mask1d])


def apply_projection(projector, operator):
    """
    Project an operator onto the subspace defined by a projector.

    Given:
      - projector (np.ndarray or csr_matrix): a (N, k) matrix
      - operator (np.ndarray or csr_matrix): a (N, N) operator

    Returns:
      - O_eff: the effective operator $O_eff = P^{\dagger}\cdot O \cdot P$
    """
    # Check the shape of the projector
    if projector.shape[0] != operator.shape[0]:
        msg = f"Projector and operator have incompatible shapes {projector.shape} {operator.shape}"
        raise ValueError(msg)
    # Return the projected operator
    if isinstance(operator, np.ndarray) and isinstance(projector, np.ndarray):
        return np.dot(projector.conj().T, np.dot(operator, projector))
    elif isspmatrix(operator) and isspmatrix(projector):
        return projector.conj().transpose() @ operator @ projector
    else:
        logger.info(f"{type(operator)} {type(projector)}")
