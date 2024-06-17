import numpy as np
from scipy.linalg import eig
from scipy.sparse import csc_matrix
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
    get_operators_nbody_term as nbops,
    link_abelian_sector,
    global_abelian_sector,
    momentum_basis_k0,
    symmetry_sector_configs,
    nbody_term,
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
        # Initialize the Hamiltonian
        self.H = QMB_hamiltonian(0, self.lvals)

    def default_params(self):
        if self.momentum_basis:
            self.B = momentum_basis_k0(self.sector_configs, self.logical_unit_size)
            logger.info(f"Momentum basis shape {self.B.shape}")
        else:
            self.B = None

        self.def_params = {
            "lvals": self.lvals,
            "has_obc": self.has_obc,
            "gauge_basis": self.gauge_basis,
            "sector_configs": self.sector_configs,
            "staggered_basis": self.staggered_basis,
            "momentum_basis": self.B,
            "momentum_k": 0,
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
                configs=self.sector_configs,
            )
        elif link_ops is not None:
            self.sector_indices, self.sector_configs = link_abelian_sector(
                loc_dims=self.loc_dims,
                sym_op_diags=link_ops,
                sym_sectors=link_sectors,
                pair_list=pair_list,
                configs=self.sector_configs,
            )

    def diagonalize_Hamiltonian(self, n_eigs, format):
        self.n_eigs = n_eigs
        # DIAGONALIZE THE HAMILTONIAN
        self.H.diagonalize(self.n_eigs, format, self.loc_dims)
        self.res["energy"] = self.H.Nenergies

    def time_evolution_Hamiltonian(self, initial_state, start, stop, n_steps):
        self.H.time_evolution(initial_state, start, stop, n_steps, self.loc_dims)

    def momentum_basis_projection(self, operator):
        # Project the Hamiltonian onto the momentum sector with k=0
        if isinstance(operator, str) and operator == "H":
            self.H.Ham = self.B.transpose() * self.H.Ham * self.B
        else:
            return self.B.transpose() * operator * self.B

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
        return canonical_avg

    def microcanonical_avg(self, local_obs, state):
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
        # Initialize a state as the superposition of all the eigenstates within
        # an energy shell of amplitude Delta E around Eq
        psi_thermal = np.zeros(self.H.Ham.shape[0], dtype=np.complex128)
        list_states = []
        for ii in range(self.n_eigs):
            if np.abs(self.H.Nenergies[ii] - Eq) < DeltaE:
                logger.info(f"{ii} {self.H.Nenergies[ii]}")
                list_states.append(ii)
                psi_thermal += self.H.Npsi[ii].psi
        norm = len(list_states)
        psi_thermal /= np.sqrt(norm)
        # Compute the microcanonical average of the local observable
        microcanonical_avg = 0
        for ii, state_indx in enumerate(list_states):
            microcanonical_avg += self.H.Npsi[state_indx].expectation_value(
                op_matrix
            ) / (norm * self.n_sites)
        logger.info(f"Microcanonical avg: {microcanonical_avg}")
        return psi_thermal, microcanonical_avg

    def diagonal_avg(self, local_obs, state):
        op_matrix = LocalTerm(
            operator=self.ops[local_obs], op_name=local_obs, **self.def_params
        ).get_Hamiltonian(1)
        diagonal_avg = 0
        for ii in range(self.n_eigs):
            prob = self.measure_fidelity(state, ii, False)
            exp_val = self.H.Npsi[ii].expectation_value(op_matrix) / self.n_sites
            diagonal_avg += prob * exp_val
        logger.info(f"Diagonal avg: {diagonal_avg}")
        return diagonal_avg

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
            self.res[obs] = self.obs_list[obs].corr

    def get_energy_gap(self, ex_counts, ops_names, H_info):
        logger.info(f"momentum_basis {self.momentum_basis}")
        # Compute the total matrix sizes for the generalized eigenvalue problem
        matrix_size = 0
        for ii in range(len(ex_counts)):
            matrix_size += ex_counts[ii] * (self.n_sites ** (ii + 1))
        # Initialize to zero the matrices for estimating the gap
        N = np.zeros((matrix_size, matrix_size), dtype=float)
        M = np.zeros((matrix_size, matrix_size), dtype=float)
        for idxA in range(matrix_size):
            A_sites, A_ops = ex_info_ops(
                idxA, self.n_sites, ex_counts, self.ops, ops_names
            )
            if not A_sites:
                continue
            for idxB in range(matrix_size):
                B_sites, B_ops = ex_info_ops(
                    idxB, self.n_sites, ex_counts, self.ops, ops_names, True
                )
                if not B_sites:
                    continue
                ex_ops = {"A": A_ops, "B": B_ops}
                ex_sites = {"A": A_sites, "B": B_sites}
                logger.info("=================================================")
                logger.info(f"ex_sites {ex_sites}")
                # Compute the N expectation value
                N[idxA, idxB] = self.N_expval(ex_ops, ex_sites, self.H.Npsi[0])
                # Compute the M expectation value
                M[idxA, idxB] = self.M_expval(ex_ops, ex_sites, H_info, self.H.Npsi[0])
                logger.info(f"{idxA}, {idxB}: N {N[idxA, idxB]} M {M[idxA, idxB]}")
        # Check for condition numbers
        cond_M = np.linalg.cond(M)
        cond_N = np.linalg.cond(N)
        logger.info(f"Condition number of M: {cond_M}")
        logger.info(f"Condition number of N: {cond_N}")

        # Check for eigenvalues of N
        eigvals_N = np.linalg.eigvals(N)
        logger.info(f"Eigenvalues of N: {eigvals_N}")

        # Regularize N if necessary
        if np.any(eigvals_N <= 1e-10):
            eps = 1e-10
            N += eps * np.eye(N.shape[0])
            logger.info("Regularized N to improve numerical stability")

        # Solve the generalized eigenvalue problem
        try:
            omega = eig(M, N, left=False, right=False)[0]
            logger.info(f"mass gap: {omega}")
        except Exception as e:
            logger.error(f"Generalized eigenvalue problem failed: {e}")
            omega = np.nan
        # Solve the generalized eigenvalue problem
        omega = eig(M, N, left=False, right=False)[0]
        logger.info(f"mass gap: {omega}")

    def N_expval(self, ex_ops, ex_sites, psi):
        if not isinstance(psi, QMB_state):
            raise TypeError("psi must be a QMB state")
        # Build a dictionary for sites: shared and unshared between A, B
        sites = {"ushd": {}, "shd": []}
        ops = {"ushd": {}, "shd": {}}
        # Separate the indices in shared and unshared
        sites["ushd"]["A"], sites["ushd"]["B"], sites["shd"] = compare_indices(
            ex_sites["A"], ex_sites["B"]
        )
        logger.info(f'ind {sites["ushd"]["A"]} + {sites["ushd"]["B"]} + {sites["shd"]}')
        # Check that A and B admits shared indices
        if sites["shd"]:
            for idx in ["A", "B"]:
                ops["shd"][idx] = filter_ops(ex_ops[idx], ex_sites[idx], sites["shd"])
                ops["ushd"][idx] = filter_ops(
                    ex_ops[idx], ex_sites[idx], sites["ushd"][idx]
                )
            # Combine the list of sites into a unique list
            op_sites_list = np.array(
                sites["ushd"]["A"] + sites["ushd"]["B"] + sites["shd"]
            )
            # List of operators sorting the (un)shared indices of idxA, of idxB
            op_list = ops["ushd"]["A"] + ops["ushd"]["B"]
            AB = [op1 @ op2 for op1, op2 in zip(ops["shd"]["A"], ops["shd"]["B"])]
            BA = [op2 @ op1 for op1, op2 in zip(ops["shd"]["A"], ops["shd"]["B"])]
            # Compute the expectation value of the commutator
            return psi.expectation_value(
                nbody_term(
                    nbops(op_list + AB, self.loc_dims),
                    op_sites_list,
                    self.sector_configs,
                )
                - nbody_term(
                    nbops(op_list + BA, self.loc_dims),
                    op_sites_list,
                    self.sector_configs,
                )
            )
        else:
            return 0

    def M_expval(self, ex_ops, ex_sites, H, psi):
        if not isinstance(psi, QMB_state):
            raise TypeError("psi must be a QMB state")
        n_sites = np.prod(psi.lvals)
        expval = 0
        # Organize a dictionary to filter operators according to (un)shared indices
        op_list = {
            "H": {"shd": {}, "ushd": []},
            "A": {"shd": {}, "ushd": []},
            "B": {"shd": {}, "ushd": []},
        }
        # Build a dictionary for sites: shared and unshared between A, B, H
        sites = {"ushd": {}, "shd": {}}
        # Consider the Hamiltonian
        for H_type in range(len(H["ops"])):
            for op_id in range(len(H["ops"][H_type])):
                # List of Hamiltonian operators
                ex_ops["H"] = [self.ops[op_name] for op_name in H["ops"][H_type][op_id]]
                # Get the corresponding Hamiltonian coefficient
                H_coeff = H["coeffs"][H_type][op_id]
                logger.info(f"coeff {H_coeff}")
                # Run over all the lattice sites
                for ii in range(n_sites):
                    # Save the sites where the Hamiltonian term is acting
                    ex_sites["H"] = [ii] + [
                        (ii + kk + 1) % n_sites for kk in range(H_type)
                    ]
                    logger.info(f"H sites {ex_sites['H']}")
                    # Separate the B indices in shared and unshared wrt the Hamiltonian
                    ushd_H, ushd_B, shd_HB = compare_indices(
                        ex_sites["H"], ex_sites["B"]
                    )
                    HuB = ushd_H + ushd_B + shd_HB
                    # If B and H admits shared site-indices
                    if shd_HB:
                        # Organize A site-indices based of sharing with B and H
                        sites["shd"]["HAB"] = [s for s in ex_sites["A"] if s in shd_HB]
                        sites["shd"]["HA"] = [s for s in ex_sites["A"] if s in ushd_H]
                        sites["shd"]["AB"] = [s for s in ex_sites["A"] if s in ushd_B]
                        sites["ushd"]["A"] = [s for s in ex_sites["A"] if s not in HuB]
                        # Update the list of sites for H and B
                        sites["ushd"]["B"] = [
                            s for s in ushd_B if s not in sites["shd"]["AB"]
                        ]
                        sites["ushd"]["H"] = [
                            s for s in ushd_H if s not in sites["shd"]["HA"]
                        ]
                        sites["shd"]["HB"] = [
                            s for s in shd_HB if s not in ex_sites["A"]
                        ]
                        logger.info(f"shd {sites['shd']}")
                        logger.info(f"ushd {sites['ushd']}")
                        # Filter the shared operators
                        for group in ["HA", "HAB", "AB"]:
                            op_list["A"]["shd"][group] = filter_ops(
                                ex_ops["A"], ex_sites["A"], sites["shd"][group]
                            )
                        for group in ["HB", "HAB", "AB"]:
                            op_list["B"]["shd"][group] = filter_ops(
                                ex_ops["B"], ex_sites["B"], sites["shd"][group]
                            )
                        for group in ["HA", "HB", "HAB"]:
                            op_list["H"]["shd"][group] = filter_ops(
                                ex_ops["H"], ex_sites["H"], sites["shd"][group]
                            )
                        # Filter the unshared operators and build a unique list
                        ushd_op_list = []
                        ushd_sites_list = []
                        for group in ["A", "B", "H"]:
                            op_list[group]["ushd"] = filter_ops(
                                ex_ops[group], ex_sites[group], sites["ushd"][group]
                            )
                            ushd_op_list += op_list[group]["ushd"]
                            ushd_sites_list += sites["ushd"][group]
                        # Filter the list with products of two operators
                        for combo in ["AB", "HA", "HB"]:
                            op_list[combo[0] + combo[1]] = [
                                op1 @ op2
                                for op1, op2 in zip(
                                    op_list[combo[0]]["shd"][combo],
                                    op_list[combo[1]]["shd"][combo],
                                )
                            ]
                            op_list[combo[1] + combo[0]] = [
                                op1 @ op2
                                for op1, op2 in zip(
                                    op_list[combo[1]]["shd"][combo],
                                    op_list[combo[0]]["shd"][combo],
                                )
                            ]
                        # Filter the list with products of three operators
                        op_name = "HAB"
                        for combo in ["AHB", "ABH", "HBA", "BHA"]:
                            op_list[combo] = [
                                op1 @ op2 @ op3
                                for op1, op2, op3 in zip(
                                    op_list[combo[0]]["shd"][op_name],
                                    op_list[combo[1]]["shd"][op_name],
                                    op_list[combo[2]]["shd"][op_name],
                                )
                            ]
                        # Define the combinations for operators and sites
                        op_combinations = {
                            "AHB": ["AB", "AHB", "HB", "AH"],
                            "ABH": ["AB", "ABH", "BH", "AH"],
                            "HBA": ["BA", "HBA", "HB", "HA"],
                            "BHA": ["BA", "BHA", "BH", "HA"],
                        }
                        # Combine the operators for each combination
                        shd_ops = {
                            key: [op for comb in combinations for op in op_list[comb]]
                            for key, combinations in op_combinations.items()
                        }
                        # Combine the shared sites
                        shd_list = [
                            site
                            for key in ["AB", "HAB", "HB", "HA"]
                            for site in sites["shd"][key]
                        ]
                        sites_list = np.array(ushd_sites_list + shd_list)
                        # Define the 4 four terms contributing to M
                        expval += psi.expectation_value(
                            H_coeff
                            * (
                                nbody_term(
                                    nbops(ushd_op_list + shd_ops["AHB"], self.loc_dims),
                                    sites_list,
                                    self.sector_configs,
                                )
                                - nbody_term(
                                    nbops(ushd_op_list + shd_ops["ABH"], self.loc_dims),
                                    sites_list,
                                    self.sector_configs,
                                )
                                - nbody_term(
                                    nbops(ushd_op_list + shd_ops["HBA"], self.loc_dims),
                                    sites_list,
                                    self.sector_configs,
                                )
                                + nbody_term(
                                    nbops(ushd_op_list + shd_ops["BHA"], self.loc_dims),
                                    sites_list,
                                    self.sector_configs,
                                )
                            )
                        )
        return expval


def ex_info_ops(idxA, n_sites, excitation_counts, ops, ops_names, dagger=False):
    """
    Calculate the excitation information and combine operators for duplicate sites.

    Parameters
    ----------
    idxA : int
        The excitation index.
    n_sites : int
        The number of sites in the system.
    excitation_counts : np.ndarray
        Array containing the number of excitations per type.
    ops : dict
        Dictionary of available operators.
    ops_names : list
        List of operator names.

    Returns
    -------
    Tuple[ List[int], List[np.ndarray]]
        A tuple containing the site indices, and the combined operators.
    """
    info, sites = get_excitation_info(idxA, n_sites, excitation_counts)
    if not info:
        return [], [], []

    n_body, ex_index = info
    if not dagger:
        ops_list = [ops[op_name] for op_name in ops_names[n_body][ex_index]]
    else:
        ops_list = [ops[op_name].conj().T for op_name in ops_names[n_body][ex_index]]
    # Combine the operators acting on the same site
    combined_sites = []
    combined_ops = []
    for idx, site in enumerate(sites):
        if site in combined_sites:
            # Find the position of the existing site in combined_sites
            pos = combined_sites.index(site)
            # Multiply the existing operator with the new one
            combined_ops[pos] = combined_ops[pos] @ ops_list[idx]
        else:
            combined_sites.append(site)
            combined_ops.append(ops_list[idx])
    # Filter out null operators
    filtered_sites = []
    filtered_ops = []
    for idx, op in enumerate(combined_ops):
        if op.nnz > 0:
            filtered_sites.append(combined_sites[idx])
            filtered_ops.append(op)
    return filtered_sites, filtered_ops


def get_excitation_info(idxA, n_sites, excitation_counts):
    """
    Calculate the excitation information for a given idxA index in a many-body system.

    Parameters
    ----------
    idxA : int
        The excitation index.
    n_sites : int
        The number of sites in the system.
    excitation_counts : np.ndarray
        Array containing the number of excitations per type.

    Returns
    -------
    Tuple[List[int], List[int]]
        A tuple containing the excitation type and the site indices.
    """
    n_body_max = len(excitation_counts)
    sizes = np.zeros(n_body_max, dtype=np.int64)

    for i in range(n_body_max):
        sizes[i] = excitation_counts[i] * (n_sites ** (i + 1))

    for n_body in range(n_body_max):
        if idxA < sizes[n_body]:
            ex_index = idxA // (n_sites ** (n_body + 1))
            R = idxA % (n_sites ** (n_body + 1))

            site_indices = []
            for i in range(n_body + 1):
                site_indices.append(R // (n_sites ** (n_body - i)))
                R %= n_sites ** (n_body - i)

            return [n_body, ex_index], site_indices

        idxA -= sizes[n_body]

    return [], []


def compare_indices(inds1, inds2):
    """
    Compare two lists of indices and return the shared and unshared indices.

    Parameters
    ----------
    inds1 : List[int]
        The first list of indices.
    inds2 : List[int]
        The second list of indices.

    Returns
    -------
    Tuple[List[int], List[int], List[int]]
        A tuple containing the unshared indices in inds1,
        unshared indices in inds2, and shared indices.
    """
    shd_inds = []
    ushd_inds1 = []
    ushd_inds2 = []
    for idx1 in inds1:
        if idx1 in inds2:
            shd_inds.append(idx1)
        else:
            ushd_inds1.append(idx1)
    for idx2 in inds2:
        if idx2 not in shd_inds:
            ushd_inds2.append(idx2)

    return ushd_inds1, ushd_inds2, shd_inds


def filter_ops(op_list, indices, filter_indices):
    """
    Filter operators based on indices.

    Parameters
    ----------
    op_list : List[np.ndarray]
        The list of operators.
    inds : List[int]
        The list of indices corresponding to the operators.
    filter_inds : List[int]
        The list of indices to filter by.

    Returns
    -------
    List[np.ndarray]
        A list of filtered operators.
    """
    filtered_ops = []
    for ii in range(len(op_list)):
        if indices[ii] in filter_indices:
            filtered_ops.append(op_list[ii])
    return filtered_ops
