"""High-level quantum lattice model workflow utilities.

This module defines :class:`QuantumModel`, a convenience class that combines
local operator projection, symmetry-sector construction, Hamiltonian assembly,
diagonalization, dynamics, and observable measurements for lattice gauge and
spin models.
"""

import numpy as np
from scipy.sparse import csc_matrix, isspmatrix, csr_matrix, identity
from scipy.sparse.linalg import expm, norm
from math import prod
from edlgt.modeling import (
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
from edlgt.symmetries import (
    get_symmetry_sector_generators,
    get_link_sector_configs,
    global_abelian_sector,
    get_momentum_basis,
    symmetry_sector_configs,
    config_to_index_binarysearch,
    config_to_index,
    subenv_map_to_unique_indices,
)
from edlgt.tools import exclude_columns
import logging

logger = logging.getLogger(__name__)
__all__ = ["QuantumModel"]


class QuantumModel:
    """High-level container orchestrating model building and measurements."""

    def __init__(
        self,
        lvals: list[int],
        has_obc: list[bool],
        ham_format="sparse",
        basis_projector: np.ndarray | None = None,
    ):
        """Initialize a lattice quantum model container.

        Parameters
        ----------
        lvals : list
            Lattice dimensions.
        has_obc : list
            Boundary-condition flags per axis (``True`` for open boundaries).
        ham_format : str, optional
            Preferred Hamiltonian representation (for example ``"sparse"``).
        basis_projector : numpy.ndarray, optional
            Optional site-local projector used to reduce the effective local
            Hilbert space uniformly on all sites.
        """
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
        self.gauge_states = None
        # Staggered Basis
        self.staggered_basis = False
        # Momentum Basis
        self.momentum_basis = None
        # Hamiltonian format
        self.ham_format = ham_format
        # Efficient reduced basis projector
        self.basis_projector = basis_projector
        if basis_projector is not None:
            logger.info(f"Efficient basis projector: {basis_projector.shape}")
        # Dictionary for system partition
        # A cache keyed by each bipartition (keep-indices tuple), storing everything you need for that cut.
        self._partition_cache: dict[tuple[int, ...], dict] = {}
        # Dictionary for results
        self.res = {}

    def default_params(self):
        """Initialize default keyword arguments and the Hamiltonian container.

        Returns
        -------
        None
            Updates ``self.def_params`` and (when possible) creates ``self.H``.
        """
        if self.momentum_basis is not None and self.sector_configs is not None:
            pair_mode = self.momentum_basis.get("pair_mode", False)
            if pair_mode:
                if self.momentum_basis["k_left"] != self.momentum_basis["k_right"]:
                    # rectangular H(k1,k2); can't create a square H
                    hamiltonian_size = None
                else:
                    hamiltonian_size = self.momentum_basis["n_cols_L"]
                    self.sector_dim = hamiltonian_size
            else:
                hamiltonian_size = self.momentum_basis["n_cols"]
                self.sector_dim = hamiltonian_size
        elif self.sector_configs is not None:
            hamiltonian_size = self.sector_configs.shape[0]
            self.sector_dim = hamiltonian_size
        else:
            hamiltonian_size = np.prod(self.loc_dims)
            self.sector_dim = hamiltonian_size
        # Define the default parameters as a dictionary
        self.def_params = {
            "lvals": self.lvals,
            "has_obc": self.has_obc,
            "sector_configs": self.sector_configs,
            "momentum_basis": self.momentum_basis,
        }
        if hamiltonian_size is not None:
            # Initialize the Hamiltonian
            self.H = QMB_hamiltonian(self.lvals, size=hamiltonian_size)
        else:
            self.H = None

    def set_momentum_sector(
        self,
        k_unit_cell_size: list[int],
        k_vals: list[int],
        TC_symmetry: bool = False,
    ):
        """Build and store a momentum-sector projector.

        Parameters
        ----------
        k_unit_cell_size : list
            Translation unit-cell size used to define momentum sectors.
        k_vals : list
            Target momentum quantum numbers (one per lattice dimension).
        TC_symmetry : bool, optional
            Whether to include translation-charge symmetry reduction in the
            momentum basis construction.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If symmetry-sector configurations are missing, OBC are present, or
            the momentum shape is inconsistent with the lattice dimension.
        """
        logger.info(k_vals)
        if self.sector_configs is None:
            raise ValueError("symmetry sector_configs not defined yet")
        if self.has_obc[0]:
            raise ValueError("Momentum is not conserved with OBC.")
        if np.ndim(k_vals) == 0:
            k_vals = np.array([int(k_vals)], dtype=np.int32)
        else:
            k_vals = np.ascontiguousarray(k_vals, dtype=np.int32)
        if k_vals.size != self.dim:
            raise ValueError(f"expected k_vals shape {self.dim}, got {k_vals.shape}")
        k_unit_cell_size = np.ascontiguousarray(k_unit_cell_size, dtype=np.int32)
        k_vals = np.ascontiguousarray(k_vals, dtype=np.int32)
        logger.info(f"====================================================")
        logger.info(f"Momentum sector {k_vals}")
        L_col_ptr, L_row_idx, L_data, R_row_ptr, R_col_idx, R_data = get_momentum_basis(
            sector_configs=self.sector_configs,
            lvals=self.lvals,
            unit_cell_size=k_unit_cell_size,
            k_vals=k_vals,
            TC_symmetry=TC_symmetry,
        )
        n_rows = self.sector_configs.shape[0]
        n_cols = L_col_ptr.shape[0] - 1
        # store for later use in projectors
        self.momentum_basis = {
            "pair_mode": False,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "L_col_ptr": L_col_ptr,
            "L_row_idx": L_row_idx,
            "L_data": L_data,
            "R_row_ptr": R_row_ptr,
            "R_col_idx": R_col_idx,
            "R_data": R_data,
        }
        logger.info(f"Momentum basis shape ({n_rows},{n_cols})")

    def set_momentum_pair(
        self,
        k_left: list[int],
        k_right: list[int],
        k_unit_cell_size: list[int],
        TC_symmetry: bool,
    ):
        """Build two momentum projectors for rectangular ``k_L``-``k_R`` blocks.

        Parameters
        ----------
        k_left, k_right : list
            Left and right momentum quantum numbers.
        k_unit_cell_size : list
            Translation unit-cell size.
        TC_symmetry : bool
            Whether to include translation-charge symmetry reduction.

        Returns
        -------
        None

        Notes
        -----
        This prepares rectangular projections of the form ``P_kL^dagger O P_kR``
        and stores them in ``self.momentum_basis`` in pair mode.
        """
        if self.sector_configs is None:
            raise ValueError("symmetry sector_configs not defined yet")
        if self.has_obc[0]:
            raise ValueError("Momentum is not conserved with OBC.")
        k_unit_cell_size = np.ascontiguousarray(k_unit_cell_size, dtype=np.int32)
        k_left = np.ascontiguousarray(k_left, dtype=np.int32)
        k_right = np.ascontiguousarray(k_right, dtype=np.int32)
        logger.info(f"====================================================")
        logger.info(f"Combined Momenta {k_left} {k_right}")
        # Left momentum kL
        L_col_ptr, L_row_idx, L_data, _, _, _ = get_momentum_basis(
            sector_configs=self.sector_configs,
            lvals=self.lvals,
            unit_cell_size=k_unit_cell_size,
            k_vals=k_left,
            TC_symmetry=TC_symmetry,
        )
        n_cols_L = L_col_ptr.shape[0] - 1
        # Right momentum kR
        _, _, _, R_row_ptr, R_col_idx, R_data = get_momentum_basis(
            sector_configs=self.sector_configs,
            lvals=self.lvals,
            unit_cell_size=k_unit_cell_size,
            k_vals=k_right,
            TC_symmetry=TC_symmetry,
        )
        n_cols_R = int(R_col_idx.max()) + 1
        # Save the momentum basis
        self.momentum_basis = {
            "pair_mode": True,
            "n_rows_full": self.sector_configs.shape[0],
            "n_cols_L": n_cols_L,
            "n_cols_R": n_cols_R,
            "L_col_ptr": L_col_ptr,
            "L_row_idx": L_row_idx,
            "L_data": L_data,
            "R_row_ptr": R_row_ptr,
            "R_col_idx": R_col_idx,
            "R_data": R_data,
            "k_left": k_left,
            "k_right": k_right,
        }
        msg = f"Momenta kL={k_left} (cols={n_cols_L}), kR={k_right} (cols={n_cols_R})"
        logger.info(msg)

    def check_momentum_pair(self):
        """Validate orthonormality/orthogonality of the stored momentum pair basis.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If either projector is not orthonormal or two distinct momentum
            sectors are not orthogonal.
        """
        N = self.sector_configs.shape[0]
        B_L = self.momentum_basis["L_col_ptr"].shape[0] - 1
        B_R = int(self.momentum_basis["R_col_idx"].max()) + 1
        P_L = csc_matrix(
            (
                self.momentum_basis["L_data"],
                self.momentum_basis["L_row_idx"],
                self.momentum_basis["L_col_ptr"],
            ),
            shape=(N, B_L),
        )
        P_R = csr_matrix(
            (
                self.momentum_basis["R_data"],
                self.momentum_basis["R_col_idx"],
                self.momentum_basis["R_row_ptr"],
            ),
            shape=(N, B_R),
        )
        G_L = P_L.conj().T @ P_L
        G_R = P_R.conj().T @ P_R
        MIX = P_L.conj().T @ P_R
        normL2 = norm(G_L - identity(B_L))
        normR2 = norm(G_R - identity(B_R))
        normMIX = norm(MIX)
        logger.info(f"norm: (PL^dag @ PL) -1: {normL2}")
        logger.info(f"norm: (PR^dag @ PR) -1: {normR2}")
        logger.info(f"norm: (PL^dag @ PR): {normMIX}")
        if normL2 > 1e-12:
            raise ValueError("PL is not a projector")
        if normR2 > 1e-12:
            raise ValueError("RL is not a projector")
        k1 = self.momentum_basis["k_left"]
        k2 = self.momentum_basis["k_right"]
        if (not np.array_equal(k1, k2)) and normMIX > 1e-12:
            raise ValueError("The two basis are not orthogonal")

    def project_operators(self, ops_dict: dict[str, csr_matrix], bg_sector_list=None):
        """Project local operators into the effective per-site basis.

        Parameters
        ----------
        ops_dict : dict
            Dictionary of bare local operators.
        bg_sector_list : list, optional
            Optional per-site background-sector labels used to restrict the
            gauge-invariant basis on each site.

        Returns
        -------
        None
            Stores projected operators in ``self.ops`` and local dimensions in
            ``self.loc_dims``.
        """
        if self.gauge_basis is not None:
            lattice_labels, loc_dims = lattice_base_configs(
                self.gauge_basis, self.lvals, self.has_obc, self.staggered_basis
            )
            loc_dims = loc_dims.transpose().reshape(self.n_sites)
            self.lattice_labels = lattice_labels.transpose().reshape(self.n_sites)
        else:
            local_dim = ops_dict[list(ops_dict.keys())[0]].shape[0]
            loc_dims = np.array([local_dim] * self.n_sites, dtype=int)
            self.lattice_labels = None
        # If a global basis_projector is used, keep old behavior (uniform dim)
        # Determine effective local dimension
        # NOTE: we assume that the basis projector is the same for all sites
        # This will be eventually generalized
        if self.basis_projector is not None:
            local_dim = self.basis_projector.shape[1]
            self.loc_dims = np.array([local_dim] * self.n_sites, dtype=int)
            # background slicing would need to apply after this projector
            bg_sector_list = None
        else:
            self.loc_dims = loc_dims.copy()
        # -------------------------------------------------------------------------
        self.gauge_states_per_site = []
        # build per-site background-sector column selections (if requested)
        sector_cols_per_site = None
        if (bg_sector_list is not None) and (self.gauge_basis is not None):
            if len(bg_sector_list) != self.n_sites:
                raise ValueError("bg_sector_list must have length self.n_sites")
            sector_cols_per_site = [None] * self.n_sites
            # For each site the specific GI states compatible with the given BG charge
            for ii in range(self.n_sites):
                site_label = self.lattice_labels[ii]
                bg_value = float(bg_sector_list[ii])
                # Extract the columns with that bg_value
                bg_col = self.gauge_states[site_label][:, 0]
                cols = np.flatnonzero(
                    np.isclose(bg_col, bg_value, rtol=0.0, atol=1e-12)
                ).astype(np.int64)
                if cols.size == 0:
                    msg = f"No GI states at site {ii} label={site_label} with bg {bg_value}"
                    raise ValueError(msg)
                # store cols (explicit list is safest; contiguous slicing is just an optimization)
                sector_cols_per_site[ii] = cols
                self.loc_dims[ii] = cols.size
                # Save the selected states for each site
                self.gauge_states_per_site.append(self.gauge_states[site_label][cols])
        elif (bg_sector_list is None) and (self.gauge_basis is not None):
            for ii in range(self.n_sites):
                site_label = self.lattice_labels[ii]
                self.gauge_states_per_site.append(self.gauge_states[site_label])
        logger.info(f"====================================================")
        logger.info("LOCAL DIMENSION per SITE")
        format_loc_dims(self.loc_dims, self.lvals)
        # Determine the maximum local dimension
        max_loc_dim = int(np.max(self.loc_dims))
        # -----------------------------------------------------------------------------
        # Initialize new dictionary with the projected operators
        logger.debug(f"Projecting operators to the effective Hilbert space")
        self.ops = {}
        # Iterate over operators
        for op, operator in ops_dict.items():
            op_shape = (self.n_sites, max_loc_dim, max_loc_dim)
            self.ops[op] = np.zeros(op_shape, dtype=ops_dict[op].dtype)
            # Run over the sites
            for jj, loc_dim in enumerate(self.loc_dims):
                # For Lattice Gauge Theories where sites have different Hilbert Bases
                if self.gauge_basis is not None and self.lattice_labels is not None:
                    # Get the label of the site
                    site_label = self.lattice_labels[jj]
                    # Get the projector
                    projector = self.gauge_basis[site_label]
                    if sector_cols_per_site is not None:
                        projector = projector[:, sector_cols_per_site[jj]]
                    eff_op = apply_projection(
                        projector=projector, operator=operator
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
                self.ops[op][jj, :loc_dim, :loc_dim] = eff_op

    def get_abelian_symmetry_sector(
        self,
        global_ops: list[np.ndarray] | None,
        global_sectors: list | None = None,
        global_sym_type: str = "U",
        link_ops: list[list[np.ndarray]] | None = None,
        link_sectors: list | None = None,
        nbody_ops: list[np.ndarray] | None = None,
        nbody_sectors: list | None = None,
        nbody_sites_list=None,
        nbody_sym_type: str | None = None,
    ):
        """Build symmetry-sector configurations from abelian constraints.

        Parameters
        ----------
        global_ops : list or None
            Generators for global abelian symmetries.
        global_sectors : list or None, optional
            Target sectors for global symmetries.
        global_sym_type : str, optional
            Global symmetry type flag passed to symmetry-sector routines.
        link_ops : list or None
            Link-symmetry generators.
        link_sectors : list or None, optional
            Target sectors for link symmetries.
        nbody_ops : list or None
            Optional n-body symmetry generators.
        nbody_sectors : list or None, optional
            Target sectors for n-body symmetries.
        nbody_sites_list : object, optional
            Site patterns for n-body symmetries.
        nbody_sym_type : str or None, optional
            Symmetry type flag for n-body constraints.

        Returns
        -------
        None
            Stores the resulting sector configurations in ``self.sector_configs``.
        """
        logger.info(f"----------------------------------------------------")
        # ================================================================================
        # GLOBAL ABELIAN SYMMETRIES
        if global_ops is not None:
            logger.info("Global Symmetry operators")
            global_ops_diags = get_symmetry_sector_generators(
                global_ops, action="global"
            )
        # ================================================================================
        # ABELIAN Z2 SYMMETRIES
        if link_ops is not None:
            logger.info("Link Symmetry operators")
            link_ops_diags = get_symmetry_sector_generators(link_ops, action="link")
            pair_list = get_lattice_link_site_pairs(self.lvals, self.has_obc)
        # ================================================================================
        # nBODY ABELIAN SYMMETRIES
        if nbody_ops is not None:
            logger.info("Nbody Symmetry operators")
            nbody_ops_diags = get_symmetry_sector_generators(nbody_ops, action="nbody")
        else:
            nbody_ops_diags = None
        # ================================================================================
        if global_ops is not None and link_ops is not None:
            logger.info("Global & Link Symmetry sector")
            self.sector_configs = symmetry_sector_configs(
                loc_dims=self.loc_dims,
                glob_op_diags=global_ops_diags,
                glob_sectors=np.array(global_sectors, dtype=float),
                sym_type_flag=global_sym_type,
                link_op_diags=link_ops_diags,
                link_sectors=link_sectors,
                pair_list=pair_list,
            )
        elif global_ops is not None:
            logger.info("Global Symmetry sector")
            self.sector_configs = global_abelian_sector(
                loc_dims=self.loc_dims,
                sym_op_diags=global_ops_diags,
                sym_sectors=np.array(global_sectors, dtype=float),
                sym_type=global_sym_type,
            )
        elif link_ops is not None:
            if nbody_ops is not None:
                logger.info("Link & Nbody Symmetry sector")
            else:
                logger.info("Link Symmetry sector")
            self.sector_configs = get_link_sector_configs(
                loc_dims=self.loc_dims,
                link_op_diags=link_ops_diags,
                link_sectors=link_sectors,
                pair_list=pair_list,
                nbody_op_diags=nbody_ops_diags,
                nbody_sectors=nbody_sectors,
                nbody_sites_list=nbody_sites_list,
                nbody_sym_type=nbody_sym_type,
            )

    def diagonalize_Hamiltonian(self, n_eigs, format, print_results=False):
        """Diagonalize the model Hamiltonian and cache energies/eigenstates."""
        # DIAGONALIZE THE HAMILTONIAN
        self.H.diagonalize(n_eigs, format, self.loc_dims, print_results)
        self.n_eigs = self.H.n_eigs
        self.res["energy"] = self.H.Nenergies

    def time_evolution_Hamiltonian(self, initial_state, time_line):
        """Evolve an initial state with the current Hamiltonian."""
        self.H.time_evolution(initial_state, time_line, self.loc_dims)

    # ---- Momentum-basis wrappers (no copies) ----
    def _basis_Pk_as_csc(self):
        """Return B as a SciPy CSC matrix using the stored arrays (no copy)."""
        if self.momentum_basis is None:
            return None
        mb = self.momentum_basis
        shape = (mb["n_rows"], mb["n_cols"])
        return csc_matrix((mb["L_data"], mb["L_row_idx"], mb["L_col_ptr"]), shape=shape)

    def _basis_Pk_as_csr(self):
        """Return B as a SciPy CSR matrix using the stored arrays (no copy)."""
        if self.momentum_basis is None:
            return None
        mb = self.momentum_basis
        shape = (mb["n_rows"], mb["n_cols"])
        return csr_matrix((mb["R_data"], mb["R_col_idx"], mb["R_row_ptr"]), shape=shape)

    def _project_state_with_basis(self, state_realspace):
        """Project a state into the stored momentum basis.

        Parameters
        ----------
        state_realspace : numpy.ndarray
            State coefficients in the symmetry-sector basis.

        Returns
        -------
        numpy.ndarray
            Momentum-basis coefficients (complex128).
        """
        if self.momentum_basis is None:
            # no momentum sector → identity map
            return state_realspace.astype(np.complex128, copy=False)

        mb = self.momentum_basis
        n_rows = mb["n_rows"]
        n_cols = mb["n_cols"]
        row_ptr = mb["R_row_ptr"]
        col_idx = mb["R_col_idx"]
        data = mb["R_data"]

        psi_out = np.zeros(n_cols, dtype=np.complex128)
        x = state_realspace  # (N,)
        # Accumulate: psi_out[j] += conj(B[r,j]) * x[r]
        for r in range(n_rows):
            xr = x[r]
            if xr == 0:  # tiny micro-optimization
                continue
            start = row_ptr[r]
            stop = row_ptr[r + 1]
            for p in range(start, stop):
                j = col_idx[p]
                v = data[p]
                psi_out[j] += np.conj(v) * xr
        return psi_out

    def momentum_basis_projection(self, operator):
        """Project an operator into the currently stored momentum basis.

        Parameters
        ----------
        operator : str or numpy.ndarray or scipy.sparse.spmatrix
            Operator to project. If ``"H"``, projects ``self.H.Ham`` in place.

        Returns
        -------
        scipy.sparse.csr_matrix or None
            Projected operator. Returns ``None`` when projecting ``"H"`` in
            place.

        Raises
        ------
        ValueError
            If no momentum basis has been set.
        """
        if self.momentum_basis is None:
            raise ValueError("Basis projector B is not set.")
        B_csc = self._basis_Pk_as_csc()
        # 1) pick A
        if isinstance(operator, str) and operator == "H":
            A = self.H.Ham
        else:
            A = operator
        # Coerce to sparse
        if not isspmatrix(A):
            A = csr_matrix(A)
        else:
            A = A.tocsr()  # good for left-multiply
        # 2) Y = A @ B  (N × M), keep sparse
        Y = A @ B_csc
        # 3) A' = B^† @ Y = (B_csc.conj().T) @ Y  (M × M)
        A_proj = (B_csc.conj().T) @ Y  # returns sparse
        # If projecting H, store it back in your Hamiltonian container
        if isinstance(operator, str) and operator == "H":
            self.H.Ham = A_proj.tocsr()
        else:
            return A_proj.tocsr()

    def _get_partition(self, keep_indices):
        """Build and cache metadata for a subsystem/environment bipartition.

        Parameters
        ----------
        keep_indices : list or tuple
            Lattice-site indices kept in the subsystem.

        Returns
        -------
        dict
            Cached partition metadata. For symmetry-sector calculations this
            includes subsystem/environment configuration maps and unique
            configurations; otherwise only dimensions and index lists are stored.
        """
        key = tuple(sorted(keep_indices))
        logger.info("----------------------------------------------------")
        logger.info(f"Bipartite the system: SUBSYS {keep_indices}")
        if key not in self._partition_cache:
            # Determine the environmental indices
            env_indices = [ii for ii in range(self.n_sites) if ii not in keep_indices]
            env_indices = np.array(env_indices)
            keep_indices = np.array(keep_indices)
            # ---------------------------------------------------------------------------------
            # Distinguish between the case of symmetry sector and the standard case
            if self.sector_configs is not None:
                # SYMMETRY SECTOR
                # Separate subsystem and environment configurations
                logger.info("get sybsystem configs")
                subsys_configs = exclude_columns(self.sector_configs, env_indices)
                logger.info("get environment configs")
                env_configs = exclude_columns(self.sector_configs, keep_indices)
                # Find unique subsystem and environment configurations
                unique_subsys_configs = np.unique(subsys_configs, axis=0)
                # Initialize the RDM with shape = number of unique subsys configs
                unique_env_configs = np.unique(env_configs, axis=0)
                # Dimensions of the partitions
                subsys_dim = unique_subsys_configs.shape[0]
                env_dim = unique_env_configs.shape[0]
                # Get the maps from subsys_configs to unique_subsys_configs (same for env)
                subsys_map, env_map = subenv_map_to_unique_indices(
                    subsystem_configs=subsys_configs,
                    environment_configs=env_configs,
                    unique_subsys_configs=unique_subsys_configs,
                    unique_env_configs=unique_env_configs,
                )
                # Check that maps are correct. Namely unique_cfgs contain all the given rows
                if not np.all(subsys_map >= 0):
                    raise ValueError("Invalid subsys_map: some entries are -1")
                if not np.all(env_map >= 0):
                    raise ValueError("Invalid env_map: some entries are -1")
            else:
                # NO SYMMETRY SECTOR
                # Determine the dimensions of the subsystem and environment for the bipartition
                subsys_dim = np.prod([self.loc_dims[ii] for ii in keep_indices])
                env_dim = np.prod([self.loc_dims[ii] for ii in env_indices])
            # ---------------------------------------------------------------------------------
            # Save the partition information
            if self.sector_configs is not None:
                self._partition_cache[key] = {
                    "subsys_indices": keep_indices,
                    "env_indices": env_indices,
                    "unique_subsys_configs": unique_subsys_configs,
                    "unique_env_configs": unique_env_configs,
                    "subsys_dim": subsys_dim,
                    "env_dim": env_dim,
                    "subsys_map": subsys_map,
                    "env_map": env_map,
                }
            else:
                self._partition_cache[key] = {
                    "subsys_indices": keep_indices,
                    "env_indices": env_indices,
                    "subsys_dim": subsys_dim,
                    "env_dim": env_dim,
                }
        return self._partition_cache[key]

    def build_projector_from_sector_to_fullspace(
        self, indices: list[int]
    ) -> csc_matrix:
        """Build a projector from a symmetry-reduced subsystem basis to the full basis.

        Parameters
        ----------
        indices : list
            Subsystem site indices.

        Returns
        -------
        scipy.sparse.csc_matrix
            Column projector from the symmetry-sector subsystem basis to the full
            subsystem computational basis.
        """
        unique_subsys_configs = self._get_partition(indices)["unique_subsys_configs"]
        subsys_dim = unique_subsys_configs.shape[0]
        subsys_loc_dims = self.loc_dims[indices]
        tot_subsys_dim = np.prod(subsys_loc_dims)
        # Build entries of the projector
        rows = np.empty(subsys_dim, dtype=np.int64)
        cols = np.arange(subsys_dim, dtype=np.int64)
        data = np.ones(subsys_dim, dtype=np.float64)
        # Run over the subsystem_configs
        for idx in range(subsys_dim):
            rows[idx] = config_to_index(
                config=unique_subsys_configs[idx], loc_dims=subsys_loc_dims
            )
        # Build the project as a sparse column matrix
        P = csc_matrix((data, (rows, cols)), shape=(tot_subsys_dim, subsys_dim))
        return P

    def get_qmb_state_from_configs(self, configs):
        """Build an equal-weight state from explicit sector configurations.

        Parameters
        ----------
        configs : sequence or list
            Collection of basis configurations compatible with
            ``self.sector_configs``.

        Returns
        -------
        numpy.ndarray
            Normalized state vector in the symmetry-sector basis.

        Raises
        ------
        NotImplementedError
            If a momentum sector is active.
        ValueError
            If one configuration is not compatible with the current symmetry
            sector.
        """
        if self.momentum_basis is not None:
            raise NotImplementedError("cannot get state configs within momentum sector")
        # INITIALIZE the STATE
        state = np.zeros(len(self.sector_configs), dtype=np.complex128)
        # Get the corresponding QMB index of each config
        logger.info("----------------------------------------------------")
        logger.info("GET QMB STATE FROM CONFIGS")
        for config in configs:
            index = config_to_index_binarysearch(config, self.sector_configs)
            if index < 0:
                logger.info(f"{config}")
                raise ValueError(f"config not compatible with the symmetry sector")
            logger.info(f"config {config} in state {index}")
            state[index] = complex(1 / np.sqrt(len(configs)), 0)
        return state

    def measure_fidelity(
        self,
        state: np.ndarray,
        index: int,
        dynamics: bool = False,
        print_value: bool = False,
    ):
        """Compute fidelity with an eigenstate or a time-evolved state.

        Parameters
        ----------
        state : numpy.ndarray
            Reference state vector.
        index : int
            Index of the comparison state.
        dynamics : bool, optional
            If ``True``, compare against ``self.H.psi_time[index]`` instead of
            ``self.H.Npsi[index]``.
        print_value : bool, optional
            If ``True``, log the fidelity value.

        Returns
        -------
        float
            Fidelity ``|<ref|state>|^2``.
        """
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
            logger.info("----------------------------------------------------")
            logger.info(f"FIDELITY: {fidelity}")
        return fidelity

    def compute_expH(self, beta):
        """Return ``exp(-beta H)`` as a sparse CSC matrix."""
        return csc_matrix(expm(-beta * self.H.Ham))

    def get_thermal_beta(self, state, threshold):
        """Estimate an effective thermal ``beta`` for a reference state."""
        return self.H.get_beta(state, threshold)

    def canonical_avg(self, local_obs, beta):
        """Compute the canonical-ensemble average of a local observable.

        Parameters
        ----------
        local_obs : str
            Observable key in ``self.ops``.
        beta : float
            Inverse temperature.

        Returns
        -------
        float
            Canonical average per lattice site.
        """
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

    def microcanonical_avg(
        self,
        local_obs_list: list[str],
        state: np.ndarray,
        special_norms: dict | None = None,
        staggered_avgs: dict | None = None,
    ):
        """Compute microcanonical averages for multiple local observables.

        The energy shell is defined using the energy density of the reference
        state and its uncertainty:

        - ``e_q = <H> / L``
        - ``delta_e = sqrt(<H^2> - <H>^2) / L``

        All eigenstates satisfying ``abs(e_i - e_q) < delta_e`` are included.
        For each observable in ``local_obs_list``, the expectation value is
        measured on every state in the shell and then averaged.

        Parameters
        ----------
        local_obs_list : list
            Keys of local observables. Each key must be present in ``self.ops``.
        state : numpy.ndarray
            Reference state vector used to define the energy shell.
        special_norms : dict, optional
            Optional mapping from observable key to a custom normalization array.
        staggered_avgs : dict, optional
            Optional mapping from observable key to a staggered-averaging label
            or rule used by the local observable measurement.

        Returns
        -------
        tuple
            ``(psi_thermal, ME_avg)`` where ``psi_thermal`` is the normalized
            coherent superposition of shell eigenstates and ``ME_avg`` is a
            dictionary mapping observable names to microcanonical averages.
        """
        # Defaults for optional dicts
        if special_norms is None:
            special_norms = {obs: None for obs in local_obs_list}
        if staggered_avgs is None:
            staggered_avgs = {obs: None for obs in local_obs_list}
        logger.info("----------------------------------------------------")
        logger.info("MICRO-CANONICAL ENSEMBLE (MULTI-OBSERVABLE)")
        # Compute the reference energy and its variance for the energy shell.
        psi_ref = QMB_state(state)
        Eq = psi_ref.expectation_value(self.H.Ham)
        E2q = psi_ref.expectation_value(self.H.Ham @ self.H.Ham)
        DeltaE = np.sqrt(E2q - Eq**2)
        logger.info(f"Energy ref: {Eq}")
        logger.info(f"DeltaE: {DeltaE}")
        # Convert to energy densities to compare with self.H.Nenergies
        e_q = Eq / self.n_sites
        delta_e = DeltaE / self.n_sites
        """# Basic sanity: is the shell inside the covered spectrum?
        E_min = np.min(self.H.Nenergies)
        E_max = np.max(self.H.Nenergies)
        # Check that the energy shell is covered by our eigenstates.
        if e_q - delta_e < E_min or e_q + delta_e > E_max:
            shell_min = e_q - delta_e
            shell_max = e_q + delta_e
            msg = f"ME shell [{shell_min}, {shell_max}] not in [{E_min}, {E_max}]."
            raise ValueError(msg)"""
        # Pre-select eigenstate indices that are within the energy shell.
        shell_mask = np.abs(self.H.Nenergies - e_q) < delta_e
        shell_indices = np.where(shell_mask)[0]
        n_shell_states = len(shell_indices)
        if n_shell_states == 0:
            msg = f"ME shell empty: no eigstate |e_i-e_q|<delta_e={delta_e}."
            raise ValueError(msg)
        logger.info(f"Number of states in shell: {n_shell_states}")
        # Initialize the thermal state (for coherent superposition) and the observable accumulators.
        psi_thermal = np.zeros(self.H.Ham.shape[0], dtype=np.complex128)
        # Pre-build LocalTerm objects
        local_ops = {}
        for obs in local_obs_list:
            local_ops[obs] = LocalTerm(self.ops[obs], obs, **self.def_params)
        # Degine a dictionary to save the ME prediction for every obs
        ME_avg = {f"ME_{obs}": 0.0 for obs in local_obs_list}
        # Loop over the shell eigenstates.
        for ii in shell_indices:
            psi_thermal += self.H.Npsi[ii].psi
            # For each observable, build the local operator and measure its expectation value.
            for obs in local_obs_list:
                local_ops[obs].get_expval(self.H.Npsi[ii], print_values=False)
                if special_norms[obs] is not None:
                    val = np.dot(local_ops[obs].obs, special_norms[obs]) / self.n_sites
                elif staggered_avgs[obs] is not None:
                    val = self.stag_avg(local_ops[obs].obs, staggered_avgs[obs])
                else:
                    val = np.mean(local_ops[obs].obs)
                ME_avg[f"ME_{obs}"] += val
        # Normalize the coherent superposition
        psi_thermal /= np.sqrt(n_shell_states)
        # Normalize the averages (equal weights)
        for obs in local_obs_list:
            ME_avg[f"ME_{obs}"] /= n_shell_states
        # Print microcanonical averages
        logger.info(f"Microcanonical averages")
        for obs in local_obs_list:
            logger.info(f"{obs}: {ME_avg[f'ME_{obs}']}")
        return psi_thermal, ME_avg

    def diagonal_avg(
        self,
        local_obs_list: list[str],
        state: np.ndarray,
        special_norms: dict | None = None,
        staggered_avgs: dict | None = None,
        tol_deg: float = 1e-10,
    ):
        """Compute diagonal-ensemble averages for several local observables.

        Parameters
        ----------
        local_obs_list : list
            Observable keys in ``self.ops``.
        state : numpy.ndarray
            Reference state used to compute diagonal weights.
        special_norms : dict, optional
            Optional observable-specific normalization arrays.
        staggered_avgs : dict, optional
            Optional observable-specific staggered averaging labels.
        tol_deg : float, optional
            Tolerance used to group degenerate eigenvalues into blocks.

        Returns
        -------
        dict
            Dictionary of diagonal-ensemble averages keyed as ``DE_<obs>``.
        """
        logger.info("----------------------------------------------------")
        logger.info("DIAGONAL ENSEMBLE AVERAGE (MULTI-OBSERVABLE)")
        # Ensure full diagonalization is available.
        if self.n_eigs != self.H.Ham.shape[0]:
            msg = f"Need all H eigvals {self.H.Ham.shape[0]}, not only {self.n_eigs}"
            raise ValueError(msg)
        # Defaults for optional dicts
        if special_norms is None:
            special_norms = {obs: None for obs in local_obs_list}
        if staggered_avgs is None:
            staggered_avgs = {obs: None for obs in local_obs_list}
        # Pre-build LocalTerm objects
        local_ops = {}
        for obs in local_obs_list:
            local_ops[obs] = LocalTerm(self.ops[obs], obs, **self.def_params)
        # Initialize accumulators for the diagonal ensemble average.
        DE_avg = {f"DE_{obs}": 0.0 for obs in local_obs_list}
        # Detect degeneracy structure in the spectrum
        block_id, n_blocks = build_energy_block_ids(self.H.Nenergies, tol=tol_deg)
        # Quick check: any block size > 1?
        counts = np.bincount(block_id, minlength=n_blocks)
        has_degeneracy = np.any(counts > 1)
        logger.info(f"Degeneracy-aware DE: {has_degeneracy} (tol={tol_deg})")
        # Loop over every eigenstate; full diagonalization is assumed.
        if not has_degeneracy:
            for ii in range(self.n_eigs):
                # Fidelity of eigenstate ii.
                prob = self.measure_fidelity(state, ii, False)
                # For each observable, measure the expectation value in eigenstate ii.
                for obs in local_obs_list:
                    local_ops[obs].get_expval(self.H.Npsi[ii], print_values=False)
                    if special_norms[obs] is not None:
                        val = np.dot(local_ops[obs].obs, special_norms[obs])
                        val /= self.n_sites
                    elif staggered_avgs[obs] is not None:
                        val = self.stag_avg(local_ops[obs].obs, staggered_avgs[obs])
                    else:
                        # Default: use the average over all sites.
                        val = np.mean(local_ops[obs].obs)
                    DE_avg[f"DE_{obs}"] += prob * val
        else:
            logger.info("BLOCK DIAGONAL AVG")
            # block-aware path (correct with degeneracies)
            # Precompute overlaps coeffs_i = <E_i|psi0>
            coeffs = np.empty(self.n_eigs, dtype=np.complex128)
            for ii in range(self.n_eigs):
                coeffs[ii] = np.vdot(self.H.Npsi[ii].psi, state)
            # Loop blocks
            for block_idx in range(n_blocks):
                idxs = np.where(block_id == block_idx)[0]
                if idxs.size == 0:
                    continue
                # Build |Psi_b> = sum_{i in block} C_i |E_i>
                psi_block = np.zeros_like(state)
                for jj in idxs:
                    Ci = coeffs[jj]
                    if Ci != 0.0:
                        psi_block += Ci * self.H.Npsi[jj].psi
                # Skip if no weight in this block
                if np.vdot(psi_block, psi_block).real < 1e-14:
                    continue
                psi_block_state = QMB_state(psi_block)
                for obs in local_obs_list:
                    local_ops[obs].get_expval(psi_block_state, print_values=False)
                    if special_norms[obs] is not None:
                        val = np.dot(local_ops[obs].obs, special_norms[obs])
                        val /= self.n_sites
                    elif staggered_avgs[obs] is not None:
                        val = self.stag_avg(local_ops[obs].obs, staggered_avgs[obs])
                    else:
                        val = np.mean(local_ops[obs].obs)
                    # NOTE: do not normalize psi_block; this is correct:
                    # sum_b <Psi_b|O|Psi_b>
                    DE_avg[f"DE_{obs}"] += val
        # Log the individual averages.
        logger.info("Diagonal Ensemble Averages:")
        for obs in local_obs_list:
            logger.info(f"{obs}: {DE_avg[f'DE_{obs}']}")
        logger.info("----------------------------------------------------")
        return DE_avg

    def get_observables(
        self,
        local_obs=[],
        twobody_obs=[],
        plaquette_obs=[],
        nbody_obs=[],
        nbody_dist=[],
        twobody_axes=None,
    ):
        """Instantiate observable objects and cache them in ``self.obs_list``.

        Parameters
        ----------
        local_obs : list, optional
            Names of local observables.
        twobody_obs : list, optional
            List of two-body operator-name pairs.
        plaquette_obs : list, optional
            List of plaquette operator-name lists.
        nbody_obs : list, optional
            List of n-body operator-name lists.
        nbody_dist : list, optional
            Relative distances for each n-body observable.
        twobody_axes : list, optional
            Axis labels for each two-body observable.

        Returns
        -------
        None
        """
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
        """Measure all observables stored in ``self.obs_list`` on one state.

        Parameters
        ----------
        index : int
            Eigenstate or time-step index.
        dynamics : bool, optional
            If ``True``, measure on ``self.H.psi_time[index]``.

        Returns
        -------
        None
            Results are stored in ``self.res``.
        """
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

    def link_avg(self, obs_name):
        """Average a directional link observable over all valid lattice links.

        Parameters
        ----------
        obs_name : str
            Base observable name. Directional components are expected in
            ``self.res`` as ``f"{obs_name}_p<dir>"``.

        Returns
        -------
        float
            Average over all valid directed links.
        """
        avg = 0
        tmp = 0
        for ii in range(prod(self.lvals)):
            # Compute the corresponding coords
            coords = zig_zag(self.lvals, ii)
            for dir in self.directions:
                # Check if it admits a link in that direction according to the lattice geometry
                _, sites_list = get_neighbor_sites(
                    coords, self.lvals, dir, self.has_obc
                )
                if sites_list is not None:
                    avg += self.res[f"{obs_name}_p{dir}"][ii]
                    tmp += 1
        logger.debug(f"{tmp} effective links")
        return avg / tmp

    def stag_avg(self, arr_flat: np.ndarray, staggered_avg=None):
        """Average a lattice quantity on all, even, or odd staggered sites.

        Parameters
        ----------
        arr_flat : numpy.ndarray
            Data flattened in the model zig-zag site ordering.
        staggered_avg : str or None, optional
            ``None`` for the full average, or ``"even"`` / ``"odd"`` for a
            staggered sublattice average.

        Returns
        -------
        float
            Requested average value.
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


def apply_projection(
    projector: np.ndarray | csr_matrix | csr_matrix,
    operator: np.ndarray | csr_matrix | csr_matrix,
):
    """Project an operator onto the subspace spanned by a column projector.

    Parameters
    ----------
    projector : numpy.ndarray or scipy.sparse.csr_matrix
        Projector-like matrix of shape ``(N, k)``.
    operator : numpy.ndarray or scipy.sparse.csr_matrix
        Operator of shape ``(N, N)``.

    Returns
    -------
    numpy.ndarray or scipy.sparse.spmatrix
        Effective operator ``P^dagger O P`` in the same dense/sparse family as
        the inputs.

    Raises
    ------
    ValueError
        If the shapes are incompatible.
    TypeError
        If dense and sparse input types are mixed.
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
        raise TypeError("Operator and projector must be both dense or both sparse.")


def build_energy_block_ids(evals: np.ndarray, tol: float = 1e-10):
    """Group sorted eigenvalues into degenerate blocks.

    Parameters
    ----------
    evals : numpy.ndarray
        Sorted eigenvalues.
    tol : float, optional
        Energies separated by less than ``tol`` are treated as degenerate.

    Returns
    -------
    tuple
        ``(block_id, n_blocks)`` where ``block_id`` labels each eigenstate's
        degenerate block.
    """
    E = np.real_if_close(np.asarray(evals))
    N = E.shape[0]
    block_id = np.empty(N, dtype=np.int32)
    b = 0
    block_id[0] = b
    for kk in range(1, N):
        if abs(E[kk] - E[kk - 1]) < tol:
            block_id[kk] = b
        else:
            b += 1
            block_id[kk] = b
    return block_id, (b + 1)


def format_loc_dims(loc_dims: np.ndarray, lvals: list[int], pad: int = 3) -> str:
    """Log local Hilbert-space dimensions arranged on the lattice geometry.

    Parameters
    ----------
    loc_dims : numpy.ndarray
        Per-site local dimensions.
    lvals : list
        Lattice dimensions.
    pad : int, optional
        Field width used for alignment in the log output.

    Returns
    -------
    None
    """
    dim = len(lvals)
    if dim == 1:
        Lx = lvals[0]
        arr = np.asarray(loc_dims).reshape(Lx)
        logger.info(f"[{' '.join(f'{val:>{pad}d}' for val in arr)}]")
    if dim == 2:
        Lx, Ly = lvals
        arr = np.asarray(loc_dims).reshape(Ly, Lx)
        lines = []
        for yy in range(Ly):
            logger.info(f"[{' '.join(f'{val:>{pad}d}' for val in arr[yy])}]")
    if dim == 3:
        Lx, Ly, Lz = lvals
        arr = np.asarray(loc_dims).reshape(Lz, Ly, Lx)  # [z, y, x]
        block_sep = "   "  # spacing between z-layers
        lines = []
        # optional header indicating z layers
        header = block_sep.join(
            [f"z={zz}".ljust((pad + 1) * Lx - 1) for zz in range(Lz)]
        )
        lines.append(header)
        for yy in range(Ly):
            row_blocks = []
            for zz in range(Lz):
                layer = f"[{' '.join(f'{val:>{pad}d}' for val in arr[zz, yy])}]"
                row_blocks.append(layer)
            logger.info(f"{block_sep.join(row_blocks)}")
