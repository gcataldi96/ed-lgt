"""U(1) lattice gauge-theory (QED) model helpers."""

import numpy as np
from numba import typed
from edlgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm, NBodyTerm
from edlgt.modeling import check_link_symmetry, staggered_mask, get_origin_surfaces
from .quantum_model import QuantumModel
from scipy.sparse import identity
from edlgt.operators import (
    QED_dressed_site_operators,
    QED_gauge_invariant_states,
    QED_gauge_integrated_operators,
)
import logging

logger = logging.getLogger(__name__)
__all__ = ["QED_Model"]


class QED_Model(QuantumModel):
    """QED lattice gauge model with optional matter fields."""

    def __init__(
        self,
        spin,
        pure_theory,
        bg_list=None,
        plaq_basis=False,
        link_symmetries=True,
        get_only_bulk=False,
        **kwargs,
    ):
        """Initialize the QED model and construct its symmetry sector.

        Parameters
        ----------
        spin : float or str = integrated
            Gauge-link spin representation.
        pure_theory : bool
            If ``True``, build the pure-gauge theory (no matter fields).
        bg_list : list, optional
            Optional background-charge configuration used during local-basis
            projection.
        get_only_bulk : bool, optional
            Restrict gauge-invariant local states to bulk-compatible ones when
            supported by the operator factory.
        **kwargs
            Arguments forwarded to :class:`~edlgt.models.quantum_model.QuantumModel`.
        """
        # Initialize base class with the common parameters
        super().__init__(**kwargs)
        # Build local operator dictionaries in complex mode, then optionally
        # switch to real mode at Hamiltonian-assembly time when allowed.
        self.configure_dtype_mode(dtype_mode="complex", auto_mode="complex")
        # Check Background charge list
        if bg_list is None:
            self.bg_list = None
        else:
            static_charges = np.asarray(bg_list, dtype=float)
            if static_charges.shape != (self.n_sites,):
                msg = f"len(bg_list) must match n_sites = {self.n_sites}: got {static_charges.shape}"
                raise ValueError(msg)
            self.bg_list = static_charges.tolist()
        # Check spin truncation: it can be integer or "integrated"
        if isinstance(spin, str):
            if spin != "integrated":
                raise ValueError(f"spin must be integer or 'integrated': got {spin}")
            else:
                if not np.all([pure_theory == False, self.dim == 1]):
                    msg = "QED with integrated gauge fields is only in 1D with matter"
                    raise ValueError(msg)
            self.spin = spin
            self.pure_theory = pure_theory
            self.plaq_basis = False
            self.background = 0
            self.staggered_basis = False
            logger.info(f"----------------------------------------------------")
            logger.info(f"({self.dim}+1)D QED MODEL with gauge fields integrated")
            if self.bg_list is not None:
                logger.info(f"bg_list={self.bg_list}")
            ops = QED_gauge_integrated_operators()
            self.project_operators(ops)
        elif np.isscalar(spin):
            self.spin = spin
            self.pure_theory = pure_theory
            self.plaq_basis = plaq_basis
            self.link_symmetries = link_symmetries
            self.staggered_basis = False if self.pure_theory else True
            pure_label = "pure" if self.pure_theory else "with matter"
            logger.info(f"----------------------------------------------------")
            msg = f"({self.dim}+1)D QED MODEL {pure_label} j={spin}"
            logger.info(msg)
            self.background = 0
            if self.bg_list is not None:
                self.background = int(max(np.abs(self.bg_list)))
                if self.background != 0:
                    logger.info(f"bg_list={self.bg_list}")
                else:
                    self.bg_list = None
            # -------------------------------------------------------------------------------
            # Acquire gauge invariant basis and states
            self.gauge_basis, self.gauge_states = QED_gauge_invariant_states(
                self.spin,
                self.pure_theory,
                lattice_dim=self.dim,
                background=self.background,
                get_only_bulk=get_only_bulk,
            )
            # -------------------------------------------------------------------------------
            # Acquire operators
            ops = QED_dressed_site_operators(
                self.spin, self.pure_theory, self.dim, background=self.background
            )
            # Initialize the operators, local dimension and lattice labels
            self.project_operators(ops, self.bg_list)
        else:
            raise ValueError(f"spin must be integer or 'integrated': got {spin}")
        # -------------------------------------------------------------------------------
        # GLOBAL SYMMETRIES
        if self.pure_theory:
            global_ops = None
            global_sectors = None
        else:
            global_ops = [self.ops["N"]]
            global_sectors = [int(self.n_sites / 2)]
        # -------------------------------------------------------------------------------
        # LINK SYMMETRIES (only present in the truncated version of gauge fields)
        if self.spin == "integrated":
            link_ops = None
            link_sectors = None
        else:
            if not self.plaq_basis:
                if self.link_symmetries:
                    dirs = self.directions
                    link_ops = [
                        [self.ops[f"E_p{d}"], -self.ops[f"E_m{d}"]] for d in dirs
                    ]
                    link_sectors = [0 for _ in self.directions]
                else:
                    link_ops = None
                    link_sectors = None
            else:
                link_ops = [
                    [self.ops[f"E_plqt_p{d}1"], self.ops[f"E_plqt_m{d}1"]]
                    for d in self.directions
                ]
                link_ops += [
                    [self.ops[f"E_plqt_p{d}2"], self.ops[f"E_plqt_m{d}2"]]
                    for d in self.directions
                ]
                link_sectors = [0 for _ in self.directions]
                link_sectors += [0 for _ in self.directions]
        # ---------------------------------------------------------------------------
        # ELECTRIC-FLUX “NBODY” SYMMETRIES
        # only in the pure (no-matter) theory, more than 1D, *and* PBC
        # Constrain, for each cartesian direction, the corresponding
        # Electric flux on the face/line through the origin:
        # 3D: (Ex → yz-face at x=0) (Ey → xz-face at y=0) (Ez → xy-face at z=0)
        # 2D: (Ex → y-axis at x=0) (Ey → x-axis at y=0)
        if self.pure_theory and not any(self.has_obc):
            logger.info("fixing surface electric fluxes")
            # one flux‐constraint per cartesian direction
            nbody_sectors = np.zeros(self.dim, dtype=float)
            nbody_ops = []
            nbody_sites_list = typed.List()
            surfaces = get_origin_surfaces(self.lvals)
            nbody_sym_type = "U"
            if self.dim == 2:
                # in 2D we have two lines through (0,0):
                line_of = {"x": "y", "y": "x"}
                for dir in self.directions:
                    sites = np.array(surfaces[line_of[dir]][1], dtype=np.uint8)
                    nbody_sites_list.append(sites)
                    nbody_ops.append(self.ops[f"E_p{dir}"])
            elif self.dim == 3:
                # in 3D we have three faces through (0,0,0):
                face_of = {"x": "yz", "y": "xz", "z": "xy"}
                for dir in self.directions:
                    sites = np.array(surfaces[face_of[dir]][1], dtype=np.uint8)
                    nbody_sites_list.append(sites)
                    logger.debug(f"{dir} sites: {sites} {surfaces[face_of[dir]][0]}")
                    nbody_ops.append(self.ops[f"E_p{dir}"])
        else:
            # no electric‐flux constraint in 1D, or in OBC or with matter
            nbody_sectors = None
            nbody_ops = None
            nbody_sites_list = None
            nbody_sym_type = None
        # GET SYMMETRY SECTOR
        self.get_abelian_symmetry_sector(
            global_ops=global_ops,
            global_sectors=global_sectors,
            link_ops=link_ops,
            link_sectors=link_sectors,
            nbody_ops=nbody_ops,
            nbody_sectors=nbody_sectors,
            nbody_sites_list=nbody_sites_list,
            nbody_sym_type=nbody_sym_type,
        )
        # DEFAULT PARAMS
        self.default_params()

    def _get_auto_dtype_mode(self, m=None, theta=0.0):
        """Heuristic dtype mode for QED Hamiltonian assembly."""
        if not self.pure_theory:
            return "complex"
        if np.abs(theta) <= 1e-12:
            return "real"
        return "complex"

    def build_Hamiltonian(self, g, m=None, theta=0.0, dtype_mode="auto"):
        """Dispatch to the appropriate QED Hamiltonian builder.

        Parameters
        ----------
        dtype_mode : str or bool, optional
            ``"auto"``, ``"real"``, ``"complex"``, or legacy bool flag.
        """
        resolved_mode = self.configure_dtype_mode(
            dtype_mode=dtype_mode,
            auto_mode=self._get_auto_dtype_mode(m=m, theta=theta),
        )
        if not self.plaq_basis:
            if self.spin == "integrated":
                return self.build_integrated_Hamiltonian(
                    g, m, theta, dtype_mode=resolved_mode
                )
            else:
                return self.build_truncated_Hamiltonian(
                    g, m, theta, dtype_mode=resolved_mode
                )
        else:
            return self.build_plaquette_Hamiltonian(g, dtype_mode=resolved_mode)

    def build_truncated_Hamiltonian(self, g, m=None, theta=0.0, dtype_mode="auto"):
        """Assemble the QED Hamiltonian.

        Parameters
        ----------
        g : float
            Gauge coupling.
        m : float, optional
            Bare fermion mass (used only when matter is present).
        theta : float, optional
            Topological-angle coupling parameter.
        dtype_mode : str or bool, optional
            ``"auto"``, ``"real"``, ``"complex"``, or legacy bool flag.
        """
        self.configure_dtype_mode(
            dtype_mode=dtype_mode,
            auto_mode=self._get_auto_dtype_mode(m=m, theta=theta),
        )
        logger.info(f"----------------------------------------------------")
        logger.info("BUILDING truncated HAMILTONIAN")
        # Hamiltonian Coefficients
        self.QED_Hamiltonian_couplings(g, m, theta)
        h_terms = {}
        # -------------------------------------------------------------------------------
        # ELECTRIC ENERGY
        op_name = "E2"
        h_terms["E2"] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
        self.H.add_term(h_terms["E2"].get_Hamiltonian(strength=self.coeffs["E"]))
        # -------------------------------------------------------------------------------
        # PLAQUETTE TERM: MAGNETIC INTERACTION
        if self.dim > 1:
            op_names_list = ["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]
            op_list = [self.ops[op] for op in op_names_list]
            h_terms["plaq_xy"] = PlaquetteTerm(
                ["x", "y"], op_list, op_names_list, **self.def_params
            )
            self.H.add_term(
                h_terms["plaq_xy"].get_Hamiltonian(
                    strength=self.coeffs["B"], add_dagger=True
                )
            )
        if self.dim == 3:
            # XZ Plane
            op_names_list = ["C_px,pz", "C_pz,mx", "C_mz,px", "C_mx,mz"]
            op_list = [self.ops[op] for op in op_names_list]
            h_terms["plaq_xz"] = PlaquetteTerm(
                ["x", "z"], op_list, op_names_list, **self.def_params
            )
            self.H.add_term(
                h_terms["plaq_xz"].get_Hamiltonian(
                    strength=self.coeffs["B"], add_dagger=True
                )
            )
            # YZ Plane
            op_names_list = ["C_py,pz", "C_pz,my", "C_mz,py", "C_my,mz"]
            op_list = [self.ops[op] for op in op_names_list]
            h_terms["plaq_yz"] = PlaquetteTerm(
                ["y", "z"], op_list, op_names_list, **self.def_params
            )
            self.H.add_term(
                h_terms["plaq_yz"].get_Hamiltonian(
                    strength=self.coeffs["B"], add_dagger=True
                )
            )
        # -------------------------------------------------------------------------------
        # TOPOLOGICAL TERM
        # -------------------------------------------------------------------------------
        if self.dim == 2 and np.abs(self.coeffs["theta"]) > 10e-10:
            logger.info("Adding topological term")
            op_names_list = ["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]
            op_list = [self.ops[op] for op in op_names_list]
            h_terms["plaq_xy"] = PlaquetteTerm(
                ["x", "y"], op_list, op_names_list, **self.def_params
            )
            self.H.add_term(
                h_terms["plaq_xy"].get_Hamiltonian(
                    strength=self.coeffs["theta"], add_dagger=True
                )
            )
        if self.dim == 3 and np.abs(self.coeffs["theta"]) > 10e-10:
            logger.info("Adding topological term")
            # XY Plane
            op_names_list = ["EzC_px,py", "C_py,mx", "C_my,px", "C_mx,my"]
            op_list = [self.ops[op] for op in op_names_list]
            h_terms["Ez_Bxy"] = PlaquetteTerm(
                ["x", "y"], op_list, op_names_list, **self.def_params
            )
            self.H.add_term(
                h_terms["Ez_Bxy"].get_Hamiltonian(
                    strength=self.coeffs["theta"], add_dagger=True
                )
            )
            # XZ Plane
            op_names_list = ["EyC_px,pz", "C_pz,mx", "C_mz,px", "C_mx,mz"]
            op_list = [self.ops[op] for op in op_names_list]
            h_terms["Ey_Bxz"] = PlaquetteTerm(
                ["x", "z"], op_list, op_names_list, **self.def_params
            )
            self.H.add_term(
                h_terms["Ey_Bxz"].get_Hamiltonian(
                    strength=self.coeffs["theta"], add_dagger=True
                )
            )
            # YZ Plane
            op_names_list = ["ExC_py,pz", "C_pz,my", "C_mz,py", "C_my,mz"]
            op_list = [self.ops[op] for op in op_names_list]
            h_terms["Ex_Byz"] = PlaquetteTerm(
                ["y", "z"], op_list, op_names_list, **self.def_params
            )
            self.H.add_term(
                h_terms["Ex_Byz"].get_Hamiltonian(
                    strength=self.coeffs["theta"], add_dagger=True
                )
            )
        # -------------------------------------------------------------------------------
        if not self.pure_theory:
            # ---------------------------------------------------------------------------
            # STAGGERED MASS TERM
            for stag_label in ["even", "odd"]:
                h_terms[f"N_{stag_label}"] = LocalTerm(
                    operator=self.ops["N"], op_name="N", **self.def_params
                )
                self.H.add_term(
                    h_terms[f"N_{stag_label}"].get_Hamiltonian(
                        self.coeffs[f"m_{stag_label}"],
                        staggered_mask(self.lvals, stag_label),
                    )
                )
            # ---------------------------------------------------------------------------
            # HOPPING
            for d in self.directions:
                for stag_label in ["even", "odd"]:
                    # Define the list of the 2 non trivial operators
                    op_names_list = [f"Q_p{d}_dag", f"Q_m{d}"]
                    op_list = [self.ops[op] for op in op_names_list]
                    # Define the Hamiltonian term
                    h_terms[f"{d}_hop_{stag_label}"] = TwoBodyTerm(
                        d, op_list, op_names_list, **self.def_params
                    )
                    self.H.add_term(
                        h_terms[f"{d}_hop_{stag_label}"].get_Hamiltonian(
                            strength=self.coeffs[f"t{d}_{stag_label}"],
                            add_dagger=True,
                            mask=staggered_mask(self.lvals, stag_label),
                        )
                    )
        # ---------------------------------------------------------------------------
        # APPLY LINK PENALTIES IN CASE LINK SYMMETRIES ARE NOT APPLIED
        if not self.link_symmetries:
            self.H.add_term(h_terms["E2"].get_Hamiltonian(strength=100))
            for d in self.directions:
                # Define the list of the 2 non trivial operators
                op_names_list = [f"E_p{d}", f"E_m{d}"]
                op_list = [self.ops[op] for op in op_names_list]
                # Define the Hamiltonian term
                h_terms["penalty"] = TwoBodyTerm(
                    d, op_list, op_names_list, **self.def_params
                )
                self.H.add_term(h_terms["penalty"].get_Hamiltonian(strength=-100))
        self.H.build(format=self.ham_format)

    def build_integrated_Hamiltonian(self, g, m, theta=0.0, dtype_mode="auto"):
        """Assemble the integrated-gauge 1D QED Hamiltonian.

        Parameters
        ----------
        dtype_mode : str or bool, optional
            ``"auto"``, ``"real"``, ``"complex"``, or legacy bool flag.
        """
        self.configure_dtype_mode(dtype_mode=dtype_mode, auto_mode="complex")
        # Hamiltonian Coefficients
        self.QED_Hamiltonian_couplings(g, m, theta)
        logger.info("BUILDING integrated HAMILTONIAN")
        if self.dim != 1 or self.pure_theory:
            msg = "Integrated QED Hamiltonian is defined only in 1D with matter."
            raise ValueError(msg)
        h_terms = {}
        n_links = self.n_sites - 1
        # ---------------------------------------------------------------------------
        # STAGGERED MASS TERM
        h_terms["N"] = LocalTerm(operator=self.ops["N"], op_name="N", **self.def_params)
        for _, stag_label in enumerate(["even", "odd"]):
            mask = staggered_mask(self.lvals, stag_label)
            mass_coeff = self.coeffs[f"m_{stag_label}"]
            self.H.add_term(h_terms["N"].get_Hamiltonian(mass_coeff, mask))
        # ---------------------------------------------------------------------------
        # HOPPING TERM
        op_names_list = ["Sp", "Sm"]
        op_list = [self.ops[op] for op in op_names_list]
        # Define the Hamiltonian term
        h_terms["hop"] = TwoBodyTerm("x", op_list, op_names_list, **self.def_params)
        self.H.add_term(
            h_terms["hop"].get_Hamiltonian(strength=self.coeffs["t"], add_dagger=True)
        )
        # ---------------------------------------------------------------------------
        # ELECTRIC ENERGY: long-range Hamiltonian
        # Eq. structure:
        # H_E = g^2/2 * sum_{n=0}^{L-2}[sum_{k=0}^{n}(q_k + (Sz_k + (-1)^k)/2)]^2
        #     = sum_i h_i * Sz_i + sum_{i<j} J_ij * Sz_i * Sz_j + const
        static_charges = self._integrated_static_charges()
        stagger_sign = np.ones(self.n_sites, dtype=float)
        for ii in range(self.n_sites):
            if ii % 2 == 1:
                stagger_sign[ii] = -1.0
        charge_offsets = static_charges + 0.5 * stagger_sign
        # A[n] constant factors
        prefix_charge = np.cumsum(charge_offsets)
        # ---------------------------------------------------------------------------
        # One-body coefficients h_i = E * sum_{n=i}^{L-2} prefix_charge[n]
        linear_sz_coeff = np.zeros(self.n_sites, dtype=float)
        for kk in range(n_links):
            linear_sz_coeff[kk] = self.coeffs["E"] * np.sum(prefix_charge[kk:n_links])
        # Build and cache single-site masks once
        site_masks = [self._single_site_mask(ii) for ii in range(self.n_sites)]
        # One-body Sz part
        h_terms["Sz"] = LocalTerm(
            operator=self.ops["Sz"], op_name="Sz", **self.def_params
        )
        for ii in range(self.n_sites):
            coeff_ii = linear_sz_coeff[ii]
            if np.abs(coeff_ii) < 1e-14:
                continue
            self.H.add_term(
                h_terms["Sz"].get_Hamiltonian(strength=coeff_ii, mask=site_masks[ii])
            )
        # ---------------------------------------------------------------------------
        # Two-body Sz_i Sz_j part with J_ij = g^2/2 * 0.5 * (L-1-j), i<j
        szsz_terms = {}
        op_list = [self.ops["Sz"], self.ops["Sz"]]
        op_names_list = ["Sz", "Sz"]
        for dist in range(1, self.n_sites):
            szsz_terms[dist] = NBodyTerm(
                op_list, op_names_list, distances=[(dist,)], **self.def_params
            )
        for jj in range(1, n_links):
            right_weight = 0.5 * self.coeffs["E"] * (self.n_sites - 1 - jj)
            if np.abs(right_weight) > 1e-14:
                for ii in range(jj):
                    dist = jj - ii
                    self.H.add_term(
                        szsz_terms[dist].get_Hamiltonian(
                            strength=right_weight, mask=site_masks[ii]
                        )
                    )
        # ---------------------------------------------------------------------------
        # Constant electric energy term:
        # E * sum_n [ A_n^2 + (1/4) * sum_{k<=n} (Sz_k)^2 ], with A_n as prefix_charge
        # and (Sz_k)^2 = I for Pauli operators.
        if n_links > 0:
            sum_sz2_counts = 0.25 * 0.5 * n_links * (n_links + 1)
            electric_constant = np.sum(prefix_charge[:n_links] ** 2) + sum_sz2_counts
            electric_constant *= self.coeffs["E"]
        else:
            electric_constant = 0.0
        if np.abs(electric_constant) > 1e-14:
            hdim = self.H.shape[0]
            self.H.add_term(float(electric_constant) * identity(hdim, format="csc"))
        self.H.build(format=self.ham_format)

    def _integrated_static_charges(self) -> np.ndarray:
        """Return static charges q_n for integrated-QED sectors."""
        if self.bg_list is None:
            return np.zeros(self.n_sites, dtype=float)
        static_charges = np.asarray(self.bg_list, dtype=float)
        if static_charges.shape != (self.n_sites,):
            logger.info("Integrated QED static charges must have one value per site")
            msg = f"expected shape ({self.n_sites},), got {static_charges.shape}."
            raise ValueError(msg)
        return static_charges

    def _single_site_mask(self, site_index: int) -> np.ndarray:
        """Boolean mask selecting one 1D lattice site."""
        site_mask = np.zeros(tuple(self.lvals), dtype=np.bool_)
        site_mask[(site_index,)] = True
        return site_mask

    def reconstruct_integrated_E2_from_N(
        self,
        density_obs_name: str = "N",
        density_corr_obs_name: str = "N_N",
        state_index: int | None = None,
        dynamics: bool = False,
        compute_density_corr: bool = True,
        print_values: bool = True,
    ) -> np.ndarray:
        """Reconstruct link-resolved ``<E^2>`` in integrated 1D QED from matter density.

        Parameters
        ----------
        density_obs_name : str, optional
            Key in ``self.res`` containing measured site-resolved ``<N_k>``.
        density_corr_obs_name : str, optional
            Key in ``self.res`` containing measured two-point correlator
            ``<N_k N_l>``.
        state_index : int or None, optional
            Eigenstate index used to compute ``<N_k N_l>`` on the fly when
            ``density_corr_obs_name`` is missing and ``compute_density_corr=True``.
            If ``None`` and only one eigenstate is available, index ``0`` is used.
        dynamics : bool, optional
            If ``True``, interpret ``state_index`` as a time index and compute
            missing correlators on ``self.H.psi_time[state_index]`` instead of
            ``self.H.Npsi[state_index]``.
        compute_density_corr : bool, optional
            If ``True``, compute ``<N_k N_l>`` when not already present in
            ``self.res``. If ``False``, missing correlations raise ``KeyError``.
        print_values : bool, optional
            If ``True``, print link-resolved reconstructed values and their
            average using the same style as local observable measurements.

        Returns
        -------
        numpy.ndarray
            Link-resolved reconstructed Casimir values ``<E_{k,k+1}^2>`` with
            shape ``(n_sites - 1,)``.

        Notes
        -----
        The reconstruction uses

        ``E_n = sum_{k=0}^n [ q_k + N_k + ((-1)^k-1)/2 ]``.

        Therefore:

        ``<E_n^2> = B_n^2 + 2 B_n sum_{k<=n}<N_k> + sum_{k,l<=n}<N_k N_l>``,

        where ``B_n = sum_{k=0}^n [q_k + ((-1)^k-1)/2]``.
        """
        if self.spin != "integrated" or self.dim != 1 or self.pure_theory:
            msg = "Integrated E2 reconstruction is defined only for 1D QED with matter"
            raise ValueError(msg)
        if density_obs_name not in self.res:
            msg = f"Missing '{density_obs_name}' in self.res. Measure local density first."
            raise KeyError(msg)
        density_values = np.asarray(self.res[density_obs_name], dtype=float)
        if density_values.shape != (self.n_sites,):
            msg = f"Expected {density_obs_name}.shape=({self.n_sites},) got {density_values.shape}."
            raise ValueError(msg)
        n_links = self.n_sites - 1
        if n_links <= 0:
            link_casimir = np.zeros(0, dtype=float)
            self.res["E2"] = link_casimir
            self.res["E2_avg"] = 0.0
            return link_casimir
        # Acquire or compute density correlator <N_k N_l>
        if density_corr_obs_name in self.res:
            density_corr = np.asarray(self.res[density_corr_obs_name], dtype=float)
        else:
            if not compute_density_corr:
                raise KeyError(
                    f"Missing '{density_corr_obs_name}' in self.res. "
                    "Measure ['N','N_N'] or set compute_density_corr=True."
                )
            if not hasattr(self, "H"):
                raise ValueError(
                    "Cannot compute density correlator: Hamiltonian not available."
                )
            if dynamics:
                if not hasattr(self.H, "psi_time"):
                    raise ValueError(
                        "Cannot compute density correlator in dynamics mode: "
                        "time-evolved states (H.psi_time) are not available."
                    )
                if state_index is None:
                    if len(self.H.psi_time) == 1:
                        state_index = 0
                    else:
                        raise ValueError(
                            "state_index is required to compute density correlator "
                            "when multiple time steps are available."
                        )
                target_state = self.H.psi_time[state_index]
            else:
                if not hasattr(self.H, "Npsi"):
                    raise ValueError(
                        "Cannot compute density correlator: Hamiltonian eigenstates not available."
                    )
                if state_index is None:
                    if len(self.H.Npsi) == 1:
                        state_index = 0
                    else:
                        raise ValueError(
                            "state_index is required to compute density correlator "
                            "when multiple eigenstates are available."
                        )
                target_state = self.H.Npsi[state_index]
            op_list = [self.ops["N"], self.ops["N"]]
            op_names_list = ["N", "N"]
            density_corr_term = TwoBodyTerm(
                "x", op_list=op_list, op_names_list=op_names_list, **self.def_params
            )
            density_corr_term.get_expval(target_state)
            density_corr = np.asarray(density_corr_term.corr, dtype=float)
            self.res[density_corr_obs_name] = density_corr
        if density_corr.shape != (self.n_sites, self.n_sites):
            raise ValueError(
                f"Expected '{density_corr_obs_name}' shape ({self.n_sites}, {self.n_sites}), "
                f"got {density_corr.shape}."
            )
        density_corr = np.array(density_corr, dtype=float, copy=True)
        # For fermions N_k^2 = N_k, so force the diagonal explicitly.
        np.fill_diagonal(density_corr, density_values)
        static_charges = self._integrated_static_charges()
        site_indices = np.arange(self.n_sites, dtype=float)
        staggered_offsets = 0.5 * ((-1.0) ** site_indices - 1.0)
        charge_offsets = static_charges + staggered_offsets
        prefix_charge = np.cumsum(charge_offsets)
        prefix_density = np.cumsum(density_values)
        # Prefix sums of <N_k N_l>: top-left block sum at link n is [n,n].
        density_corr_prefix = density_corr.cumsum(axis=0).cumsum(axis=1)
        prefix_density_corr = np.diag(density_corr_prefix)
        link_casimir = (
            prefix_charge[:n_links] ** 2
            + 2.0 * prefix_charge[:n_links] * prefix_density[:n_links]
            + prefix_density_corr[:n_links]
        )
        self.res["E2"] = link_casimir
        self.res["E2_avg"] = float(np.mean(link_casimir))
        if print_values:
            logger.info("----------------------------------------------------")
            logger.info("E2 ")
            for ii in range(n_links):
                logger.info(f"{(ii,)} {format(link_casimir[ii], '.16f')}")
            logger.info(f"{format(self.res['E2_avg'], '.16f')}")
        return link_casimir

    def build_plaquette_Hamiltonian(self, g, dtype_mode="auto"):
        """Assemble the plaquette-basis QED Hamiltonian.

        Parameters
        ----------
        dtype_mode : str or bool, optional
            ``"auto"``, ``"real"``, ``"complex"``, or legacy bool flag.
        """
        self.configure_dtype_mode(
            dtype_mode=dtype_mode,
            auto_mode=self._get_auto_dtype_mode(m=None, theta=0.0),
        )
        logger.info("BUILDING plaquette HAMILTONIAN")
        # Hamiltonian Coefficients
        self.coeffs = {"E": g, "B": -1 / (2 * g)}
        h_terms = {}
        assert self.dim == 2, "Plaquette Hamiltonian only defined for dim >=2"
        # -------------------------------------------------------------------------------
        # ELECTRIC ENERGY
        for op_name in ["E2_plq", "E2_plq_px", "E2_plq_py"]:
            h_terms[op_name] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
            self.H.add_term(h_terms[op_name].get_Hamiltonian(strength=self.coeffs["E"]))
        # -------------------------------------------------------------------------------
        # MAGNETIC ENERGY
        # PLAQUETTE TERM: MAGNETIC INTERACTION
        h_terms["B2_plq"] = LocalTerm(self.ops["B2_plq"], "B2_plq", **self.def_params)
        self.H.add_term(h_terms[op_name].get_Hamiltonian(strength=self.coeffs["B"]))
        for d in self.directions:
            # Define the list of the 2 non trivial operators
            op_names_list = [f"B2_plq_p{d}", f"B2_plq_m{d}"]
            op_list = [self.ops[op] for op in op_names_list]
            # Define the Hamiltonian term
            h_terms[f"plq_{d}"] = TwoBodyTerm(
                d, op_list, op_names_list, **self.def_params
            )
            self.H.add_term(
                h_terms[f"plq_{d}"].get_Hamiltonian(
                    strength=self.coeffs["B"], add_dagger=True
                )
            )
        # Plaquette operator between sites:
        op_names_list = ["B2_plq_px_py", "B2_plq_mx_py", "B2_plq_px_my", "B2_plq_mx_my"]
        op_list = [self.ops[op] for op in op_names_list]
        h_terms["B2_plaq_xy"] = PlaquetteTerm(
            ["x", "y"], op_list, op_names_list, **self.def_params
        )
        self.H.add_term(
            h_terms["B2_plaq_xy"].get_Hamiltonian(
                strength=self.coeffs["B"], add_dagger=True
            )
        )
        self.H.build(format=self.ham_format)

    def check_symmetries(self):
        """Check link-symmetry constraints on measured electric fields."""
        # CHECK LINK SYMMETRIES
        for ax in self.directions:
            check_link_symmetry(
                ax,
                self.obs_list[f"E_p{ax}"],
                self.obs_list[f"E_m{ax}"],
                value=0,
                sign=1,
            )

    def QED_Hamiltonian_couplings(self, g, m=None, theta=0.0, magnetic_basis=False):
        """Set QED Hamiltonian couplings from physical parameters.

        Parameters
        ----------
        g : float
            Gauge coupling.
        m : float, optional
            Bare mass parameter (used only with matter).
        theta : float, optional
            Topological-angle parameter.
        magnetic_basis : bool, optional
            If ``True``, use the alternative magnetic-basis normalization.

        Returns
        -------
        None
            Stores couplings in ``self.coeffs``.
        """
        if not magnetic_basis:
            if self.dim == 1:
                E = g / 2
                B = -1 / (2 * g)
            elif self.dim == 2:
                E = g / 2
                B = -1 / (2 * g)
            else:
                E = g / 2
                B = -1 / (2 * g)
            # DICTIONARY WITH MODEL COEFFICIENTS
            coeffs = {
                "g": g,
                "E": E,  # ELECTRIC FIELD coupling
                "B": B,  # MAGNETIC FIELD coupling
                "theta": -theta * g,  # THETA TERM coupling
            }
            if not self.pure_theory and m is not None:
                t = 1 / 2
                coeffs |= {
                    "t": -complex(0, t),  # integrated-gauge hopping coefficient
                    "tx_even": -complex(0, t),  # x HOPPING (EVEN SITES)
                    "tx_odd": -complex(0, t),  # x HOPPING (ODD SITES)
                    "ty_even": -t,  # y HOPPING (EVEN SITES)
                    "ty_odd": t,  # y HOPPING (ODD SITES)
                    "tz_even": -complex(0, t),  # z HOPPING (EVEN SITES)
                    "tz_odd": complex(0, t),  # z HOPPING (ODD SITES)
                    "m": m,
                    "m_odd": -m,  # EFFECTIVE MASS for ODD SITES
                    "m_even": m,  # EFFECTIVE MASS for EVEN SITES
                }
        else:
            # DICTIONARY WITH MODEL COEFFICIENTS
            coeffs = {
                "g": g,
                "E": -g,
                "B": -0.5 / g,
            }
        self.coeffs = coeffs

    def overlap_QMB_state(self, name):
        """Return predefined benchmark basis configurations for selected labels.

        Parameters
        ----------
        name : str
            Label of a predefined reference configuration.

        Returns
        -------
        numpy.ndarray
            Configuration in the model symmetry-sector basis.
        """
        # ---------------------------------------------------------------------------
        # 1D QED with matter: simple benchmark labels
        #   V            : vacuum
        #   meson_center : local meson in the central (even, odd) pair
        # Backward-compatible alias:
        #   PV -> V
        if self.dim == 1 and not self.pure_theory:
            name_norm = name.strip().lower().replace(" ", "_")
            if name_norm == "pv":
                name_norm = "v"
            if name_norm in ("v", "meson_center"):
                # Integrated case (matter-only local basis: 0/1 occupancies)
                if self.spin == "integrated":
                    config_state = np.zeros(self.n_sites, dtype=np.int32)
                    # Integrated Pauli convention:
                    # basis index 0 -> N=1 (occupied), index 1 -> N=0 (empty).
                    # Vacuum occupancy (even empty, odd occupied) is [1,0,1,0,...].
                    for ii in range(self.n_sites):
                        config_state[ii] = 1 if (ii % 2 == 0) else 0
                    if name_norm == "meson_center":
                        even_site = self.n_sites // 2
                        if even_site % 2 == 1:
                            even_site -= 1
                        even_site = max(0, min(even_site, self.n_sites - 2))
                        odd_site = even_site + 1
                        # Meson on central (even, odd) pair relative to V.
                        config_state[even_site] = 0
                        config_state[odd_site] = 1
                    return config_state
                # Truncated case (hard-coded local-state labels for spin=1)
                if np.isscalar(self.spin) and int(self.spin) == 1:
                    config_state = np.zeros(self.n_sites, dtype=np.int32)
                    if all(self.has_obc):
                        # OBC:
                        # left edge = 0, right edge = 1
                        # core: even -> 1, odd -> 3
                        if self.n_sites >= 1:
                            config_state[0] = 0
                        for ii in range(1, self.n_sites - 1):
                            config_state[ii] = 1 if (ii % 2 == 0) else 3
                        if self.n_sites >= 2:
                            config_state[self.n_sites - 1] = 1
                    else:
                        # PBC: alternance 1,3
                        for ii in range(self.n_sites):
                            config_state[ii] = 1 if (ii % 2 == 0) else 3
                    if name_norm == "meson_center":
                        even_site = self.n_sites // 2
                        if even_site % 2 == 1:
                            even_site -= 1
                        even_site = max(0, min(even_site, self.n_sites - 2))
                        odd_site = even_site + 1
                        config_state[even_site] = 4
                        config_state[odd_site] = 1
                    return config_state
                raise NotImplementedError(
                    "1D labels 'V'/'meson_center' are implemented for "
                    "spin='integrated' or truncated spin=1."
                )
        # MINIMAL STRING CONFIGURATIONS IN 3D QED WITH BACKGROUND charges
        if len(self.lvals) == 3 and self.bg_list == [-1, 0, 0, 0, 0, 0, 0, 1]:
            if self.spin == 1:
                if name == "S1":
                    config_state = np.array([4, 9, 9, 3, 6, 0, 3, 10])
                elif name == "S2":
                    config_state = np.array([3, 9, 6, 0, 9, 3, 3, 11])
                elif name == "S3":
                    config_state = np.array([1, 7, 9, 3, 9, 2, 3, 10])
                elif name == "S4":
                    config_state = np.array([4, 9, 9, 3, 7, 3, 0, 8])
                elif name == "S5":
                    config_state = np.array([1, 6, 9, 2, 9, 3, 3, 11])
                elif name == "S6":
                    config_state = np.array([3, 9, 7, 3, 9, 3, 2, 8])
                else:
                    raise ValueError(f"Unknown String name: {name}")
            elif self.spin == 2:
                if name == "S1":
                    config_state = np.array([11, 27, 27, 9, 22, 4, 9, 29])
                elif name == "S2":
                    config_state = np.array([10, 27, 22, 4, 27, 9, 9, 30])
                elif name == "S3":
                    config_state = np.array([10, 27, 23, 9, 27, 9, 8, 25])
                elif name == "S4":
                    config_state = np.array([6, 22, 27, 8, 27, 9, 9, 30])
                elif name == "S5":
                    config_state = np.array([6, 23, 27, 9, 27, 8, 9, 29])
                elif name == "S6":
                    config_state = np.array([11, 27, 27, 9, 23, 9, 4, 25])
                else:
                    raise ValueError(f"Unknown String name: {name}")
            elif self.spin == 3:
                if name == "S1":
                    config_state = np.array([20, 54, 47, 11, 54, 18, 18, 58])
                elif name == "S2":
                    config_state = np.array([21, 54, 54, 18, 47, 11, 18, 57])
                elif name == "S3":
                    config_state = np.array([21, 54, 54, 18, 48, 18, 11, 51])
                elif name == "S4":
                    config_state = np.array([20, 54, 48, 18, 54, 18, 17, 51])
                elif name == "S5":
                    config_state = np.array([14, 48, 54, 18, 54, 17, 18, 57])
                elif name == "S6":
                    config_state = np.array([14, 47, 54, 17, 54, 18, 18, 58])
                else:
                    raise ValueError(f"Unknown String name: {name}")
            return config_state
        else:
            raise NotImplementedError("Only 3D QED states are supported")

    def print_state_config(self, config, amplitude=None):
        """Log a readable per-site decomposition of a QED basis configuration."""
        if self.spin == "integrated":
            raise NotImplementedError("It does not work with spin='integrated'")
        logger.info(f"----------------------------------------------------")
        msg = f"CONFIG {config}"
        if amplitude is not None:
            msg += f" |psi|^2={np.abs(amplitude)**2:.8f}"
        logger.info(msg)
        logger.info(f"----------------------------------------------------")
        # Choose width (sign + digits).
        max_abs = 1
        max_abs = max(max_abs, int(abs(getattr(self, "spin", 1))))
        max_abs = max(max_abs, int(abs(getattr(self, "background", 0))))
        entry_width = len(str(max_abs)) + 2
        logger.info(f"{'':>19s}{self._format_header(entry_width)}")
        for ii, cfg_idx in enumerate(config):
            loc_basis_state = self.gauge_states_per_site[ii][cfg_idx]
            state_str = (
                "["
                + " ".join(f"{val:>{entry_width}d}" for val in loc_basis_state)
                + " ]"
            )
            logger.info(f"SITE {ii:>2d} state {cfg_idx:>3d}: {state_str}")

    def _local_state_labels(self) -> list[str]:
        """Return column labels used by :meth:`print_state_config`."""
        labels: list[str] = []
        # Background first (only if present in your effective gauge states)
        if getattr(self, "background", 0) > 0:
            labels.append("bg")
        # Matter occupation (only if not pure theory)
        if not getattr(self, "pure_theory", True):
            labels.append("M")
        # Links: -x,-y,-z,+x,+y,+z up to dim
        dim = int(getattr(self, "dim", 0))
        dirs = "xyz"[:dim]
        labels += [f"-{d}" for d in dirs] + [f"+{d}" for d in dirs]
        return labels

    def _format_header(self, entry_width: int) -> str:
        """Format the header row for :meth:`print_state_config`."""
        labels = self._local_state_labels()
        header = "[" + " ".join(f"{lab:>{entry_width}s}" for lab in labels) + " ]"
        return header
