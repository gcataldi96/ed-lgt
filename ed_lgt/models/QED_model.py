import numpy as np
from numba import typed
from scipy.sparse import csr_matrix, kron as sp_kron, identity as sp_identity
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm
from ed_lgt.modeling import check_link_symmetry, staggered_mask, get_origin_surfaces
from .quantum_model import QuantumModel
from ed_lgt.operators import QED_dressed_site_operators, QED_gauge_invariant_states
from ed_lgt.modeling import QMB_liouvillian
import logging

logger = logging.getLogger(__name__)
__all__ = ["QED_Model"]


class QED_Model(QuantumModel):
    def __init__(self, spin, pure_theory, get_only_bulk=False, **kwargs):
        # Initialize base class with the common parameters
        super().__init__(**kwargs)
        self.spin = spin
        self.pure_theory = pure_theory
        self.staggered_basis = False if self.pure_theory else True
        # -------------------------------------------------------------------------------
        # Acquire gauge invariant basis and states
        self.gauge_basis, self.gauge_states = QED_gauge_invariant_states(
            self.spin,
            self.pure_theory,
            lattice_dim=self.dim,
            get_only_bulk=get_only_bulk,
        )
        # -------------------------------------------------------------------------------
        # Acquire operators
        ops = QED_dressed_site_operators(self.spin, self.pure_theory, self.dim)
        # Initialize the operators, local dimension and lattice labels
        self.project_operators(ops)
        # -------------------------------------------------------------------------------
        # GLOBAL SYMMETRIES
        if self.pure_theory:
            global_ops = None
            global_sectors = None
        else:
            global_ops = [self.ops["N"]]
            global_sectors = [int(self.n_sites / 2)]
        # -------------------------------------------------------------------------------
        # LINK SYMMETRIES
        link_ops = [[self.ops[f"E_p{d}"], self.ops[f"E_m{d}"]] for d in self.directions]
        link_sectors = [0 for _ in self.directions]
        """
        # -------------------------------------------------------------------------------
        ELECTRIC-FLUX “NBODY” SYMMETRIES ———
        only in the pure (no-matter) theory, more than 1D, *and* PBC
        Constrain, for each cartesian direction, the corresponding 
        Electric flux on the face/line through the origin:
        3D:
            for 'Ex' → the yz-face at x=0
            for 'Ey' → the xz-face at y=0
            for 'Ez' → the xy-face at z=0
        2D:
            for 'Ex' → the y-axis at x=0
            for 'Ey' → the x-axis at y=0
        """
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

    def build_Hamiltonian(self, g, m=None, t_coef=1 / 2, theta=0.0):
        logger.info("BUILDING HAMILTONIAN")
        # Hamiltonian Coefficients
        self.QED_Hamiltonian_couplings(g, m, t_coef, theta)
        h_terms = {}
        # -------------------------------------------------------------------------------
        # ELECTRIC ENERGY
        op_name = "E_square"
        h_terms[op_name] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
        self.H.add_term(h_terms[op_name].get_Hamiltonian(strength=self.coeffs["E"]))
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
                    strength=-self.coeffs["theta"], add_dagger=True
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
        self.H.build(format=self.ham_format)

    # TODO: For now it will only work for 1 dimension
    # NOTE: It is NOT more efficient to write a Numba/JIT function to do the commutators. the csr@csr is well optimized. What we could
    # do to optimize the commutators is to not do the commutator with the whole hamiltonian but only the relevant part OR, I think
    # ideally we could just write the theoretical result of the commutator I got but I need Giovanni's help to understand how to
    # use the Q operators
    def build_Liouvillian(
        self,
        g,
        m=None,
        t_coef=1 / 2,
        theta=0.0,
        beta=0.1,
        D0_coef=0,
        corr_type="Delta",
        sigma=0,
        lamb_shift_existence=False,
    ):
        logger.info("BUILDING LIOUVILLIAN")
        # Initialize the Liouvillian
        hilbertDim = self.H.shape[0]
        liouDim = self.H.shape[0] ** 2
        self.L = QMB_liouvillian(self.lvals, size=liouDim)
        T_coef = 1 / beta

        # Get Hamiltonian
        self.build_Hamiltonian(
            g, m, t_coef=t_coef, theta=theta
        )  # Now self.H.Ham, = The hamiltonian, but also self.H.(value, row, col)_list
        self.L.effH = self.H.Ham.copy()

        # Get Jump Operators [For SU(2) the for loop would also go over color dof]
        N_matter = self.lvals[0]
        jump_ops = {}
        for n in range(N_matter):
            single_site_mask = np.arange(N_matter) == n

            # Create zeroth order jump operator O(n)
            temp_op = LocalTerm(operator=self.ops["N"], op_name="N", **self.def_params)
            r_list, c_list, v_list = temp_op.get_Hamiltonian(
                strength=(-1) ** n, mask=single_site_mask
            )
            jump_ops[f"0th_{n}"] = csr_matrix(
                (v_list, (r_list, c_list)), shape=(hilbertDim, hilbertDim)
            )

            # Create second order jump operator L(n)
            # TODO: ASK HOW TO GET ONLY THE HOPPING TERM TO MAKE THIS COMMUTATOR SIMPLER
            jump_ops[f"2nd_{n}"] = jump_ops[f"0th_{n}"] - (
                1 / (4 * T_coef)
            ) * commutator(self.H.Ham, jump_ops[f"0th_{n}"])

        # Get the environment correlator
        env_corr = self.L.get_environment_correlator(corr_type, D0_coef, sigma)

        # Get Lamb Shift Term
        if lamb_shift_existence:
            lamb_shift = csr_matrix((hilbertDim, hilbertDim), dtype=complex)
            for n in range(N_matter):
                for m in range(N_matter):
                    if env_corr[n, m] != 0:
                        lamb_shift += (env_corr[n, m] / 2) * anticommutator(
                            jump_ops[f"0th_{n}"],
                            commutator(self.H.Ham, jump_ops[f"0th_{m}"]),
                        )
            self.L.effH += (1j / (8 * T_coef)) * lamb_shift

        # -------------------------------Construct Liouvillian------------------------------------
        # TODO: It is possible to construct this next section in a function with numba and avoid using sp_kron all together
        # Construct the hermitian part
        identity = sp_identity(hilbertDim, format="csr")
        Liouville_op = -1j * (
            sp_kron(self.L.effH, identity) - sp_kron(identity, (self.L.effH).T)
        )

        # Construct the non-hermitian part
        for n in range(N_matter):
            Ln_dag = jump_ops[f"2nd_{n}"].conj().T
            for m in range(N_matter):
                if env_corr[n, m] != 0:
                    Lm = jump_ops[f"2nd_{m}"]
                    Ln_dag_Lm = Ln_dag @ Lm

                    term1 = sp_kron(Lm, Ln_dag.T)  # L_m ⊗ L_n^dag^T
                    term2 = sp_kron(Ln_dag_Lm, identity)  # L_n^dag L_m ⊗ I
                    term3 = sp_kron(identity, (Ln_dag_Lm).T)  # I ⊗ (L_n^dag L_m)^T

                    Liouville_op += (env_corr[n, m] / 2) * (2 * term1 - term2 - term3)
            print(n, end=" ")
        self.L.add_term(Liouville_op)
        # --------------------------------------------------------------------------------------

        # Build Liouvillian
        self.L.build(format=self.ham_format)

    def check_symmetries(self):
        # CHECK LINK SYMMETRIES
        for ax in self.directions:
            check_link_symmetry(
                ax,
                self.obs_list[f"E_p{ax}"],
                self.obs_list[f"E_m{ax}"],
                value=0,
                sign=1,
            )

    def QED_Hamiltonian_couplings(
        self, g, m=None, t_coef=1 / 2, theta=0.0, magnetic_basis=False
    ):
        """
        This function provides the QED Hamiltonian coefficients
        starting from the gauge coupling g and the bare mass parameter m

        Args:
            pure_theory (bool): True if the theory does not include matter

            g (scalar): gauge coupling

            m (scalar, optional): bare mass parameter

        Returns:
            dict: dictionary of Hamiltonian coefficients
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
                "theta": -complex(0, theta * g),  # THETA TERM coupling
            }
            if m is not None:
                t = t_coef
                coeffs |= {
                    "m": m,
                    "tx_even": t,  # HORIZONTAL HOPPING
                    "tx_odd": t,
                    "ty_even": t,  # VERTICAL HOPPING (EVEN SITES)
                    "ty_odd": -t,  # VERTICAL HOPPING (ODD SITES)
                    "tz_even": t,  # VERTICAL HOPPING (EVEN SITES)
                    "tz_odd": t,  # VERTICAL HOPPING (ODD SITES)
                    "m_even": m,
                    "m_odd": -m,
                }
        else:
            # DICTIONARY WITH MODEL COEFFICIENTS
            coeffs = {
                "g": g,
                "E": -g,
                "B": -0.5 / g,
            }
        self.coeffs = coeffs


# Shorthand for commutator
def commutator(A, B):
    return A @ B - B @ A


# Shorthand for anticommutator
def anticommutator(A, B):
    return A @ B + B @ A
