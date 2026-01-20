import numpy as np
from numba import typed
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm
from ed_lgt.modeling import check_link_symmetry, staggered_mask, get_origin_surfaces
from .quantum_model import QuantumModel
from ed_lgt.operators import QED_dressed_site_operators,QED_plq_site_operators,QED_gauge_invariant_states
import logging

logger = logging.getLogger(__name__)
__all__ = ["QED_Model"]


class QED_Model(QuantumModel):
    def __init__(self, spin, pure_theory, get_only_bulk=False, **kwargs):
        #TODO flag for dressed or plaquette 
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

    def build_Hamiltonian(self, g, m=None, theta=0.0):
        logger.info("BUILDING HAMILTONIAN")
        # Hamiltonian Coefficients
        self.QED_Hamiltonian_couplings(g, m, theta)
        h_terms = {}
        # -------------------------------------------------------------------------------
        # ELECTRIC ENERGY
        op_name = "E2"
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


    def build_plq_Hamiltonian(self, g, m=None, theta=0.0):
        logger.info("BUILDING HAMILTONIAN")
        # Hamiltonian Coefficients
        self.QED_Hamiltonian_couplings(g, m, theta)
        h_terms = {}
        
        assert self.dim==2, "Plaquette Hamiltonian only defined for dim >=2"
        # -------------------------------------------------------------------------------
        # ELECTRIC ENERGY
        for op_name in ["E2_plq","E2_plq_px","E2_plq_py"]:
            h_terms[op_name] = LocalTerm(self.ops_plqt[op_name], op_name, **self.def_params)
            self.H.add_term(h_terms[op_name].get_Hamiltonian(strength=self.coeffs["E"]))
            
        # -------------------------------------------------------------------------------
        # MAGNETIC ENERGY 
        # PLAQUETTE TERM: MAGNETIC INTERACTION
        h_terms["B2_plq"] = LocalTerm(self.ops_plqt["B2_plq"], "B2_plq", **self.def_params)
        self.H.add_term(h_terms[op_name].get_Hamiltonian(strength=self.coeffs["B"]))
            
        for d in self.directions:
            # Define the list of the 2 non trivial operators
            op_names_list = [f"B2_plq_p{d}", f"B2_plq_m{d}"] 
            op_list = [self.ops[op] for op in op_names_list]
            # Define the Hamiltonian term
            h_terms[f"{d}_hop"] = TwoBodyTerm(
                d, op_list, op_names_list, **self.def_params
            )
            self.H.add_term(
                h_terms[f"{d}_hop"].get_Hamiltonian(
                strength=self.coeffs["B"],
                add_dagger=True,
                )) 
                
                
        #Plaquette operator between sites: 
        op_names_list = ["B2_plq_px_py","B2_plq_mx_py","B2_plq_mx_my","B2_plq_px_my"]
        op_list = [self.ops[op] for op in op_names_list]
        h_terms["plaq_xy"] = PlaquetteTerm(
            ["x", "y"], op_list, op_names_list, **self.def_params
        )
        self.H.add_term(
            h_terms["plaq_xy"].get_Hamiltonian(
                strength=self.coeffs["B"], add_dagger=True
            )
        )
            
        self.H.build(format=self.ham_format)
        
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

    def QED_Hamiltonian_couplings(self, g, m=None, theta=0.0, magnetic_basis=False):
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
                coeffs |= {
                    "m": m,
                    "tx_even": 0.5,  # HORIZONTAL HOPPING
                    "tx_odd": 0.5,
                    "ty_even": 0.5,  # VERTICAL HOPPING (EVEN SITES)
                    "ty_odd": -0.5,  # VERTICAL HOPPING (ODD SITES)
                    "tz_even": 0.5,  # VERTICAL HOPPING (EVEN SITES)
                    "tz_odd": 0.5,  # VERTICAL HOPPING (ODD SITES)
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
