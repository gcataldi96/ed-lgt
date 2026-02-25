import numpy as np
from numba import typed
from edlgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm, get_origin_surfaces
from .quantum_model import QuantumModel
from edlgt.operators import QED_dressed_site_operators, QED_gauge_invariant_states
import logging

logger = logging.getLogger(__name__)
__all__ = ["QED_plaq_Model"]


class QED_plaq_Model(QuantumModel):
    def __init__(self, spin, pure_theory, **kwargs):
        # Initialize base class with the common parameters
        super().__init__(**kwargs)
        if self.dim != 2:
            raise ValueError(f"This model is only defined in 2D")
        self.spin = spin
        self.pure_theory = pure_theory
        self.staggered_basis = False if self.pure_theory else True
        # -------------------------------------------------------------------------------
        # Acquire gauge invariant basis and states
        self.gauge_basis, self.gauge_states = QED_gauge_invariant_states(
            self.spin, self.pure_theory, lattice_dim=self.dim
        )
        # -------------------------------------------------------------------------------
        # Acquire operators
        ops = QED_dressed_site_operators(self.spin, self.pure_theory, self.dim)
        # Initialize the operators, local dimension and lattice labels
        self.project_operators(ops)
        # -------------------------------------------------------------------------------
        # GLOBAL SYMMETRIES
        global_ops = None
        global_sectors = None
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
            if self.dim == 2:
                # in 2D we have two lines through (0,0):
                line_of = {"x": "y", "y": "x"}
                for dir in self.directions:
                    sites = np.array(surfaces[line_of[dir]][1], dtype=np.uint8)
                    nbody_sites_list.append(sites)
                    nbody_ops.append(self.ops[f"E_p{dir}"])
        else:
            # no electric‐flux constraint in 1D, or in OBC or with matter
            nbody_sectors = None
            nbody_ops = None
            nbody_sites_list = None
        # GET SYMMETRY SECTOR
        self.get_abelian_symmetry_sector(
            global_ops=global_ops,
            global_sectors=global_sectors,
            link_ops=link_ops,
            link_sectors=link_sectors,
            nbody_ops=nbody_ops,
            nbody_sectors=nbody_sectors,
            nbody_sites_list=nbody_sites_list,
        )
        # DEFAULT PARAMS
        self.default_params()

    def build_Hamiltonian(self, g):
        logger.info("BUILDING HAMILTONIAN")
        # Hamiltonian Coefficients
        coeffs = {"E": g, "B": -1 / (2 * g)}
        h_terms = {}
        # -------------------------------------------------------------------------------
        # ELECTRIC ENERGY
        op_name = "Casimir_plaq"
        h_terms[op_name] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
        self.H.add_term(h_terms[op_name].get_Hamiltonian(strength=coeffs["E"]))
        # -------------------------------------------------------------------------------
        # MAGNETIC INTERACTION
        # Local plaquette term
        op_name = "Magnetic_plaq"
        h_terms[op_name] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
        self.H.add_term(h_terms[op_name].get_Hamiltonian(strength=coeffs["B"]))
        # Twobody plaquette terms
        # horizonal interaction
        op_names_list = ["L_C_px,py_C_my,px", "R_C_py,mx_C_px,px"]
        h_terms["x_plaq"] = TwoBodyTerm("x", op_list, op_names_list, **self.def_params)
        # vertical interaction
        op_names_list = ["B_C_px,py_C_py,mx", "T_C_my,px_C_px,px"]
        h_terms["y_plaq"] = TwoBodyTerm("y", op_list, op_names_list, **self.def_params)
        # Fourbody plaquette term
        op_names_list = ["C4_px,py", "C3_py,mx", "C1_my,px", "C2_mx,my"]
        op_list = [self.ops[op] for op in op_names_list]
        h_terms["plaq"] = PlaquetteTerm(
            ["x", "y"], op_list, op_names_list, **self.def_params
        )
        self.H.add_term(
            h_terms["plaq_xy"].get_Hamiltonian(strength=coeffs["B"], add_dagger=True)
        )
        # -------------------------------------------------------------------------------
        self.H.build(format=self.ham_format)
