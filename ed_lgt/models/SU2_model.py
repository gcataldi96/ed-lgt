import numpy as np
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm, QMB_hamiltonian
from ed_lgt.modeling import check_link_symmetry, staggered_mask
from .quantum_model import QuantumModel
from ed_lgt.operators import SU2_dressed_site_operators, SU2_gauge_invariant_states
import logging

logger = logging.getLogger(__name__)
__all__ = ["SU2_Model"]


class SU2_Model(QuantumModel):
    def __init__(self, spin, pure_theory, **kwargs):
        # Initialize base class with the common parameters
        super().__init__(**kwargs)
        self.spin = spin
        self.pure_theory = pure_theory
        self.staggered_basis = False
        # Acquire operators
        self.ops = SU2_dressed_site_operators(
            self.spin, self.pure_theory, lattice_dim=self.dim
        )
        # Acquire gauge invariant basis and states
        self.gauge_basis, self.gauge_states = SU2_gauge_invariant_states(
            self.spin, self.pure_theory, lattice_dim=self.dim
        )
        # Acquire local dimension and lattice label
        self.get_local_site_dimensions()
        # GLOBAL SYMMETRIES
        if self.pure_theory:
            global_ops = None
            global_sectors = None
        else:
            global_ops = [self.ops["N_tot"]]
            global_sectors = [self.n_sites]
        # LINK SYMMETRIES
        link_ops = [
            [self.ops[f"T2_p{d}"], -self.ops[f"T2_m{d}"]] for d in self.directions
        ]
        link_sectors = [0 for _ in self.directions]
        # GET SYMMETRY SECTOR
        self.get_abelian_symmetry_sector(
            global_ops=global_ops,
            global_sectors=global_sectors,
            link_ops=link_ops,
            link_sectors=link_sectors,
        )
        self.default_params()

    def build_Hamiltonian(self, coeffs):
        # Hamiltonian Coefficients
        self.coeffs = coeffs
        # CONSTRUCT THE HAMILTONIAN
        self.H = QMB_hamiltonian(0, self.lvals, self.loc_dims)
        h_terms = {}
        # ---------------------------------------------------------------------------
        # ELECTRIC ENERGY
        op_name = "E_square"
        h_terms[op_name] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
        self.H.Ham += h_terms[op_name].get_Hamiltonian(strength=coeffs["E"])
        # -------------------------------------------------------------------------------
        # PLAQUETTE TERM: MAGNETIC INTERACTION
        if self.dim > 1:
            op_names_list = ["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]
            op_list = [self.ops[op] for op in op_names_list]
            h_terms["plaq_xy"] = PlaquetteTerm(
                ["x", "y"], op_list, op_names_list, **self.def_params
            )
            self.H.Ham += h_terms["plaq_xy"].get_Hamiltonian(
                strength=-self.coeffs["B"], add_dagger=True
            )
        if self.dim == 3:
            # XZ Plane
            op_names_list = ["C_px,pz", "C_pz,mx", "C_mz,px", "C_mx,mz"]
            op_list = [self.ops[op] for op in op_names_list]
            h_terms["plaq_xz"] = PlaquetteTerm(
                ["x", "z"], op_list, op_names_list, **self.def_params
            )
            self.H.Ham += h_terms["plaq_xz"].get_Hamiltonian(
                strength=-self.coeffs["B"], add_dagger=True
            )
            # YZ Plane
            op_names_list = ["C_py,pz", "C_pz,my", "C_mz,py", "C_my,mz"]
            op_list = [self.ops[op] for op in op_names_list]
            h_terms["plaq_yz"] = PlaquetteTerm(
                ["y", "z"], op_list, op_names_list, **self.def_params
            )
            self.H.Ham += h_terms["plaq_yz"].get_Hamiltonian(
                strength=-self.coeffs["B"], add_dagger=True
            )
        # ---------------------------------------------------------------------------
        if not self.pure_theory:
            # -----------------------------------------------------------------------
            # STAGGERED MASS TERM
            for site in ["even", "odd"]:
                h_terms[f"N_{site}"] = LocalTerm(
                    self.ops["N_tot"], "N_tot", **self.def_params
                )
                self.H.Ham += h_terms[f"N_{site}"].get_Hamiltonian(
                    coeffs[f"m_{site}"], staggered_mask(self.lvals, site)
                )
            # -----------------------------------------------------------------------
            # HOPPING
            for d in self.directions:
                for site in ["even", "odd"]:
                    op_names_list = [f"Qp{d}_dag", f"Qm{d}"]
                    op_list = [self.ops[op] for op in op_names_list]
                    # Define the Hamiltonian term
                    h_terms[f"{d}_hop_{site}"] = TwoBodyTerm(
                        d, op_list, op_names_list, **self.def_params
                    )
                    mask = staggered_mask(self.lvals, site)
                    self.H.Ham += h_terms[f"{d}_hop_{site}"].get_Hamiltonian(
                        strength=coeffs[f"t{d}_{site}"],
                        add_dagger=True,
                        mask=mask,
                    )

    def check_symmetries(self):
        # CHECK LINK SYMMETRIES
        for ax in self.directions:
            check_link_symmetry(
                ax,
                self.obs_list[f"T2_p{ax}"],
                self.obs_list[f"T2_m{ax}"],
                value=0,
                sign=-1,
            )

    def overlap_QMB_state(self, name):
        if name == "V":
            s1, s2, L, R = 0, 4, 0, 2
        elif name == "PV":
            s1, s2, L, R = 1, 5, 1, 1
        elif name == "M":
            s1, s2, L, R = 3, 2, 1, 1
        config_state = [s1 if ii % 2 == 0 else s2 for ii in range(self.n_sites)]
        if self.has_obc[0]:
            config_state[0] = L
            config_state[-1] = R
        config_state = config_state
        return np.array(config_state)
