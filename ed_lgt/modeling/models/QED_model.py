from ed_lgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm, QMB_hamiltonian
from ed_lgt.modeling import check_link_symmetry, staggered_mask
from .quantum_model import QuantumModel
from ed_lgt.operators import QED_dressed_site_operators, QED_gauge_invariant_states

__all__ = ["QED_Model"]


class QED_Model(QuantumModel):
    def __init__(self, spin, pure_theory, **kwargs):
        # Initialize base class with the common parameters
        super().__init__(**kwargs)
        self.spin = spin
        self.pure_theory = pure_theory
        self.staggered_basis = False if self.pure_theory else True
        # Acquire operators
        self.ops = QED_dressed_site_operators(
            self.spin, self.pure_theory, U="ladder", lattice_dim=self.dim
        )
        # Acquire gauge invariant basis and states
        self.gauge_basis, self.gauge_states = QED_gauge_invariant_states(
            self.spin, self.pure_theory, lattice_dim=self.dim
        )
        # Acquire local dimension and lattice label
        self.get_local_site_dimensions()
        # GLOBAL SYMMETRIES
        if self.pure_theory:
            global_ops = None
            global_sectors = None
        else:
            global_ops = [self.ops["N"]]
            global_sectors = [int(self.n_sites / 2)]
        # LINK SYMMETRIES
        link_ops = [[self.ops[f"E_p{d}"], self.ops[f"E_m{d}"]] for d in self.directions]
        link_sectors = [0 for _ in self.directions]
        # GET SYMMETRY SECTOR
        self.get_abelian_symmetry_sector(
            global_ops=global_ops,
            global_sectors=global_sectors,
            link_ops=link_ops,
            link_sectors=link_sectors,
        )
        # DEFAULT PARAMS
        self.default_params()

    def build_Hamiltonian(self, coeffs):
        # Hamiltonian Coefficients
        self.coeffs = coeffs
        h_terms = {}
        # -------------------------------------------------------------------------------
        # ELECTRIC ENERGY
        op_name = "E_square"
        h_terms[op_name] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
        self.H.Ham += h_terms[op_name].get_Hamiltonian(strength=self.coeffs["E"])
        # -------------------------------------------------------------------------------
        # PLAQUETTE TERM: MAGNETIC INTERACTION
        if self.dim > 1:
            op_names_list = ["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]
            op_list = [self.ops[op] for op in op_names_list]
            h_terms["plaq_xy"] = PlaquetteTerm(
                ["x", "y"], op_list, op_names_list, **self.def_params
            )
            self.H.Ham += h_terms["plaq_xy"].get_Hamiltonian(
                strength=self.coeffs["B"], add_dagger=True
            )
        if self.dim == 3:
            # XZ Plane
            op_names_list = ["C_px,pz", "C_pz,mx", "C_mz,px", "C_mx,mz"]
            op_list = [self.ops[op] for op in op_names_list]
            h_terms["plaq_xz"] = PlaquetteTerm(
                ["x", "z"], op_list, op_names_list, **self.def_params
            )
            self.H.Ham += h_terms["plaq_xz"].get_Hamiltonian(
                strength=self.coeffs["B"], add_dagger=True
            )
            # YZ Plane
            op_names_list = ["C_py,pz", "C_pz,my", "C_mz,py", "C_my,mz"]
            op_list = [self.ops[op] for op in op_names_list]
            h_terms["plaq_yz"] = PlaquetteTerm(
                ["y", "z"], op_list, op_names_list, **self.def_params
            )
            self.H.Ham += h_terms["plaq_yz"].get_Hamiltonian(
                strength=self.coeffs["B"], add_dagger=True
            )
        # -------------------------------------------------------------------------------
        if not self.pure_theory:
            # ---------------------------------------------------------------------------
            # STAGGERED MASS TERM
            for stag_label in ["even", "odd"]:
                h_terms[f"N_{stag_label}"] = LocalTerm(
                    operator=self.ops["N"], op_name="N", **self.def_params
                )
                self.H.Ham += h_terms[f"N_{stag_label}"].get_Hamiltonian(
                    coeffs[f"m_{stag_label}"], staggered_mask(self.lvals, stag_label)
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
                    self.H.Ham += h_terms[f"{d}_hop_{stag_label}"].get_Hamiltonian(
                        strength=self.coeffs[f"t{d}_{stag_label}"],
                        add_dagger=True,
                        mask=staggered_mask(self.lvals, stag_label),
                    )

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
