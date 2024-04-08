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
        """
        # ------------------------------------------------------------------------
        # HOPPING
        for d in self.directions:
            for site in ["even", "odd"]:
                hopping_terms = [
                    [f"Q1_p{d}_dag", f"Q2_m{d}"],
                    [f"Q2_p{d}_dag", f"Q1_m{d}"],
                ]
                for op_names_list in hopping_terms:
                    op_list = [self.ops[op] for op in op_names_list]
                    # Define the Hamiltonian term
                    h_terms[f"{d}_hop_{site}"] = TwoBodyTerm(
                        d, op_list, op_names_list, **self.def_params
                    )
                    self.H.Ham += h_terms[f"{d}_hop_{site}"].get_Hamiltonian(
                        strength=coeffs[f"t{d}_{site}"],
                        add_dagger=True,
                        mask=staggered_mask(self.lvals, site),
                    )

        # -------------------------------------------------------------------------------
        # PLAQUETTE TERM: MAGNETIC INTERACTION
        plaq_list = []
        plaquette_directions = ["xy", "xz", "yz"]
        plaquette_set = [
            ["AB", "AB", "AB", "AB"],
            ["AA", "AB", "BB", "AB"],
            ["AB", "AB", "AA", "BB"],
            ["AB", "AB", "AA", "BB"],
            ["AB", "BB", "AB", "AA"],
            ["AA", "BB", "BB", "AA"],
            ["AB", "BB", "AA", "BA"],
            ["AA", "BB", "BA", "BA"],
            ["BB", "AA", "AB", "AB"],
            ["BA", "AA", "BB", "AB"],
            ["BB", "AA", "AA", "BB"],
            ["BA", "AA", "BA", "BB"],
            ["BB", "BA", "AB", "AA"],
            ["BA", "BA", "AB", "AA"],
            ["BB", "BA", "AA", "BA"],
            ["BA", "BA", "BA", "BA"],
        ]
        plaquette_signs = [
            -1,
            +1,
            +1,
            -1,
            +1,
            -1,
            -1,
            +1,
            +1,
            -1,
            -1,
            +1,
            -1,
            +1,
            +1,
            -1,
        ]
        for ii, pdir in enumerate(plaquette_directions):
            if (self.dim == 2 and ii == 0) or self.dim == 3:
                for jj, p_set in enumerate(plaquette_set):
                    # DEFINE THE LIST OF CORNER OPERATORS
                    op_names_list = [
                        f"C{p_set[0]}_p{pdir[0]},p{pdir[1]}",
                        f"C{p_set[1]}_p{pdir[1]},m{pdir[0]}",
                        f"C{p_set[2]}_m{pdir[1]},p{pdir[0]}",
                        f"C{p_set[3]}_m{pdir[0]},m{pdir[1]}",
                    ]
                    # CORRESPONDING LIST OF OPERATORS
                    op_list = [self.ops[op] for op in op_names_list]
                    # DEFINE THE PLAQUETTE CLASS
                    plaq_name = f"P_{pdir}_" + "".join(p_set)
                    h_terms[plaq_name] = PlaquetteTerm(
                        [pdir[0], pdir[1]],
                        op_list,
                        op_names_list,
                        print_plaq=False,
                        **self.def_params,
                    )
                    # ADD THE HAMILTONIAN TERM
                    self.H.Ham += h_terms[plaq_name].get_Hamiltonian(
                        strength=plaquette_signs[jj] * coeffs["B"], add_dagger=True
                    )
                    # ADD THE PLAQUETTE TO THE LIST OF OBSERVABLES
                    plaq_list.append(plaq_name)
        """

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
