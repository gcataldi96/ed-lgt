import numpy as np
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm
from ed_lgt.modeling import check_link_symmetry, staggered_mask
from .quantum_model import QuantumModel
from ed_lgt.operators import (
    SU2_gen_dressed_site_operators,
    SU2_gen_gauge_invariant_states,
)
import logging

logger = logging.getLogger(__name__)
__all__ = ["SU2_Model_Gen"]


class SU2_Model_Gen(QuantumModel):
    def __init__(self, spin, pure_theory, background, **kwargs):
        # Initialize base class with the common parameters
        super().__init__(**kwargs)
        self.spin = spin
        self.pure_theory = pure_theory
        self.background = background
        self.staggered_basis = False
        # Acquire operators
        self.ops = SU2_gen_dressed_site_operators(
            self.spin,
            self.pure_theory,
            lattice_dim=self.dim,
            background=self.background,
        )
        # Acquire gauge invariant basis and states
        self.gauge_basis, self.gauge_states = SU2_gen_gauge_invariant_states(
            self.spin,
            self.pure_theory,
            lattice_dim=self.dim,
            background=self.background,
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
            # --------------------------------------------------------------------
            # Generalized HOPPING
            for d in self.directions:
                for site in ["even", "odd"]:
                    hopping_terms = [
                        [f"Q1_p{d}_dag", f"Q2_m{d}"],
                        [f"Q2_p{d}_dag", f"Q1_m{d}"],
                    ]
                    for ii, op_names_list in enumerate(hopping_terms):
                        op_list = [self.ops[op] for op in op_names_list]
                        # Define the Hamiltonian term
                        h_terms[f"{d}{ii}_hop_{site}"] = TwoBodyTerm(
                            d, op_list, op_names_list, **self.def_params
                        )
                        mask = staggered_mask(self.lvals, site)
                        self.H.Ham += h_terms[f"{d}{ii}_hop_{site}"].get_Hamiltonian(
                            strength=coeffs[f"t{d}_{site}"],
                            add_dagger=True,
                            mask=mask,
                        )

        # -------------------------------------------------------------------------------
        # PLAQUETTE TERM: MAGNETIC INTERACTION
        plaq_list = []
        plaquette_directions = ["xy", "xz", "yz"]
        plaquette_set = [
            ["AB", "AB", "AB", "AB"],
            ["AA", "AB", "BB", "AB"],
            ["AB", "AB", "AA", "BB"],
            ["AA", "AB", "BA", "BB"],
            ["AB", "BB", "AB", "AA"],
            ["AA", "BB", "BB", "AA"],
            ["AB", "BB", "AA", "BA"],
            ["AA", "BB", "BA", "BA"],
            ["BB", "AA", "AB", "AB"],
            ["BA", "AA", "BB", "AB"],
            ["BB", "AA", "AA", "BB"],
            ["BA", "AA", "BA", "BB"],
            ["BB", "BA", "AB", "AA"],
            ["BA", "BA", "BB", "AA"],
            ["BB", "BA", "AA", "BA"],
            ["BA", "BA", "BA", "BA"],
        ]
        for ii, pdir in enumerate(plaquette_directions):
            if (self.dim > 1 and ii == 0) or self.dim == 3:
                for p_set in plaquette_set:
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
                    plaq_name = f"P{pdir}_" + "".join(p_set)
                    h_terms[plaq_name] = PlaquetteTerm(
                        [pdir[0], pdir[1]],
                        op_list,
                        op_names_list,
                        print_plaq=False,
                        **self.def_params,
                    )
                    # ADD THE HAMILTONIAN TERM
                    self.H.Ham += h_terms[plaq_name].get_Hamiltonian(
                        strength=-coeffs["B"], add_dagger=True
                    )
                    # ADD THE PLAQUETTE TO THE LIST OF OBSERVABLES
                    plaq_list.append(plaq_name)

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
            if self.spin == 1:
                s1, s2, L, R = 0, 7, 0, 2
            elif self.spin < 1:
                s1, s2, L, R = 0, 4, 0, 2
        elif name == "PV":
            if self.spin == 1:
                s1, s2, L, R = 1, 8, 1, 1
            elif self.spin < 1:
                s1, s2, L, R = 1, 5, 1, 1
        config_state = [s1 if ii % 2 == 0 else s2 for ii in range(self.n_sites)]
        if self.has_obc[0]:
            config_state[0] = L
            config_state[-1] = R
        config_state = config_state
        return np.array(config_state)
