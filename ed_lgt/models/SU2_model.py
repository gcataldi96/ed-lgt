import numpy as np
from .quantum_model import QuantumModel
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm
from ed_lgt.modeling import check_link_symmetry, staggered_mask
from ed_lgt.operators import (
    SU2_dressed_site_operators,
    SU2_gauge_invariant_states,
    SU2_gen_dressed_site_operators,
)
import logging

logger = logging.getLogger(__name__)
__all__ = ["SU2_Model"]


class SU2_Model(QuantumModel):
    def __init__(self, spin, pure_theory, background, sectors=None, **kwargs):
        # Initialize base class with the common parameters
        super().__init__(**kwargs)
        self.spin = spin
        self.pure_theory = pure_theory
        self.background = background
        if self.spin > 3:
            # Acquire operators
            self.ops = SU2_dressed_site_operators(
                self.spin,
                self.pure_theory,
                lattice_dim=self.dim,
                background=self.background,
            )
        else:
            # Acquire operators
            self.ops = SU2_gen_dressed_site_operators(
                self.spin,
                self.pure_theory,
                lattice_dim=self.dim,
                background=self.background,
            )
        # Acquire gauge invariant basis and states
        self.gauge_basis, self.gauge_states = SU2_gauge_invariant_states(
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
            global_sectors = sectors
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

    def build_Hamiltonian(self, g, m=None):
        logger.info("BUILDING s=1/2 HAMILTONIAN")
        # Hamiltonian Coefficients
        self.SU2_Hamiltonian_couplings(g, m)
        h_terms = {}
        # ---------------------------------------------------------------------------
        # ELECTRIC ENERGY
        h_terms["E_square"] = LocalTerm(
            self.ops["E_square"], "E_square", **self.def_params
        )
        self.H.add_term(h_terms["E_square"].get_Hamiltonian(strength=self.coeffs["E"]))
        # ---------------------------------------------------------------------------
        # PLAQUETTE TERM: MAGNETIC INTERACTION
        if self.dim > 1:
            op_names_list = ["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]
            op_list = [self.ops[op] for op in op_names_list]
            h_terms["plaq_xy"] = PlaquetteTerm(
                ["x", "y"], op_list, op_names_list, **self.def_params
            )
            self.H.add_term(
                h_terms["plaq_xy"].get_Hamiltonian(
                    strength=-self.coeffs["B"], add_dagger=True
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
                    strength=-self.coeffs["B"], add_dagger=True
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
                    strength=-self.coeffs["B"], add_dagger=True
                )
            )
        # ---------------------------------------------------------------------------
        if not self.pure_theory:
            # -----------------------------------------------------------------------
            # STAGGERED MASS TERM
            for site in ["even", "odd"]:
                h_terms[f"N_{site}"] = LocalTerm(
                    self.ops["N_tot"], "N_tot", **self.def_params
                )
                self.H.add_term(
                    h_terms[f"N_{site}"].get_Hamiltonian(
                        self.coeffs[f"m_{site}"], staggered_mask(self.lvals, site)
                    )
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
                    self.H.add_term(
                        h_terms[f"{d}_hop_{site}"].get_Hamiltonian(
                            strength=self.coeffs[f"t{d}_{site}"],
                            add_dagger=True,
                            mask=mask,
                        )
                    )
        self.H.build(self.ham_format)

    def build_gen_Hamiltonian(self, g, m=None):
        logger.info("BUILDING generalized HAMILTONIAN")
        # Hamiltonian Coefficients
        self.SU2_Hamiltonian_couplings(g, m)
        h_terms = {}
        # ---------------------------------------------------------------------------
        # ELECTRIC ENERGY
        op_name = "E_square"
        h_terms[op_name] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
        self.H.add_term(h_terms[op_name].get_Hamiltonian(strength=self.coeffs["E"]))
        # ---------------------------------------------------------------------------
        if not self.pure_theory:
            # -----------------------------------------------------------------------
            # STAGGERED MASS TERM
            for site in ["even", "odd"]:
                h_terms[f"N_{site}"] = LocalTerm(
                    self.ops["N_tot"], "N_tot", **self.def_params
                )
                self.H.add_term(
                    h_terms[f"N_{site}"].get_Hamiltonian(
                        self.coeffs[f"m_{site}"], staggered_mask(self.lvals, site)
                    )
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
                        self.H.add_term(
                            h_terms[f"{d}{ii}_hop_{site}"].get_Hamiltonian(
                                strength=self.coeffs[f"t{d}_{site}"],
                                add_dagger=True,
                                mask=mask,
                            )
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
                    self.H.add_term(
                        h_terms[plaq_name].get_Hamiltonian(
                            strength=-self.coeffs["B"], add_dagger=True
                        )
                    )
                    # ADD THE PLAQUETTE TO THE LIST OF OBSERVABLES
                    plaq_list.append(plaq_name)
        self.H.build(self.ham_format)

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
        # POLARIZED AND BARE VACUUM in 1D
        if len(self.lvals) == 1:
            if name == "V":
                if self.spin < 1:
                    s1, s2, L, R = 0, 4, 0, 2
                elif self.spin == 1:
                    s1, s2, L, R = 0, 7, 0, 2
                elif self.spin > 1:
                    s1, s2, L, R = 0, 10, 0, 2
            elif name == "PV":
                if self.spin < 1:
                    s1, s2, L, R = 1, 5, 1, 1
                elif self.spin == 1:
                    s1, s2, L, R = 1, 8, 1, 1
                elif self.spin > 1:
                    s1, s2, L, R = 1, 11, 1, 1
            elif name == "T":
                s1, s2, L, R = 6, 12, 1, 1
            elif name == "M":
                s1, s2, L, R = 2, 3, 1, 1
            elif name == "B":
                s1, s2, L, R = 7, 11, 1, 1
            else:
                s1, s2, L, R = 0, 0, 0, 0
            config_state = [s1 if ii % 2 == 0 else s2 for ii in range(self.n_sites)]
            if name == "DW":
                s1, s2, L, R = 0, 4, 0, 2
                config_state = [
                    s1 if ii < self.n_sites // 2 else s2 for ii in range(self.n_sites)
                ]
            elif name == "DW2":
                s1, s2, L, R = 0, 4, 0, 2
                config_state = [
                    s1 if (ii // 2) % 2 == 0 else s2 for ii in range(self.n_sites)
                ]
            if self.has_obc[0]:
                config_state[0] = L
                config_state[-1] = R
        # POLARIZED AND BARE VACUUM in 2D
        else:
            if not self.pure_theory:
                if self.has_obc[0]:
                    if name == "V":
                        config_state = [0, 9, 0, 4, 4, 0, 9, 0]
                    elif name == "PV1":
                        config_state = [1, 12, 3, 5, 5, 2, 11, 1]
                    elif name == "PV2":
                        config_state = [1, 11, 1, 5, 5, 3, 10, 1]
                else:
                    config_state = [0, 9, 0, 9, 9, 0, 9, 0]
            else:
                if name == "V":
                    config_state = [0 for _ in range(self.n_sites)]
        config_state = config_state
        return np.array(config_state)

    def SU2_Hamiltonian_couplings(self, g, m=None):
        """
        This function provides the couplings of the SU2 Yang-Mills Hamiltonian
        starting from the gauge coupling g and the bare mass parameter m

        Args:
            pure_theory (bool): True if the theory does not include matter

            g (scalar): gauge coupling

            m (scalar, optional): bare mass parameter

        Returns:
            dict: dictionary of Hamiltonian coefficients

        # NOTE: in the actual version of the coefficients, we rescale the Hamiltonian
        in such a way that the hopping term is dimensionless as in
        https://doi.org/10.1103/PRXQuantum.5.040309.
        To do so, we need to multiply
        - the hopping by 4*np.sqrt(2) (the original coupling is 1/2) --> 2*np.sqrt(2)
        - the electric by 8/3 (the original was g_{0}^{2}/2) --> 8g^{2}/3, g^{2}=(3/2np.sqrt(2))*g_{0}^{2}
        - the magnetic by 3 ()
        - the other convention here is g is intended to be g^{2}
        """
        if self.dim == 1:
            E = 8 * g / 3  # The correct one is g**2 / 2
            B = 0
        else:
            E = 8 * g / 3  # The correct one is g**2 / 2
            B = -3 / g  # The correct one is -1/2*g**2
        # Dictionary with Hamiltonian COEFFICIENTS
        self.coeffs = {
            "g": g,
            "E": E,  # ELECTRIC FIELD coupling
            "B": B,  # MAGNETIC FIELD coupling
        }
        if not self.pure_theory:
            # The correct hopping in original units should be 1/2
            t = 2 * np.sqrt(2)
            self.coeffs |= {
                "tx_even": -complex(0, t),  # x HOPPING (EVEN SITES)
                "tx_odd": -complex(0, t),  # x HOPPING (ODD SITES)
                "ty_even": -t,  # y HOPPING (EVEN SITES)
                "ty_odd": t,  # y HOPPING (ODD SITES)
                "tz_even": -t,  # z HOPPING (EVEN SITES)
                "tz_odd": t,  # z HOPPING (ODD SITES)
                "m": m,
                "m_odd": -m,  # EFFECTIVE MASS for ODD SITES
                "m_even": m,  # EFFECTIVE MASS for EVEN SITES
            }
