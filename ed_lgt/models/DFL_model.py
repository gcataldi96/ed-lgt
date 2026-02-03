import numpy as np
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm
from ed_lgt.modeling import check_link_symmetry, staggered_mask
from .quantum_model import QuantumModel
from ed_lgt.operators import (
    SU2_dressed_site_operators,
    SU2_gauge_invariant_states,
    SU2_gen_dressed_site_operators,
)
from ed_lgt.symmetries import get_state_configs
import logging

logger = logging.getLogger(__name__)
__all__ = ["DFL_Model"]


class DFL_Model(QuantumModel):
    def __init__(self, spin, pure_theory, background, ham_format, **kwargs):
        # Initialize base class with the common parameters
        super().__init__(**kwargs)
        self.spin = spin
        self.pure_theory = pure_theory
        self.background = background
        self.staggered_basis = False
        self.ham_format = ham_format
        # -------------------------------------------------------------------------------
        # Acquire gauge invariant basis and states
        self.gauge_basis, self.gauge_states = SU2_gauge_invariant_states(
            self.spin,
            self.pure_theory,
            lattice_dim=self.dim,
            background=self.background,
        )
        # -------------------------------------------------------------------------------
        # Acquire operators
        if self.spin < 1:
            ops = SU2_dressed_site_operators(
                self.spin,
                self.pure_theory,
                lattice_dim=self.dim,
                background=self.background,
            )
        else:
            # Acquire operators
            ops = SU2_gen_dressed_site_operators(
                self.spin,
                self.pure_theory,
                lattice_dim=self.dim,
                background=self.background,
            )
        # Initialize the operators, local dimension and lattice labels
        self.project_operators(ops)
        # Rather than for SU2, here we do not select the symmetry sector

    def build_Hamiltonian(self, g, m=None):
        logger.info("BUILDING HAMILTONIAN")
        # Hamiltonian Coefficients
        self.DFL_Hamiltonian_couplings(g, m)
        h_terms = {}
        # -------------------------------------------------------------------------
        # ELECTRIC ENERGY
        op_name = "E_square"
        h_terms[op_name] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
        if not np.isclose(self.coeffs["E"], 1e-10):
            self.H.add_term(h_terms[op_name].get_Hamiltonian(strength=self.coeffs["E"]))
        # -------------------------------------------------------------------------
        # PLAQUETTE TERM: MAGNETIC INTERACTION
        if self.dim > 1:
            op_names_list = ["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]
            op_list = [self.ops[op] for op in op_names_list]
            h_terms["plaq_xy"] = PlaquetteTerm(
                ["x", "y"], op_list, op_names_list, **self.def_params
            )
            if not np.isclose(self.coeffs["B"], 1e-10):
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
            if not np.isclose(self.coeffs["B"], 1e-10):
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
            if not np.isclose(self.coeffs["B"], 1e-10):
                self.H.add_term(
                    h_terms["plaq_yz"].get_Hamiltonian(
                        strength=self.coeffs["B"], add_dagger=True
                    )
                )
        # -------------------------------------------------------------------------
        if not self.pure_theory:
            # ---------------------------------------------------------------------
            # STAGGERED MASS TERM
            h_terms["N_tot"] = LocalTerm(self.ops["N_tot"], "N_tot", **self.def_params)
            for site in ["even", "odd"]:
                if not np.isclose(self.coeffs["m"], 1e-10):
                    self.H.add_term(
                        h_terms["N_tot"].get_Hamiltonian(
                            self.coeffs[f"m_{site}"], staggered_mask(self.lvals, site)
                        )
                    )
            # ---------------------------------------------------------------------
            # HOPPING
            for d in self.directions:
                op_names_list = [f"Qp{d}_dag", f"Qm{d}"]
                op_list = [self.ops[op] for op in op_names_list]
                # Define the Hamiltonian term
                h_terms["hopping"] = TwoBodyTerm(
                    d, op_list, op_names_list, **self.def_params
                )
                for site in ["even", "odd"]:
                    # Define the mask
                    mask = staggered_mask(self.lvals, site)
                    self.H.add_term(
                        h_terms["hopping"].get_Hamiltonian(
                            strength=self.coeffs[f"t{d}_{site}"],
                            add_dagger=True,
                            mask=mask,
                        )
                    )
        self.H.build(format=self.ham_format)

    def build_gen_Hamiltonian(self, g, m=None):
        logger.info("BUILDING generalized HAMILTONIAN")
        # Hamiltonian Coefficients
        self.DFL_Hamiltonian_couplings(g, m)
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
                            strength=self.coeffs["B"], add_dagger=True
                        )
                    )
                    # ADD THE PLAQUETTE TO THE LIST OF OBSERVABLES
                    plaq_list.append(plaq_name)
        self.H.build(self.ham_format)

    def DFL_Hamiltonian_couplings(self, g, m=None):
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

        NOTE: for the DFL project use
        E = 8 * g / 3
        B = -3 / g
        t = 2 * np.sqrt(2)
        """
        if self.dim == 1:
            E = g / 2
            B = 0
        else:
            E = g
            B = -1  # -1/ (2 * g)
        # Dictionary with Hamiltonian COEFFICIENTS
        self.coeffs = {
            "g": g,
            "E": E,  # ELECTRIC FIELD coupling
            "B": B,  # MAGNETIC FIELD coupling
        }
        if not self.pure_theory and m is not None:
            t = 1
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

    def get_background_charges_configs(self, logical_stag_basis):
        # NOTE! It works only in 1D
        if len(self.lvals) > 1:
            raise ValueError("SU2 background configs works only in 1D")
        has_obc = self.has_obc[0]

        def next_val(value, last):
            """Return the two possible next values based on the current value."""
            if value in [0, 7]:
                return [4, 11] if last else [0, 6]
            elif value in [1, 6]:
                return [5, 12] if last else [1, 7]
            elif value in [4, 12]:
                return [0, 6] if last else [4, 11]
            elif value in [5, 11]:
                return [1, 7] if last else [5, 12]
            else:
                raise ValueError(f"got unexpected value {value}")

        first_site_space = 2 if has_obc else 4
        loc_dims = np.array(
            [first_site_space] + [2 for _ in range(self.n_sites - 1)], dtype=int
        )
        A = np.zeros((np.prod(loc_dims), self.n_sites), dtype=int)
        BG = np.zeros((np.prod(loc_dims), self.n_sites), dtype=int)
        local_configs = get_state_configs(loc_dims)
        # Initial states and bg charge sector
        if not has_obc:
            init = np.array([0, 1, 6, 7], dtype=int)
            bg_init = np.array([0, 0, 1, 1], dtype=int)
        else:
            init = np.array([0, 3], dtype=int)
            bg_init = np.array([0, 1], dtype=int)
        for tmp in range(len(A)):
            local_config = local_configs[tmp]
            # Save the config of the first site
            A[tmp, 0] = init[local_config[0]]
            # Save the corresponding charge sector
            BG[tmp, 0] = bg_init[local_config[0]]
            for ii in range(1, self.n_sites, 1):
                last = ii % logical_stag_basis == 0
                # Save the config of the ii-th site
                A[tmp, ii] = next_val(A[tmp, ii - 1], last)[local_config[ii]]
                # Save the corresponding charge sector
                BG[tmp, ii] = local_config[ii]
        # Filter rows if has_obc is False to make it compatible with PBC
        if not has_obc:
            mask1 = np.isin(A[:, 0], [0, 6]) & np.isin(A[:, -1], [4, 12])
            mask2 = np.isin(A[:, 0], [1, 7]) & np.isin(A[:, -1], [5, 11])
            mask = mask1 | mask2
            A = A[mask]
            BG = BG[mask]
        else:
            mask = np.isin(A[:, -1], [2, 5])
            A = A[mask]
            BG = BG[mask]
        return A, BG

    def get_string_breaking_configs(self, finite_density=0):
        logger.info(f"finite density {finite_density}")
        if self.lvals == [5, 2]:
            self.n_min_strings = 5
            self.n_max_strings = 1
            if finite_density == 0:
                self.string_cfgs = {
                    "max0": np.array([6, 10, 2, 10, 1, 5, 3, 10, 3, 11], dtype=int),
                    "min0": np.array([7, 12, 3, 12, 1, 4, 0, 9, 0, 11], dtype=int),
                    "min1": np.array([7, 12, 3, 11, 0, 4, 0, 9, 1, 12], dtype=int),
                    "min2": np.array([7, 12, 2, 9, 0, 4, 0, 10, 2, 12], dtype=int),
                    "min3": np.array([7, 11, 0, 9, 0, 4, 1, 11, 2, 12], dtype=int),
                    "min4": np.array([6, 9, 0, 9, 0, 5, 2, 11, 2, 12], dtype=int),
                }
            elif finite_density == 2:
                self.string_cfgs = {
                    "min0": np.array([7, 12, 12, 12, 1, 4, 0, 9, 0, 11], dtype=int),
                    "min1": np.array([7, 12, 12, 11, 0, 4, 0, 9, 1, 12], dtype=int),
                    "min2": np.array([7, 12, 11, 9, 0, 4, 0, 10, 2, 12], dtype=int),
                    "min3": np.array([7, 11, 9, 9, 0, 4, 1, 11, 2, 12], dtype=int),
                    "min4": np.array([6, 9, 9, 9, 0, 5, 2, 11, 2, 12], dtype=int),
                }
            elif finite_density == 4:
                self.string_cfgs = {
                    "min0": np.array([7, 12, 12, 12, 5, 4, 0, 9, 0, 11], dtype=int),
                    "min1": np.array([7, 12, 12, 11, 4, 4, 0, 9, 1, 12], dtype=int),
                    "min2": np.array([7, 12, 11, 9, 4, 4, 0, 10, 2, 12], dtype=int),
                    "min3": np.array([7, 11, 9, 9, 4, 4, 1, 11, 2, 12], dtype=int),
                    "min4": np.array([6, 9, 9, 9, 4, 5, 2, 11, 2, 12], dtype=int),
                }
            elif finite_density == 6:
                self.string_cfgs = {
                    "min0": np.array([12, 12, 12, 12, 5, 4, 0, 9, 0, 11], dtype=int),
                    "min1": np.array([12, 12, 12, 11, 4, 4, 0, 9, 1, 12], dtype=int),
                    "min2": np.array([12, 12, 11, 9, 4, 4, 0, 10, 2, 12], dtype=int),
                    "min3": np.array([12, 11, 9, 9, 4, 4, 1, 11, 2, 12], dtype=int),
                    "min4": np.array([11, 9, 9, 9, 4, 5, 2, 11, 2, 12], dtype=int),
                }
        elif self.lvals == [4, 3]:
            self.n_min_strings = 10
            self.n_max_strings = 24
            self.string_cfgs = {
                "min0": np.array([7, 12, 3, 5, 9, 0, 21, 1, 0, 9, 0, 11], dtype=int),
                "min1": np.array([7, 12, 2, 4, 9, 0, 24, 2, 0, 9, 0, 11], dtype=int),
                "min2": np.array([7, 12, 2, 4, 9, 0, 23, 0, 0, 9, 1, 12], dtype=int),
                "min3": np.array([7, 11, 0, 4, 9, 3, 26, 2, 0, 9, 0, 11], dtype=int),
                "min4": np.array([7, 11, 0, 4, 9, 3, 25, 0, 0, 9, 1, 12], dtype=int),
                "min5": np.array([7, 11, 0, 4, 9, 2, 21, 0, 0, 10, 2, 12], dtype=int),
                "min6": np.array([6, 9, 0, 4, 12, 5, 26, 2, 0, 9, 0, 11], dtype=int),
                "min7": np.array([6, 9, 0, 4, 12, 5, 25, 0, 0, 9, 1, 12], dtype=int),
                "min8": np.array([6, 9, 0, 4, 12, 4, 21, 0, 0, 10, 2, 12], dtype=int),
                "min9": np.array([6, 9, 0, 4, 11, 0, 21, 0, 1, 11, 2, 12], dtype=int),
                # --------------------------------------------------------------------
                "max0": np.array([7, 12, 3, 5, 10, 5, 26, 3, 1, 11, 2, 12], dtype=int),
                "max1": np.array([6, 10, 3, 5, 11, 3, 26, 3, 1, 11, 2, 12], dtype=int),
                "max2": np.array([7, 11, 1, 5, 10, 6, 24, 3, 1, 11, 2, 12], dtype=int),
                "max3": np.array([7, 12, 3, 5, 10, 4, 22, 3, 1, 12, 1, 12], dtype=int),
                "max4": np.array([7, 12, 3, 5, 10, 5, 25, 1, 1, 11, 3, 11], dtype=int),
                "max5": np.array([6, 10, 3, 5, 12, 7, 26, 3, 0, 10, 2, 12], dtype=int),
                "max6": np.array([6, 10, 3, 5, 12, 8, 26, 3, 0, 10, 2, 12], dtype=int),
                "max7": np.array([6, 10, 3, 5, 11, 2, 22, 3, 1, 12, 1, 12], dtype=int),
                "max8": np.array([6, 10, 3, 5, 11, 3, 25, 1, 1, 11, 3, 11], dtype=int),
                "max9": np.array([7, 11, 1, 5, 10, 6, 23, 1, 1, 11, 3, 11], dtype=int),
                "max10": np.array([7, 12, 2, 4, 10, 5, 28, 2, 1, 11, 3, 11], dtype=int),
                "max11": np.array([7, 12, 2, 4, 10, 5, 29, 2, 1, 11, 3, 11], dtype=int),
                "max12": np.array([6, 10, 3, 5, 12, 7, 25, 1, 0, 10, 3, 11], dtype=int),
                "max13": np.array([6, 10, 3, 5, 12, 8, 25, 1, 0, 10, 3, 11], dtype=int),
                "max14": np.array([6, 9, 1, 5, 11, 1, 28, 3, 1, 12, 1, 12], dtype=int),
                "max15": np.array([6, 9, 1, 5, 11, 1, 29, 3, 1, 12, 1, 12], dtype=int),
                "max16": np.array([6, 10, 2, 4, 11, 3, 28, 2, 1, 11, 3, 11], dtype=int),
                "max17": np.array([6, 10, 2, 4, 11, 3, 29, 2, 1, 11, 3, 11], dtype=int),
                "max18": np.array([7, 11, 1, 5, 10, 7, 27, 1, 1, 12, 0, 11], dtype=int),
                "max19": np.array([7, 11, 1, 5, 10, 8, 27, 1, 1, 12, 0, 11], dtype=int),
                "max20": np.array([6, 10, 2, 4, 12, 7, 28, 2, 0, 10, 3, 11], dtype=int),
                "max21": np.array([6, 10, 2, 4, 12, 8, 28, 2, 0, 10, 3, 11], dtype=int),
                "max22": np.array([6, 10, 2, 4, 12, 7, 29, 2, 0, 10, 3, 11], dtype=int),
                "max23": np.array([6, 10, 2, 4, 12, 8, 29, 2, 0, 10, 3, 11], dtype=int),
            }
        elif self.lvals == [3, 2]:
            self.n_min_strings = 1
            self.n_max_strings = 1
            self.string_cfgs = {
                "max0": np.array([25, 30, 2, 9, 9, 35], dtype=int),
                "min0": np.array([27, 37, 2, 7, 0, 35], dtype=int),
            }
        elif self.lvals == [6, 2]:
            self.n_min_strings = 6
            self.n_max_strings = 0
            if finite_density == 0:
                self.string_cfgs = {
                    "min0": np.array([7, 12, 3, 12, 3, 5, 4, 0, 9, 0, 9, 6], dtype=int),
                    "min1": np.array(
                        [7, 12, 3, 12, 2, 4, 4, 0, 9, 0, 10, 7], dtype=int
                    ),
                    "min2": np.array(
                        [7, 12, 3, 11, 0, 4, 4, 0, 9, 1, 11, 7], dtype=int
                    ),
                    "min3": np.array(
                        [7, 12, 2, 9, 0, 4, 4, 0, 10, 2, 11, 7], dtype=int
                    ),
                    "min4": np.array(
                        [7, 11, 0, 9, 0, 4, 4, 1, 11, 2, 11, 7], dtype=int
                    ),
                    "min5": np.array([6, 9, 0, 9, 0, 4, 5, 2, 11, 2, 11, 7], dtype=int),
                }
            if finite_density == 2:
                self.string_cfgs = {
                    "min0": np.array(
                        [7, 12, 12, 12, 3, 5, 4, 0, 9, 0, 9, 6], dtype=int
                    ),
                    "min1": np.array(
                        [7, 12, 12, 12, 2, 4, 4, 0, 9, 0, 10, 7], dtype=int
                    ),
                    "min2": np.array(
                        [7, 12, 12, 11, 0, 4, 4, 0, 9, 1, 11, 7], dtype=int
                    ),
                    "min3": np.array(
                        [7, 12, 11, 9, 0, 4, 4, 0, 10, 2, 11, 7], dtype=int
                    ),
                    "min4": np.array(
                        [7, 11, 9, 9, 0, 4, 4, 1, 11, 2, 11, 7], dtype=int
                    ),
                    "min5": np.array([6, 9, 9, 9, 0, 4, 5, 2, 11, 2, 11, 7], dtype=int),
                }
            if finite_density == 4:
                self.string_cfgs = {
                    "min0": np.array(
                        [7, 12, 12, 12, 12, 5, 4, 0, 9, 0, 9, 6], dtype=int
                    ),
                    "min1": np.array(
                        [7, 12, 12, 12, 11, 4, 4, 0, 9, 0, 10, 7], dtype=int
                    ),
                    "min2": np.array(
                        [7, 12, 12, 11, 9, 4, 4, 0, 9, 1, 11, 7], dtype=int
                    ),
                    "min3": np.array(
                        [7, 12, 11, 9, 9, 4, 4, 0, 10, 2, 11, 7], dtype=int
                    ),
                    "min4": np.array(
                        [7, 11, 9, 9, 9, 4, 4, 1, 11, 2, 11, 7], dtype=int
                    ),
                    "min5": np.array([6, 9, 9, 9, 9, 4, 5, 2, 11, 2, 11, 7], dtype=int),
                }
            if finite_density == 6:
                self.string_cfgs = {
                    "min0": np.array(
                        [7, 12, 12, 12, 12, 5, 4, 0, 9, 0, 9, 11], dtype=int
                    ),
                    "min1": np.array(
                        [7, 12, 12, 12, 11, 4, 4, 0, 9, 0, 10, 12], dtype=int
                    ),
                    "min2": np.array(
                        [7, 12, 12, 11, 9, 4, 4, 0, 9, 1, 11, 12], dtype=int
                    ),
                    "min3": np.array(
                        [7, 12, 11, 9, 9, 4, 4, 0, 10, 2, 11, 12], dtype=int
                    ),
                    "min4": np.array(
                        [7, 11, 9, 9, 9, 4, 4, 1, 11, 2, 11, 12], dtype=int
                    ),
                    "min5": np.array(
                        [6, 9, 9, 9, 9, 4, 5, 2, 11, 2, 11, 12], dtype=int
                    ),
                }
            if finite_density == 8:
                self.string_cfgs = {
                    "min0": np.array(
                        [12, 12, 12, 12, 12, 5, 4, 0, 9, 0, 9, 11], dtype=int
                    ),
                    "min1": np.array(
                        [12, 12, 12, 12, 11, 4, 4, 0, 9, 0, 10, 12], dtype=int
                    ),
                    "min2": np.array(
                        [12, 12, 12, 11, 9, 4, 4, 0, 9, 1, 11, 12], dtype=int
                    ),
                    "min3": np.array(
                        [12, 12, 11, 9, 9, 4, 4, 0, 10, 2, 11, 12], dtype=int
                    ),
                    "min4": np.array(
                        [12, 11, 9, 9, 9, 4, 4, 1, 11, 2, 11, 12], dtype=int
                    ),
                    "min5": np.array(
                        [11, 9, 9, 9, 9, 4, 5, 2, 11, 2, 11, 12], dtype=int
                    ),
                }
        else:
            msg = "String breaking in ED considered only for lvals=[5,2] or [4,3] or [3,2]"
            raise ValueError(msg)

    def print_state_config(self, config):
        logger.info(f"----------------------------------------------------")
        logger.info(f"SINGLETS IN CONFIG {config}")
        logger.info(f"----------------------------------------------------")
        for ii, cfg_idx in enumerate(config):
            msg = f"site {ii} state {cfg_idx}"
            lattice_label = self.lattice_labels[ii]
            local_basis_state = self.gauge_states[lattice_label][cfg_idx]
            local_basis_state.display_singlet(msg)
