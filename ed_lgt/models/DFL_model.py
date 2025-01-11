import numpy as np
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm
from ed_lgt.modeling import check_link_symmetry, staggered_mask
from .quantum_model import QuantumModel
from ed_lgt.operators import SU2_dressed_site_operators, SU2_gauge_invariant_states
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
        # Acquire operators
        self.ops = SU2_dressed_site_operators(
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
            if not np.isclose(self.coeffs["B"], 1e-10):
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
            if not np.isclose(self.coeffs["B"], 1e-10):
                self.H.add_term(
                    h_terms["plaq_yz"].get_Hamiltonian(
                        strength=-self.coeffs["B"], add_dagger=True
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
