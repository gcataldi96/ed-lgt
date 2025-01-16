import numpy as np
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, PlaquetteTerm
from ed_lgt.modeling import check_link_symmetry, staggered_mask
from .quantum_model import QuantumModel
from ed_lgt.operators import (
    QCD_dressed_site_operators,
    QCD_gauge_invariant_states,
)
import logging

logger = logging.getLogger(__name__)
__all__ = ["QCD_Model"]


class QCD_Model(QuantumModel):
    def __init__(self, pure_theory, **kwargs):
        # Initialize base class with the common parameters
        super().__init__(**kwargs)
        self.pure_theory = pure_theory
        self.staggered_basis = False
        # Acquire operators
        self.ops = QCD_dressed_site_operators(
            self.pure_theory,
            lattice_dim=self.dim,
        )
        # Acquire gauge invariant basis and states
        self.gauge_basis, self.gauge_states = QCD_gauge_invariant_states(
            self.pure_theory,
            lattice_dim=self.dim,
        )
        # Acquire local dimension and lattice label
        self.get_local_site_dimensions()
        # GLOBAL SYMMETRIES
        if self.pure_theory:
            global_ops = None
            global_sectors = None
        else:
            global_ops = [self.ops["N_up"], self.ops["N_down"]]
            global_sectors = [self.n_sites, self.n_sites]
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
        logger.info("BUILDING HAMILTONIAN")
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
