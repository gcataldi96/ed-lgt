import numpy as np
from edlgt.operators import (
    Z2_FermiHubbard_dressed_site_operators,
    Z2_FermiHubbard_gauge_invariant_states,
)
from edlgt.modeling import LocalTerm, TwoBodyTerm, NBodyTerm
from edlgt.modeling import check_link_symmetry, border_mask
from .quantum_model import QuantumModel
import logging

logger = logging.getLogger(__name__)

__all__ = ["Z2_FermiHubbard_Model"]


class Z2_FermiHubbard_Model(QuantumModel):
    def __init__(self, sectors, ham_format, **kwargs):
        # Initialize base class with the common parameters
        super().__init__(**kwargs)
        self.ham_format = ham_format
        # Acquire operators
        self.ops = Z2_FermiHubbard_dressed_site_operators(lattice_dim=self.dim)
        # Acquire gauge invariant basis and states
        self.gauge_basis, self.gauge_states = Z2_FermiHubbard_gauge_invariant_states(
            lattice_dim=self.dim
        )
        # Acquire local dimension and lattice label
        self.get_local_site_dimensions()
        # GLOBAL SYMMETRIES
        global_ops = [self.ops["N_tot"], self.ops["N_up"]]
        global_sectors = sectors
        # LINK SYMMETRIES
        link_ops = [
            [self.ops[f"n_p{d}"], -self.ops[f"n_m{d}"]] for d in self.directions
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
        # CONSTRUCT THE HAMILTONIAN
        h_terms = {}
        # -------------------------------------------------------------------------------
        # COULOMB POTENTIAL
        op_name = "N_pair_half"
        h_terms["V"] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
        self.H.add_term(h_terms["V"].get_Hamiltonian(strength=self.coeffs["U"]))
        # -------------------------------------------------------------------------------
        # HOPPING
        for d in self.directions:
            for s in ["up", "down"]:
                # Define the list of the 2 non trivial operators
                op_names_list = [f"Q{s}_p{d}_dag", f"Q{s}_m{d}"]
                op_list = [self.ops[op] for op in op_names_list]
                # Define the Hamiltonian term
                h_terms[f"{d}_hop_{s}"] = TwoBodyTerm(
                    d, op_list, op_names_list, **self.def_params
                )
                self.H.add_term(
                    h_terms[f"{d}_hop_{s}"].get_Hamiltonian(
                        strength=self.coeffs["t"], add_dagger=True
                    )
                )
        # -------------------------------------------------------------------------------
        # EXTERNAL ELECTRIC FIELD
        """op_name = "E"
        h_terms["E"] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
        self.H.add_term(h_terms["E"].get_Hamiltonian(strength=self.coeffs["h"])"""
        for ii, d in enumerate(self.directions):
            border = f"p{d}"
            op_name = f"P_{border}"
            if self.has_obc[ii]:
                # Apply the mask on the specific border
                mask = ~border_mask(self.lvals, border)
            else:
                mask = None
            h_terms[op_name] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
            self.H.add_term(
                h_terms[op_name].get_Hamiltonian(strength=self.coeffs["h"], mask=mask)
            )
        # -------------------------------------------------------------------------------
        # STRING Z OPERATOR (only in case of PBCx)
        if not self.has_obc[0]:
            op_names_list = ["Sz_mx,px" for _ in range(self.lvals[0])]
            op_list = [self.ops[op] for op in op_names_list]
            distances = [[ii, 0] for ii in range(1, self.lvals[0], 1)]
            mask = np.zeros(tuple(self.lvals), dtype=bool)
            for ii in range(self.lvals[1]):
                mask[0, ii] = True
            h_terms["Zflux"] = NBodyTerm(
                op_list, op_names_list, distances, **self.def_params
            )
            self.H.add_term(
                h_terms["Zflux"].get_Hamiltonian(strength=-self.coeffs["J"], mask=mask)
            )
        # -------------------------------------------------------------------------------
        self.H.build(self.ham_format)

    def check_symmetries(self):
        # CHECK LINK SYMMETRIES
        for ax in self.directions:
            check_link_symmetry(
                ax,
                self.obs_list[f"n_p{ax}"],
                self.obs_list[f"n_m{ax}"],
                value=0,
                sign=-1,
            )
