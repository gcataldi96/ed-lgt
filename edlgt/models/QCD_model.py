import numpy as np
from edlgt.modeling import LocalTerm, TwoBodyTerm
from edlgt.modeling import staggered_mask
from .quantum_model import QuantumModel
import logging
import pickle


def load_dictionary(filename):
    with open(filename, "rb") as outp:
        return pickle.load(outp)


logger = logging.getLogger(__name__)
__all__ = ["QCD_Model"]


class QCD_Model(QuantumModel):
    def __init__(self, sectors, **kwargs):
        # Initialize base class with the common parameters
        super().__init__(**kwargs)
        # Initialize local dimensions
        self.loc_dims = np.array([54 for _ in range(self.n_sites)], dtype=np.uint8)
        # Acquire operators
        self.ops = load_dictionary("SU3_2flavor_site.pkl")["operators"]
        # Acquire local dimension and lattice label
        self.get_local_site_dimensions()
        # GLOBAL SYMMETRIES
        global_ops = [self.ops["uu"], self.ops["dd"]]
        global_sectors = sectors
        # LINK SYMMETRIES
        self.ops["P_px"] = self.ops["LL"].copy()
        self.ops["P_px"].data = np.imag(self.ops["LL"].data)
        self.ops["P_mx"] = self.ops["RR"].copy()
        self.ops["P_mx"].data = np.imag(self.ops["RR"].data)
        link_ops = [[self.ops["P_px"], self.ops["P_mx"]]]
        link_sectors = [0]
        # GET SYMMETRY SECTOR
        self.get_abelian_symmetry_sector(
            global_ops=global_ops,
            global_sectors=global_sectors,
            link_ops=link_ops,
            link_sectors=link_sectors,
        )
        self.default_params()

    def build_Hamiltonian(self, g, mu, md, ham_format):
        # Hamiltonian Coefficients
        self.QCD_Hamiltonian_couplings(g, mu, md)
        logger.info("BUILDING HAMILTONIAN")
        h_terms = {}
        # ---------------------------------------------------------------------------
        # ELECTRIC ENERGY SU(3) CASIMIR
        # ---------------------------------------------------------------------------
        op_name = "lnk"
        h_terms[op_name] = LocalTerm(self.ops[op_name], op_name, **self.def_params)
        self.H.add_term(h_terms[op_name].get_Hamiltonian(strength=self.coeffs["E"]))
        # ---------------------------------------------------------------------------
        # STAGGERED MASS TERMS (for up and down particles)
        # ---------------------------------------------------------------------------
        for flavor in ["u", "d"]:
            for site in ["even", "odd"]:
                name_h_term = f"N{flavor}_{site}"
                op_name = f"{flavor}{flavor}"
                h_terms[name_h_term] = LocalTerm(
                    self.ops[op_name], op_name, **self.def_params
                )
                self.H.add_term(
                    h_terms[name_h_term].get_Hamiltonian(
                        self.coeffs[f"m{flavor}_{site}"],
                        staggered_mask(self.lvals, site),
                    )
                )
        # ---------------------------------------------------------------------------
        # HOPPING
        # ---------------------------------------------------------------------------
        for flavor in ["u", "d"]:
            hop_names_list = [f"L{flavor}_hc", f"{flavor}R_hc"]
            op_list = [self.ops[op] for op in hop_names_list]
            # Define the Hamiltonian term
            h_terms[f"{flavor}_hop"] = TwoBodyTerm(
                "x", op_list, hop_names_list, **self.def_params
            )
            mask = staggered_mask(self.lvals, site)
            self.H.add_term(
                h_terms[f"{flavor}_hop"].get_Hamiltonian(
                    strength=self.coeffs["tx"],
                    add_dagger=True,
                    mask=mask,
                )
            )
        self.H.build(ham_format)

    def QCD_Hamiltonian_couplings(self, g: float, mu: float, md: float):
        """
        This function provides the couplings of the SU3 Yang-Mills 2 flavour Hamiltonian
        starting from the gauge coupling g and the bare mass parameter m

        Args:

            g (scalar): gauge coupling

            mu (scalar): bare mass parameter of u(p) particles

            md (scalar): bare mass parameter of d(own) particles

        Returns:
            dict: dictionary of Hamiltonian coefficients
        """
        t = 1 / 2
        # Dictionary with Hamiltonian COEFFICIENTS
        self.coeffs = {
            "g": g,
            "E": g / 3,  # ELECTRIC FIELD coupling
            "tx": -complex(0, t),  # x HOPPING
            "mu": mu,
            "mu_odd": -mu,  # EFFECTIVE MASS for ODD SITES
            "mu_even": mu,  # EFFECTIVE MASS for EVEN SITES
            "md": md,
            "md_odd": -md,  # EFFECTIVE MASS for ODD SITES
            "md_even": md,  # EFFECTIVE MASS for EVEN SITES
        }


"""res = load_dictionary("SU3_2flavor_site.pkl")
for op in res["operators"].keys():
    print(op)
print(res["operators"]["uR_hc"])"""
