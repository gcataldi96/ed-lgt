import numpy as np
from functools import partial
from modeling import PlaquetteTerm, TwoBodyTerm2D, LocalTerm


class SU2_Model:
    def __init__(self, params):
        if not isinstance(params, dict):
            raise TypeError(f"params should be a dict, not a {type(params)}")

        self.dim = params["dim"]
        if self.dim != 2:
            raise NotImplementedError("Only 2D lattices are currently supported.")

        self.lvals = params["lvals"]
        self.directions = params.setdefault("directions", "xyz"[: self.dim])
        self.has_obc = params["has_obc"]
        self.input_folder = params["input_folder"]
        self.output_folder = params["output_folder"]

        # Border Penalties
        if self.has_obc:
            for d in self.directions:
                for s in "mp":
                    mask = partial(self.border_mask, f"{s}{d}")
                    self += LocalTerm(
                        f"P_{s}{d}",
                        strength="eta",
                        mask=mask,
                    )

        # Link Symmetries
        axes = [[1, 0], [0, 1]]
        for i, d in enumerate(self.directions):
            self += TwoBodyTerm2D(
                [f"W_{s}{d}" for s in "pm"],
                axes[i],
                strength="eta",
                isotropy_xyz=False,
                has_obc=self.has_obc,
            )
        # E ELECTRIC ENERGY: gamma operator
        self += LocalTerm("Gamma", strength="g")

        # B MAGNETIC ENERGY: plaquette interaction
        # NOTE: The order of the operators is BL, TL, BR, TR:
        self += PlaquetteTerm(
            ["C_py_px", "C_my_px", "C_py_mx", "C_my_mx"],
            strength="B",
            has_obc=self.has_obc,
        )
        self += PlaquetteTerm(
            ["C_py_px_dag", "C_my_px_dag", "C_py_mx_dag", "C_my_mx_dag"],
            strength="B",
            has_obc=self.has_obc,
        )

        # HOPPING ACTIVITY along x AXIS
        self += TwoBodyTerm2D(
            ["Q_px_dag", "Q_mx"],
            [1, 0],
            strength="tx",
            isotropy_xyz=False,
            has_obc=self.has_obc,
        )

        self += TwoBodyTerm2D(
            ["Q_px", "Q_mx_dag"],
            [1, 0],
            strength="tx_dag",
            isotropy_xyz=False,
            has_obc=self.has_obc,
        )

        # HOPPING ACTIVITY along y AXIS
        for site in ["even", "odd"]:
            mask = mask = partial(self.staggered_mask, site)
            self += TwoBodyTerm2D(
                ["Q_py", "Q_my_dag"],
                [0, 1],
                strength=f"ty_{site}",
                isotropy_xyz=False,
                has_obc=self.has_obc,
                mask=mask,
            )

        # Staggered Mass Term
        for site in ["even", "odd"]:
            mask = mask = partial(self.staggered_mask, site)
            self += LocalTerm(
                "mass_op",
                strength=f"m_{site}",
                mask=mask,
            )

    def border_mask(self, border, params):
        """
        Defines the masks for all four sides: top, bottom, left,
        and right as well as the four corners.
        NOTE Rows and Columns of the mask array corresponds to (x,y) coordinates!
        """
        lx = self.lvals[0]
        ly = self.lvals[1]
        mask = np.zeros((lx, ly), dtype=bool)
        if border == "mx":
            mask[0, :] = True
        elif border == "px":
            mask[lx - 1, :] = True
        if border == "my":
            mask[:, 0] = True
        elif border == "py":
            mask[:, ly - 1] = True
        return mask

    def staggered_mask(self, site, params):
        lx = self.lvals[0]
        ly = self.lvals[1]
        mask = np.zeros((lx, ly), dtype=bool)
        for ii in range(lx):
            for jj in range(ly):
                stag = (-1) ** (ii + jj)
                if site == "even":
                    if stag > 0:
                        mask[ii, jj] = True
                elif site == "odd":
                    if stag < 0:
                        mask[ii, jj] = True
        return mask

    @staticmethod
    def get_Hamiltonian_couplings(g, mass, pure_theory=False):
        E = 3 * (g**2) / 16  # ELECTRIC FIELD coupling
        B = -4 / (g**2)  # MAGNETIC FIELD coupling
        coeffs = {
            "g": g,  # Gauge Coupling
            "E": E,  # ELECTRIC FIELD coupling
            "B": B,  # MAGNETIC FIELD coupling
        }
        if not pure_theory:
            coeffs |= {
                "tx": -0.5j,  # HORIZONTAL HOPPING
                "tx_dag": 0.5j,  # HORIZONTAL HOPPING DAGGER
                "ty_even": -0.5,  # VERTICAL HOPPING (EVEN SITES)
                "ty_odd": 0.5,  # VERTICAL HOPPING (ODD SITES)
                "m_odd": -mass,  # EFFECTIVE MASS for ODD SITES
                "m_even": mass,  # EFFECTIVE MASS for EVEN SITES
            }
        if pure_theory:
            coeffs["eta"] = 10 * max(E, np.abs(B))
        else:
            coeffs["eta"] = 10 * max(E, np.abs(B), mass)
        return coeffs

    def get_operators(self):
        ed_ops = TNOperators("SU2operators")
        ops = get_operators()
        return ed_ops

    def get_observables(self):
        # TODO: Computing Plaquettes
        tn_obs = TNObservables()
        # Border Penalties
        if self.has_obc:
            for d in self.directions:
                for s in "mp":
                    tn_obs += TNObsLocal(f"P_{s}{d}", f"P_{s}{d}")
        # Link Symmetry Correlators
        for d in self.directions:
            tn_obs += TNObsCorr(f"W_{d}_link", [f"W_{s}{d}" for s in "pm"])
        # Gamma Electric Energy
        tn_obs += TNObsLocal("gamma", "Gamma")
        # Number Operators
        for n in ["n_single", "n_pair", "n_tot"]:
            tn_obs += TNObsLocal(n, n)
        # Density Correlator
        tn_obs += TNObsCorr("nn_density", ["n_tot", "n_tot"])
        # S-wave SCOP Correlator
        # tn_obs += TNObsCorr("S_Wave", ["Delta_Dagger", "Delta"])
        return tn_obs

    def get_conv_parameters(self, params):
        tn_conv = TNConvergenceParameters(**params["conv"])
        return tn_conv
