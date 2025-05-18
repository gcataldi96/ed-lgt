# %%
import numpy as np

# Import TenPy modules
# for custom sites and merging symmetries
from tenpy.networks.site import Site
from tenpy.linalg import np_conserved as npc  # for handling conserved charges
from tenpy.models.lattice import Lattice, Square  # for constructing the lattice
from tenpy.models.model import CouplingMPOModel  # class for models with coupling terms
from tenpy.networks.mps import MPS  # MPS construction
from tenpy.algorithms import dmrg  # DMRG algorithm

from ed_lgt.operators import (
    Z2_FermiHubbard_dressed_site_operators,
    Z2_FermiHubbard_gauge_invariant_states,
)


def Z2Hubbard_gauge_invariant_ops(lattice_dim):
    in_ops = Z2_FermiHubbard_dressed_site_operators(lattice_dim)
    gauge_basis, _ = Z2_FermiHubbard_gauge_invariant_states(lattice_dim)
    ops = {}
    label = "site"
    for op in in_ops.keys():
        ops[op] = (
            gauge_basis[label].transpose() @ in_ops[op] @ gauge_basis[label]
        ).toarray()
    return ops


ops = Z2Hubbard_gauge_invariant_ops(lattice_dim=2)
# Define the ChargeInfo for 2 conserved quantum numbers:
# Two U(1) charges: 'N_tot', 'N_up'
# For U(1) we use 1, and for Z2 we use modulus 2.
charge_ops_list = ["N_tot", "N_up"]
chinfo = npc.ChargeInfo([1, 1], charge_ops_list)
# Create a list with the charge assignments for each state.
# Each entry is a list: [N_tot, N_up].
# You need to fill in the actual charge assignments for your model.
charges = []
state_labels = []
for ii in range(32):
    charge_i = [np.diagonal(ops[opname])[ii] for opname in charge_ops_list]
    charges.append(charge_i)
    state_labels.append(f"state_{ii}")

# Create the LegCharge from the list of charges
leg = npc.LegCharge.from_qflat(chinfo, charges, qconj=+1)
# Build the custom Site with the 30 basis states.
custom_site = Site(leg, state_labels=state_labels, sort_charge=True)
# Now, attach your operators from the dictionary to the custom site.
# The add_op method adds an operator that can later be accessed via custom_site.ops
for key, op in ops.items():
    custom_site.add_op(key, op)

# Generic Square Lattice
lat = Square(Lx=8, Ly=2, site=custom_site, bc=["open", "open"], bc_MPS="finite")
coeffs = {"U": 1.0, "t": 1.0, "h": 0.5}


class BosonicLadderModel(CouplingMPOModel):
    def init_terms(self, model_params):
        # 1) Onsite term:
        # For every site i, add the term U * N_pair_half.
        for site in self.lat.unit_cell:
            self.add_onsite(strength=model_params["U"], u=site, opname="N_pair_half")
        # 2) Hopping terms:
        for spin in ["up", "down"]:
            # along the x direction:
            self.add_coupling(
                strength=-model_params["t"],
                u1=0,
                op1=f"Q{spin}_px_dag",
                u2=0,
                op2=f"Q{spin}_mx",
                dx=[1, 0],
                plus_hc=True,
            )
            # along the y direction:
            self.add_coupling(
                strength=-model_params["t"],
                u1=0,
                op1=f"Q{spin}_py_dag",
                u2=0,
                op2=f"Q{spin}_my",
                dx=[1, 0],
                plus_hc=True,
            )


# Build an initial MPS from a product state.
init_state = ["state_0"] * lat.N_sites
psi = MPS.from_product_state(lat.mps_sites(), init_state, bc="finite")

# %%


class LadderZ2GaugeModel(CouplingMPOModel):
    def __init__(self, model_params):
        """
        model_params is expected to be a dictionary containing at least:
          - 'lat': the lattice (an instance of TenPy's Lattice class)
          - 'ops': dictionary of local operators
          - 'coeffs': dictionary of Hamiltonian coefficients (e.g. "U", "t", "h")
          - 'def_params': a dict of additional parameters for term construction (if any)
          - 'directions': list of directions, e.g. ['x', 'y'] (or similar strings)
          - 'has_obc': list (or tuple) indicating if open boundary conditions apply for each direction
          - 'lvals': a list/array used for the border_mask function
          - 'ham_format': a string that specifies the MPO format (e.g. "dense")
        """
        # Save model parameters for later use.
        self.n_sites = 10
        self.lat = model_params["lat"]
        self.ops = model_params["ops"]
        model_params = model_params["coeffs"]
        self.def_params = model_params.get("def_params", {})
        self.directions = model_params["directions"]
        self.has_obc = model_params["has_obc"]
        self.lvals = model_params["lvals"]
        self.ham_format = model_params["ham_format"]

        # Initialize the CouplingMPOModel with the lattice.
        super().__init__(self.lat)

    def init_terms(self, model_params: dict):
        # 0) Read and set parameters.
        t = model_params.get("t", 1.0)
        U = model_params.get("U", 1.0)
        h = model_params.get("V", 0.0)
        for site_in_unit_cell in range(len(self.lat.unit_cell)):
            self.add_onsite(U, site_in_unit_cell, "N_pair_half")
            self.add_onsite(100, site_in_unit_cell, "n_total")
        for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:

            self.add_coupling(hop, u1, "Bd", u2, "B", dx, plus_hc=True)
            self.add_coupling(V, u1, "N", u2, "N", dx)

    def init_terms(self):
        # Penalties on the borders:
        self.add_onsite_term(strength=model_params["alpha"], i=0, op="n_mx")
        self.add_onsite_term(
            strength=model_params["alpha"], i=self.n_sites - 1, op="n_px"
        )
        # --- COULOMB POTENTIAL (Local term) ---
        # add_onsite takes (strength, lat.unit_cell, operator matrix)
        self.add_onsite(
            strength=model_params["U"], u=self.lat.unit_cell, opname="N_pair_half"
        )

        # --- HOPPING TERMS (Two-body terms) ---
        # Loop over directions (e.g., 'x', 'y') and over spin species ("up", "down")
        for d in self.directions:
            for s in ["up", "down"]:
                # Define the names of the operators to be used on each bond.
                # Example: for direction 'x' and species 'up', these might be:
                #   "Qup_px_dag" for the creation operator on the "plus" side,
                #   "Qup_mx" for the annihilation operator on the "minus" side.
                op_names_list = [f"Q{s}_p{d}_dag", f"Q{s}_m{d}"]
                # Extract the corresponding operators from self.ops.
                op_list = [self.ops[op] for op in op_names_list]
                # Now, add the coupling term on every bond in direction d.
                # In TenPy, the lattice stores bond information.
                # For simplicity, we loop over all bonds in self.lat.pairs['nearest_neighbors'].
                # In a more refined implementation, you would filter these bonds by direction.
                for bond in self.lat.pairs["nearest_neighbors"]:
                    # Each bond is typically a tuple (i, j, vec) where i and j are site indices
                    # and vec is the bond vector.
                    i, j, vec = bond
                    # Check if the bond vector corresponds to the desired direction d.
                    # Here we assume that the bond vector's first component is non-zero for 'x' and
                    # second for 'y'. Adjust as needed.
                    if d == "x" and abs(vec[0]) > 0:
                        self.add_coupling(
                            model_params["t"],  # hopping amplitude
                            i,
                            op_list[0],
                            j,
                            op_list[1],
                            plus_hc=True,  # add the Hermitian conjugate
                        )
                    elif d == "y" and abs(vec[1]) > 0:
                        self.add_coupling(
                            model_params["t"],
                            i,
                            op_list[0],
                            j,
                            op_list[1],
                            plus_hc=True,
                        )

        # --- EXTERNAL ELECTRIC FIELD (Local term on borders) ---
        # Loop over each direction, and if open boundary conditions are applied, use a mask.
        for ii, d in enumerate(self.directions):
            border = f"p{d}"
            op_name = f"P_{border}"  # Example: "P_px" or "P_py"
            # Determine the mask if open boundary conditions are applied.
            if self.has_obc[ii]:
                mask = ~border_mask(
                    self.lvals, border
                )  # mask should be an array of booleans
            else:
                mask = None
            # Add the local term on each site.
            # Here we loop over all sites; in a more refined implementation you might add it only
            # on the border sites.
            for i in range(self.lat.N_sites):
                # The add_onsite method can accept an optional mask argument.
                self.add_onsite(model_params["h"], i, self.ops[op_name], mask=mask)


# Example usage:
if __name__ == "__main__":
    # TODO: Build your lattice (for instance, a ladder or chain) and custom site(s) first.
    # For simplicity, assume 'lat' is your pre-constructed lattice.
    # For example, for a 1D chain:
    from tenpy.models.lattice import Chain

    lvals = [10]  # number of sites in the chain (or adjust for a ladder)
    # custom_site has been built earlier as shown in your script.
    lat = Chain(L, custom_site)

    # Define your dictionary of operators, self.ops, from your ED code.
    # For this example, we assume 'ops' has been defined earlier.
    # Similarly, define coefficients and other parameters.
    model_params = {
        "lat": lat,
        "ops": ops,  # your dictionary of operators
        "coeffs": {"U": 1.0, "t": 1.0, "h": 0.5},
        "def_params": {},  # add any default parameters needed for term construction
        "directions": ["x", "y"],
        "has_obc": [True, True],  # assume open boundaries in both directions
        "lvals": np.arange(lat.N_sites),  # dummy lvals array; replace as needed
        "ham_format": "dense",  # or "sparse" if you prefer
    }
    # Instantiate the model
    model = LadderZ2GaugeModel(model_params)

    # Build an initial MPS from a product state.
    # Here we simply use the state "state_0" on all sites.
    from tenpy.networks.mps import MPS

    init_state = ["state_0"] * lat.N_sites
    psi = MPS.from_product_state(lat.mps_sites(), init_state, bc="finite")

    # Run DMRG:
    from tenpy.algorithms import dmrg

    dmrg_params = {
        "mixer": True,
        "chi_list": {0: 32},
        "max_sweeps": 10,
        "trunc_params": {"svd_min": 1e-10},
    }
    result = dmrg.run(psi, model, dmrg_params)

    # Print the final ground state energy.
    print("Final ground state energy:", result["E"])
# %%
#########################################
# 2. Prepare the unit cell for the ladder #
#########################################

# To enforce the Z2 flux matching on each link, we need to merge the Z2 charges of adjacent sites.
# Since each site has four flux operators (P_px, P_mx, P_py, P_my), and the constraint is:
#   - P_px(n) = P_mx(n+ux)  (horizontal link)
#   - P_py(n) = P_my(n+uy)  (vertical link)
# we will create a 2×2 unit cell (four sites) and then merge:
#   - siteA's 'Z2_px' with siteB's 'Z2_mx'
#   - siteC's 'Z2_px' with siteD's 'Z2_mx'
#   - siteA's 'Z2_py' with siteC's 'Z2_my'
#   - siteB's 'Z2_py' with siteD's 'Z2_my'
#
# This assumes a tiling of the ladder into 2 columns; for a ladder (2 rows) this is convenient.
#
# Create four copies of the custom_site for the unit cell:
siteA = custom_site
siteB = custom_site.copy()
siteC = custom_site.copy()
siteD = custom_site.copy()

# Merge the Z2 charges on the relevant links:
multi_sites_combine_charges(
    [siteA, siteB, siteC, siteD],
    same_charges=[
        [
            (0, "Z2_px"),
            (1, "Z2_mx"),
        ],  # Horizontal: right flux of siteA equals left flux of siteB.
        [
            (2, "Z2_px"),
            (3, "Z2_mx"),
        ],  # Horizontal: right flux of siteC equals left flux of siteD.
        [
            (0, "Z2_py"),
            (2, "Z2_my"),
        ],  # Vertical: up flux of siteA equals down flux of siteC.
        [
            (1, "Z2_py"),
            (3, "Z2_my"),
        ],  # Vertical: up flux of siteB equals down flux of siteD.
    ],
)

######################################
# 3. Construct the Ladder Lattice    #
######################################

# We define a ladder as a 2-row lattice.
# Let Lx be the number of columns (must be even to match the 2-column unit cell),
# and Ly = 2 (for a ladder). The total number of sites is Lx*Ly.
Lx = 4  # TODO: set the desired number of columns (must be even)
Ly = 2  # Ladder: 2 rows

# We tile the 2×2 unit cell to cover the ladder.
# For a ladder, one unit cell in the y-direction covers both rows,
# and Lx//2 unit cells in the x-direction cover the Lx columns.
from tenpy.models.lattice import Lattice

lat = Lattice(
    [Lx // 2, 1],
    [siteA, siteB, siteC, siteD],
    unit_cell_dimensions=[2, 2],
    bc=["open", "open"],
    bc_MPS="finite",
)

##################################################
# 4. Initialize the MPS using a product state    #
##################################################

# For the product state, you plan to use the dominant configuration
# from an ED simulation on a smaller system.
# Here, we use a placeholder: we set all sites to a chosen state.
# Replace "state_0" with your ED-derived state label if available.
init_state = ["state_0"] * lat.N_sites
psi = MPS.from_product_state(lat.mps_sites(), init_state, bc="finite")

#######################################
# 5. Define the Model with Hamiltonian #
#######################################

from tenpy.models.model import CouplingModel


class LadderZ2GaugeModel(CouplingModel):
    def __init__(self, model_params):
        # Retrieve the lattice from the parameters.
        lat = model_params["lat"]
        CouplingModel.__init__(self, lat)

        # Example: Add on-site potential term (e.g., chemical potential times N_tot)
        # TODO: Replace 'N_tot' with your actual operator name if different.
        mu = model_params.get("mu", 1.0)
        for i in range(lat.N_sites):
            self.add_onsite(mu, i, "N_tot")  # Adjust the sign and magnitude as needed.

        # Example: Add hopping terms along x-direction.
        # TODO: Define your hopping operator (e.g., boson creation/annihilation operators)
        # and add horizontal couplings. The following is a placeholder.
        Jx = model_params.get("Jx", 1.0)
        # Loop over horizontal bonds (adapt indices according to your lattice structure)
        for bond in lat.pairs["nearest_neighbors"]:
            # Check if the bond is horizontal:
            # (Here, you might use bond vector info from lat.coupling_info)
            # TODO: Insert a condition to select only horizontal bonds.
            # For each horizontal bond, add the hopping coupling:
            self.add_coupling(-Jx, bond[0], "bdag", bond[1], "b", plus_hc=True)

        # Example: Add hopping terms along y-direction.
        Jy = model_params.get("Jy", 1.0)
        # Loop over vertical bonds (adapt indices accordingly)
        for bond in lat.pairs["nearest_neighbors"]:
            # TODO: Insert a condition to select only vertical bonds.
            self.add_coupling(-Jy, bond[0], "bdag", bond[1], "b", plus_hc=True)

        # TODO: Add additional local operator terms (e.g., another local operator)
        # You can use self.add_onsite(...) or self.add_coupling(...) as needed.

        # NOTE: Ensure that all operators (like 'b', 'bdag', 'N_tot', etc.)
        # are defined in your custom site operator dictionary.
        # If not, you must add them, e.g., via site.add_op('b', b_matrix)

        # End of model definition.
        pass


# Define model parameters (adjust numerical values as needed)
model_params = {
    "lat": lat,
    "Jx": 1.0,  # hopping along x-direction
    "Jy": 1.0,  # hopping along y-direction
    "mu": 1.0,  # on-site potential
    # Add any additional parameters here.
}

# Instantiate the model
model = LadderZ2GaugeModel(model_params)

#########################################
# 6. Configure and Run DMRG             #
#########################################

# Set up DMRG parameters (tweak as needed for your simulation)
dmrg_params = {
    "mixer": True,  # enables mixing (helps convergence sometimes)
    "chi_list": {0: 32},  # maximum bond dimension (adjust as needed)
    "max_sweeps": 10,  # number of DMRG sweeps
    "trunc_params": {"svd_min": 1e-10},
}

# Run DMRG on the model with the initial MPS.
result = dmrg.run(psi, model, dmrg_params)

# Print final energy and basic observables:
print("Final ground state energy: ", result["E"])
print("Entanglement entropies:", psi.entanglement_entropy())

# Example: Measure a local observable (e.g., N_tot) on each site.
# TODO: Replace 'N_tot' with the proper operator name if different.
N_vals = [psi.expectation_value("N_tot", i) for i in range(lat.N_sites)]
print("Local N_tot expectation values:", N_vals)

# End of script.
