import numpy as np
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from ed_lgt.models import QED_Model
from simsio import run_sim
from time import perf_counter
from ed_lgt.modeling import get_projector_for_efficient_density_matrix
import logging


def apply_projection(projector: np.ndarray, operator: np.ndarray):
    """
    Project an operator onto the subspace defined by a projector.

    Given:
      - projector (np.ndarray): a (N, k) matrix
      - operator (np.ndarray): a (N, N) operator

    Returns:
      - O_eff: the effective operator $O_eff = P^{\dagger}\cdot O \cdot P$
    """
    # Check the shape of the projector
    if projector.shape[0] != operator.shape[0]:
        msg = f"Projector and operator have incompatible shapes {projector.shape} {operator.shape}"
        raise ValueError(msg)
    # Return the projected operator
    return np.dot(projector.conj().T, np.dot(operator, projector))


def project_operators(
    ops_dict: dict,
    loc_dims: np.ndarray,
    gauge_basis: dict = None,
    lattice_labels=None,
    extra_projector: np.ndarray = None,
):
    """
    Project a dictionary of operators into a gauge-invariant or optimal subspace.

    Parameters:
        ops_dict (dict): Dictionary of operators (expected to be scipy.sparse.csr_matrix).
        loc_dims (np.ndarray): Array containing local Hilbert space dimensions.
        gauge_basis (dict, optional): Dictionary mapping lattice sites to their gauge-invariant basis projectors.
        lattice_labels (list, optional): List of labels for each lattice site.
        extra_projector (np.ndarray, optional): Additional projector to further reduce the local Hilbert space.

    Returns:
        dict: New dictionary of projected operators with shape (n_sites, max_loc_dim, max_loc_dim).
        the keys are the same as the input dictionary.
        For each operator, the value is a 3D array of shape (n_sites, max_loc_dim, max_loc_dim)
        which contains the effective matrix for each site, accounting for the possibility of
        different local Hilbert spaces among the sites.
    """
    # Set the number of sites
    n_sites = len(loc_dims)
    # Determine effective local dimensions
    eff_loc_dims = (
        np.array([extra_projector.shape[1]] * n_sites, dtype=loc_dims.dtype)
        if extra_projector is not None
        else np.copy(loc_dims)
    )
    max_loc_dim = max(eff_loc_dims)
    # Initialize new dictionary
    new_ops_dict = {
        op: np.zeros((n_sites, max_loc_dim, max_loc_dim), dtype=ops_dict[op].dtype)
        for op in ops_dict
    }
    # Iterate over operators
    for op, operator in ops_dict.items():
        # Run over the sites
        for jj, loc_dim in enumerate(eff_loc_dims):
            # For Lattice Gauge Theories where sites have different Hilbert Bases
            if gauge_basis is not None:
                # Get the label of the site
                site_label = lattice_labels[jj]
                # Get the projected operator
                eff_op = apply_projection(
                    projector=gauge_basis[site_label].toarray(),
                    operator=operator.toarray(),
                )
            # For Theories where all the sites have the same Hilber basis
            else:
                eff_op = operator.toarray()
            # If an extra projector is given (like the one to reduce the local Hilbert space)
            if extra_projector is not None:
                eff_op = apply_projection(projector=extra_projector, operator=eff_op)
            # Save it inside the new list of operators
            new_ops_dict[op][jj, :loc_dim, :loc_dim] = np.real(eff_op)
    return new_ops_dict


logger = logging.getLogger(__name__)
with run_sim() as sim:
    # Start measuring time
    start_time = perf_counter()
    # -------------------------------------------------------------------------------
    # MODEL HAMILTONIAN
    model = QED_Model(**sim.par["model"])
    m = sim.par["m"] if not model.pure_theory else None
    model.build_Hamiltonian(sim.par["g"], m)
    # -------------------------------------------------------------------------------
    # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
    n_eigs = sim.par["hamiltonian"]["n_eigs"]
    model.diagonalize_Hamiltonian(n_eigs, model.ham_format)
    sim.res["energy"] = model.H.Nenergies
    # ---------------------------------------------------------------------
    # LIST OF LOCAL OBSERVABLES
    local_obs = ["E_px", "E_mx", "E_my", "E_py", "E_square"]
    if not model.pure_theory:
        local_obs += ["N"]
    for obs in local_obs:
        sim.res[obs] = np.zeros(n_eigs, dtype=float)
    # LIST OF TWOBODY CORRELATORS
    twobody_obs = [[f"P_p{d}", f"P_m{d}"] for d in model.directions]
    twobody_axes = [d for d in model.directions]
    # LIST OF PLAQUETTE OPERATORS
    if model.dim == 2:
        plaquette_obs = [["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"]]
    else:
        plaquette_obs = None
    # DEFINE OBSERVABLES
    model.get_observables(
        local_obs, twobody_obs, plaquette_obs, twobody_axes=twobody_axes
    )
    # ENTROPY
    partition_indices = [0]
    # Build the list of environment and subsystem sites configurations
    model.get_subsystem_environment_configs(keep_indices=partition_indices)
    sim.res["entropy"] = np.zeros(n_eigs, dtype=float)
    # -------------------------------------------------------------------------------
    for ii in range(model.H.n_eigs):
        model.H.print_energy(ii)
        if not model.momentum_basis:
            # -----------------------------------------------------------------------
            # ENTROPY
            if sim.par["get_entropy"]:
                sim.res["entropy"][ii] = model.H.Npsi[ii].entanglement_entropy(
                    partition_indices,
                    model.subsystem_configs,
                    model.env_configs,
                    model.unique_subsys_configs,
                    model.unique_env_configs,
                )
            # -----------------------------------------------------------------------
            # STATE CONFIGURATIONS
            if sim.par["get_state_configs"]:
                model.H.Npsi[ii].get_state_configurations(1e-1, model.sector_configs)
        # ---------------------------------------------------------------------------
        # MEASURE OBSERVABLES
        model.measure_observables(ii)
        for obs in local_obs:
            sim.res[obs][ii] = np.mean(model.res[obs])
    # -------------------------------------------------------------------------------
    # LOCAL BASIS OPTIMIZATION
    # Get the reduced density matrix of a single site in the ground state
    RDM = model.H.Npsi[0].reduced_density_matrix(
        partition_indices,
        model.subsystem_configs,
        model.env_configs,
        model.unique_subsys_configs,
        model.unique_env_configs,
    )
    logger.info(f"{norm(RDM-np.diag(np.diag(RDM)), ord='nuc')}")
    # Get the reduced and optimized operators of a single site in the ground state
    proj, sim.res["eigvals"], sim.res["vecs"] = (
        get_projector_for_efficient_density_matrix(RDM, 1e-3)
    )
    new_ops = {}
    true_ops = {}
    for op in model.ops.keys():
        true_ops[op] = (
            model.gauge_basis["site"].transpose()
            @ model.ops[op]
            @ model.gauge_basis["site"]
        ).toarray()
        if op in ["E_px"]:
            logger.info(f"OPERATOR {op} {true_ops[op].shape}")
            logger.info(f"{csr_matrix(true_ops[op])}")
    # project the operators
    for op in true_ops.keys():
        new_ops[op] = (
            csr_matrix(proj).transpose().conj() @ true_ops[op] @ csr_matrix(proj)
        )
        if op in ["E_px"]:
            logger.info(f"{csr_matrix(new_ops[op])}")
    sim.res["expvals"] = np.zeros(len(sim.res["eigvals"]), dtype=float)
    for obs in ["E_px", "E_mx", "E_my", "E_py", "E_square"]:
        sim.res[f"exp_{obs}"] = np.zeros(len(sim.res["eigvals"]), dtype=float)
        for ii in range(len(sim.res["eigvals"])):
            sim.res[f"exp_{obs}"][ii] = np.real(
                np.dot(
                    sim.res["vecs"][:, ii].conj(),
                    np.dot(true_ops[obs], sim.res["vecs"][:, ii]),
                )
            )

    # sim.res["E_old"] = model.gauge_basis["site"].transpose()model.ops["E_px"].data
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
