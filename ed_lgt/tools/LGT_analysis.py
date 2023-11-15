import numpy as np

__all__ = ["get_energy_density"]


def get_energy_density(
    tot_energy,
    lvals,
    penalty,
    border_penalty=False,
    link_penalty=False,
    plaquette_penalty=False,
    has_obc=True,
):
    # ACQUIRE LATTICE DIMENSIONS
    Lx = lvals[0]
    Ly = lvals[1]
    n_borders = 0
    n_links = 0
    n_plaquettes = 0
    if has_obc:
        if border_penalty:
            n_borders += 2 * Lx + 2 * Ly
        if link_penalty:
            n_links += Lx * (Ly - 1) + Ly * (Lx - 1)
        if plaquette_penalty:
            n_plaquettes += (Lx - 1) * (Ly - 1)
    else:
        if link_penalty:
            n_links += 2 * (Lx * Ly)
        if plaquette_penalty:
            n_plaquettes += Lx * Ly
    # COUNTING THE TOTAL NUMBER OF PENALTIES
    n_penalties = n_borders + n_links + n_plaquettes
    # RESCALE ENERGY
    energy_density = (tot_energy - n_penalties * penalty) / (Lx * Ly)
    return energy_density


def define_measurements(obs_list, stag_obs_list=None, has_obc=False):
    if not isinstance(obs_list, list):
        raise TypeError(f"obs_list must be a LIST, not a {type(obs_list)}")
    else:
        for obs in obs_list:
            if not isinstance(obs, str):
                raise TypeError(f"obs_list elements are STR, not a {type(obs)}")
    if not isinstance(has_obc, bool):
        raise TypeError(f"has_obc should be a BOOL, not a {type(has_obc)}")
    # ===========================================================================
    # Default observables
    measures = {}
    measures["energy"] = []
    measures["energy_density"] = []
    measures["entropy"] = []
    if not has_obc:
        measures["rho_eigvals"] = []
    else:
        measures["state_configurations"] = []
    # ===========================================================================
    # Observables resulting from Operators
    for obs in obs_list:
        measures[obs] = []
        measures[f"delta_{obs}"] = []
    # Observables resulting from STAGGERED Operators
    if stag_obs_list is not None:
        if not isinstance(stag_obs_list, list):
            raise TypeError(
                f"stag_obs_list must be a LIST, not a {type(stag_obs_list)}"
            )
        else:
            for obs in stag_obs_list:
                if not isinstance(obs, str):
                    raise TypeError(
                        f"stag_obs_list elements are STR, not a {type(obs)}"
                    )
        for site in ["even", "odd"]:
            for obs in stag_obs_list:
                measures[f"{obs}_{site}"] = []
                measures[f"delta_{obs}_{site}"] = []
    return measures


def get_SU2_topological_invariant(link_parity_op, lvals, psi, axis):
    # NOTE: it works only on a 2x2 system
    op_list = [link_parity_op, link_parity_op]
    if axis == "x":
        op_sites_list = [0, 1]
    elif axis == "y":
        op_sites_list = [0, lvals[0]]
    else:
        raise ValueError(f"axis can be only x or y not {axis}")
    sector = np.real(
        np.dot(
            np.conjugate(psi),
            two_body_op(op_list, op_sites_list, lvals, has_obc=True).dot(psi),
        )
    )
    print(f"P{axis} TOPOLOGICAL SECTOR: {sector}")
    print("----------------------------------------------------")
    return sector
