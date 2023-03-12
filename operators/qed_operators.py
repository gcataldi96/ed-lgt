# %%
import numpy as np
from itertools import product


def gauge_invariant_states(rishons_number, lattice_dim=2):
    single_rishon_configs = np.arange(rishons_number + 1)
    gauge_states = {"odd": [], "even": []}
    # Run over even and odd sites
    for ii, site in enumerate(list(gauge_states.keys())):
        # Define a counter
        counter = 0
        # Define the parity of odd (-1) and even (+1) sites
        parity = -1 if site == "odd" else +1
        # Possible matter occupation number
        for matter in [0, 1]:
            # Possible rishon occupation number
            for n_py, n_my, n_px, n_mx in product(single_rishon_configs, repeat=4):
                # Define the Gauss law left and right hand sides
                left = matter + n_mx + n_px + n_my + n_py
                right = lattice_dim * rishons_number + 1 / 2 * (1 + parity)
                if left == right:
                    counter += 1
                    # if the Gauss Law is satisfied, save the state
                    gauge_states[site].append(
                        [counter, matter, n_mx, n_px, n_my, n_py, parity]
                    )
                    if (n_px == n_mx == n_py == n_my == 2) or (
                        n_px == n_mx == n_py == n_my == 0
                    ):
                        print("ATTTENZIONE")
        # Convert list of gauge inv state into matrices
        gauge_states[site] = np.matrix(gauge_states[site])
    return gauge_states


from scipy.sparse import csr_matrix
def hopping(gauge_states, dim):
    # Spatial directions
    directions = "xyz"[:dim]
    # Dictionary for operators
    ops = {}
    # Distinguish between even & odd sites
    for site in gauge_states:
        # Get the dimensionality of the gauge-invariant local basis (even/odd)
        dim_ops = len(gauge_states[site])
        # Run over directions and versus
        for d in directions:
            for s in "mp":
                ops[f"Q_dagger_{s}{d}_{site}"] = np.zeros((dim_ops, dim_ops))
        for j in gauge_states[site]:
            # Q_mx_dagger
            if j[1] == 0 and j[2] > 0:
                for i in gauge_states[site]:
                    if (i[1], i[2], i[3], i[4], i[5]) == (
                        1,
                        j[2] - 1,
                        j[3],
                        j[4],
                        j[5],
                    ):
                        ops[f"Q_dagger_mx_{site}"][i[0] - 1, j[0] - 1] = np.power(
                            -1, j[2] + 1
                        )
            # Q_my_dagger
            if j[1] == 0 and j[4] > 0:
                for i in gauge_states[site]:
                    if (i[1], i[2], i[3], i[4], i[5]) == (
                        1,
                        j[2],
                        j[3],
                        j[4] - 1,
                        j[5],
                    ):
                        ops[f"Q_dagger_my_{site}"][i[0] - 1, j[0] - 1] = np.power(
                            -1, j[2] + j[4] + 1
                        )
            # Q_px_dagger
            if j[1] == 0 and j[3] > 0:
                for i in gauge_states[site]:
                    if (i[1], i[2], i[3], i[4], i[5]) == (
                        1,
                        j[2],
                        j[3] - 1,
                        j[4],
                        j[5],
                    ):
                        ops[f"Q_dagger_px_{site}"][i[0] - 1, j[0] - 1] = np.power(
                            -1, j[2] + j[4]
                        )
            # Q_py_dagger
            if j[1] == 0 and j[5] > 0:
                for i in gauge_states[site]:
                    if (i[1], i[2], i[3], i[4], i[5]) == (
                        1,
                        j[2],
                        j[3],
                        j[4],
                        j[5] - 1,
                    ):
                        ops[f"Q_dagger_px_{site}"][i[0] - 1, j[0] - 1] = np.power(
                            -1, j[2] + j[3] + j[4]
                        )
        # Run over directions and versus
        for d in directions:
            for s in "mp":
                ops[f"Q_dagger_{s}{d}_{site}"] = csr_matrix(
                    ops[f"Q_dagger_{s}{d}_{site}"]
                )
                # ADD THE HERMITIAN CONJUGATE
                ops[f"Q_{s}{d}_{site}"] = csr_matrix(
                    ops[f"Q_dagger_{s}{d}_{site}"].conj().transpose()
                )
    return ops


def electric_field(gauge_states):
    # Dictionary for operators
    ops = {}
    # Distinguish between even & odd sites
    for site in gauge_states:
        # Get the dimensionality of the gauge-invariant local basis (even/odd)
        dim_ops = len(gauge_states[site])
        ops[f"E_square_{site}"] = np.zeros((dim_ops, dim_ops))
        for i in gauge_states[site]:
            for j in gauge_states[site]:
                if i[0] == j[0]:
                    ops[f"E_square_{site}"][i[0] - 1, j[0] - 1] = (1 / 2) * (
                        (j[2] - 1) ** 2
                        + (j[3] - 1) ** 2
                        + (j[4] - 1) ** 2
                        + (j[5] - 1) ** 2
                    )
        # GET THE SPARSE MATRIX
        ops[f"E_square_{site}"] = csr_matrix(ops[f"E_square_{site}"])
    return ops


def matter_operator(gauge_states):
    # Dictionary for operators
    ops = {}
    # Distinguish between even & odd sites
    for site in gauge_states:
        # Get the dimensionality of the gauge-invariant local basis (even/odd)
        dim_ops = len(gauge_states[site])
        ops[f"Mass_{site}"] = np.zeros((dim_ops, dim_ops))
        for i in gauge_states[site]:
            for j in gauge_states[site]:
                if i[0] == j[0]:
                    if site == "odd":
                        ops[f"Mass_{site}"][i[0] - 1, j[0] - 1] = j[1]
                    else:
                        ops[f"Mass_{site}"][i[0] - 1, j[0] - 1] = 1 - j[1]
        ops[f"Mass_{site}"] = csr_matrix(ops[f"Mass_{site}"])
    return ops


def corner_operators(gauge_states):
    # List of possible corners
    corner_dir = ["mx_my", "my_px", "px_py", "py_mx"]
    # Dictionary for operators
    ops = {}
    # Distinguish between even & odd sites
    for site in gauge_states:
        # Get the dimensionality of the gauge-invariant local basis (even/odd)
        dim_ops = len(gauge_states[site])
        for corner in corner_dir:
            ops[f"C_{corner}_{site}"] = np.zeros((dim_ops, dim_ops))
            for j in gauge_states[site]:
                if j[5] < 2 and j[3] > 0:
                    for i in gauge_states[site]:
                        if (i[1], i[2], i[3], i[4], i[5]) == (
                            j[1],
                            j[2],
                            j[3] - 1,
                            j[4],
                            j[5] + 1,
                        ):
                            ops[f"C_px_py_{site}"][i[0] - 1, j[0] - 1] = np.power(
                                -1, j[3] + 1
                            )
                if j[4] < 2 and j[2] > 0:
                    for i in gauge_states[site]:
                        if (i[1], i[2], i[3], i[4], i[5]) == (
                            j[1],
                            j[2] - 1,
                            j[3],
                            j[4] + 1,
                            j[5],
                        ):
                            ops[f"C_mx_my_{site}"][i[0] - 1, j[0] - 1] = np.power(
                                -1, j[4]
                            )
                if j[2] < 2 and j[5] > 0:
                    for i in gauge_states[site]:
                        if (i[1], i[2], i[3], i[4], i[5]) == (
                            j[1],
                            j[2] + 1,
                            j[3],
                            j[4],
                            j[5] - 1,
                        ):
                            ops[f"C_py_mx_{site}"][i[0] - 1, j[0] - 1] = np.power(
                                -1, j[3] + j[4]
                            )
            ops[f"C_{corner}_{site}"] = csr_matrix(ops[f"C_{corner}_{site}"])
    return ops


# %%
def get_qed_operators(gauge_states):
    ops = {}
    # ops |= ID(pure_theory)
    # ops |= W_operators(pure_theory)
    #ops |= number_operator(gauge_states)
    ops |= corner_operators(gauge_states)
    ops |= electric_field(gauge_states)
    ops |= hopping(gauge_states)
    ops |= matter_operator(gauge_states)

    return ops


# %%
rishons_number = 2
gauge_states = gauge_invariant_states(rishons_number)
# %%
ops=get_qed_operators(gauge_states)
# %%
