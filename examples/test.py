# %%
from numba import njit
import numpy as np
from itertools import product
from scipy.sparse import csr_matrix
from ed_lgt.modeling import get_lattice_borders_labels, LGT_border_configs
from ed_lgt.operators import (
    couple_two_spins,
    add_new_spin,
    SU2_singlet_canonical_vector,
    get_spin_Hilbert_spaces,
    get_SU2_singlets,
    SU2_gen_gauge_invariant_states,
)
from ed_lgt.tools import get_time
from time import perf_counter
import logging

logger = logging.getLogger(__name__)

# %%
import numpy as np

# Example matrix
matrix = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 2, 3]])

# Step 1: Identify rows with at least one nonzero entry
nonzero_rows = np.any(matrix != 0, axis=1)

# Step 2: Get the indices of the rows with nonzero entries
valid_row_indices = np.where(nonzero_rows)[0]

# Step 3: Loop over valid rows and their nonzero column entries
for row in valid_row_indices:
    # Find the column indices with nonzero entries for this row
    nonzero_cols = np.where(matrix[row, :] != 0)[0]

    # Loop over the nonzero entries in the row
    for col in nonzero_cols:
        value = matrix[row, col]
        print(f"Row {row}, Column {col} has nonzero value: {value}")


# %%
@njit
def get_factorials(max_n):
    """Calculate log of factorials up to max_n for use in later functions."""
    log_factorials = np.zeros(max_n + 1)
    for i in range(1, max_n + 1):
        log_factorials[i] = log_factorials[i - 1] + np.log(i)
    return log_factorials


@njit
def wigner2_3j(j1, j2, j3, m1, m2, m3):
    if int(j1 * 2) != j1 * 2 or int(j2 * 2) != j2 * 2 or int(j3 * 2) != j3 * 2:
        raise ValueError("j values must be integer or half integer")
    if int(m1 * 2) != m1 * 2 or int(m2 * 2) != m2 * 2 or int(m3 * 2) != m3 * 2:
        raise ValueError("m values must be integer or half integer")
    if m1 + m2 + m3 != 0:
        return 0
    if (abs(m1) > j1) or (abs(m2) > j2) or (abs(-m3) > j3):
        return 0

    maxfact = max(j1 + j2 + j3 + 1, j1 + abs(m1), j2 + abs(m2), j3 + abs(m3))
    log_factorials = get_factorials(int(maxfact))

    argsqrt = (
        log_factorials[int(j1 + j2 - j3)]
        + log_factorials[int(j1 - j2 + j3)]
        + log_factorials[int(-j1 + j2 + j3)]
        + log_factorials[int(j1 - m1)]
        + log_factorials[int(j1 + m1)]
        + log_factorials[int(j2 - m2)]
        + log_factorials[int(j2 + m2)]
        + log_factorials[int(j3 + m3)]
        + log_factorials[int(j3 - m3)]
        - log_factorials[int(j1 + j2 + j3 + 1)]
    )

    if argsqrt < 0:
        raise ValueError("Square root of negative number encountered.")

    ressqrt = np.sqrt(np.exp(argsqrt))
    imin = int(max(-j3 + j1 + m2, -j3 + j2 - m1, 0))
    imax = int(min(j2 + m2, j1 - m1, j1 + j2 - j3))
    sumres = 0
    for ii in range(imin, imax + 1):
        log_term = (
            log_factorials[ii]
            + log_factorials[int(ii + j3 - j1 - m2)]
            + log_factorials[int(j2 + m2 - ii)]
            + log_factorials[int(j1 - ii - m1)]
            + log_factorials[int(ii + j3 - j2 + m1)]
            + log_factorials[int(j1 + j2 - j3 - ii)]
        )
        if ii % 2 == 1:  # Adjust for alternating signs
            log_term = -log_term
        sumres += np.exp(
            log_term
        )  # sum exponentiated terms directly if not using logsumexp

    res = ressqrt * sumres * ((-1) ** int(j1 - j2 - m3))
    return res


@njit
def m_vals(spin):
    return np.arange(-spin, spin + 1)[::-1]


@njit
def spin_space(spin):
    """
    Returns the size of the Hilbert space for a given spin,
    which can be an integer or half-integer to reflect physical spins.
    """
    return int(2 * spin + 1)


@njit
def get_factorials(max_n):
    """Calculate factorials up to max_n for use in later functions."""
    factorials = np.zeros(max_n + 1)
    factorials[0] = 1
    for i in range(1, max_n + 1):
        factorials[i] = factorials[i - 1] * i
    return factorials


@njit
def check_triangle_rule(j1, j2, j3):
    return (
        abs(int(2 * (j1 - j2))) <= int(2 * j3) <= int(2 * (j1 + j2))
        and int(2 * (j1 + j2 + j3)) % 1 == 0
    )


@njit
def wigner_3j(j1, j2, j3, m1, m2, m3):
    if int(j1 * 2) != j1 * 2 or int(j2 * 2) != j2 * 2 or int(j3 * 2) != j3 * 2:
        raise ValueError("j values must be integer or half integer")
    if int(m1 * 2) != m1 * 2 or int(m2 * 2) != m2 * 2 or int(m3 * 2) != m3 * 2:
        raise ValueError("m values must be integer or half integer")
    if m1 + m2 + m3 != 0:
        return 0
    prefid = int((-1) ** int(j1 - j2 - m3))
    m3 = -m3
    a1 = j1 + j2 - j3
    a2 = j1 - j2 + j3
    a3 = -j1 + j2 + j3
    if np.any([a1 < 0, a2 < 0, a3 < 0]):
        return 0

    maxfact = max(j1 + j2 + j3 + 1, j1 + abs(m1), j2 + abs(m2), j3 + abs(m3))
    factorials = get_factorials(int(maxfact))

    argsqrt = int(
        factorials[int(j1 + j2 - j3)]
        * factorials[int(j1 - j2 + j3)]
        * factorials[int(-j1 + j2 + j3)]
        * factorials[int(j1 - m1)]
        * factorials[int(j1 + m1)]
        * factorials[int(j2 - m2)]
        * factorials[int(j2 + m2)]
        * factorials[int(j3 - m3)]
        * factorials[int(j3 + m3)]
        / factorials[int(j1 + j2 + j3 + 1)]
    )
    print(argsqrt)
    ressqrt = np.sqrt(argsqrt)
    imin = int(max(-j3 + j1 + m2, -j3 + j2 - m1, 0))
    imax = int(min(j2 + m2, j1 - m1, j1 + j2 - j3))
    sumres = 0
    for ii in range(imin, imax + 1):
        den = (
            factorials[ii]
            * factorials[int(ii + j3 - j1 - m2)]
            * factorials[int(j2 + m2 - ii)]
            * factorials[int(j1 - ii - m1)]
            * factorials[int(ii + j3 - j2 + m1)]
            * factorials[int(j1 + j2 - j3 - ii)]
        )
        sumres = sumres + int((-1) ** ii) / den

    res = ressqrt * sumres * prefid
    return res


@njit
def clebsch_gordan(j1, j2, j3, m1, m2, m3):
    """Calculate the Clebsch-Gordan coefficients using Wigner 3j symbols."""
    return (
        int((-1) ** int(j1 - j2 + m3))
        * np.sqrt(2 * j3 + 1)
        * wigner2_3j(j1, j2, j3, m1, m2, -m3)
    )


@njit
def estimate_max_states(j1, j2, get_singlet, m1values):
    """Estimate the maximum number of valid states for coupling spins."""
    if get_singlet:
        # Only the singlet state, j3 = 0, m3 = 0
        # Count pairs (m1, m2) such that m1 + m2 = 0
        valid_pairs = 0
        for m1 in m1values:
            for m2 in m_vals(j2):
                if int(2 * m1) + int(2 * m2) == 0:
                    valid_pairs += 1
        return valid_pairs
    else:
        total_states = 0
        for j3 in np.arange(np.abs(j1 - j2), j1 + j2 + 1):
            for m3 in m_vals(j3):
                valid_pairs = 0
                for m1 in m1values:
                    for m2 in m_vals(j2):
                        if int(2 * m1) + int(2 * m2) == int(2 * m3):
                            valid_pairs += 1
                total_states += valid_pairs
        return total_states


@njit
def couple_two_spins2(j1, j2, get_singlet, M=None):
    """
    This function computes SU(2) states obtained by combining two spins j1, j2
    and computing Clebsh-Gordan coefficients CG. The possible outcomes j3,m3 are
    the ones with non null CG

    Args:
        j1 (integer/half-integer): spin of the 1st particle
        j2 (integer/half-integer): spin of the 2nd particle
        get_singlet (bool, optional): if true, look only at the (j1, j2)-combinations providing an SU(2) singlet.
            Defaults to False.
        M (integer/half-integer, optional): spin-z component of the 1st particle. Defaults to None.

    Returns:
        np.array: [(j1, m1,) j2, m2, CG, j3, m3]
    """
    if M is not None:
        m1values = np.array([M], dtype=np.float64)
        size = 5
    else:
        m1values = m_vals(j1)
        size = 7
    if get_singlet:
        j3_values = np.array([0], dtype=np.float64)
    else:
        j3_values = np.arange(np.abs(j1 - j2), j1 + j2 + 1, 1.0)
    # Compute the maximum number of nonzero CG coefficients
    max_states = estimate_max_states(j1, j2, get_singlet, m1values)
    SU2_states = np.zeros((max_states, size), dtype=np.float64)
    count = 0
    if M is not None:
        for m1 in m1values:
            for m2 in m_vals(j2):
                for j3 in j3_values:
                    for m3 in m_vals(j3):
                        if int(2 * m1) + int(2 * m2) == int(2 * m3):
                            CG = clebsch_gordan(j1, j2, j3, m1, m2, m3)
                            if CG != 0:
                                SU2_states[count] = [j2, m2, CG, j3, m3]
                                count += 1
    else:
        for m1 in m1values:
            for m2 in m_vals(j2):
                for j3 in j3_values:
                    for m3 in m_vals(j3):
                        if int(2 * m1) + int(2 * m2) == int(2 * m3):
                            CG = clebsch_gordan(j1, j2, j3, m1, m2, m3)
                            if CG != 0:
                                SU2_states[count] = [j1, m1, j2, m2, CG, j3, m3]
                                count += 1
    return SU2_states[:count]


@njit
def add_new_spin2(previous_configs, new_spin, get_singlet):
    """
    Couples a list of spin configurations with a new spin and determines the resulting configurations.
    Overall, this function is a wrapper of the previous function "couple_two_spins".

    Args:
        previous_configs (list): The current list of spin-configurations.
            Each spin-configuration is a list whose last two terms corresponds to:

            - spin_config[-2]= J1 total spin associated to the spin-configuration

            - spin_config[-1]= M1 correspinding z-component of the total spin of the given configuration

            These two variables will be combined with the new_spin to obtain a new configuation

        new_spin (half/integer): The new spin to be coupled with each configuration in previous_config.

        get_singlet (bool): Specify if the resulting configuration must be a singlet.

    Returns:
        list: Updated list of spin configurations after coupling with new_spin.
    """
    old_size = previous_configs.shape[1]
    new_size = old_size + 5
    # Estimate the amount of configs
    n_configs = 0
    for config in previous_configs:
        n_configs += estimate_max_states(
            j1=config[-2],
            j2=new_spin,
            get_singlet=get_singlet,
            m1values=np.array([config[-1]], dtype=np.float64),
        )
    updated_configs = np.zeros((n_configs, new_size), dtype=np.float64)
    # Update the configs
    tmp = 0
    for config in previous_configs:
        tmp_configs = couple_two_spins2(
            j1=config[-2], j2=new_spin, get_singlet=get_singlet, M=config[-1]
        )
        for new_config in tmp_configs:
            if new_config.shape[0] != 0:
                updated_configs[tmp, :old_size] = config
                updated_configs[tmp, old_size:new_size] = new_config
                tmp += 1
    return updated_configs[:tmp, :]


# %%
j1 = 4
j2 = 4
j3 = 8
j4 = 2
spin_set = np.array([j1, j2, j3, j4], dtype=float)
pure_theory = True
lattice_dim = 2
s_max = 1
# %%
start_time = perf_counter()
b = couple_two_spins2(j1, j2, False)
b = add_new_spin2(b, j3, True)
end_time = perf_counter()
logger.info(f"TIME SIMS {format(end_time-start_time, '.5f')}")
# %%
start_time = perf_counter()
a = couple_two_spins(j1, j2, False)
a = np.asarray(add_new_spin(a, j3, True), dtype=float)
end_time = perf_counter()
logger.info(f"TIME SIMS {format(end_time-start_time, '.5f')}")

# %%
for ii in range(len(b)):
    if not np.allclose(a[ii], b[ii]):
        print(np.isclose(a[ii], b[ii]))
        print(a[ii])
        print(b[ii])


# %%
@get_time
@njit
def get_SU2_singlets_configs(spin_list):
    """
    This function aims to identify all possible SU(2) singlet configs that can be
    formed from a given set (list) of spin representations.
    Singlet states are those where the total spin equals zero, meaning they are
    invariant under SU(2) transformations (rotationally invariant).

    Args:
        spin_list (list): list of spin representations to be coupled in order to get a singlet

    Returns:
        list: instances of SU2_singlet; if there is no singlet, just None
    """
    # Perform the first coupling (if there are only two spins, the result is a singlet)
    spin_configs = couple_two_spins2(
        spin_list[0], spin_list[1], get_singlet=(len(spin_list) == 2)
    )
    # Couple the resulting spin with the next ones
    for ii in range(2, len(spin_list)):
        # If ii corresponds to the last spin, then we want to select the singlet
        get_singlet = ii == len(spin_list) - 1
        # Add the ii spin to the list of coupled spins
        spin_configs = add_new_spin2(spin_configs, spin_list[ii], get_singlet)
    return spin_configs


@get_time
def group_SU2_singlets1(spin_list, pure_theory, psi_vacuum):
    """
    Groups (and sort) spin-configurations based on their total spin and individual spin z-components.

    Args:
        spin_list (list): list of spin representations giving rise to all the spin-configs

        pure_theory (bool, optional): if True, only gauge fields

        psi_vacuum (bool, optional): If True, the first element of spin_list is the vacuum of matter.
            If False, the first element of spin_list is the pair (up & down) of matter. Default to None.

    Returns:
        list: List of grouped and sorted spin configurations.
    """
    spin_configs = get_SU2_singlets_configs(spin_list)
    config_size = spin_configs.shape[1]
    # Save the positions of each resulting spin out of a combined pair
    Jtot_indices = np.arange(5, config_size, 5, dtype=int)
    # Sort the spin-configurations in terms of different SINGLETS
    sorted_indices = np.lexsort([spin_configs[:, idx] for idx in Jtot_indices[::-1]])
    spin_configs = spin_configs[sorted_indices]
    # ----------------------------------------------------------------------------
    # Save the positions of the Sz/M component of each spin in the chain
    M_indices = np.concatenate(([1], np.arange(3, config_size, 5)))
    # Save the positions of the CG value of each pair of spin irreps
    CG_indices = np.arange(4, config_size, 5)

    # Initialize lists to store the grouped configurations
    grouped_spin_configs = []
    current_group = []
    current_jtot = spin_configs[0, Jtot_indices]

    # Run over all the spin configuarions
    for config in spin_configs:
        # If it's the first config or the total spins of intermediate spin combinations are the same
        if np.array_equal(config[Jtot_indices], current_jtot):
            current_group.append(config)
        else:
            # Process the current group into an SU2_singlet object
            grouped_spin_configs.append(
                process_group_to_SU2_singlet(
                    current_group,
                    spin_list,
                    M_indices,
                    CG_indices,
                    pure_theory,
                    psi_vacuum,
                )
            )
            current_group = [config]
            current_jtot = config[Jtot_indices]

    # Append the last group processed into an SU2_singlet object
    grouped_spin_configs.append(
        process_group_to_SU2_singlet(
            current_group, spin_list, M_indices, CG_indices, pure_theory, psi_vacuum
        )
    )

    return grouped_spin_configs


@get_time
def group_SU2_singlets2(spin_list, pure_theory, psi_vacuum):
    """
    Groups (and sort) spin-configurations based on their total spin and individual spin z-components.

    Args:
        spin_list (list): list of spin representations giving rise to all the spin-configs

        pure_theory (bool, optional): if True, only gauge fields

        psi_vacuum (bool, optional): If True, the first element of spin_list is the vacuum of matter.
            If False, the first element of spin_list is the pair (up & down) of matter. Default to None.

    Returns:
        list: List of grouped and sorted spin configurations.
    """
    spin_configs = get_SU2_singlets_configs(spin_list)
    config_size = spin_configs.shape[1]
    # Save the positions of each resulting spin out of a combined pair
    Jtot_indices = np.arange(5, config_size, 5, dtype=int)
    # Sort the spin-configurations in terms of different SINGLETS
    sorted_indices = np.lexsort([spin_configs[:, idx] for idx in Jtot_indices[::-1]])
    spin_configs = spin_configs[sorted_indices]
    # ----------------------------------------------------------------------------
    # Save the positions of the Sz/M component of each spin in the chain
    M_indices = np.concatenate(([1], np.arange(3, config_size, 5)))
    # Save the positions of the CG value of each pair of spin irreps
    CG_indices = np.arange(4, config_size, 5)

    # Initialize lists to store the grouped configurations
    grouped_spin_configs = []
    # Assuming spin_configs is sorted
    J_values = spin_configs[:, Jtot_indices]
    change_points = np.any(np.diff(J_values, axis=0) != 0, axis=1)
    group_boundaries = np.where(change_points)[0] + 1
    groups = np.split(spin_configs, group_boundaries)

    # Process each group
    for group in groups:
        grouped_spin_configs.append(
            process_group_to_SU2_singlet(
                group, spin_list, M_indices, CG_indices, pure_theory, psi_vacuum
            )
        )
    return grouped_spin_configs


def process_group_to_SU2_singlet(
    group, spin_list, M_indices, CG_indices, pure_theory, psi_vacuum
):
    """Converts a group of configurations into an SU2_singlet object."""
    group_array = np.array(group)
    # Extract M configurations
    M_configs = group_array[:, M_indices]
    # Calculate product of CG values for each configuration
    CG_values = group_array[:, CG_indices]
    # Create the SU2_singlet object
    return SU2_singlet(
        J_config=spin_list,
        M_configs=M_configs,
        CG_values=CG_values,
        pure_theory=pure_theory,
        psi_vacuum=psi_vacuum,
    )


class SU2_singlet:
    def __init__(
        self, J_config, M_configs, CG_values, pure_theory=True, psi_vacuum=None
    ):
        self.J_config = J_config
        self.pure_theory = pure_theory
        self.psi_vacuum = psi_vacuum
        # If not pure_theory, make a difference between vacuum=0 and pair=up&down singlets
        if not self.pure_theory:
            if self.psi_vacuum is True:
                self.J_config[0] = "V"
            if self.psi_vacuum is False:
                self.J_config[0] = "P"
        # Acquire M configs with corresponding
        self.M_configs = M_configs
        self.CG_values = np.prod(CG_values, axis=1)
        repeated_J_config = np.tile(self.J_config, (M_configs.shape[0], 1))
        self.JM_configs = np.hstack((repeated_J_config, M_configs))

    def display_singlets(self):
        """
        Print the list of singlets (s1 ... sN) in the following way:
            Js   J1, J2, J3, ... JN

            s1 [ m1, m2, m3, ... mN] CG1

            s2 [ m1, m2, m3, ... mN] CG2

            s3 [ m1, m2, m3, ... mN] CG3
        """
        logger.info("====================================================")
        logger.info(f"J: {self.J_config}")
        for m, CG in zip(self.M_configs, self.CG_values):
            logger.info(f"M:{m} CG:{CG}")
        logger.info("----------------------------------------------------")


# %%
spin_configs = get_SU2_singlets_configs(spin_set[:2])
singlets = group_SU2_singlets1(spin_set[:2], pure_theory, None)
for s in singlets:
    s.display_singlets()
# %%
singlets = group_SU2_singlets2(spin_set[:2], pure_theory, None)
for s in singlets:
    s.display_singlets()
# %%
singlets1 = get_SU2_singlets(list(spin_set[:2]), pure_theory, None)
for s in singlets1:
    s.display_singlets()


# %%
@njit
def check_singlet_recursive(spins):
    """
    Recursively checks if a zero total spin (singlet) can be formed.
    """
    if len(spins) == 1:
        # Base case: one spin cannot form a singlet alone
        return spins[0] == 0
    elif len(spins) == 2:
        # Base case: two spins form a singlet if they are equal
        return spins[0] == spins[1]
    else:
        # General case: try to form a singlet by combining first two spins and checking the rest
        # Start with the minimum possible resultant spin
        new_j = abs(spins[0] - spins[1])
        # Maximum possible resultant spin
        max_j = spins[0] + spins[1]
        while new_j <= max_j:
            if check_singlet_recursive(np.append(spins[2:], new_j)):
                return True
            new_j += 1  # Increment to the next possible resultant spin
        return False


@njit
def can_form_singlet(spins):
    """
    Checks if a given configuration of spins can potentially form an SU(2) singlet.

    Parameters:
    - spins (np.ndarray): An array of spins.

    Returns:
    - bool: True if the spins can form a singlet, False otherwise.
    """
    if len(spins) < 2:
        return False  # Need at least two spins to form a singlet
    # Sort spins to facilitate singlet checking
    sorted_spins = np.sort(spins)
    # Attempt to form singlets by recursive subtraction
    return check_singlet_recursive(sorted_spins)


@get_time
def SU2_gauge_invariant_states1(s_max, pure_theory, lattice_dim):
    spin_list = [s_max for _ in range(2 * lattice_dim)]
    spins = []
    # For each single spin particle in the list,
    # consider all the spin irrep up to the max one
    for s in spin_list:
        tmp = np.arange(0, spin_space(s), 1)
        spins.append(tmp / 2)
    if not pure_theory:
        spins.insert(0, np.asarray([0, 1 / 2, 0]))
        # Check the matter spin (0 (vacuum),1/2,0 (up & down))
        v_sector = np.prod([len(l) for l in [[spins[0][0]]] + spins[1:]])
    else:
        psi_vacuum = None
    # Set rows and col counters list for the basis
    gauge_states = {"site": []}
    gauge_basis = {"site": []}
    # List of borders/corners of the lattice
    borders = get_lattice_borders_labels(lattice_dim)
    for label in borders:
        gauge_states[f"site_{label}"] = []
        gauge_basis[f"site_{label}"] = []
    for ii, spins_config in enumerate(product(*spins)):
        spins_config = np.asarray(spins_config)
        # Check the existence of a SU2 singlet state
        if can_form_singlet(spins_config):
            if not pure_theory:
                if ii < v_sector:
                    psi_vacuum = True
                elif 2 * v_sector - 1 < ii < 3 * v_sector:
                    psi_vacuum = False
                else:
                    psi_vacuum = None
            singlets = group_SU2_singlets(spins_config, pure_theory, psi_vacuum)
            for s in singlets:
                # Save the singlet state
                gauge_states["site"].append(s)
                # Save the singlet state written in the canonical basis
                singlet_state = SU2_singlet_canonical_vector(spin_list, s)
                gauge_basis["site"].append(singlet_state)
                # GET THE CONFIG LABEL
                spin_sizes = [spin_space(s) for s in spins_config]
                label = LGT_border_configs(
                    config=spin_sizes, offset=1, pure_theory=pure_theory
                )
                if label:
                    # Save the config state also in the specific subset of borders
                    for ll in label:
                        gauge_states[f"site_{ll}"].append(s)
                        gauge_basis[f"site_{ll}"].append(singlet_state)
    # Build the basis combining the states into a matrix
    for label in list(gauge_basis.keys()):
        gauge_basis[label] = csr_matrix(np.column_stack(tuple(gauge_basis[label])))
    return gauge_basis, gauge_states


@get_time
def SU2_singlet_canonical_vector(spin_list, singlet):
    """
    Constructs the canonical state vector representing a specific SU2 singlet configuration
    within the total Hilbert space formed by the tensor product of individual spin Hilbert spaces.
    Each spin space is defined (via "get_spin_Hilbert_spaces") up to the maximum irrep given in spin_list.
    This function calculates the state vector in the composite system space that corresponds to
    the combination of individual spin degrees of freedom forming the specified singlet state.

    Args:
        spin_list (list of (half)integers): List of the maximal spin irreps for each degree of freedom (dof)
            in the system. The Hilbert space for each dof spans from the singlet state
            up to the maximal irrep specified in this list.

        singlet (SU2_singlet): An instance of the SU2_singlet class representing the specific singlet state
            for which the canonical state vector is to be constructed. This singlet should be compatible
            with the spin configurations defined by spin_list.

    Returns:
        np.ndarray: A one-dimensional array representing the normalized canonical state vector of the
            specified singlet in the total Hilbert space. The length of this vector equals the product
            of the dimensions of the individual spin Hilbert spaces defined by spin_list.
    """
    # Acquire the combined Hilbert spaces for the spin configuration.
    j_list, m_list = get_spin_Hilbert_spaces(spin_list, singlet.pure_theory)
    # Compute the length of the basis in the total Hilbert space.
    len_basis = len(list(product(*m_list)))
    # Initialize the state vector as a zero vector.
    state = np.zeros(len_basis)
    # Construct the state vector for the singlet.
    for ii, (j_config, m_config) in enumerate(zip(product(*j_list), product(*m_list))):
        JM_config = list(j_config) + list(m_config)
        # Check if the configuration contributes to the singlet.
        if JM_config in singlet.JM_configs:
            # Find the corresponding index in the singlet's configurations.
            index = np.where(singlet.JM_configs == JM_config)[0]
            # Assign the appropriate coefficient from the singlet configuration.
            state[ii] = singlet.CG_values[index]
    # Check the norm of the state in the canonical basis
    state_norm = np.sum(state**2)
    if np.abs(state_norm - 1) > 1e-10:
        raise ValueError(f"The state is not normalized: norm {state_norm}")
    return state


# %%
c, d = SU2_gen_gauge_invariant_states(s_max, pure_theory, lattice_dim)
# %%
a, b = SU2_gauge_invariant_states1(s_max, pure_theory, lattice_dim)
# %%
