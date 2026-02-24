import numpy as np
from numba import njit
from itertools import product
from sympy import S
from sympy.physics.wigner import clebsch_gordan as CG_coeff
from copy import deepcopy
from .spin_operators import spin_space, m_values
from ed_lgt.tools import validate_parameters, get_time
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "SU2_singlet",
    "get_SU2_singlets",
    "get_spin_Hilbert_spaces",
    "couple_two_spins",
    "add_new_spin",
    "group_sorted_spin_configs",
    "SU2_singlet_canonical_vector",
    "can_form_singlet",
]
S0 = S(0)
S12 = S(1) / 2
S1 = S(1)


def couple_two_spins(j1, j2, get_singlet=False, M=None):
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
        list: [(j1, m1,) j2, m2, CG, j3, m3]
    """
    if M is not None:
        validate_parameters(spin_list=[j1, j2], sz_list=[M], get_singlet=get_singlet)
        m1values = [M]
    else:
        validate_parameters(spin_list=[j1, j2], get_singlet=get_singlet)
        m1values = m_values(j1)
    if get_singlet:
        # if we expect a singlet, the only resulting j3 can be 0
        j3_values = np.array([0])
    else:
        # otherwise, it is bounded between |j1 - j2| and j1 + j2
        j3_values = np.arange(np.abs(j1 - j2), j1 + j2 + 1, 1)
    # Define a list of possible SU2 states
    SU2_states = []
    # Run over all the possible combinations of z-components of j1, j2, j3
    # searching for non-zero CG coefficients
    Sj1, Sj2 = S(j1), S(j2)
    for m1, m2, j3 in product(m1values, m_values(j2), j3_values):
        m3 = m1 + m2
        if abs(m3) > j3:
            continue
        # Compute the CG coefficient
        CG = CG_coeff(Sj1, Sj2, S(j3), S(m1), S(m2), S(m3))
        # Save the configuration if it exists
        if CG != 0:
            if M is not None:
                SU2_states.append([Sj2, S(m2), CG, S(j3), S(m3)])
            else:
                SU2_states.append([Sj1, S(m1), Sj2, S(m2), CG, S(j3), S(m3)])
    return SU2_states


class SU2_singlet:
    def __init__(
        self,
        J_config,
        M_configs,
        CG_values,
        pure_theory=True,
        psi_vacuum=None,
        background=0,
    ):
        """
        This class collects a configuration of a set of angular momenta Js
        (and their corresponding Z-momentum) that form an SU(2) singlet state
        with a certain Clebsh-Gordon coeffiicient.
        The set of momenta is typically referred to SU(2) gauge fields,
        but it can eventually include a matter state (in first position)
        describing Flavorless Color 1/2 Dirac fermions with 4 possible states:

        - (J,M)=(0,0)

        - (J,M)=(1/2,1/2)

        - (J,M)=(1/2,-1/2)

        - (J,M)=(1/2,0)

        Args:
            J_config (list): list of Total angular momentum J of a set of particles/entities

            M_configs (list): list of possible sets of angular z-momentum
                (each set has the same length of J_config) of the J_config that allows for a singlet.

            CG_values (list): list of intermediate Clebsh-Gordon coefficients, to be multiplied
                together for the overall CG coefficient of the J configuration.

            pure_theory (bool, optional): If False, the theory also involves Flavorless Color 1/2 Dirac fermions.
                Defaults to True.

            psi_vacuum (bool, optional): If it used, it specifies which type of 0 singlet the matter state
                is corresponding to. Defaults to None.

            background (bool, optional): If True, it adds an extra charge (j=0,1/2) to the combination of the sites.

        Raises:
            TypeError: If the input arguments are of incorrect types or formats.

            ValueError: If M_configs and CG_values do not have the same # of entries

            ValueError: If any of M config is NOT made of len(J_config)

            ValueError: If any entry of M_config does not have len(J_config)-1 CGs,
        """
        # CHECK ON TYPES
        validate_parameters(
            spin_list=J_config, pure_theory=pure_theory, psi_vacuum=psi_vacuum
        )
        n_spins = len(J_config)
        if not isinstance(M_configs, list):
            raise TypeError(f"M_configs must be a list of lists, not {type(M_configs)}")
        if not isinstance(CG_values, list):
            raise TypeError(f"CG_values must be a list of lists, not {type(CG_values)}")
        if len(M_configs) != len(CG_values):
            raise ValueError(f"M_configs and CG_values must have the same # of entries")
        # Check each configuration of M values
        for ii, conf in enumerate(M_configs):
            validate_parameters(sz_list=conf)
            if len(conf) != n_spins:
                raise ValueError(f"{ii} M-config has {len(conf)} vals, not {n_spins}")
            if len(CG_values[ii]) != (n_spins - 1):
                n_CGs = len(CG_values[ii])
                raise ValueError(f"{ii} M config has {n_CGs} CGs, not {n_spins-1}")
        # ----------------------------------------------------------------------------------
        self.n_spins = n_spins
        self.J_config = deepcopy(J_config)
        self.pure_theory = pure_theory
        self.psi_vacuum = psi_vacuum
        # If not pure_theory, make a difference between vacuum=0 and pair=up&down singlets
        matter_ind = 0 if background == 0 else 1
        if not self.pure_theory:
            if self.psi_vacuum is True:
                self.J_config[matter_ind] = "V"
            if self.psi_vacuum is False:
                self.J_config[matter_ind] = "P"
        # Acquire M configs with corresponding
        self.M_configs = deepcopy(M_configs)
        self.CG_values = []
        self.JM_configs = []
        for ii, CG_set in enumerate(CG_values):
            self.CG_values.append(np.prod(CG_set))
        for ii, m in enumerate(self.M_configs):
            self.JM_configs.append(self.J_config + m)

    def display_singlet(self, msg: str | None = None):
        """
        Print the list of singlets (s1 ... sN) in the following way:
            Js   J1, J2, J3, ... JN

            s1 [ m1, m2, m3, ... mN] CG1

            s2 [ m1, m2, m3, ... mN] CG2

            s3 [ m1, m2, m3, ... mN] CG3
        """
        if msg is not None:
            msg_len = len(msg)
            line = "=" * int(np.ceil((52 - 2 - msg_len) / 2))
            logger.info(line + " " + msg + " " + line)
        else:
            logger.info("====================================================")
        logger.info(f"J: {self.J_config}")
        for m, CG in zip(self.M_configs, self.CG_values):
            logger.info(f"M: {m} CG:{float(CG)}")


@get_time
def get_SU2_singlets(spin_list, pure_theory=True, psi_vacuum=None, background=0):
    """
    This function aims to identify all possible SU(2) singlet states that can be
    formed from a given set (list) of spin representations.
    Singlet states are those where the total spin equals zero, meaning they are
    invariant under SU(2) transformations (rotationally invariant).

    Args:
        spin_list (list): list of spin representations to be coupled in order to get a singlet

        pure_theory (bool, optional): if True, only gauge fields

        psi_vacuum (bool, optional): If True, the first element of spin_list is the vacuum of matter.
            If False, the first element of spin_list is the pair (up & down) of matter. Default to None.

        background (bool, optional): If True, it adds an extra charge (j=0,1/2) to the combination of the sites.

    Returns:
        list: instances of SU2_singlet; if there is no singlet, just None
    """
    # CHECK ON TYPES
    validate_parameters(
        spin_list=spin_list, pure_theory=pure_theory, psi_vacuum=psi_vacuum
    )
    if len(spin_list) < 2:
        raise ValueError("2 is the minimum number of spins to form a singlet")
    # Perform the first coupling (if there are only two spins, the result is a singlet)
    spin_configs = couple_two_spins(
        spin_list[0], spin_list[1], get_singlet=(len(spin_list) == 2)
    )
    # Couple the resulting spin with the next ones
    for ii in range(2, len(spin_list)):
        # If ii corresponds to the last spin, then we want to select the singlet
        get_singlet = ii == len(spin_list) - 1
        # Add the ii spin to the list of coupled spins
        spin_configs = add_new_spin(
            spin_configs, spin_list[ii], get_singlet=get_singlet
        )
    # If there are valid configurations, sort and group them into singlets; otherwise, return None
    if spin_configs:
        return group_sorted_spin_configs(
            spin_configs, spin_list, pure_theory, psi_vacuum, background
        )
    else:
        return None


@get_time
def add_new_spin(previous_configs, new_spin, get_singlet):
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
    updated_configs = []
    for config in previous_configs:
        # couple the resulting spin of each configuration with the new one, eventually obtaining a singlet
        tmp_configs = couple_two_spins(
            j1=config[-2], j2=new_spin, get_singlet=get_singlet, M=config[-1]
        )
        for new_config in tmp_configs:
            updated_configs.append(config + new_config)
    return updated_configs


@get_time
def group_sorted_spin_configs(
    spin_configs, spin_list, pure_theory, psi_vacuum, background=0
):
    """
    Groups and sorts spin-configurations based on their total spin and individual spin z-components.

    Args:
        spin_configs (list of lists): The list of spin-configurations to sort and group.

        spin_list (list): list of spin representations giving rise to all the spin-configs

        pure_theory (bool, optional): if True, only gauge fields

        psi_vacuum (bool, optional): If True, the first element of spin_list is the vacuum of matter.
            If False, the first element of spin_list is the pair (up & down) of matter. Default to None.

        background (bool, optional): If True, it adds an extra charge (j=0,1/2) to the combination of the sites.

    Returns:
        list: List of grouped and sorted spin configurations.
    """
    validate_parameters(
        spin_list=spin_list, pure_theory=pure_theory, psi_vacuum=psi_vacuum
    )
    # Save the positions of the Sz/M component of each spin in the chain
    M_inds = [1] + list(np.arange(3, len(spin_configs[0]), 5))
    # Save the positions of the CG value of each pair of spin irreps
    CG_inds = list(np.arange(4, len(spin_configs[0]), 5))
    # Save the positions of each resulting spin out of a combined pair
    Jtot_inds = list(np.arange(5, len(spin_configs[0]), 5))
    # Sort the spin-configurations in terms of different SINGLETS
    spin_configs = sorted(spin_configs, key=lambda x: tuple(x[kk] for kk in Jtot_inds))
    # ----------------------------------------------------------------------------------
    SU2_singlet_list = []  # List of sorted SU2 singlets
    M_configs = []  # List of lists of Sz components (one list per spin configuration)
    CG_configs = []  # List of lists of CG vals (one list per spin configuration)
    # Run over all the spin configuarions
    for ii, conf in enumerate(spin_configs):
        # If it's the first config or if the total spins of intermediate spin combinations are the same
        if ii == 0 or [conf[kk] for kk in Jtot_inds] == Jtot_values:
            # Acquire intermediate spin combinations
            Jtot_values = [conf[kk] for kk in Jtot_inds]
            # Acquire corresponding Sz components of the spin configuratin
            M_configs.append([conf[m] for m in M_inds])
            # Acquire the corresponding CG value
            CG_configs.append([conf[cg] for cg in CG_inds])
        else:
            Jtot_values = [conf[kk] for kk in Jtot_inds]
            # The spin-config has a new set of intermediate spin combinations.
            # The previous one was a singlet
            SU2_singlet_list.append(
                SU2_singlet(
                    J_config=spin_list,
                    M_configs=M_configs,
                    CG_values=CG_configs,
                    pure_theory=pure_theory,
                    psi_vacuum=psi_vacuum,
                    background=background,
                )
            )
            # Re-define a list with sets of Sz and CG configurations associated to this new spin config
            M_configs = [[conf[m] for m in M_inds]]
            CG_configs = [[conf[cg] for cg in CG_inds]]
    # Save the last singlet configuration
    SU2_singlet_list.append(
        SU2_singlet(
            J_config=spin_list,
            M_configs=M_configs,
            CG_values=CG_configs,
            pure_theory=pure_theory,
            psi_vacuum=psi_vacuum,
            background=background,
        )
    )
    return SU2_singlet_list


@get_time
def get_spin_Hilbert_spaces(max_spin_irrep_list, pure_theory, background=0):
    """
    This function generates the Hilbert spaces for quantum systems characterized
    by their spin degrees of freedom.
    For each degree of freedom, it calculates the possible spin states (J)
    and their corresponding magnetic quantum numbers (M),
    spanning from the singlet state J,M=0 up to the maximum spin representation provided.
    The function supports both pure gauge field configurations and mixed systems including
    Fermionic particles by adjusting the `pure_theory` flag.
    In non-pure theory mode, additional states are included to represent the vacuum and fermion pairs.

    Args:
        max_spin_irrep_list (list): List of the maximal spin-irrep of each degree of freedom (dof) in the system.
            For each d.o.f, the Hilbert space will span from the smallest=0-irrep to the maximal irrep.

        pure_theory (bool, optional): Defaults to True. If False, it will add Fermionic spin 1/2 particles,
            whose Hilbert space is 4dimensional and can be characterized by the following values of
            spin and its z-component:

            - J = [0, 1 / 2, 1 / 2, 0]

            - M = [0, 1 / 2, -1 / 2, 0]

            Where the first J=0 is referred to the absence of spin particles V=vacuum,
            while the second J=0 is referred to the case of a pair of particles with opposite spin P=pair

        background (bool, optional): If True, it adds an extra charge (j=0,1/2) to the combination of the sites.

    Returns:
        - j_list: A list of lists, where each sublist represents the possible J values (total spin values) for a dof.
        - m_list: A list of lists parallel to j_list, representing the corresponding M values (magnetic quantum numbers) for each J value.

        Each list has len = len(max_spin_irrep_list)
        The length of each internal list is exactly the sum of (2s+1) for s from 0 to the coorresponding max_spin

    Example:
        if spins=[1,1], we will get
            - J=[[0, 1/2, 1/2, 1, 1, 1],
                 [0, 1/2, 1/2, 1, 1, 1]]
            - M=[[0, 1/2, -1/2, 1, 0, -1],
                 [0, 1/2, -1/2, 1, 0, -1]]
    """
    validate_parameters(spin_list=max_spin_irrep_list, pure_theory=pure_theory)
    spin_dof = []
    m_list = []
    j_list = []
    # For each single spin particle in the list
    for max_irrep in max_spin_irrep_list:
        # Create an array with all the spin irreps up to the max one
        spin_irreps = np.arange(S0, spin_space(max_irrep), 1) / 2
        # For each spin dof save the list of allowed irreps
        spin_dof.append(spin_irreps)
    # run over each spin dof
    for spin_irreps in spin_dof:
        m_set = []
        j_set = []
        # run over the irreps of each spin dof
        for irrep in spin_irreps:
            # save the values of z-component of the irrep
            m_set += list(m_values(irrep))
            # save the irrep for each z-component
            j_set += [irrep for i in m_values(irrep)]
        # At the end, or each spin dof, len(m_set) = len(j_set) = direct sum of each irrep hilbert space
        m_list.append(m_set)
        j_list.append(j_set)
    if not pure_theory:
        # add the Hilbert space of 2 fermionic spin 1/2 particles (see docs)
        j_list.insert(0, ["V", S12, S12, "P"])
        m_list.insert(0, [S0, S12, -S12, S0])
    if background != 0:
        m_set_bg = []
        j_set_bg = []
        for irrep in np.arange(S0, spin_space(background), 1) / 2:
            # save the values of z-component of the irrep
            m_set_bg += list(m_values(irrep))
            # save the irrep for each z-component
            j_set_bg += [irrep for i in m_values(irrep)]
        # add the Hilbert space of the background charge
        j_list.insert(0, j_set_bg)
        m_list.insert(0, m_set_bg)
    return j_list, m_list


@get_time
def SU2_singlet_canonical_vector(spin_list, singlet, background=False):
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
    validate_parameters(spin_list=spin_list)
    if not isinstance(singlet, SU2_singlet):
        raise TypeError(f"singlet is not {SU2_singlet}, not {type(singlet)}")
    # Acquire the combined Hilbert spaces for the spin configuration.
    j_list, m_list = get_spin_Hilbert_spaces(spin_list, singlet.pure_theory, background)
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
            index = singlet.JM_configs.index(JM_config)
            # Assign the appropriate coefficient from the singlet configuration.
            state[ii] = singlet.CG_values[index]
    # Check the norm of the state in the canonical basis
    state_norm = np.sum(state**2)
    if np.abs(state_norm - 1) > 1e-10:
        raise ValueError(f"The state is not normalized: norm {state_norm}")
    return state


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
