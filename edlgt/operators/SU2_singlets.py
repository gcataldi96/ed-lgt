"""SU(2) singlet construction utilities and helper data structures."""

import numpy as np
from numba import njit
from itertools import product
from sympy import S
from sympy.physics.wigner import clebsch_gordan as CG_coeff
from copy import deepcopy
from .spin_operators import spin_space, m_values
from edlgt.tools import validate_parameters, get_time
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
    """Couple two SU(2) spins and return nonzero Clebsch-Gordan contributions.

    Parameters
    ----------
    j1, j2 : scalar
        Input spin irreps (integer or half-integer values).
    get_singlet : bool, optional
        If ``True``, keep only couplings whose total spin is zero.
    M : scalar, optional
        If provided, fix the ``m`` component of the first spin to ``M``.

    Returns
    -------
    list
        List of coupling configurations with nonzero Clebsch-Gordan
        coefficients.
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
    """Representation of one SU(2)-singlet multiplet in coupled-spin form."""

    def __init__(
        self,
        J_config,
        M_configs,
        CG_values,
        pure_theory=True,
        psi_vacuum=None,
        background=0,
    ):
        """Initialize an SU(2)-singlet descriptor.

        Parameters
        ----------
        J_config : list
            Total-spin labels for the constituent degrees of freedom.
        M_configs : list
            Allowed sets of magnetic quantum numbers producing the singlet.
        CG_values : list
            Lists of intermediate Clebsch-Gordan coefficients associated with
            each ``M`` configuration.
        pure_theory : bool, optional
            If ``False``, matter degrees of freedom are included in the first
            entries of the configuration.
        psi_vacuum : bool, optional
            Distinguishes the two matter ``J=0`` singlets when matter is present.
        background : int, optional
            Background-charge sector information used for labeling.

        Raises
        ------
        TypeError
            If inputs are not in the expected formats.
        ValueError
            If configuration lengths are inconsistent.
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
        """Log the singlet components and their Clebsch-Gordan coefficients."""
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
    """Enumerate all SU(2) singlets compatible with a list of spin irreps.

    Parameters
    ----------
    spin_list : list
        Spin irreps to be coupled.
    pure_theory : bool, optional
        If ``True``, consider only gauge degrees of freedom.
    psi_vacuum : bool, optional
        Matter-sector selector used when ``pure_theory=False``.
    background : int, optional
        Background-charge sector information.

    Returns
    -------
    list or None
        List of :class:`SU2_singlet` objects, or ``None`` if no singlet exists.
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
    """Couple an additional spin to a list of intermediate coupling configurations.

    Parameters
    ----------
    previous_configs : list
        Intermediate coupling configurations produced by
        :func:`couple_two_spins` / previous recursive steps.
    new_spin : scalar
        Spin irrep to add (integer or half-integer value).
    get_singlet : bool
        If ``True``, keep only resulting singlet configurations.

    Returns
    -------
    list
        Updated list of coupling configurations.
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
    """Group intermediate coupling configurations into singlet multiplets.

    Parameters
    ----------
    spin_configs : list
        Intermediate coupling configurations.
    spin_list : list
        Original list of spin irreps.
    pure_theory : bool
        If ``True``, consider only gauge degrees of freedom.
    psi_vacuum : bool
        Matter-sector selector used when ``pure_theory=False``.
    background : int, optional
        Background-charge sector information.

    Returns
    -------
    list
        List of grouped :class:`SU2_singlet` objects.
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
    """Build local spin Hilbert spaces used in singlet construction.

    Parameters
    ----------
    max_spin_irrep_list : list
        Maximum spin irrep kept for each degree of freedom.
    pure_theory : bool
        If ``False``, prepend the matter Hilbert space used by the SU(2)
        dressed-site construction.
    background : int, optional
        If nonzero, prepend the background-charge Hilbert space.

    Returns
    -------
    tuple
        ``(j_list, m_list)`` with per-degree-of-freedom lists of spin irreps and
        magnetic quantum numbers.
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
    """Construct the canonical basis vector of a specific SU(2) singlet.

    Parameters
    ----------
    spin_list : list
        Maximum spin irreps for each degree of freedom.
    singlet : SU2_singlet
        Singlet descriptor returned by :func:`get_SU2_singlets`.
    background : bool or int, optional
        Background-charge flag/sector forwarded to
        :func:`get_spin_Hilbert_spaces`.

    Returns
    -------
    numpy.ndarray
        Normalized canonical state vector for the requested singlet.
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
    """Recursively test whether a set of spins can yield total spin zero."""
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
    """Check whether a spin configuration can form an SU(2) singlet.

    Parameters
    ----------
    spins : numpy.ndarray
        Array of spins (encoded as integers or doubled half-integers depending on
        the calling code).

    Returns
    -------
    bool
        ``True`` if a singlet is possible, ``False`` otherwise.
    """
    if len(spins) < 2:
        return False  # Need at least two spins to form a singlet
    # Sort spins to facilitate singlet checking
    sorted_spins = np.sort(spins)
    # Attempt to form singlets by recursive subtraction
    return check_singlet_recursive(sorted_spins)
