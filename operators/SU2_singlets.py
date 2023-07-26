import numpy as np
from itertools import product
from sympy import S
from sympy.physics.wigner import clebsch_gordan as CG_coeff
from scipy.sparse import diags
from simsio import logger

__all__ = ["get_spin_operators", "spin_space", "m_values"]


def get_spin_operators(s):
    if not np.isscalar(s):
        raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(s)}")
    """
    This function computes the spin (sparse) matrices: 
    [Sz, Sp=S+, Sm=S-, Sx, Sy, S2=Casimir]
    in any arbitrary spin-s representation

    Args:
        s (scalar, real): spin value, assumed to be integer or semi-integer

    Returns:
        dict: dictionary with the spin matrices
    """
    # Size of the spin matrix
    size = spin_space(s)
    shape = (size, size)
    # Diagonal entries of the Sz matrix
    sz_diag = m_values(s)
    # Diagonal entries of the S+ matrix
    sp_diag = np.sqrt(s * (s + 1) - sz_diag[1:] * (sz_diag[1:] + 1))
    ops = {}
    ops["Sz"] = diags(sz_diag, 0, shape)
    ops["Sp"] = diags(sp_diag, 1, shape)
    ops["Sm"] = ops["Sp"].transpose()
    ops["Sx"] = 0.5 * (ops["Sp"] + ops["Sm"])
    ops["Sy"] = complex(0, -0.5) * (ops["Sp"] - ops["Sm"])
    ops["S2"] = diags([s * (s + 1) for i in range(size)], 0, shape)
    return ops


def spin_space(s):
    if not np.isscalar(s):
        raise TypeError(f"s must be scalar (int or real), not {type(s)}")
    # Given the spin value s, it returns the size of its Hilber space
    return int(2 * s + 1)


def m_values(s):
    if not np.isscalar(s):
        raise TypeError(f"s must be scalar (int or real), not {type(s)}")
    # Given the spin value s, it returns an array with the possible spin-z components
    return np.arange(-s, s + 1)[::-1]


def spin_couple(j1, j2, singlet=False, M=None):
    if not np.isscalar(j1):
        raise TypeError(f"j1 must be scalar (int or real), not {type(j1)}")
    if not np.isscalar(j2):
        raise TypeError(f"j2 must be scalar (int or real), not {type(j2)}")
    if not isinstance(singlet, bool):
        raise TypeError(f"singlet must be bool, not {type(singlet)}")
    """
    This function computes states obtained by combining two spins j1, j2
    by computing Clebsh-Gordan coefficients CG. 
    Args:
        j1 (real & scalar): spin of the 1st particle

        j2 (real & scalar): spin of the 2nd particle

        singlet (bool, optional): if true, look only at the combinations
        if j1 and j2 that provides an SU(2) single. Defaults to False.

        M (real & scalar, optional): spin-z component of the 1st particle.
        Defaults to None.

    Returns:
        list: [(j1, m1,) j2, m2, CG, j3, m3]
    """
    set = []
    if singlet:
        # if we expect a singlet, the only resulting j can only be 0
        j3_values = np.array([0])
    else:
        # otherwise, it is bounded between |j1 - j2| and j1 + j2
        j3_values = np.arange(np.abs(j1 - j2), j1 + j2 + 1, 1)
    if M is not None:
        if not np.isscalar(M):
            raise TypeError(f"M must be scalar (int or real), not {type(M)}")
        else:
            m1values = [M]
    else:
        m1values = m_values(j1)
    # Run over all the possible combinations of z-components of j1, j2, j3
    # searching for non-zero CG coefficients
    for m1, m2, j3 in product(m1values, m_values(j2), j3_values):
        for m3 in m_values(j3):
            # Compute the CG coefficient
            CG = CG_coeff(S(j1), S(j2), S(j3), S(m1), S(m2), S(m3))
            # Save the configuration if it exists
            if CG != 0:
                if M is not None:
                    set.append([S(j2), S(m2), CG, S(j3), S(m3)])
                else:
                    set.append([S(j1), S(m1), S(j2), S(m2), CG, S(j3), S(m3)])
    return set


class SU2_singlet:
    def __init__(
        self, J_config, M_configs, CG_values, pure_theory=True, psi_vacuum=None
    ):
        # CHECK ON TYPES
        if not isinstance(J_config, list):
            raise TypeError(f"J_config must be a list, not {type(J_config)}")
        self.n_spins = len(J_config)
        if not isinstance(M_configs, list):
            raise TypeError(
                f"M_configs must be a list (of lists), not {type(M_configs)}"
            )
        if not isinstance(CG_values, list):
            raise TypeError(
                f"CG_values must be a list (of lists), not {type(CG_values)}"
            )
        if len(M_configs) != len(CG_values):
            raise ValueError(f"M_configs and CG_values must have the same # of entries")
        for ii, conf in enumerate(M_configs):
            if len(conf) != self.n_spins:
                raise ValueError(
                    f"Every M config should be made of {self.n_spins} not {len(conf)}"
                )
            if len(CG_values[ii]) != (self.n_spins - 1):
                raise ValueError(
                    f"The {ii} M config should have {self.n_spins-1} CGs, not {len(CG_values[ii])}"
                )
        if not isinstance(pure_theory, bool):
            raise TypeError(f"pure_theory must be a list, not {type(pure_theory)}")
        if psi_vacuum is not None:
            if not isinstance(psi_vacuum, bool):
                raise TypeError(f"psi_vacuum must be bool, not {type(psi_vacuum)}")
        # ----------------------------------------------------------------------------------
        self.J_config = J_config.copy()
        # If not pure_theory, make a difference between vacuum=0 and pair=up&down singlets
        self.pure_theory = pure_theory
        self.psi_vacuum = psi_vacuum
        if not self.pure_theory:
            if self.psi_vacuum is True:
                self.J_config[0] = "V"
            if self.psi_vacuum is False:
                self.J_config[0] = "P"
        # Acquire M configs with corresponding
        self.M_configs = M_configs.copy()
        self.CG_values = []
        self.JM_configs = []
        for ii, CG_set in enumerate(CG_values):
            self.CG_values.append(np.prod(CG_set))
        for ii, m in enumerate(self.M_configs):
            self.JM_configs.append(self.J_config + m)

    def show(self):
        logger.info("====================================================")
        logger.info(f"J: {self.J_config}")
        for m, CG in zip(self.M_configs, self.CG_values):
            logger.info(f"M:{m} CG:{CG}")
        logger.info("----------------------------------------------------")


def get_SU2_singlets(spin_list, pure_theory=True, psi_vacuum=None):
    # CHECK ON TYPES
    if not isinstance(spin_list, list):
        raise TypeError(f"spin_list must be a list, not {type(spin_list)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory must be a list, not {type(pure_theory)}")
    if psi_vacuum is not None:
        if not isinstance(psi_vacuum, bool):
            raise TypeError(f"psi_vacuum must be bool, not {type(psi_vacuum)}")
    """
    This function computes the form of an SU(2) singlet out of a list of
    spin representations
    Args:
        spin_list (list): list of spins to be coupled in order to get a singlet
        pure_theory (bool, optional): if True, only gauge fields

        psi_vacuum (bool, optional):
            If True, the first element of spin_list is the vacuum of matter
            If False, the first element of spin_list is the pair (up & down) of matter
            Default to None.
    Returns:
        list: list of instances of SU2_singlet; 
            if there is no singlet, just None
    """
    n_spins = len(spin_list)
    if n_spins < 2:
        raise ValueError("2 is the minimum number of spins to be coupled")
    # Perform the first coupling
    if len(spin_list) == 2:
        spin_config = spin_couple(spin_list[0], spin_list[1], singlet=True)
    else:
        spin_config = spin_couple(spin_list[0], spin_list[1])
    # Couple the resulting spin with the next ones
    for ii in np.arange(2, n_spins, 1):
        if ii == n_spins - 1:
            singlet = True
        else:
            singlet = False
        # Make a copy of the spin configurations
        tmp1 = spin_config.copy()
        spin_config = []
        for j in tmp1:
            # Couple the resulting spin of the previous combination with the next spin
            tmp2 = spin_couple(j1=j[-2], j2=spin_list[ii], singlet=singlet, M=j[-1])
            for J in tmp2:
                spin_config.append(j + J)
    if spin_config:
        M_sites = [1] + list(np.arange(3, len(spin_config[0]), 5))
        CG_sites = list(np.arange(4, len(spin_config[0]), 5))
        K_sites = list(np.arange(5, len(spin_config[0]), 5))
        # Sort the the spin_config in terms of different SINGLETS
        spin_config = sorted(spin_config, key=lambda x: tuple(x[k] for k in K_sites))
        SU2_singlets = []
        M_configs = []
        CG_configs = []
        for ii, conf in enumerate(spin_config):
            if ii == 0:
                M_configs.append([conf[M] for M in M_sites])
                CG_configs.append([conf[CG] for CG in CG_sites])
                K_values = [conf[k] for k in K_sites]
            else:
                K_values_new = [conf[k] for k in K_sites]
                if K_values_new == K_values:
                    M_configs.append([conf[M] for M in M_sites])
                    CG_configs.append([conf[CG] for CG in CG_sites])
                else:
                    SU2_singlets.append(
                        SU2_singlet(
                            spin_list, M_configs, CG_configs, pure_theory, psi_vacuum
                        )
                    )
                    K_values = K_values_new.copy()
                    M_configs = [[conf[M] for M in M_sites]]
                    CG_configs = [[conf[CG] for CG in CG_sites]]
        SU2_singlets.append(
            SU2_singlet(spin_list, M_configs, CG_configs, pure_theory, psi_vacuum)
        )
    else:
        SU2_singlets = None
    return SU2_singlets
