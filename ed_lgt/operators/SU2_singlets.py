import numpy as np
from itertools import product
from sympy import S
from sympy.physics.wigner import clebsch_gordan as CG_coeff
from scipy.sparse import diags, block_diag, identity, csr_matrix
from copy import deepcopy

__all__ = [
    "get_spin_operators",
    "get_Pauli_operators",
    "SU2_singlet",
    "SU2_generators",
    "get_SU2_singlets",
]


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


def get_spin_operators(s):
    """
    This function computes the spin (sparse) matrices:
    [Sz, Sp=S+, Sm=S-, Sx, Sy, S2=Casimir] in any arbitrary spin-s representation

    Args:
        s (scalar, real): spin value, assumed to be integer or semi-integer

    Returns:
        dict: dictionary with the spin matrices
    """
    if not np.isscalar(s):
        raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(s)}")
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
    ops["Sx"] = (ops["Sp"] + ops["Sm"]) / 2
    ops["Sy"] = complex(0, -0.5) * (ops["Sp"] - ops["Sm"])
    ops["S2"] = diags([s * (s + 1) for i in range(size)], 0, shape)
    return ops


def get_Pauli_operators():
    shape = (2, 2)
    ops = {}
    ops["Sz"] = diags([1, -1], 0, shape)
    ops["Sp"] = diags([1], 1, shape)
    ops["Sm"] = ops["Sp"].transpose()
    ops["Sx"] = ops["Sp"] + ops["Sm"]
    ops["Sy"] = complex(0, -1) * (ops["Sp"] - ops["Sm"])
    return ops


def SU2_generators(s, matter=False):
    """
    This function computes the generators of the group for the SU2 Lattice Gauge Theory:
    [Tz, Tp=T+, Tm=T-, Tx, Ty, T2=Casimir] in any arbitrary spin-s representation

    Args:
        s (scalar, real): spin value, assumed to be integer or semi-integer

        matter (bool, optional):
            if true, it yields the SU2 generators of flavorless SU(2) 1/2 particles
            if false it yields the SU2 generators of Rishon modes in the LGT

    Returns:
        dict: dictionary with the spin matrices
    """
    if not np.isscalar(s):
        raise TypeError(f"s must be SCALAR & (semi)INTEGER, not {type(s)}")
    if not isinstance(matter, bool):
        raise TypeError(f"matter should be a BOOL, not a {type(matter)}")
    largest_s_size = int(2 * s + 1)
    matrices = {"Tz": [0], "Tp": [0], "T2": [0]}
    if not matter:
        for s_size in range(1, largest_s_size):
            spin = s_size / 2
            spin_ops = get_spin_operators(spin)
            for op in ["z", "p", "2"]:
                matrices[f"T{op}"] += [spin_ops[f"S{op}"]]
        SU2_gen = {}
        for op in ["Tz", "Tp", "T2"]:
            SU2_gen[op] = block_diag(tuple(matrices[op]), format="csr")
        SU2_gen["Tm"] = SU2_gen["Tp"].transpose()
        SU2_gen["Tx"] = 0.5 * (SU2_gen["Tp"] + SU2_gen["Tm"])
        SU2_gen["Ty"] = complex(0, -0.5) * (SU2_gen["Tp"] - SU2_gen["Tm"])
        SU2_gen["T4"] = SU2_gen["T2"] ** 2
        # Introduce the effective Casimir operator on which a single rishon is acting
        gen_size = SU2_gen["T2"].shape[0]
        ID = identity(gen_size)
        SU2_gen["T2_root"] = 0.5 * (csr_matrix(ID + 4 * SU2_gen["T2"]).sqrt() - ID)
    else:
        spin_ops = get_spin_operators(1 / 2)
        for op in ["z", "p", "2"]:
            matrices[f"T{op}"] += [spin_ops[f"S{op}"]]
            matrices[f"T{op}"] += [0]
        SU2_gen = {}
        for op in ["z", "p", "2"]:
            SU2_gen[f"S{op}_psi"] = block_diag(tuple(matrices[f"T{op}"]), format="csr")
        SU2_gen["Sm_psi"] = SU2_gen["Sp_psi"].transpose()
        SU2_gen["Sx_psi"] = 0.5 * (SU2_gen["Sp_psi"] + SU2_gen["Sm_psi"])
        SU2_gen["Sy_psi"] = complex(0, -0.5) * (SU2_gen["Sp_psi"] - SU2_gen["Sm_psi"])
    return SU2_gen


def spin_couple(j1, j2, singlet=False, M=None):
    """
    This function computes SU(2) states obtained by combining two spins j1, j2
    and computing Clebsh-Gordan coefficients CG. The possible outcomes j3,m3 are
    the ones with non null CG

    Args:
        j1 (real & scalar): spin of the 1st particle

        j2 (real & scalar): spin of the 2nd particle

        singlet (bool, optional): if true, look only at the combinations of j1 and j2 that provides an SU(2) singlet. Defaults to False.

        M (real & scalar, optional): spin-z component of the 1st particle. Defaults to None.

    Returns:
        list: [(j1, m1,) j2, m2, CG, j3, m3]
    """
    if not np.isscalar(j1):
        raise TypeError(f"j1 must be scalar (int or real), not {type(j1)}")
    if not np.isscalar(j2):
        raise TypeError(f"j2 must be scalar (int or real), not {type(j2)}")
    if not isinstance(singlet, bool):
        raise TypeError(f"singlet must be bool, not {type(singlet)}")
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
        """
        This class collects a configuration of a set of angular momenta Js (and their corresponding Z-momentum) that form an SU(2) singlet state with a certain Clebsh-Gordon coeffiicient.

        The set of momenta is typically referred to SU(2) gauge fields, but it can eventually include a matter state (in first position) describing Flavorless Color 1/2 Dirac fermions with 4 possible states:

        (J,M)=(0,0)

        (J,M)=(1/2,1/2)

        (J,M)=(1/2,-1/2)

        (J,M)=(1/2,0)

        Args:
            J_config (list): list of Total angular momentum J of a set of particles/entities

            M_configs (list): list of possible sets of angular z-momentum (each set has the same length of J_config) of the J_config that allows for a singlet.

            CG_values (list): list of intermediate Clebsh-Gordon coefficients, to be multiplied
                together for the overall CG coefficient of the J configuration.

            pure_theory (bool, optional): If False, the theory also involves Flavorless Color 1/2 Dirac fermions. Defaults to True.

            psi_vacuum (bool, optional): If it used, it specifies with type of 0 singlet the matter state is corresponding to. Defaults to None.

        Raises:
            TypeError: If the input arguments are of incorrect types or formats.

            ValueError: If M_configs and CG_values do not have the same # of entries

            ValueError: If any of M config is NOT made of len(J_config)

            ValueError: If any entry of M_config does not have len(J_config)-1 CGs,
        """
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
        self.J_config = deepcopy(J_config)
        # If not pure_theory, make a difference between vacuum=0 and pair=up&down singlets
        self.pure_theory = pure_theory
        self.psi_vacuum = psi_vacuum
        if not self.pure_theory:
            if self.psi_vacuum is True:
                self.J_config[0] = "V"
            if self.psi_vacuum is False:
                self.J_config[0] = "P"
        # Acquire M configs with corresponding
        self.M_configs = deepcopy(M_configs)
        self.CG_values = []
        self.JM_configs = []
        for ii, CG_set in enumerate(CG_values):
            self.CG_values.append(np.prod(CG_set))
        for ii, m in enumerate(self.M_configs):
            self.JM_configs.append(self.J_config + m)

    def show(self):
        """
        Print the list of singlets (s1 ... sN) with the following shape:
            Js   J1, J2, J3, ... JN

            s1 [ m1, m2, m3, ... mn] CG1

            s2 [ m1, m2, m3, ... mn] CG2

            s3 [ m1, m2, m3, ... mn] CG3
        """
        print("====================================================")
        print(f"J: {self.J_config}")
        for m, CG in zip(self.M_configs, self.CG_values):
            print(f"M:{m} CG:{CG}")
        print("----------------------------------------------------")


def get_SU2_singlets(spin_list, pure_theory=True, psi_vacuum=None):
    """
    This function computes the form of an SU(2) singlet out of a list of
    spin representations

    Args:
        spin_list (list): list of spins to be coupled in order to get a singlet

        pure_theory (bool, optional): if True, only gauge fields

        psi_vacuum (bool, optional): If True, the first element of spin_list is the vacuum of matter.
            If False, the first element of spin_list is the pair (up & down) of matter. Default to None.

    Returns:
        list: list of instances of SU2_singlet; if there is no singlet, just None
    """
    # CHECK ON TYPES
    if not isinstance(spin_list, list):
        raise TypeError(f"spin_list must be a list, not {type(spin_list)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory must be a list, not {type(pure_theory)}")
    if psi_vacuum is not None:
        if not isinstance(psi_vacuum, bool):
            raise TypeError(f"psi_vacuum must be bool, not {type(psi_vacuum)}")
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
        tmp1 = deepcopy(spin_config)
        spin_config = []
        for j in tmp1:
            # Couple the resulting spin of the previous combination with the next spin
            tmp2 = spin_couple(j1=j[-2], j2=spin_list[ii], singlet=singlet, M=j[-1])
            for J in tmp2:
                spin_config.append(j + J)
    if spin_config:
        M_sites = [1] + list(np.arange(3, len(spin_config[0]), 5))
        CG_sites = list(np.arange(4, len(spin_config[0]), 5))
        # The resulting of a spin couling
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


def get_list(spins, pure_theory=True):
    if not isinstance(spins, list):
        raise TypeError(f"spins must be a list, not {type(spins)}")
    if not isinstance(pure_theory, bool):
        raise TypeError(f"pure_theory should be a BOOL, not a {type(pure_theory)}")
    spin_list = []
    # For each single spin particle in the list,
    # consider all the spin irrep up to the max one
    for s in spins:
        tmp = np.arange(S(0), spin_space(s), 1)
        spin_list.append(tmp / 2)
    m_list = []
    j_list = []
    for s in spin_list:
        m_set = []
        j_set = []
        for ss in s:
            m_set += list(m_values(ss))
            j_set += [ss for i in m_values(ss)]
        m_list.append(m_set)
        j_list.append(j_set)
    if not pure_theory:
        j_list.insert(0, ["V", S(1) / 2, S(1) / 2, "P"])
        m_list.insert(0, [S(0), S(1) / 2, -S(1) / 2, S(0)])
    return j_list, m_list


def canonical_vector(spin_list, singlet):
    if not isinstance(spin_list, list):
        raise TypeError(f"spin_list must be a list, not {type(spin_list)}")
    if not isinstance(singlet, SU2_singlet):
        raise TypeError(f"singlet is not {SU2_singlet}, but {type(singlet)}")
    # Acquire the list of matter and rishons
    j_list, m_list = get_list(spin_list, singlet.pure_theory)
    len_basis = len(list(product(*m_list)))
    state = np.zeros(len_basis)
    for ii, (j_config, m_config) in enumerate(zip(product(*j_list), product(*m_list))):
        JM_config = list(j_config) + list(m_config)
        if JM_config in singlet.JM_configs:
            index = singlet.JM_configs.index(JM_config)
            state[ii] = singlet.CG_values[index]
    # Check the norm of the state in the canonical basis
    state_norm = np.sum(state**2)
    if np.abs(state_norm - 1) > 1e-10:
        raise ValueError(f"The state is not normalized: norm {state_norm}")
    return state
