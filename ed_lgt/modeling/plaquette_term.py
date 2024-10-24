"""
:class:`PlaquetteTerm` computes plaquette terms on a D>=2 lattice model, 
providing methods for their calculation and visualization. 
Plaquette terms are used to compute properties relevant to lattice gauge theories.
"""

import numpy as np
from math import prod
from scipy.sparse import isspmatrix, csr_matrix
from .lattice_geometry import get_plaquette_neighbors
from .lattice_mappings import zig_zag
from .qmb_operations import four_body_op
from .qmb_state import QMB_state
from .qmb_term import QMBTerm
from ed_lgt.tools import validate_parameters
from ed_lgt.symmetries import nbody_term
import logging

logger = logging.getLogger(__name__)

__all__ = ["PlaquetteTerm"]


class PlaquetteTerm(QMBTerm):
    def __init__(self, axes, op_list, op_names_list, print_plaq=True, **kwargs):
        """
        This function introduce all the fundamental information to define a Plaquette Hamiltonian
        Term and possible eventual measures of it.

        Args:
            axes (list of str): list of 2 axes along which the Plaquette term should be applied

            op_list (list of 2 scipy.sparse.matrices): list of the two operators involved in the 2Body Term

            op_names_list (list of 2 str): list of the names of the two operators
        """
        validate_parameters(
            axes=axes,
            op_list=op_list,
            op_names_list=op_names_list,
            print_plaq=print_plaq,
        )
        # Preprocess arguments
        super().__init__(op_list=op_list, op_names_list=op_names_list, **kwargs)
        self.axes = axes
        self.print_plaq = print_plaq

    def get_Hamiltonian(self, strength, add_dagger=False, mask=None):
        """
        The function calculates the Plaquette Hamiltonian by summing up 4body terms for each lattice site,
        potentially with some sites excluded based on the mask.
        The result is scaled by the strength parameter before being returned.
        Eventually, it is possible to sum also the dagger part of the Hamiltonian.

        Args:
            strength (scalar): Coupling of the Hamiltonian term.

            add_dagger (bool, optional): If true, it add the hermitian conjugate of
                the resulting Hamiltonian. Defaults to False.

            mask (np.ndarray, optional): 2D array with bool variables specifying
                (if True) where to apply the local term. Defaults to None.

        Returns:
            scipy.sparse: Plaquette Hamiltonian term ready to be used for exact diagonalization/expectation values.
        """
        # CHECK ON TYPES
        if not np.isscalar(strength):
            raise TypeError(f"strength must be SCALAR not a {type(strength)}")
        validate_parameters(add_dagger=add_dagger)
        # Define the Hamiltonian
        H_plaq = 0
        for ii in range(prod(self.lvals)):
            # Compute the corresponding coords of the BL site of the Plaquette
            coords = zig_zag(self.lvals, ii)
            _, sites_list = get_plaquette_neighbors(
                coords, self.lvals, self.axes, self.has_obc
            )
            if sites_list is None:
                continue
            # CHECK MASK CONDITION ON THE SITE
            if self.get_mask_conditions(coords, mask):
                # ADD THE TERM TO THE HAMILTONIAN
                if self.sector_configs is None:
                    H_plaq += strength * four_body_op(
                        op_list=self.op_list,
                        op_sites_list=sites_list,
                        **self.def_params,
                    )
                else:
                    H_plaq += strength * nbody_term(
                        self.sym_ops, np.array(sites_list), self.sector_configs
                    )
        if not isspmatrix(H_plaq):
            H_plaq = csr_matrix(H_plaq)
        if add_dagger:
            H_plaq += csr_matrix(H_plaq.conj().transpose())
        return H_plaq

    def get_expval(self, psi, get_imag=False, stag_label=None):
        """
        The function calculates the expectation value (and it variance) of the Plaquette Hamiltonian
        and its average over all the lattice sites.

        Args:
            psi (numpy.ndarray): QMB state where the expectation value has to be computed

            get_imag(bool, optional): if true, it results the imaginary part of the expectation value, otherwise, the real part. Default to False.

            stag_label (str, optional): if odd/even, then the expectation value is performed only on that kind of sites. Defaults to None.

        Raises:
            TypeError: If the input arguments are of incorrect types or formats.
        """
        # Check on parameters
        if not isinstance(psi, QMB_state):
            raise TypeError(f"psi must be instance of class:QMB_state not {type(psi)}")
        validate_parameters(stag_label=stag_label, get_imag=get_imag)
        # ADVERTISE OF THE CHOSEN PART OF THE PLAQUETTE YOU WANT TO COMPUTE
        if self.print_plaq:
            logger.info(f"----------------------------------------------------")
            if stag_label is None:
                logger.info(f"PLAQUETTE: {'_'.join(self.op_names_list)}")
            else:
                logger.info(f"PLAQUETTE: {'_'.join(self.op_names_list)}")
            logger.info(f"----------------------------------------------------")
        # MEASURE NUMBER OF PLAQUETTES:
        list_of_plaq_sites = []
        list_of_plaq_strings = []
        for ii in range(prod(self.lvals)):
            # Compute the corresponding coords of the BL site of the Plaquette
            coords = zig_zag(self.lvals, ii)
            coords_list, sites_list = get_plaquette_neighbors(
                coords, self.lvals, self.axes, self.has_obc
            )
            if sites_list is None:
                continue
            else:
                list_of_plaq_sites.append(sites_list)
                list_of_plaq_strings.append([f"{c}" for c in coords_list])
        self.obs = np.zeros(len(list_of_plaq_sites), dtype=float)
        self.var = np.zeros(len(list_of_plaq_sites), dtype=float)
        # IN CASE OF NO SYMMETRY SECTOR
        if self.sector_configs is None:
            for ii, sites_list in enumerate(list_of_plaq_sites):
                self.obs[ii] = psi.expectation_value(
                    four_body_op(
                        op_list=self.op_list,
                        op_sites_list=sites_list,
                        **self.def_params,
                    )
                )
                self.var[ii] = (
                    psi.expectation_value(
                        four_body_op(
                            op_list=self.op_list,
                            op_sites_list=sites_list,
                            **self.def_params,
                        )
                        ** 2,
                    )
                    - self.obs[ii] ** 2
                )
                if self.print_plaq:
                    self.print_Plaquette(list_of_plaq_strings[ii], self.obs[ii])
        # GET THE EXPVAL ON THE SYMMETRY SECTOR
        else:
            for ii, sites_list in enumerate(list_of_plaq_sites):
                self.obs[ii] = psi.expectation_value(
                    nbody_term(self.sym_ops, np.array(sites_list), self.sector_configs)
                )
                self.var[ii] = (
                    psi.expectation_value(
                        nbody_term(
                            self.sym_ops,
                            np.array(sites_list),
                            self.sector_configs,
                        )
                        ** 2
                    )
                    - self.obs[ii] ** 2
                )
                if self.print_plaq:
                    self.print_Plaquette(list_of_plaq_strings[ii], self.obs[ii])
        # GET STATISTICS
        self.avg = np.mean(self.obs)
        self.std = np.sqrt(np.mean(np.abs(self.var)))
        if self.print_plaq:
            logger.info(f"{format(self.avg, '.10f')} +/- {format(self.std, '.10f')}")

    def print_Plaquette(self, sites_list, value):
        if not isinstance(sites_list, list):
            raise TypeError(f"sites_list should be a LIST, not a {type(sites_list)}")
        if len(sites_list) != 4:
            raise ValueError(f"sites_list has 4 elements, not {str(len(sites_list))}")
        if not isinstance(value, float):
            raise TypeError(f"sites_list should be FLOAT, not a {type(value)}")
        if value > 0:
            value = format(value, ".10f")
        else:
            if np.abs(value) < 10 ** (-10):
                value = format(np.abs(value), ".10f")
            else:
                value = format(value, ".9f")
        logger.info(f"{sites_list[2]}------------{sites_list[3]}")
        logger.info(f"  |                  |")
        logger.info(f"  |   {value}   |")
        logger.info(f"  |                  |")
        logger.info(f"{sites_list[0]}------------{sites_list[1]}")
        logger.info("")
