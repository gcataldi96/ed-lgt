"""
:class:`PlaquetteTerm2D` computes plaquette terms on a 2D lattice model, 
providing methods for their calculation and visualization. 
Plaquette terms are used to compute properties relevant to lattice gauge theories.
"""

import numpy as np
from math import prod
from copy import deepcopy
from scipy.sparse import isspmatrix, csr_matrix
from .lattice_mappings import zig_zag, inverse_zig_zag
from .qmb_operations import four_body_op
from .qmb_state import QMB_state
from ed_lgt.tools import validate_parameters

__all__ = ["PlaquetteTerm"]


class PlaquetteTerm:
    def __init__(
        self,
        axes,
        op_list,
        op_names_list,
        lvals,
        has_obc,
        staggered_basis=False,
        site_basis=None,
        print_plaq=True,
    ):
        """
        This function introduce all the fundamental information to define a Plaquette Hamiltonian Term and possible eventual measures of it.

        Args:
            axes (list of str): list of 2 axes along which the Plaquette term should be applied

            op_list (list of 2 scipy.sparse.matrices): list of the two operators involved in the 2Body Term

            op_names_list (list of 2 str): list of the names of the two operators

            lvals (list of ints): Dimensions (# of sites) of a d-dimensional hypercubic lattice

            has_obc (bool): It specifies the type of boundary conditions. If False, the topology is a thorus

            staggered_basis (bool, optional): Whether the lattice has staggered basis. Defaults to False.

            site_basis (dict, optional): Dictionary of Basis Projectors (sparse matrices) for lattice sites
            (corners, borders, lattice core, even/odd sites). Defaults to None.

        Raises:
            TypeError: If the input arguments are of incorrect types or formats.
        """
        validate_parameters(
            axes=axes,
            op_list=op_list,
            op_names_list=op_names_list,
            lvals=lvals,
            has_obc=has_obc,
            print_plaq=print_plaq,
        )
        self.axes = axes
        self.BL = op_list[0]
        self.BR = op_list[1]
        self.TL = op_list[2]
        self.TR = op_list[3]
        self.BL_name = op_names_list[0]
        self.BR_name = op_names_list[1]
        self.TL_name = op_names_list[2]
        self.TR_name = op_names_list[3]
        self.op_list = [self.BL, self.BR, self.TL, self.TR]
        self.lvals = lvals
        self.dimensions = "xyz"[: len(lvals)]
        self.has_obc = has_obc
        self.stag_basis = staggered_basis
        self.site_basis = site_basis
        self.print_plaq = print_plaq
        print(
            f"Plaquette {op_names_list[0]}-{op_names_list[1]}-{op_names_list[2]}-{op_names_list[3]}"
        )

    def get_Hamiltonian(self, strength, add_dagger=False, mask=None):
        """
        The function calculates the Plaquette Hamiltonian by summing up 4body terms for each lattice site,
        potentially with some sites excluded based on the mask.
        The result is scaled by the strength parameter before being returned.
        Eventually, it is possible to sum also the dagger part of the Hamiltonian.

        Args:
            strength (scalar): Coupling of the Hamiltonian term.

            add_dagger (bool, optional): If true, it add the hermitian conjugate of the resulting Hamiltonian. Defaults to False.

            mask (np.ndarray, optional): 2D array with bool variables specifying (if True) where to apply the local term. Defaults to None.

        Raises:
            TypeError: If the input arguments are of incorrect types or formats.

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
            coords1 = zig_zag(self.lvals, ii)
            _, sites_list = self.get_Plaquette_coords(coords1)
            if sites_list is None:
                continue
            # Check Mask conditions:
            if mask is None:
                mask_conditions = True
            elif mask is not None and mask[coords1]:
                mask_conditions = True
            elif mask is not None and not mask[coords1]:
                mask_conditions = False
            # Add the Plaquette to the Hamiltonian
            if mask_conditions:
                H_plaq += strength * four_body_op(
                    op_list=self.op_list,
                    op_sites_list=sites_list,
                    lvals=self.lvals,
                    has_obc=self.has_obc,
                    staggered_basis=self.stag_basis,
                    site_basis=self.site_basis,
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
            print(f"----------------------------------------------------")
            if get_imag:
                chosen_part = "IMAG"
            else:
                chosen_part = "REAL"
            if stag_label is None:
                print(f"PLAQUETTE: {chosen_part}")
            else:
                print(f"PLAQUETTE: {chosen_part} PART {stag_label}")
            print(f"----------------------------------------------------")
        self.avg = 0.0
        self.std = 0.0
        counter = 0
        for ii in range(prod(self.lvals)):
            # Compute the corresponding coords of the BL site of the Plaquette
            coords = zig_zag(self.lvals, ii)
            coords_list, sites_list = self.get_Plaquette_coords(coords)
            if sites_list is None:
                continue
            # COMPUTE THE PLAQUETTE only for the appropriate site
            stag = (-1) ** sum(coords)
            site_conditions = [
                stag_label is None,
                (stag_label == "even" and stag > 0),
                (stag_label == "odd" and stag < 0),
            ]
            if any(site_conditions):
                plaq = psi.expectation_value(
                    four_body_op(
                        op_list=self.op_list,
                        op_sites_list=sites_list,
                        lvals=self.lvals,
                        has_obc=self.has_obc,
                        staggered_basis=self.stag_basis,
                        site_basis=self.site_basis,
                        get_real=False,
                    )
                )
                print(plaq)
                delta_plaq = (
                    psi.expectation_value(
                        four_body_op(
                            op_list=self.op_list,
                            op_sites_list=sites_list,
                            lvals=self.lvals,
                            has_obc=self.has_obc,
                            staggered_basis=self.stag_basis,
                            site_basis=self.site_basis,
                            get_real=False,
                        )
                        ** 2,
                    )
                    - plaq**2
                )
                # PRINT THE PLAQUETTE
                plaq_string = [f"{c}" for c in coords_list]
                if self.print_plaq:
                    self.print_Plaquette(plaq_string, plaq)
                # Update the average and the variance
                counter += 1
                self.avg += plaq
                self.std += delta_plaq
        self.avg = self.avg / counter
        self.std = np.sqrt(np.abs(self.std) / counter)
        if self.print_plaq:
            print(f"{format(self.avg, '.10f')} +/- {format(self.std, '.10f')}")

    def get_Plaquette_coords(self, coords1):
        coords1 = list(coords1)
        i1 = inverse_zig_zag(self.lvals, coords1)
        coords2 = deepcopy(coords1)
        coords3 = deepcopy(coords1)
        coords4 = deepcopy(coords1)
        # Look at the specific indices of the two axis along which plaquettes are applied
        indx1 = self.dimensions.index(self.axes[0])
        indx2 = self.dimensions.index(self.axes[1])
        # Check the possibility of applying the plaquette according to the position of the first site
        if (
            coords1[indx1] < self.lvals[indx1] - 1
            and coords1[indx2] < self.lvals[indx2] - 1
        ):
            # Coordinates of sites where to apply Operators
            coords2[indx1] += 1
            coords3[indx2] += 1
            coords4[indx1] += 1
            coords4[indx2] += 1
            i2 = inverse_zig_zag(self.lvals, coords2)
            i3 = inverse_zig_zag(self.lvals, coords3)
            i4 = inverse_zig_zag(self.lvals, coords4)
            sites_list = [i1, i2, i3, i4]
            coords_list = [
                tuple(coords1),
                tuple(coords2),
                tuple(coords3),
                tuple(coords4),
            ]
        else:
            # PERIODIC BOUNDARY CONDITIONS
            if not self.has_obc:
                # UPPER BORDER
                if (
                    coords1[indx1] < self.lvals[indx1] - 1
                    and coords1[indx2] == self.lvals[indx2] - 1
                ):
                    coords2[indx1] += 1
                    coords3[indx2] = 0
                    coords4[indx1] += 1
                    coords4[indx2] = 0
                # RIGHT BORDER
                elif (
                    coords1[indx1] == self.lvals[indx1] - 1
                    and coords1[indx2] < self.lvals[indx2] - 1
                ):
                    coords2[indx1] = 0
                    coords3[indx2] += 1
                    coords4[indx1] = 0
                    coords4[indx2] += 1
                # UPPER RIGHT CORNER
                else:
                    coords2[indx1] = 0
                    coords3[indx2] = 0
                    coords4[indx1] = 0
                    coords4[indx2] = 0
                # Get the corresponding 1d coordinates of the lattice sites
                i2 = inverse_zig_zag(self.lvals, coords2)
                i3 = inverse_zig_zag(self.lvals, coords3)
                i4 = inverse_zig_zag(self.lvals, coords4)
                sites_list = [i1, i2, i3, i4]
                coords_list = [
                    tuple(coords1),
                    tuple(coords2),
                    tuple(coords3),
                    tuple(coords4),
                ]
            else:
                sites_list = None
                coords_list = None
        return coords_list, sites_list

    def print_Plaquette(self, sites_list, value):
        if not isinstance(sites_list, list):
            raise TypeError(f"sites_list should be a LIST, not a {type(sites_list)}")
        if len(sites_list) != 4:
            raise ValueError(
                f"sites_list should have 4 elements, not {str(len(sites_list))}"
            )
        if not isinstance(value, float):
            raise TypeError(
                f"sites_list should be a FLOAT REAL NUMBER, not a {type(value)}"
            )
        if value > 0:
            value = format(value, ".10f")
        else:
            if np.abs(value) < 10 ** (-10):
                value = format(np.abs(value), ".10f")
            else:
                value = format(value, ".9f")
        print(f"{sites_list[2]}------------{sites_list[3]}")
        print(f"  |                  |")
        print(f"  |   {value}   |")
        print(f"  |                  |")
        print(f"{sites_list[0]}------------{sites_list[1]}")
        print("")
