import numpy as np

class TwoBodyTerm2D():
    """
    The term defines an interaction between two sites of the 2D lattice.
    **Arguments**

    operators : list of two strings
        String identifier for the operators. Before launching the simulation,
        the python API will check that the operator is defined.

    strength : str, callable, numeric (optional)
        Defines the coupling strength of the local terms. It can
        be parameterized via a value in the dictionary for the
        simulation or a function.
        Default to 1.

    isotropy_xyz : bool, optional
        If False, the defined shift will only be applied as it is. 
        If true, we permute the defined shift to cover all spatial directions.
        Default to True.

    add_complex_conjg : bool, optional

    has_obc : bool or list of bools, optional
        Defines the boundary condition along each spatial dimension.
        If scalar is given, the boundary condition along each
        spatial dimension is assumed to be equal.
        Default to True

    mask : callable or ``None``, optional
        The true-false-mask allows to apply the Hamiltonians terms
        only to specific sites, i.e., with true values. The function
        takes the dictionary with the simulation parameters as an
        argument. The mask is applied to the site where the left
        operator is acting on.
        Default to ``None`` (all sites have the interaction)
    """

    def __init__(self, operator, strength=1, mask=None):
        self.operator = operator
        self.strength = strength
        self.mask = mask
    
    def get_interactions(self,params):
        