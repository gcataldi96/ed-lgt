import numpy as np


class PlaquetteTerm:
    """
    The plaquette term is applied to 2x2 nearest-neighbor sites
    in a 2d model.

    **Arguments**
    operators : list of two strings
        String identifier for the operators. Before launching the simulation,
        the python API will check that the operator is defined.

    strength : str, callable, numeric (optional)
        Defines the coupling strength of the local terms. It can
        be parameterized via a value in the dictionary for the
        simulation or a function.
        Default to 1.

    has_obc : bool or list of bools, optional
        Defines the boundary condition along each spatial dimension.
        If scalar is given, the boundary condition along each
        spatial dimension is assumed to be equal.
        Default to True
        If [False, True], the topology is a strip on the x axis
        If [True, False], the topology is a strip on the y axis
        If [False,False], the topology is a thorus


    The order of the operators is for the shifts (0,0), (0,1), (1,0), (1,1)
    """

    def __init__(self, operators, strength=1, has_obc=True):
        self.operators = operators
        self.strength = strength

        if isinstance(has_obc, bool):
            self.has_obc = [has_obc] * 2
        else:
            self.has_obc = has_obc

    def get_interactions(self, ll, params, **kwargs):
        """
        Description of interactions close to the TPO formulation.
        It works for both Periodic and Open Boundary conditions,
        depending on the argument has_obc.
        NOTE the order of the operators is
                (x2,y2) ---- (x4,y4)
                   |            |
                   |            |
                (x1,y1) ---- (x3,y3)
        """

        elem = {"operators": self.operators}

        for x1 in range(ll[0]):
            for y1 in range(ll[1]):
                if x1 < ll[0] - 1:
                    x2 = x1
                    x3 = x1 + 1
                    x4 = x1 + 1
                else:
                    if not self.has_obc[0]:
                        # Periodic boundary conditions on x.
                        # The right part of the plaquette is on the left side of the lattice,
                        # i.e. x3=x4=0
                        x2 = x1
                        x3 = 0
                        x4 = 0
                    else:
                        continue
                if y1 < ll[1] - 1:
                    y2 = y1 + 1
                    y3 = y1
                    y4 = y1 + 1
                else:
                    if not self.has_obc[1]:
                        # Periodic boundary conditions on y.
                        # The upper part of the plaquette is on the lower side of the lattice,
                        # i.e. y2=y4=0
                        y2 = 0
                        y3 = y1
                        y4 = 0
                    else:
                        continue

                coords_1d = [
                    map_to_1d[elem] for elem in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                ]

                yield elem, coords_1d
