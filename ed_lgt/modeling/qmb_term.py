import numpy as np
from math import prod
from .lattice_geometry import lattice_base_configs
from ed_lgt.tools import validate_parameters
from ed_lgt.symmetries import get_operators_nbody_term
import logging

logger = logging.getLogger(__name__)

__all__ = ["QMBTerm"]


class QMBTerm:
    def __init__(
        self,
        lvals,
        has_obc,
        operator=None,
        op_name=None,
        op_list=None,
        op_names_list=None,
        staggered_basis=False,
        gauge_basis=None,
        sector_configs=None,
    ):
        # Validate type of parameters
        validate_parameters(
            lvals=lvals,
            has_obc=has_obc,
            staggered_basis=staggered_basis,
            gauge_basis=gauge_basis,
        )
        # Lattice Geometry
        self.lvals = lvals
        self.dimensions = "xyz"[: len(lvals)]
        self.has_obc = has_obc
        # Operators Info
        self.op = operator
        self.op_name = op_name
        self.op_list = op_list
        self.op_names_list = op_names_list
        # Dressed site informations
        self.staggered_basis = staggered_basis
        self.gauge_basis = gauge_basis
        # Symmetry sector
        self.sector_configs = sector_configs
        # Get default parameters
        self.def_params = {
            "lvals": self.lvals,
            "has_obc": self.has_obc,
            "gauge_basis": self.gauge_basis,
            "staggered_basis": self.staggered_basis,
        }
        # Get Symmetry operator
        self.get_symmetry_operator()

    def get_symmetry_operator(self):
        if self.sector_configs is not None:
            # If the basis of lattice sites depends on the site
            if self.gauge_basis is not None:
                # Get Label of each site and corresponding local dimension
                lattice_labels, loc_dims = lattice_base_configs(**self.def_params)
                self.loc_dims = loc_dims.transpose().reshape(prod(self.lvals))
                self.lattice_labels = lattice_labels.transpose().reshape(
                    prod(self.lvals)
                )
            else:
                self.lattice_labels = None
                # Acquire local dimension from operators
                loc_dim = (
                    self.op.shape[0]
                    if self.op is not None
                    else self.op_list[0].shape[0]
                )
                self.loc_dims = np.array(
                    [loc_dim for _ in range(prod(self.lvals))], dtype=np.uint8
                )
            # Construct the symmetry operators
            sym_op_list = [self.op] if self.op is not None else self.op_list
            self.sym_ops = get_operators_nbody_term(
                sym_op_list, self.loc_dims, self.gauge_basis, self.lattice_labels
            )

    def get_staggered_conditions(self, coords, stag_label):
        # Compute the staggered factor
        stag = (-1) ** (sum(coords))
        stag_conditions = [
            stag_label is None,
            ((stag_label == "even") and (stag > 0)),
            ((stag_label == "odd") and (stag < 0)),
        ]
        return any(stag_conditions)

    def get_mask_conditions(self, coords, mask):
        # CHECK MASK CONDITION ON THE SITE
        mask_conditions = (
            True if mask is None or all([mask is not None, mask[coords]]) else False
        )
        return mask_conditions
