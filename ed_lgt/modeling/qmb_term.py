import numpy as np
from ed_lgt.tools import validate_parameters
import logging

logger = logging.getLogger(__name__)

__all__ = ["QMBTerm"]


class QMBTerm:
    def __init__(
        self,
        lvals: list[int],
        has_obc: list[bool],
        operator: np.ndarray = None,
        op_name: str = None,
        op_list: list[np.ndarray] = None,
        op_names_list: list[str] = None,
        sector_configs: np.ndarray = None,
        momentum_basis=None,
        momentum_k=None,
    ):
        # Validate type of parameters
        validate_parameters(lvals=lvals, has_obc=has_obc)
        # Lattice Geometry
        self.lvals = lvals
        self.dimensions = "xyz"[: len(lvals)]
        self.has_obc = has_obc
        # Operators Info
        self.op = operator
        self.op_name = op_name
        self.op_list = op_list
        self.op_names_list = op_names_list
        # Symmetry sector
        self.sector_configs = sector_configs
        # Get default parameters
        self.def_params = {"lvals": self.lvals, "has_obc": self.has_obc}
        # Get Symmetry operator
        self.get_symmetry_operator()
        # Momentum basis
        self.momentum_basis = momentum_basis
        self.momentum_k = momentum_k

    def get_symmetry_operator(self):
        if self.sector_configs is not None:
            # Construct the symmetry operators
            sym_op_list = [self.op] if self.op is not None else self.op_list
            self.sym_ops = np.array(sym_op_list)

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
