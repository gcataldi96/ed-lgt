from . import (
    # SU2_operators_old,
    SU2_operators,
    SU2_singlets,
    bose_fermi_operators,
    QED_operators,
    Zn_operators,
)

# from .SU2_operators_old import *
from .SU2_operators import *
from .SU2_singlets import *
from .QED_operators import *
from .Zn_operators import *
from .bose_fermi_operators import *

# All modules have an __all__ defined
__all__ = SU2_operators.__all__.copy()
__all__ += SU2_singlets.__all__.copy()
__all__ += QED_operators.__all__.copy()
__all__ += bose_fermi_operators.__all__.copy()
# __all__ += SU2_operators_old.__all__.copy()
__all__ += Zn_operators.__all__.copy()
