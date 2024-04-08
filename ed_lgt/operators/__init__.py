from . import (
    # SU2_operators_old,
    SU2_operators,
    SU2_singlets,
    SU2_rishons,
    bose_fermi_operators,
    spin_operators,
    QED_operators,
    Zn_operators,
    Z2_FermiHubbard_operators,
)

# from .SU2_operators_old import *
from .SU2_operators import *
from .SU2_singlets import *
from .SU2_rishons import *
from .QED_operators import *
from .Zn_operators import *
from .bose_fermi_operators import *
from .spin_operators import *
from .Z2_FermiHubbard_operators import *

# All modules have an __all__ defined
__all__ = SU2_operators.__all__.copy()
__all__ += SU2_singlets.__all__.copy()
__all__ += SU2_rishons.__all__.copy()
__all__ += QED_operators.__all__.copy()
__all__ += bose_fermi_operators.__all__.copy()
__all__ += spin_operators.__all__.copy()
# __all__ += SU2_operators_old.__all__.copy()
__all__ += Zn_operators.__all__.copy()
__all__ += Z2_FermiHubbard_operators.__all__.copy()
