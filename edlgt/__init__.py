import logging
from importlib.metadata import PackageNotFoundError, version
from .dtype_config import (
    set_default_dtype_mode,
    get_default_dtype_mode,
    set_hamiltonian_is_complex,
    get_hamiltonian_is_complex,
)

try:
    __version__ = version("edlgt")
except PackageNotFoundError:
    # Source tree usage without installed package metadata.
    __version__ = "0.0.0+local"

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

__all__ = [
    "__version__",
    "set_default_dtype_mode",
    "get_default_dtype_mode",
    "set_hamiltonian_is_complex",
    "get_hamiltonian_is_complex",
]
