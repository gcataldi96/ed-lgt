import numpy as np
from scipy.sparse import isspmatrix

__all__ = [
    "set_default_dtype_mode",
    "get_default_dtype_mode",
    "set_hamiltonian_is_complex",
    "get_hamiltonian_is_complex",
    "get_numeric_dtype",
    "coerce_numeric_scalar",
    "coerce_numeric_array",
    "coerce_matrix_dtype",
]


_HAMILTONIAN_IS_COMPLEX = True
_REAL_CAST_TOL = 1e-12


def set_default_dtype_mode(mode: str) -> None:
    if mode not in ("real", "complex"):
        raise ValueError(f"mode must be 'real' or 'complex', got {mode!r}")
    set_hamiltonian_is_complex(mode == "complex")


def get_default_dtype_mode() -> str:
    return "complex" if _HAMILTONIAN_IS_COMPLEX else "real"


def set_hamiltonian_is_complex(is_complex: bool) -> None:
    global _HAMILTONIAN_IS_COMPLEX
    if not isinstance(is_complex, (bool, np.bool_)):
        raise TypeError(f"is_complex must be bool, got {type(is_complex)}")
    _HAMILTONIAN_IS_COMPLEX = bool(is_complex)


def get_hamiltonian_is_complex() -> bool:
    return _HAMILTONIAN_IS_COMPLEX


def get_numeric_dtype() -> np.dtype:
    return np.dtype(np.complex128 if _HAMILTONIAN_IS_COMPLEX else np.float64)


def _ensure_real_compatible(values, name: str, tol: float) -> None:
    arr = np.asarray(values)
    if not np.iscomplexobj(arr):
        return
    if arr.size == 0:
        return
    max_imag = float(np.max(np.abs(np.imag(arr))))
    if max_imag > tol:
        raise ValueError(
            f"{name} has non-negligible imaginary part (max |Im|={max_imag:.3e}) "
            "while global dtype mode is 'real'. Set complex mode instead."
        )


def coerce_numeric_scalar(value, name: str = "value", tol: float = _REAL_CAST_TOL):
    if not np.isscalar(value):
        raise TypeError(f"{name} must be a scalar, got {type(value)}")
    if _HAMILTONIAN_IS_COMPLEX:
        return np.complex128(value)
    if np.iscomplexobj(value):
        _ensure_real_compatible(value, name, tol)
        return float(np.real(value))
    return float(value)


def coerce_numeric_array(
    values,
    name: str = "array",
    tol: float = _REAL_CAST_TOL,
):
    arr = np.asarray(values)
    if _HAMILTONIAN_IS_COMPLEX:
        return np.asarray(arr, dtype=np.complex128)
    _ensure_real_compatible(arr, name, tol)
    if np.iscomplexobj(arr):
        arr = np.real(arr)
    return np.asarray(arr, dtype=np.float64)


def coerce_matrix_dtype(matrix, name: str = "matrix", tol: float = _REAL_CAST_TOL):
    if isspmatrix(matrix):
        if _HAMILTONIAN_IS_COMPLEX:
            return matrix.astype(np.complex128)
        _ensure_real_compatible(matrix.data, f"{name}.data", tol)
        return matrix.astype(np.float64)
    return coerce_numeric_array(matrix, name=name, tol=tol)
