.. _numba_functions:

========================================
Numba Functions
========================================

Low-level Numba-accelerated utilities for array indexing, row comparisons, and
column filtering used by higher-level basis and symmetry routines.

The API section below documents the functions exported in
``edlgt.tools.numba_functions.__all__``.

.. automodule:: edlgt.tools.numba_functions
   :synopsis: Numba-accelerated array helpers for indexing and filtering.
   :members: rowcol_to_index, get_nonzero_indices, precompute_nonzero_indices, arrays_equal, exclude_columns, filter_compatible_rows
   :member-order: bysource

