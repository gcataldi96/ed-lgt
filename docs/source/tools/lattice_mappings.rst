.. _lattice_mappings:

Mappings Between 1D Indices and Lattice Coordinates
===================================================

Utilities for converting between a 0-based linear site index and lattice
coordinates.

The exported ``zig_zag`` / ``inverse_zig_zag`` functions are generic helpers
for multi-dimensional lattices. In practice, the documented and recommended use
of the generic interface is square / hypercubic geometries (equal linear size
along each axis), which is the convention used in this codebase.

Indexing convention
-------------------

- Linear indices start at ``0``.
- Coordinates start at ``0`` on each axis.
- ``coords(x, y)`` is only a display helper and returns a 1-based string label.

Examples
--------

2D (square lattice), 1D -> 2D -> 1D:

.. code-block:: python

   from edlgt.tools.lattice_mappings import zig_zag, inverse_zig_zag

   lvals = [4, 4]
   d = 6
   xy = zig_zag(lvals, d)              # (2, 1)
   d_back = inverse_zig_zag(lvals, xy) # 6

3D (cubic lattice), 1D -> 3D -> 1D:

.. code-block:: python

   from edlgt.tools.lattice_mappings import zig_zag, inverse_zig_zag

   lvals = [3, 3, 3]
   d = 17
   xyz = zig_zag(lvals, d)               # (2, 2, 1)
   d_back = inverse_zig_zag(lvals, xyz)  # 17

The module also contains legacy 2D-specific helpers (snake and Hilbert
variants), but the API documented below focuses on the exported functions.

.. automodule:: edlgt.tools.lattice_mappings
    :members: zig_zag, inverse_zig_zag, coords
    :member-order: bysource
