.. _masks:

=================
Masks
=================

Boolean masks for borders, corners, staggered sublattices, and OBC lattice
regions.

Examples
========

For a 2D lattice ``lvals = (4, 5)``:

- ``border_mask(lvals, "mx")`` selects the first row,
- ``border_mask(lvals, "py")`` selects the last column,
- ``corner_mask(lvals, ["mx", "py"])`` selects the top-right corner,
- ``obc_mask(lvals)`` returns a dictionary with ``"core"``, borders, and corners.

.. automodule:: edlgt.modeling.masks
   :synopsis: Lattice-region masks for selective operator application.
   :members:
   :member-order: bysource
