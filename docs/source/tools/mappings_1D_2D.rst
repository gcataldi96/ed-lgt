.. _mappings_1D_2D:

Mappings from 1D to 2D lattices
===============================

This module provides functions for mappings between 1D points and 2D coordinates in various space-filling curves: zigzag, snake, and Hilbert curves. They are designed to work with 0-based indexing for points and coordinates. 
For the zigzag and snake curve mappings, the 1D points are counted from 0 to (nx * ny) - 1, and the coordinates (x, y) range from 0 to nx - 1 (ny - 1). 
For the Hilbert curve mappings, the 1D points are counted from 0 to (n**2) - 1, and the coordinates (x, y) range from 0 to n - 1. The functions provide the option to start indexing from 1 by adding/subtracting 1 as needed.

.. automodule:: edlgt.tools.mappings_1D_2D
    :noindex:
    :members:
    :undoc-members:
    :show-inheritance:






