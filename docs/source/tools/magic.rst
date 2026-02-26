.. _magic:

========================================
Magic
========================================

Numba-accelerated helpers for encoding local configurations as integer keys and
for building the X-string data used in stabilizer calculations.

This module is mostly a low-level backend. A typical workflow is:

1. Build ``strides`` from ``loc_dims`` with :func:`edlgt.tools.magic.compute_strides`.
2. Encode support configurations (or decode keys when debugging).
3. Generate pairwise X-string keys from the support and deduplicate them.
4. Evaluate the stabilizer Rényi-2 sum on the truncated support.

Public API
----------

.. automodule:: edlgt.tools.magic
   :synopsis: Encoded-configuration utilities and stabilizer support kernels.
   :members: compute_strides, encode_config, encode_all_configs, decode_key_to_config, binary_search_sorted, decode_Xstrings, unique_sorted_int64, all_pairwise_pkeys_support, stabilizer_renyi_sum
   :member-order: bysource

