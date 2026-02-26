.. _qmb_state:

=========================================
Tools for a Quantum Many Body (QMB) State
=========================================

State-vector utilities for expectation values, reduced density matrices,
entanglement measures, and support-based diagnostics.

Typical workflow:

- build a :class:`~edlgt.modeling.qmb_state.QMB_state` from a state vector,
- normalize or truncate it if needed,
- evaluate observables and reduced density matrices,
- extract support configurations for diagnostics in a symmetry-sector basis.

.. automodule:: edlgt.modeling.qmb_state
   :synopsis: Fundamental tools to access and study a QMB state/wavefunction
   :members:
   :member-order: bysource
