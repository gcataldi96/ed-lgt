.. Submodule
    sphinx-quickstart on Sat Jul 29 15:01:45 2023.
    You can adapt this file completely to your liking, but it should at least
    contain the root `toctree` directive.

========
Modeling
========

This section contains the core building blocks used to assemble quantum
many-body (QMB) Hamiltonians (local, two-body, and plaquette interactions) and
to analyze QMB states (expectation values, reduced density matrices,
entanglement entropy, and basis-state configurations). It also includes base
classes and lattice masks used by the term builders.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    qmb_term
    local_term
    twobody_term
    plaquette_term
    nbody_term
    qmb_operations
    qmb_hamiltonian
    qmb_state
    masks
