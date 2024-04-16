.. Submodule
    sphinx-quickstart on Sat Jul 29 15:01:45 2023.
    You can adapt this file completely to your liking, but it should at least
    contain the root `toctree` directive.

========
Modeling
========

This module contains the main functions to build a Quantum Many-Body (QMB) Hamiltonian with Local, TwoBody and Plaquette Interactions.
In addition, in qmb_state, it provides all the tools to access a QMB state, such as entanglment entropy, (reduced) density matrix, expectation values, single site configurations.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    qmb_state
    qmb_operations
    local_term
    twobody_term
    plaquette_term
    masks