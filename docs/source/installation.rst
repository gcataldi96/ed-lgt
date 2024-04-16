Installation and Setup
=====================================

1. Setup: instructions for downloading the repository and installing the dependencies.

2. Configure Simsio Simulations: explanation of how to create and configure configuration files for simulations using the `simsio` library.

3. Run Simulations: instructions on how to run the simulations.


Setup
-----

1. Download the repository from git:

    .. code-block:: bash

        git clone --recursive git@github.com:gcataldi96/ed-lgt.git

2. Create the environment with all the required Python packages:

    .. code-block:: bash

        conda env create -f ed-lgt/environment.yml
        conda activate su2

3. Install the library
   
    .. code-block:: bash

        cd ed-lgt/
        pip install -e .

Exploiting SIMSIO library to manage I/O of simulations 
------------------------------------------------------

1. Add the `simsio` library as a submodule (it should already be there):

    .. code-block:: bash

        git submodule add https://github.com/rgbmrc/simsio.git
        git add .
        git commit -m "Add simsio submodule to the TTN code"

This is an example of a configuration file that should be created inside the `configs` folder (if this folder does not exist, create it):

.. code-block:: yaml

    ===:
    template: |
        n$enum:
        <<<: common
        g: $g
    common:
        dim: 2
        lvals: [2, 2]
        pure: false
        has_obc: false
        DeltaN: 2
        m: 1.0
    n0:
        <<<: common
        g: j0
    n1:
        <<<: common
        g: j1

In this file, different simulations are defined with different values of the `g` parameter. 
For example, two sets of simulations are defined with `g` values equal to `j0` and `j1`.

If you want to create a larger set of simulations automatically, run a script like the following:

.. code-block:: python

    from simsio import gen_configs
    import numpy as np

    params = {"g": np.logspace(-1, 1, 10)}
    gen_configs("template", params, "config_NAME_FILE")

This script will generate a series of simulations with different `g` values as specified in the "config_NAME_FILE.yaml" file.

Run Simulations
---------------

To run simulations, type the following command on the shell:

.. code-block:: bash

    nohup bash -c "printf 'n%s\n' {0..N} | shuf | xargs -PA -i python SU2_model.py config_NAME_FILE.yaml {} B" &>/dev/null &

Where:

1. N is the total number of simulations specified in the configuration file `config_NAME_FILE.yaml`.

2. A is the number of processes in parallel.

3. B is the number of single-node threads per simulation.