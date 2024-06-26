# Exact_Diagonalization
Code for Exact Diagonalization of Quantum Many-Body Hamiltonians and Lattice Gauge Theories.

Read the whole Documentation on https://ed-su2.readthedocs.io/en/latest/

# Setup
1) Download from git the repository

        git clone --recursive git@github.com:gcataldi96/ed-lgt.git

2) Create the Environment with all the needed python packages

        conda env create -f ed-lgt/environment.yml
        conda activate ed

3) Install the library

        cd ed-lgt/
        pip install -e .

Enjoy 👏

# Configure Simsio Simulations
Just in case you want to use Simsio to run simulations, do the following steps:

1) (ignore it unless you create the repo for the first time) Add the simsio library as a submodule (it should be already there)

        git submodule add https://github.com/rgbmrc/simsio.git
        git add .
        git commit -m "Add simsio submodule to the TTN code"

2) Update and get all the submodules
        
        git submodule update
        git submodule recursive

This is an example of a config file that should be created inside the folder *configs* (if this latter does not exist, create the directory):

    ===:
    template: |
        n$enum:
        <<<: common
        g: $g
    common:
        dim: 2
        lvals: [2,2]
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

where j0 and j1 are two values of g that one would like to simulate. 

If you want to create a larger set of simulations automatically, run a script like the following:

    from simsio import gen_configs
    import numpy as np

    params = {"g": np.logspace(-1, 1, 10)}
    gen_configs("template", params, "config_NAME_FILE")

Then, in "config_NAME_FILE.yaml" it will add simulations like

        ni:
        <<<: common
        g: j

where 

$i$ is the $i^{th}$ simulation corresponding to the model with the g-parameter (which is not common to all the other simulations) equal to $j$.

# Run Simulations
To run simulations, just type on the command shell the following command. On linux:

    nohup bash -c "printf 'n%s\n' {0..N} | shuf | xargs -PA -i python script.py config_NAME_FILE {} B" &>/dev/null &

On MAC:

    nohup bash -c "printf 'n%s\n' {0..N} | xargs -PA -I% python script.py config_NAME_FILE % B" &>/dev/null &

where 

1) N is the total number of simulations in the *config_file_name*,

2) A is the number of processes in parallel 

3) B is the number of single-node threads per simulation
