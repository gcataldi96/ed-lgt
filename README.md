# Exact_Diagonalization
Code for Exact Diagonalization of Quantum Many-Body Hamiltonians
## Setup
    git clone --recursive git@github.com:gcataldi96/ed-su2.git
    git submodule add https://github.com/rgbmrc/simsio.git
    git add .
    git commit -m "Add simsio submodule to ED code"
    conda env create -f ed-su2/environment.yml

enjoy 👏

## Configure Simsio Simulations
This is an example of a config file that should be created inside the folder *configs*:

    ===:
    template: |
        n$enum:
        <<<: common
        g: $g
    common:
        dim: 2
        lvals: 4
        PBC: true
        DeltaN: 0
        m: 1.0
        conv:
            max_iter: 50
            abs_deviation: 1e-5
            rel_deviation: 1e-5
            n_points_conv_check: 2
            max_bond_dimension: 10
            arnoldi_min_tolerance: 1e-4

To create configurations write a script gen_configs.py like the following:

    from simsio import gen_configs
    import numpy as np

    params = {"g": np.logspace(-1, 1, 10)}
    gen_configs("template", params, "config_NAME_FILE")

If you run it, then, in "config_NAME_FILE.yaml" will add simulations like

        nN:
        <<<: common
        g: A

where N is the number of the Nth simulation with the parameter g (which is not in common) is equal to A
# Run Simulations
If the *n* is the total number of simulations in the *config_file_name*, to launch just type on the command shell the following command:

    nohup bash -c "printf 'n%s\n' {0..n} | shuf | xargs -PA -i python SU2_ED_simsio.py config_file_name {} B" &>/dev/null &
    nohup bash -c "export OMP_NUM_THREADS=4; printf 'n%s\n' {0..1} | shuf | xargs -P1 -i python SU2_ED_simsio.py prova {} 10" &>/dev/null &

where 

A is the number of processes in parallel 

B is the number of single-node threads per simulation
