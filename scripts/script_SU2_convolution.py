import os
import sys

# Ensure NUMBA_NUM_THREADS is set properly before importing anything else
B = int(sys.argv[-1])
# Read the B parameter from command-line arguments
os.environ["NUMBA_NUM_THREADS"] = str(B)

import numpy as np
from scipy.sparse import csr_matrix, identity
from itertools import product
from ed_lgt.models import SU2_Model
from scipy.sparse.linalg import norm
from ed_lgt.modeling import LocalTerm, TwoBodyTerm, exp_val_data2
from simsio import run_sim
from time import perf_counter
import logging
from scipy.sparse import csc_matrix, csr_matrix


def get_mask(lvals, sites_list):
    mask = np.zeros(lvals, dtype=bool)
    for site in sites_list:
        mask[site] = True
    return mask


logger = logging.getLogger(__name__)
with run_sim() as sim:
    start_time = perf_counter()
    # -------------------------------------------------------------------------------
    # SU(2) MODEL in (1+1)D with MOMENTUM SYMMETRY
    model = SU2_Model(**sim.par["model"])
    zero_density = False if sim.par["model"]["sectors"][0] != model.n_sites else True
    logger.info(f"zero density {zero_density}")
    m = sim.par["m"] if not model.pure_theory else None
    # Build momentum grid
    kdict = {}
    # Choose if TC symmetry is enabled
    TC_symmetry = sim.par.get("TC_symmetry", False)
    k_unit_cell_size = [1] if TC_symmetry else [2]
    n_momenta = model.n_sites if TC_symmetry else model.n_sites // 2
    k_vals = np.arange(0, n_momenta, 1)
    if TC_symmetry:
        k_phys = 2 * np.pi * k_vals / model.n_sites
    else:
        k_phys = 4 * np.pi * k_vals / model.n_sites
    # -------------------------------------------------------------------------------
    for kidx in k_vals:
        kdict[f"{kidx}"] = {}
        # Set the momentum basis on the model
        model.set_momentum_sector(k_unit_cell_size, [kidx], TC_symmetry)
        model.default_params()
        # Generate HAMILTONIAN
        if model.spin > 0.5:
            model.build_gen_Hamiltonian(sim.par["g"], m)
        else:
            model.build_Hamiltonian(sim.par["g"], m)
        # -------------------------------------------------------------------------------
        # DIAGONALIZE THE HAMILTONIAN and SAVE ENERGY EIGVALS
        n_eigs = 1 if (kidx == 0 and zero_density) else 0
        model.diagonalize_Hamiltonian(n_eigs, model.ham_format)
        # Save the states of the energy band
        idx = 2 if (kidx == 0 and zero_density) else 1
        kdict[f"{kidx}"]["psi"] = model.H.Npsi[idx].psi
        if kidx == 0 and zero_density:
            kdict[f"{kidx}"]["gs"] = model.H.Npsi[0].psi
        for ii in range(model.H.n_eigs):
            model.H.print_energy(ii)
    # -------------------------------------------------------------------------------
    kdict["overlaps"] = np.zeros((len(k_vals), len(k_vals)), dtype=np.complex128)

    def Hjk1k2(psi1, psi2, R0):
        hterms = {}
        val = 0.0 + 0.0j
        if TC_symmetry:
            # ---------------------------------------------------------------------------
            hterms["E2"] = LocalTerm(
                model.ops["E_square"], "E_square", **model.def_params
            )
            row, col, val_list = hterms["E2"].get_Hamiltonian(
                model.coeffs["E"], get_mask(model.lvals, [R0])
            )
            val += exp_val_data2(psi1, psi2, row, col, val_list)
            # ---------------------------------------------------------------------------
            hterms["N"] = LocalTerm(model.ops["N-1"], "N-1", **model.def_params)
            row, col, val_list = hterms["N"].get_Hamiltonian(
                ((-1) ** R0) * model.coeffs["m"], get_mask(model.lvals, [R0])
            )
            val += exp_val_data2(psi1, psi2, row, col, val_list)
            # ---------------------------------------------------------------------------
            op_names_list = ["Qpx_dag", "Qmx"]
            op_list = [model.ops[op] for op in op_names_list]
            hterms["hop"] = TwoBodyTerm("x", op_list, op_names_list, **model.def_params)
            # Add the term j, j+1
            row, col, val_list = hterms["hop"].get_Hamiltonian(
                0.5 * model.coeffs["tx_even"],
                mask=get_mask(model.lvals, [R0]),
            )
            val += exp_val_data2(psi1, psi2, row, col, val_list)
            # Add the term j-1, j
            row, col, val_list = hterms["hop"].get_Hamiltonian(
                0.5 * model.coeffs["tx_even"],
                mask=get_mask(model.lvals, [(R0 - 1) % model.n_sites]),
            )
            val += exp_val_data2(psi1, psi2, row, col, val_list)
            # Add the hermitian conjugate
            op_names_list = ["Qpx", "Qmx_dag"]
            op_list = [model.ops[op] for op in op_names_list]
            hterms["hop"] = TwoBodyTerm("x", op_list, op_names_list, **model.def_params)
            # Add the term j, j+1
            row, col, val_list = hterms["hop"].get_Hamiltonian(
                -0.5 * model.coeffs["tx_even"],
                mask=get_mask(model.lvals, [R0]),
            )
            val += exp_val_data2(psi1, psi2, row, col, val_list)
            # Add the term j-1, j
            row, col, val_list = hterms["hop"].get_Hamiltonian(
                -0.5 * model.coeffs["tx_even"],
                mask=get_mask(model.lvals, [(R0 - 1) % model.n_sites]),
            )
            val += exp_val_data2(psi1, psi2, row, col, val_list)
        else:
            # ---------------------------------------------------------------------------
            hterms["E2"] = LocalTerm(
                model.ops["E_square"], "E_square", **model.def_params
            )
            row, col, val_list = hterms["E2"].get_Hamiltonian(
                model.coeffs["E"], get_mask(model.lvals, [R0, (R0 + 1) % model.n_sites])
            )
            val += exp_val_data2(psi1, psi2, row, col, val_list)
            # ---------------------------------------------------------------------------
            hterms["N"] = LocalTerm(model.ops["N_tot"], "N_tot", **model.def_params)
            row, col, val_list = hterms["N"].get_Hamiltonian(
                ((-1) ** R0) * model.coeffs["m"], get_mask(model.lvals, [R0])
            )
            val += exp_val_data2(psi1, psi2, row, col, val_list)
            row, col, val_list = hterms["N"].get_Hamiltonian(
                ((-1) ** (R0 + 1)) * model.coeffs["m"],
                get_mask(model.lvals, [(R0 + 1) % model.n_sites]),
            )
            val += exp_val_data2(psi1, psi2, row, col, val_list)
            # ---------------------------------------------------------------------------
            op_names_list = ["Qpx_dag", "Qmx"]
            op_list = [model.ops[op] for op in op_names_list]
            hterms["hop"] = TwoBodyTerm("x", op_list, op_names_list, **model.def_params)
            row, col, val_list = hterms["hop"].get_Hamiltonian(
                0.5 * model.coeffs["tx_even"],
                mask=get_mask(
                    model.lvals, [(R0 - 1) % model.n_sites, (R0 + 1) % model.n_sites]
                ),
            )
            val += exp_val_data2(psi1, psi2, row, col, val_list)
            row, col, val_list = hterms["hop"].get_Hamiltonian(
                model.coeffs["tx_even"],
                mask=get_mask(model.lvals, [R0]),
            )
            val += exp_val_data2(psi1, psi2, row, col, val_list)
            # Add the hermitian conjugate
            op_names_list = ["Qpx", "Qmx_dag"]
            op_list = [model.ops[op] for op in op_names_list]
            hterms["hop"] = TwoBodyTerm("x", op_list, op_names_list, **model.def_params)
            row, col, val_list = hterms["hop"].get_Hamiltonian(
                -0.5 * model.coeffs["tx_even"],
                mask=get_mask(
                    model.lvals, [(R0 - 1) % model.n_sites, (R0 + 1) % model.n_sites]
                ),
            )
            val += exp_val_data2(psi1, psi2, row, col, val_list)
            row, col, val_list = hterms["hop"].get_Hamiltonian(
                -model.coeffs["tx_even"],
                mask=get_mask(model.lvals, [R0]),
            )
            val += exp_val_data2(psi1, psi2, row, col, val_list)
        # ---------------------------------------------------------------------------
        return val

    def check_momentum_bases():
        N = model.sector_configs.shape[0]
        B_L = model.momentum_basis["L_col_ptr"].shape[0] - 1
        B_R = int(model.momentum_basis["R_col_idx"].max()) + 1
        P_L = csc_matrix(
            (
                model.momentum_basis["L_data"],
                model.momentum_basis["L_row_idx"],
                model.momentum_basis["L_col_ptr"],
            ),
            shape=(N, B_L),
        )
        P_R = csr_matrix(
            (
                model.momentum_basis["R_data"],
                model.momentum_basis["R_col_idx"],
                model.momentum_basis["R_row_ptr"],
            ),
            shape=(N, B_R),
        )
        G_L = P_L.conj().T @ P_L
        G_R = P_R.conj().T @ P_R
        MIX = P_L.conj().T @ P_R
        normL2 = norm(G_L - identity(B_L))
        normR2 = norm(G_R - identity(B_R))
        normMIX = norm(MIX)
        logger.info(f"norm: (PL^dag @ PL) -1: {normL2}")
        logger.info(f"norm: (PL^dag @ PL) -1: {normR2}")
        logger.info(f"norm: (PL^dag @ PR): {normMIX}")
        if normL2 > 1e-12:
            raise ValueError("PL is not a projector")
        if normR2 > 1e-12:
            raise ValueError("RL is not a projector")
        if k1 != k2 and normMIX > 1e-12:
            raise ValueError("The two basis are not orthogonal")

    if zero_density:
        # Check Translational Hamiltonian
        model.set_momentum_pair([0], [0], k_unit_cell_size, TC_symmetry)
        model.default_params()
        for ii in range(n_momenta):
            eg_single_block = Hjk1k2(kdict["0"]["gs"], kdict["0"]["gs"], ii)
            logger.info(f"E0 single [block size {k_unit_cell_size}]: {eg_single_block}")
            logger.info(f"E0 {eg_single_block*n_momenta}")
    R0 = 0
    # CONVOLUTIONAL expectation values
    for k1, k2 in product(k_vals, k_vals):
        model.set_momentum_pair([k1], [k2], k_unit_cell_size, TC_symmetry)
        model.default_params()
        check_momentum_bases()
        # Get the states
        psi1, psi2 = kdict[f"{k1}"]["psi"], kdict[f"{k2}"]["psi"]
        kdict["overlaps"][k1, k2] = Hjk1k2(psi1, psi2, R0)

    sim.res["kvals"] = k_vals
    sim.res["k1k2matrix"] = kdict["overlaps"]
    if zero_density:
        sim.res["gs_energy"] = eg_single_block
    # -------------------------------------------------------------------------------
    end_time = perf_counter()
    logger.info(f"TIME SIMS {round(end_time-start_time, 5)}")
