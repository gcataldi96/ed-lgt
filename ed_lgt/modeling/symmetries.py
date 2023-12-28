# %%
import numpy as np
from itertools import product


def sector_indices(loc_dims, op_list, sectors_list):
    basis = []
    for n, config in enumerate(list(product(*[range(dim) for dim in loc_dims]))):
        include_state = True
        for op, sector_value in zip(op_list, sectors_list):
            # Calculate the value of the operator in the configuration config
            tmp = sum(op[c, c] for c in config)
            # Check if the state belongs to the sector
            if tmp != sector_value:
                include_state = False
                break
        if include_state:
            # Save the configuration state in the basis projector
            basis.append(n)
            print(n, config)
    print(len(basis))
    return basis


def build_H_block(H, loc_dims, site_basis, op_list, sectors_list):
    basis = sector_indices(loc_dims, site_basis, op_list, sectors_list)
    H_sector = H[basis, :][:, basis]
    return H_sector


loc_dims = [4, 4, 4, 4]
op1 = np.diag([0, 1, 1, 2])
op2 = np.diag([0, 1, 0, 1])
sectors = [6, 3]
op_list = [op1, op2]
basis = sector_indices(loc_dims, op_list, sectors)

# %%
