import numpy as np

__all__=["entanglement_entropy"]

def entanglement_entropy(psi, loc_dim, partition_size):
    if not isinstance(psi, np.ndarray):
        raise TypeError(f"psi should be an ndarray, not a {type(psi)}")
    if not np.isscalar(loc_dim) and not isinstance(loc_dim, int):
        raise TypeError(f"loc_dim must be an SCALAR & INTEGER, not a {type(loc_dim)}")
    if not np.isscalar(partition_size) and not isinstance(loc_dim, int):
        raise TypeError(
            f"partition_size must be an SCALAR & INTEGER, not a {type(partition_size)}"
        )
    # COMPUTE THE ENTANGLEMENT ENTROPY OF A SPECIFIC SUBSYSTEM
    tmp = psi.reshape((loc_dim**partition_size, loc_dim**partition_size))
    S, V, D = np.linalg.svd(tmp)
    tmp = np.array(
        [-np.abs(llambda**2) * np.log(np.abs(llambda**2)) for llambda in V]
    )
    return np.sum(tmp)