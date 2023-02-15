import numpy as np

import tensorly as tl
from tensorly.decomposition._cp import initialize_cp


def kr(matrices):
    start = ord("a")
    common_dim = "z"
    target = "".join(chr(start + i) for i in range(len(matrices)))
    source = ",".join(i + common_dim for i in target)
    operation = source + "->" + target + common_dim
    return np.einsum(operation, *matrices).reshape((-1, matrices[0].shape[1]))


def parafac(
    tensor,
    rank,
    n_iter_max=100,
    init="svd",
    svd="truncated_svd",
):
    """A simple CP ALS."""
    _, factors = initialize_cp(
        tensor,
        rank,
        init=init,
        svd=svd,
    )
    unfolds = [tl.unfold(tensor, mode) for mode in range(tensor.ndim)]

    for _ in range(n_iter_max):
        for mode in range(tensor.ndim):
            other_factors = [factors[i] for i in range(len(factors)) if i != mode]

            pinv = np.ones((rank, rank))
            for f in other_factors:
                pinv *= np.dot(f.T, f)

            mttkrp = unfolds[mode] @ kr(other_factors)
            factors[mode] = np.linalg.solve(pinv.T, mttkrp.T).T

    return factors
