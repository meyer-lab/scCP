import numpy as np
from tensorly.decomposition._cp import initialize_cp


def unfolding_dot_khatri_rao(tensor, factors, mode):
    """mode-n unfolding times khatri-rao product of factors"""
    tensor_idx = "".join(chr(ord("a") + i) for i in range(tensor.ndim))
    rank = chr(ord("a") + tensor.ndim + 1)
    op = tensor_idx
    for i in range(tensor.ndim):
        if i != mode:
            op += "," + "".join([tensor_idx[i], rank])
        else:
            result = "".join([tensor_idx[i], rank])
    op += "->" + result
    factors = [f for (i, f) in enumerate(factors) if i != mode]
    return np.einsum(op, tensor, *factors)


def parafac(
    tensor,
    rank,
    n_iter_max=100,
    init="svd",
    svd="truncated_svd",
):
    """A simple CP ALS."""
    _, factors = initialize_cp(tensor, rank, init=init, svd=svd)

    for _ in range(n_iter_max):
        for mode in range(tensor.ndim):
            other_factors = [f for (i, f) in enumerate(factors) if i != mode]

            pinv = np.ones((rank, rank))
            for f in other_factors:
                pinv *= np.dot(f.T, f)

            mttkrp = unfolding_dot_khatri_rao(tensor, factors, mode)

            factors[mode] = np.linalg.solve(pinv.T, mttkrp.T).T

    return factors
