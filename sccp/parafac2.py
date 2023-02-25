import cupy as cp
from tqdm import tqdm
import tensorly as tl
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from tensorly.tenalg import khatri_rao
from tensorly.decomposition import parafac
from tensorly.decomposition._parafac2 import (
    _project_tensor_slices,
    _compute_projections,
    _parafac2_reconstruction_error
)
from tlviz.model_evaluation import core_consistency


def parafac2_nd(
    X_nd,
    rank: int,
    n_iter_max: int=50,
    init="svd",
    tol=1e-7,
    verbose=False,
    n_iter_parafac=30,
):
    r"""The same interface as regular PARAFAC2."""
    # *** THIS IMPLEMENTATION REQUIRES A SINGLE ZERO-PADDED TENSOR. ***
    tl.set_backend("cupy")
    X_nd = cp.array(X_nd)
    X = cp.reshape(X_nd, (-1, X_nd.shape[-2], X_nd.shape[-1]))

    # Initialization
    unfolded = tl.unfold(X, 2)
    assert cp.shape(unfolded)[0] > rank
    C = tl.svd_interface(unfolded, n_eigenvecs=rank, method="randomized_svd")[0]
    factors = [cp.ones((X.shape[0], rank)), cp.eye(rank), C]
    projections = _compute_projections(X, factors, "truncated_svd")

    errs = []
    norm_tensor = cp.asnumpy(cp.linalg.norm(X) ** 2)

    err = cp.asnumpy(_parafac2_reconstruction_error(X, (None, factors, projections)) ** 2)
    errs.append(err / norm_tensor)

    tq = tqdm(range(n_iter_max), disable=(not verbose))
    for iter in tq:
        projections = _compute_projections(X, factors, "truncated_svd")
        projected_X = _project_tensor_slices(X, projections)

        # Convert projections and projected tensor to nD
        projected_X = cp.stack(projected_X, axis=0)
        projected_X_nD = cp.reshape(
            projected_X, (*X_nd.shape[0:-2], rank, X_nd.shape[-1])
        )

        CP_nD = parafac(
            projected_X_nD,
            rank,
            n_iter_max=n_iter_parafac,
            init=init if iter == 0 else CP_nD,
            svd="truncated_svd",
            tol=False,
            normalize_factors=False,
        )

        # Convert factors to 3D
        factors = [khatri_rao(CP_nD.factors[:-2]), CP_nD.factors[-2], CP_nD.factors[-1]]

        err = cp.asnumpy(_parafac2_reconstruction_error(X, (None, factors, projections)) ** 2)
        errs.append(err / norm_tensor)

        delta = errs[-2] - errs[-1]
        tq.set_postfix(R2X=1.0 - errs[-1], Δ=delta, refresh=False)

        if delta < tol:
            break

    CP_nD = cp_normalize(CP_nD)
    CP_nD = cp_flip_sign(CP_nD, mode=1)

    coreC = core_consistency(CP_nD, projected_X_nD, normalised=True)
    print(f"Core consistency = {coreC}.")

    projections = cp.stack(projections, axis=0)
    projections_nD = cp.reshape(projections, (*X_nd.shape[0:-1], rank))

    R2X = 1 - errs[-1]
    tl.set_backend("numpy")
    return cp.asnumpy(CP_nD[0]), [cp.asnumpy(f) for f in CP_nD[1]], cp.asnumpy(projections_nD), R2X, coreC
