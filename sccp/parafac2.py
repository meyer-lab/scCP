import numpy as np
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
import tensorly as tl
from tensorly.decomposition import parafac
import tlviz


def _compute_projections(X_nd, recon_nD, rank):
    X = np.reshape(X_nd, (-1, X_nd.shape[-2], X_nd.shape[-1]))
    recon = np.reshape(recon_nD, (-1, rank, X_nd.shape[-1]))

    projections = np.empty((X.shape[0], X.shape[1], rank))
    proj_X = np.empty((X.shape[0], rank, X.shape[2]))
    error = 0.0

    for i in range(X.shape[0]):
        U, _, Vh = tl.tenalg.svd.truncated_svd(recon[i] @ X[i].T, rank)
        projections[i] = (U @ Vh).T
        error += np.linalg.norm(X[i] - projections[i] @ recon[i]) ** 2
        proj_X[i] = projections[i].T @ X[i]

    # Convert projections and projected tensor to nD
    projections_nD = np.reshape(projections, (*X_nd.shape[0:-1], rank))
    projected_X_nD = np.reshape(
        proj_X, (*X_nd.shape[0:-2], rank, X_nd.shape[-1])
    )

    return projections_nD, projected_X_nD, error


def parafac2_nd(
    X_nd,
    rank,
    n_iter_max=200,
    tol=1e-5,
    verbose=False,
):
    r"""The same interface as regular PARAFAC2."""
    # *** THIS IMPLEMENTATION REQUIRES A SINGLE ZERO-PADDED TENSOR. ***

    # Initialization
    unfolded_mode_2 = tl.unfold(X_nd, X_nd.ndim - 1)
    assert rank < np.shape(unfolded_mode_2)[0]
    S, _, D = tl.svd_interface(unfolded_mode_2, n_eigenvecs=rank, method="randomized_svd")
    projections_nD = np.reshape(D, (*X_nd.shape[0:-1], rank))
    projected_tensor_nD = np.einsum("...jk,...jl->...kl", projections_nD, X_nd)

    factors_nD = [np.ones((X_nd.shape[i], rank)) for i in range(X_nd.ndim - 2)]
    factors_nD += [np.eye(rank), S]
    # End initialization

    errs = []
    norm_tensor = np.linalg.norm(X_nd) ** 2

    for iter in range(n_iter_max):
        _, factors_nD = parafac(
            projected_tensor_nD,
            rank,
            init=(None, factors_nD),
            n_iter_max=4,
            svd="no svd", # should never be used anyway
            tol=None,
        )

        reconstruction_nD = tl.cp_to_tensor((None, factors_nD))

        projections_nD, projected_tensor_nD, rec_error = _compute_projections(X_nd, reconstruction_nD, rank)
        errs.append(rec_error / norm_tensor)

        if iter > 0:
            if verbose:
                print(
                    f"iteration {iter + 1}: error={errs[-1]}, Δ={errs[-2] - errs[-1]}."
                )

            if (errs[-2] - errs[-1]) < (tol * errs[-2]) or errs[-1] < 1e-12:
                if verbose:
                    print(f"converged in {iter + 1} iterations.")
                break
    
    weights, factors_nD = cp_normalize((None, factors_nD))
    weights, factors_nD = cp_flip_sign((weights, factors_nD), mode=1)

    coreC = tlviz.model_evaluation.core_consistency((weights, factors_nD), projected_tensor_nD, normalised=True)
    print(f"Core consistency = {coreC}.")

    r2x = 1 - errs[-1]
    print(f"R2X = {r2x}.")

    return weights, factors_nD, projections_nD, r2x, coreC
