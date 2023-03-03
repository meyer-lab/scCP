import torch
from tqdm import tqdm
import tensorly as tl
from tensorly.cp_tensor import cp_flip_sign, cp_normalize, cp_to_tensor
from tensorly.tenalg import khatri_rao
from tensorly.decomposition import parafac
from tensorly.decomposition._parafac2 import (
    _parafac2_reconstruction_error
)
from tlviz.model_evaluation import core_consistency


def initialize_cp(tensor, rank):
    factors = []
    for mode in range(tl.ndim(tensor)):
        U, S, _ = tl.truncated_svd(tl.unfold(tensor, mode), rank)

        # Put SVD initialization on the same scaling as the tensor in case normalize_factors=False
        if mode == 0:
            idx = min(rank, tl.shape(S)[0])
            U = tl.index_update(U, tl.index[:, :idx], U[:, :idx] * S[:idx])

        if tensor.shape[mode] < rank:
            # TODO: this is a hack but it seems to do the job for now
            random_part = torch.randn(U.shape[0], rank - tl.shape(tensor)[mode]).cuda()
            U = tl.concatenate([U, random_part], axis=1)

        factors.append(U[:, :rank])

    return (None, factors)


def _compute_projections(X, factors):
    recon = cp_to_tensor((None, factors))
    svd_int = recon @ tl.moveaxis(X, 1, 2)
    U, _, Vh = torch.svd_lowrank(svd_int, factors[0].shape[1], niter=4)
    projs = U @ tl.moveaxis(Vh, 1, 2)
    return tl.moveaxis(projs, 1, 2)


@torch.inference_mode()
def parafac2_nd(
    X_nd,
    rank: int,
    n_iter_max: int=100,
    tol=1e-9,
    verbose=False,
):
    r"""The same interface as regular PARAFAC2."""
    # *** THIS IMPLEMENTATION REQUIRES A SINGLE ZERO-PADDED TENSOR. ***
    tl.set_backend("pytorch")
    X_nd = tl.tensor(X_nd).cuda()
    X = tl.reshape(X_nd, (-1, X_nd.shape[-2], X_nd.shape[-1]))

    # Initialization
    unfolded = tl.unfold(X, 2)
    assert tl.shape(unfolded)[0] > rank
    C = torch.svd_lowrank(unfolded, rank)[0]
    factors = [tl.ones((X.shape[0], rank)).cuda(), tl.eye(rank).cuda(), C]

    errs = []
    norm_tensor = tl.norm(X) ** 2

    tq = tqdm(range(n_iter_max), disable=(not verbose))
    for iter in tq:
        projections = _compute_projections(X, factors)
        projections_nD = tl.reshape(projections, (*X_nd.shape[0:-1], rank))

        # Project tensor slices
        projected_X_nD = tl.einsum("...ji,...jk->...ik", projections_nD, X_nd)

        if iter == 0:
            CP_nD = initialize_cp(projected_X_nD, rank)

        CP_nD = parafac(
            projected_X_nD,
            rank,
            n_iter_max=20,
            init=CP_nD,
            svd="no svd",
            tol=False,
            normalize_factors=False,
        )

        # Convert factors to 3D
        factors = [khatri_rao(CP_nD.factors[:-2]), CP_nD.factors[-2], CP_nD.factors[-1]]

        err = _parafac2_reconstruction_error(X, (None, factors, projections)) ** 2
        errs.append(tl.to_numpy((err / norm_tensor).cpu()))

        if iter > 0:
            delta = errs[-2] - errs[-1]
            tq.set_postfix(R2X=1.0 - errs[-1], Δ=delta, refresh=False)

            if delta < tol:
                break

    CP_nD = cp_normalize(CP_nD)
    CP_nD = cp_flip_sign(CP_nD, mode=X_nd.ndim - 2)

    coreC = core_consistency(CP_nD, projected_X_nD, normalised=True)
    print(f"Core consistency = {coreC}.")

    R2X = 1 - errs[-1]
    tl.set_backend("numpy")

    weights = tl.to_numpy(CP_nD[0].cpu())
    factors_nD = [tl.to_numpy(f.cpu()) for f in CP_nD[1]]
    projections_nD = tl.to_numpy(projections_nD.cpu())
    return weights, factors_nD, projections_nD, R2X, coreC
