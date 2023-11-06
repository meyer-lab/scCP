from typing import Sequence
import numpy as np
from tqdm import tqdm
import anndata
import tensorly as tl
import cupy as cp
from pacmap import PaCMAP
from .imports import import_citeseq, import_lupus, import_thomson
from tensorly.parafac2_tensor import parafac2_to_slice
from tensorly.tenalg.svd import randomized_svd
from tensorly.decomposition._parafac2 import parafac2, _parafac2_reconstruction_error
from tensorly.preprocessing import svd_decompress_parafac2_tensor
from scipy.optimize import linear_sum_assignment


def cwSNR(
    matrices: Sequence,
    weights: np.ndarray,
    factors: list[np.ndarray],
    projections: list[np.ndarray],
):
    """Calculate the columnwise signal-to-noise ratio for each dataset."""
    SNR = np.empty((len(matrices), len(weights)), dtype=float)

    for i, mat in enumerate(matrices):
        xx = parafac2_to_slice((weights, factors, projections), i, validate=False)

        SNR[i, :] = (factors[0][i, :] * weights) ** 2.0
        SNR[i, :] /= np.linalg.norm(mat - xx) ** 2.0

    return SNR


def pf2(
    X: anndata.AnnData,
    condition_name: str,
    rank: int,
    random_state=1,
    doEmbedding: bool = True,
):
    # TensorFy
    # Get the indices for subsetting the data
    sgUnique, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)

    # We are going to center as we make the matrices
    means = np.mean(X.X, axis=0)

    X_pf = [X[sgIndex == sgi, :].X.toarray() - means for sgi in range(len(sgUnique))]

    # Quantify the variation in cross-products since this is an assumption of Pf2
    covs = np.stack([xx.T @ xx for xx in X_pf])
    cov_total = tl.norm(covs) ** 2
    cov_var = tl.norm(covs - np.mean(covs, axis=0)) ** 2

    weight, factors, projs, _ = parafac2_nd(
        X_pf,
        rank=rank,
        random_state=random_state,
    )

    X.uns["Pf2_weights"] = weight
    X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"] = factors
    X.uns["cov_ratio"] = cov_var / cov_total
    X.uns["cvSNR"] = cwSNR(X_pf, weight, factors, projs)

    X.obsm["projections"] = np.zeros((X.shape[0], rank))
    for i, p in enumerate(projs):
        X.obsm["projections"][sgIndex == i, :] = p  # type: ignore

    X.obsm["weighted_projections"] = X.obsm["projections"] @ X.uns["Pf2_B"]

    if doEmbedding:
        pcm = PaCMAP(random_state=random_state)
        X.obsm["embedding"] = pcm.fit_transform(X.obsm["projections"])  # type: ignore

    return X


def runAndSavePf2():
    """Runs the analysis and saves the cache files."""
    X = import_citeseq()
    X = pf2(X, "Condition", 80)
    X.write("CITEseq_analyzed_80comps.h5ad")

    X = import_lupus()
    X = pf2(X, "Condition", 40)
    X.write("Lupus_analyzed_40comps.h5ad")

    X = import_thomson()
    X = pf2(X, "Condition", 30)
    X.write("Thomson_analyzed_30comps.h5ad")


def svd_compress_tensor_slices(tensor_slices, rng, maxrank=500):
    r"""Compress data with the SVD for running PARAFAC2."""
    loading_matrices = [None for _ in tensor_slices]
    score_matrices = [None for _ in tensor_slices]

    for i, tensor_slice in tqdm(enumerate(tensor_slices), total=len(tensor_slices)):
        n_rows, n_cols = np.shape(tensor_slice)
        if n_rows <= n_cols:
            score_matrices[i] = cp.array(tensor_slice)
            continue

        U, s, Vh = randomized_svd(
            cp.array(tensor_slice),
            n_eigenvecs=min(n_cols, maxrank),
            random_state=rng,
        )

        # Array broadcasting happens at the last dimension, since Vh is num_svds x n_cols
        # we need to transpose it, multiply in the singular values and then transpose
        # it again. This is equivalen to writing diag(s) @ Vh. If we skip the
        # transposes, we would get Vh @ diag(s), which is wrong.
        score_matrices[i] = (s * Vh.T).T
        loading_matrices[i] = U.get()

    return score_matrices, loading_matrices


def parafac2_nd(
    X_in: Sequence,
    rank: int,
    n_iter_max: int = 300,
    tol: float = 1e-7,
    verbose=False,
    random_state=None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], float]:
    r"""The same interface as regular PARAFAC2."""
    rng = np.random.RandomState(random_state)

    # Checks size of each experiment is bigger than rank
    for i in range(len(X_in)):
        assert np.shape(X_in[i])[0] > rank

    # Checks size of signal measured is bigger than rank
    assert np.shape(X_in[0])[1] > rank

    print("Compressing tensor slices")
    tl.set_backend("cupy")
    score_matrices, loading_matrices = svd_compress_tensor_slices(X_in, rng)

    # Assemble covariance matrix rather than concatenation
    # This saves memory and should be faster
    covM = score_matrices[0].T @ score_matrices[0]
    for i in range(1, len(score_matrices)):
        covM += score_matrices[i].T @ score_matrices[i]

    # Since we've reduced to cov matrix, calculate overall norm
    norm_tensor = cp.asnumpy(cp.trace(covM))

    _, C = cp.linalg.eigh(cp.asarray(covM))

    CPinit = (
        None,
        [
            cp.ones((len(X_in), rank)),
            cp.eye(rank),
            C[:, -rank:],
        ],
    )

    weights, factors, projections = parafac2(
        score_matrices,
        rank,
        n_iter_max=n_iter_max,
        init=CPinit,  # type: ignore
        svd="truncated_svd",
        normalize_factors=True,
        tol=tol,
        nn_modes=(0,),
        random_state=rng,
        verbose=verbose,
        linesearch=True,
    )
    tl.set_backend("numpy")

    weights = weights.get()
    factors = [f.get() for f in factors]
    projections = [p.get() for p in projections]

    weights, factors, projections = svd_decompress_parafac2_tensor(
        (weights, factors, projections), loading_matrices
    )

    gini_idx = giniIndex(factors[0])

    factors = [f[:, gini_idx] for f in factors]
    weights = weights[gini_idx]

    # Maximize the diagonal of B
    _, col_ind = linear_sum_assignment(np.abs(factors[1].T), maximize=True)
    factors[1] = factors[1][col_ind, :]
    projections = [p[:, col_ind] for p in projections]

    # Flip the sign based on B
    signn = np.sign(np.diag(factors[1]))
    factors[1] *= signn[:, np.newaxis]
    projections = [p * signn for p in projections]

    # Calculate R2X
    err = _parafac2_reconstruction_error(X_in, (weights, factors, projections)) ** 2
    R2X = 1.0 - err / norm_tensor

    return weights, factors, projections, R2X


def giniIndex(X: np.ndarray) -> np.ndarray:
    """Calculates the Gini Coeff for each component and returns the index rearrangment"""
    X = np.abs(X)
    gini = np.var(X, axis=0) / np.mean(X, axis=0)

    return np.argsort(gini)
