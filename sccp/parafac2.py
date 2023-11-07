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
from tensorly.decomposition._parafac2 import parafac2
from tensorly.preprocessing import svd_decompress_parafac2_tensor
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from scipy.optimize import linear_sum_assignment
from sklearn.utils.sparsefuncs import mean_variance_axis


def cwSNR(
    X: anndata.AnnData,
    condition_name: str,
    weights: np.ndarray,
    factors: list[np.ndarray],
    projections: list[np.ndarray],
):
    """Calculate the columnwise signal-to-noise ratio for each dataset and overall error."""
    SNR = np.empty((factors[0].shape[0], len(weights)), dtype=float)
    norm_overall = 0.0
    err_norm = 0.0

    for i in range(factors[0].shape[0]):
        xx = parafac2_to_slice((weights, factors, projections), i, validate=False)
        X_cond = getCondition(X, condition_name, i)
        norm_overall += np.linalg.norm(X_cond) ** 2.0
        err_norm_here = np.linalg.norm(X_cond - xx) ** 2.0
        err_norm += err_norm_here

        SNR[i, :] = (factors[0][i, :] * weights) ** 2.0
        SNR[i, :] /= err_norm_here

    return SNR, 1.0 - err_norm / norm_overall


def getCondition(
    X: anndata.AnnData, condition_name: str, matrix_idx: int
) -> np.ndarray:
    """Separate out one condition from within the anndata object. Note that this is
    dependent on the ordering of cells in the dataset. This also centers based on
    the whole-dataset mean."""
    # Get the indices for subsetting the data
    _, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)

    # We are going to center as we make the matrices
    means, _ = mean_variance_axis(X.X, axis=0)  # type: ignore

    X_condition = X[sgIndex == matrix_idx, :]
    X_condition_arr = X_condition.X.toarray()
    assert X_condition_arr.shape == X_condition.shape

    return X_condition_arr - means


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

    X_pf = []
    loadings_pf = []

    tl.set_backend("cupy")

    print("Loading and compressing tensor slices")
    for sgi in tqdm(range(len(sgUnique)), total=len(sgUnique)):
        X_cond = getCondition(X, condition_name, sgi)
        scores, loadings = svd_compress_tensor_slice(X_cond, maxrank=500)
        X_pf.append(scores)
        loadings_pf.append(loadings)

    tl.set_backend("numpy")

    weight, factors, projs = parafac2_nd(
        X_pf,
        rank=rank,
    )

    weight, factors, projs = svd_decompress_parafac2_tensor(
        (weight, factors, projs), loadings_pf
    )

    X.uns["Pf2_weights"] = weight
    X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"] = factors
    X.uns["cvSNR"], X.uns["R2X"] = cwSNR(X, condition_name, weight, factors, projs)
    print(X.uns["R2X"])

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


def svd_compress_tensor_slice(tensor_slice, maxrank=500):
    r"""Compress data with the SVD for running PARAFAC2."""
    n_rows, n_cols = np.shape(tensor_slice)
    n_cols = min(n_cols, maxrank)
    if n_rows <= n_cols:
        return cp.array(tensor_slice), None

    U, s, Vh = randomized_svd(
        cp.array(tensor_slice),
        n_eigenvecs=n_cols,
        random_state=1,
    )

    # Array broadcasting happens at the last dimension, since Vh is num_svds x n_cols
    # we need to transpose it, multiply in the singular values and then transpose
    # it again. This is equivalen to writing diag(s) @ Vh. If we skip the
    # transposes, we would get Vh @ diag(s), which is wrong.
    return (s * Vh.T).T, U.get() # scores, loadings


def parafac2_nd(
    X_in: Sequence,
    rank: int,
    n_iter_max: int = 300,
    tol: float = 1e-7,
    verbose=False,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    r"""The same interface as regular PARAFAC2."""

    tl.set_backend("cupy")

    if isinstance(X_in[0], np.ndarray):
        X_in = [cp.array(x) for x in X_in]

    # Assemble covariance matrix rather than concatenation
    # This saves memory and should be faster
    covM = X_in[0].T @ X_in[0]
    for i in range(1, len(X_in)):
        covM += X_in[i].T @ X_in[i]

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
        X_in,
        rank,
        n_iter_max=n_iter_max,
        init=CPinit,  # type: ignore
        svd="truncated_svd",
        normalize_factors=True,
        tol=tol,
        nn_modes=(0,),
        verbose=verbose,
        linesearch=True,
    )
    tl.set_backend("numpy")

    weights = weights.get()
    factors = [f.get() for f in factors]
    projections = [p.get() for p in projections]

    weights, factors, projections = standardize_pf2(
        weights, factors, projections
    )

    return weights, factors, projections


def standardize_pf2(
    weights: np.ndarray, factors: list[np.ndarray], projections: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    # Order components by condition variance
    gini = np.var(factors[0], axis=0) / np.mean(factors[0], axis=0)
    gini_idx = np.argsort(gini)
    factors = [f[:, gini_idx] for f in factors]
    weights = weights[gini_idx]

    weights, factors = cp_normalize(cp_flip_sign((weights, factors), mode=1))

    # Order eigen-cells to maximize the diagonal of B
    _, col_ind = linear_sum_assignment(np.abs(factors[1].T), maximize=True)
    factors[1] = factors[1][col_ind, :]
    projections = [p[:, col_ind] for p in projections]

    # Flip the sign based on B
    signn = np.sign(np.diag(factors[1]))
    factors[1] *= signn[:, np.newaxis]
    projections = [p * signn for p in projections]

    return weights, factors, projections
