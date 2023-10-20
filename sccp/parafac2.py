import os
import numpy as np
import anndata
from pacmap import PaCMAP
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from tensorly.decomposition import parafac2 as pf2tensorly
from scipy.optimize import linear_sum_assignment


def parafac2_nd(
    X_in: list,
    rank: int,
    tol: float = 1e-6,
    random_state=None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], float]:
    r"""The same interface as regular PARAFAC2."""
    rng = np.random.RandomState(random_state)

    # Verbose if this is not an automated build
    verbose = "CI" not in os.environ

    # Checks size of each experiment is bigger than rank
    for i in range(len(X_in)):
        assert np.shape(X_in[i])[0] > rank

    # Checks size of signal measured is bigger than rank
    assert np.shape(X_in[0])[1] > rank

    (weights, factors, projections), rec_errors = pf2tensorly(
        X_in,
        rank,
        init="svd",  # type: ignore
        svd="randomized_svd",
        tol=tol,
        random_state=rng,
        verbose=verbose,
        return_errors=True,
        n_iter_parafac=5,
        linesearch=True,
    )

    R2X = 1 - rec_errors[-1]

    gini_idx = giniIndex(factors[0])
    assert gini_idx.size == rank

    CP = cp_normalize(cp_flip_sign((weights, factors), mode=1))

    # Maximize the diagonal of B
    _, col_ind = linear_sum_assignment(np.abs(CP.factors[1].T), maximize=True)
    CP.factors[1] = CP.factors[1][col_ind, :]

    # Flip the sign based on B
    signn = np.sign(np.diag(CP.factors[1]))
    CP.factors[1] *= signn[:, np.newaxis]
    projections = [p * signn for p in projections]

    return CP.weights, CP.factors, projections, R2X


def giniIndex(X: np.ndarray) -> np.ndarray:
    """Calculates the Gini Coeff for each component and returns the index rearrangment"""
    X = np.abs(X)
    gini = np.var(X, axis=0) / np.mean(X, axis=0)

    return np.argsort(gini)


def pf2(
    X: anndata.AnnData,
    condition_name: str,
    rank: int,
    random_state=1,
    doEmbedding: bool = True,
):
    # TensorFy
    # Sort so that the concatenation matches up later
    sort_idx = np.argsort(X.obs_vector(condition_name))
    X = X[sort_idx, :]

    # Get the indices for subsetting the data
    sgUnique, sgIndex = np.unique(X.obs_vector(condition_name), return_inverse=True)

    X_pf = [X[sgIndex == sgi, :].X.toarray() for sgi in range(len(sgUnique))]

    weight, factors, projs, _ = parafac2_nd(
        X_pf,
        rank=rank,
        random_state=random_state,
    )

    X.uns["Pf2_weights"] = weight
    X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"] = factors
    X.obsm["projections"] = np.concatenate(projs, axis=0)
    X.obsm["weighted_projections"] = X.obsm["projections"] @ X.uns["Pf2_B"]

    if doEmbedding:
        X.obsm["embedding"] = PaCMAP(random_state=random_state).fit_transform(
            np.concatenate(projs, axis=0)
        )

    return X
