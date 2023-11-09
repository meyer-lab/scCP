"""
Test the parafac2 method.
"""
import numpy as np
from scipy import sparse
from ..parafac2 import pf2, pf2_r2x, svd_compress_tensor_slice
from ..imports import import_thomson


def test_compress_csr():
    A = sparse.random(5000, 2000, format="csr")
    means = np.ones(2000)

    scores, loadings = svd_compress_tensor_slice(A, means, 100)

    A_centered = A.toarray() - means
    means = np.zeros_like(means)
    scores_c, loadings_c = svd_compress_tensor_slice(A_centered, means, 100)

    np.testing.assert_allclose(scores, scores_c, atol=1e-12)
    np.testing.assert_allclose(loadings, loadings_c, atol=1e-12)


def test_factor_thomson():
    """Import and factor Thomson."""
    X = import_thomson()

    X = pf2(X, 3, doEmbedding=False)

    r2x = pf2_r2x(X, 4)
    print(r2x)
