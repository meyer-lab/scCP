"""
Test the data import.
"""
import numpy as np
from ..parafac2 import parafac2
from tensorly.random import random_cp


def test_n_way():
    """Stub test."""
    rcp = random_cp((10, 20, 30), rank=3, full=True)
    rcpStack = np.stack([rcp, rcp, rcp, rcp, rcp, rcp], axis=0)

    _, factors, _ = parafac2(rcp, rank=3, n_iter_max=20, verbose=True, tol=1e-10)
    _, facStack, _ = parafac2(rcpStack, rank=3, n_iter_max=20, verbose=True, tol=1e-10)

    assert np.testing.assert_allclose(factors[0], facStack[1])
