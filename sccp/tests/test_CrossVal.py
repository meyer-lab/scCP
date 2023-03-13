"""
Test the cross validation accuracy.
"""
import numpy as np
from tensorly.random import random_parafac2
from ..crossVal import crossvalidate


def test_crosval():
    """Test for correctness of cross validation."""
    X = random_parafac2([(100, 50)] * 10, rank=1, full=True, random_state=1)
    X = np.stack(X, axis=0)
    
    err = crossvalidate(X, rank=1, trainPerc=0.75)
    
    assert err > 0.8



