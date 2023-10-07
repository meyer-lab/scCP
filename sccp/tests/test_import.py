"""
Test the cross validation accuracy.
"""
import numpy as np
from ..imports.scRNA import ThompsonXA_SCGenes, load_lupus_data
from ..imports.citeseq import import_citeseq


def test_Thomson():
    """Test for correctness of cross validation."""
    X = ThompsonXA_SCGenes()

    assert np.all(np.isfinite(X.X))


def test_Lupus():
    """Test for correctness of cross validation."""
    X = load_lupus_data()

    assert np.all(np.isfinite(X.X))


def test_CITEseq():
    """Test for correctness of cross validation."""
    X = import_citeseq()

    assert np.all(np.isfinite(X.X))
