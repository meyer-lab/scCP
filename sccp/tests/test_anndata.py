"""
Test that we have correctly setup a Theis lab method function. Functions taken from Theis Lab
"""
import anndata
import numpy as np
from ..imports.anndata import pf2Theis


def test_anndata():
    """Check formatting is correct for aData before and after Pf2"""
    aData = anndata.read_h5ad("/opt/andrew/anndata_test.h5ad")

    print(aData)

    corrected = pf2Theis(aData, "batch", rank=3)

    print(corrected)

    assert corrected.shape == aData.shape

    # proj = aData.obsm["Pf2"]
    # proj = proj[:, :-1]

    # assert np.shape(proj)[1] == rank
    # assert np.shape(aData.X)[0] == np.shape(proj)[0]
