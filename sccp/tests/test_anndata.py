"""
Test that we have correctly setup a Theis lab method function. Functions taken from Theis Lab
"""
import anndata
import numpy as np
from ..parafac2 import parafac2_nd
from ..imports.anndata import check_adata, check_batch, check_Pf2
from ..imports.scRNA import tensorFy
from ..figures.common import flattenData


def test_anndata():
    """Check formatting is correct for aData before and after Pf2"""
    aData = anndata.read_h5ad("/opt/andrew/anndata_test.h5ad")
    Pf2aData(aData,rank=2, batchName="batch")


def Pf2aData(aData, rank, batchName):
    """Run Pf2 for aData"""
    check_adata(aData)
    check_batch(batchName, aData.obs, verbose=True)
    
    X = tensorFy(aData, batchName)
    rank = 2
    _, factors, projs, _ = parafac2_nd(X, rank=rank, random_state=1, verbose=True)
    _, projDF, _ = flattenData(X, factors, projs)
    aData.obsm["Pf2"] = projDF.values
    
    check_Pf2(aData, rank)
