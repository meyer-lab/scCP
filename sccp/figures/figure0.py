"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import anndata
from .common import (
    subplotLabel,
    getSetup,
    flattenData
)
from ..parafac2 import parafac2_nd
import numpy as np
from ..imports.scRNA import ThompsonXA_SCGenes
import numpy as np

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), (2, 2))

    # Add subplot labels
    subplotLabel(ax)
    
    aData = aDataCorrected_Pf2(rank=25, annData=True)
    
    print(aData)
    
    
   
    return f

def check_adata(adata):
    """Ensures data is AnnData"""
    if type(adata) is not anndata.AnnData:
        raise TypeError("Input is not a valid AnnData object")

def check_batch(batch, obs, verbose=False):
    """Ensures categorica names for batch (i.e. drugs)"""
    if batch not in obs:
        raise ValueError(f"column {batch} is not in obs")
    elif verbose:
        print(f"Object contains {obs[batch].nunique()} batches.")
            
def split_batches(adata, batch, hvg=None, return_categories=False):
    """Split batches and preserve category information

    :param adata:
    :param batch: name of column in ``adata.obs``. The data type of the column must be of ``Category``.
    :param hvg: list of highly variable genes
    :param return_categories: whether to return the categories object of ``batch``
    """
    split = []
    batch_categories = adata.obs[batch].cat.categories
    if hvg is not None:
        adata = adata[:, hvg]
    for i in batch_categories:
        split.append(adata[adata.obs[batch] == i].copy())
    if return_categories:
        return split, batch_categories
    return split


def merge_adata(*adata_list, **kwargs):
    """Merge adatas from list while remove duplicated ``obs`` and ``var`` columns

    :param adata_list: ``anndata`` objects to be concatenated
    :param kwargs: arguments to be passed to ``anndata.AnnData.concatenate``
    """

    if len(adata_list) == 1:
        return adata_list[0]

    # Make sure that adatas do not contain duplicate columns
    for _adata in adata_list:
        for attr in ("obs", "var"):
            df = getattr(_adata, attr)
            dup_mask = df.columns.duplicated()
            if dup_mask.any():
                print(
                    f"Deleting duplicated keys `{list(df.columns[dup_mask].unique())}` from `adata.{attr}`."
                )
                setattr(_adata, attr, df.loc[:, ~dup_mask])

    return anndata.AnnData.concatenate(*adata_list, **kwargs)
    
    
def aDataCorrected_Pf2(rank, annData=True):
    """Imports data as both tensor and AnnData. Saves projections in AnnData for SCIB/Theis Lab"""
    aData, X = ThompsonXA_SCGenes(annData=annData, offset=1.0)
    obsV = aData.obs_vector("Drugs")

    check_adata(aData)
    check_batch("Drugs", aData.obs, verbose=True)
    split, categories = split_batches(aData.copy(), "Drugs", return_categories=True)
    
    _, factors, projs, _ = parafac2_nd(X, rank=rank, random_state=1, verbose=True)
    _, projDF, _ = flattenData(X, factors, projs)
    aData.obsm["Proj"] = projDF.values
    
    return aData