"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import anndata
from .common import (
    subplotLabel,
    getSetup,
    flattenData
)
# from ..imports.scRNA import import_pancreas, import_pancreas_all
from ..parafac2 import parafac2_nd
# import umap
# import scib
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from ..imports.scRNA import ThompsonXA_SCGenes
from sklearn import preprocessing

warnings.filterwarnings("ignore")

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), (2, 4))

    # Add subplot labels
    subplotLabel(ax)
    
    X, data = ThompsonXA_SCGenes(offset=1.0)
    obsV = X.obs_vector("Drugs")
    
    sgUnique, sgIndex = np.unique(obsV, return_inverse=True)
    
    # ]
    # drugs = preprocessing.label_binarize(obsV, classes=sgUnique).flatten()
    # print(drugs)
    # print(len(drugs))
    # drugNames = np.unique(obsV)
    ata = [len(X[sgIndex == sgi, :]) for sgi in range(len(sgUnique))]
    
    toal = []
    for i, rept in enumerate(ata):
        toal = np.append(toal, np.tile(i, rept))
        
        
    X.obs["batch"] = toal
    

    print(np.shape(X.X))
    check_adata(X)
    check_batch("Drugs", X.obs, verbose=True)
    
    split, categories = split_batches(X.copy(), "Drugs", return_categories=True)
    
    rank=3
    
    _, factors, projs, _ = parafac2_nd(data, rank=rank, random_state=1, verbose=True)
    _, projDF, _ = flattenData(data, factors, projs)
    X.obsm["Proj"] = projDF.values
   

    
    
    
    
    
    

            

    # print(X)
    # print(ata)
    # print(toal)
    # print(len(toal))
    # # X.obs.rename(columns={"Drugs": "batch"}, inplace=True)
    # print(X)
    # print(data)

    return f

def check_adata(adata):
    if type(adata) is not anndata.AnnData:
        raise TypeError("Input is not a valid AnnData object")

def check_batch(batch, obs, verbose=False):
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
    