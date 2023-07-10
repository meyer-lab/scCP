"""
S3: Sean's bold attempt at Pf2 on the lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup,
    flattenData,
    plotDrugUMAP,
    plotGeneUMAP,
    plotCmpUMAP,
)
from ..parafac2 import parafac2_nd, Pf2X
import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
import seaborn as sns
from os.path import dirname
from scipy.stats import linregress
import anndata
import scipy
from .common import subplotLabel, getSetup, plotFactors
import warnings



# load data (need to change filepath once dataset gets uploaded to opt/...)
# lupus_data = anndata.read_h5ad("/home/seanp/scCP/GSE174188_CLUES1_adjusted.h5ad")

# get rid of warnings <3 (this lowkey doesn't work)
warnings.filterwarnings("once")


####################################################################################
## LOAD IN DATA -- CAN BE MOVED TO imports/scRNA.py
####################################################################################

# from imports/scRNA.py
def tensorFy(annD: anndata.AnnData, obsName: str) -> Pf2X:
    obsV = annD.obs_vector(obsName)
    sgUnique, sgIndex = np.unique(obsV, return_inverse=True)

    data = [annD[sgIndex == sgi, :].X.toarray() for sgi in range(len(sgUnique))]

    return Pf2X(data, sgUnique, annD.var_names)


def lupus_data(third_axis= "ind_cov") -> Pf2X:
    """Import Thompson lab PBMC dataset."""
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus.h5ad")
    X = X[::10, :]

    assert np.all(np.isfinite(X.X.data)) # yeah this should be true

    # X.X /= np.sum(X.X, axis=0) don't need to normalize because "we" already did <3

    # Center the genes [already done]
    # X.X -= np.mean(X.X, axis=0)

    # Assign cells a count per-experiment so we can reindex
    return tensorFy(X, third_axis)

####################################################################################

rank = 40

lupus_tensor = lupus_data()

_, factors, projs, _ = parafac2_nd(lupus_tensor, 
                                   rank = rank, 
                                   random_state = 1, 
                                   verbose=True)


# dataDF, projDF, _ = flattenData(lupus_tensor, factors, projs)

# UMAP dimension reduction
# pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))

# PCA dimension reduction
# pc = PCA(n_components=rank)
# pcaPoints = pc.fit_transform(lupus_tensor.unfold())
# pcaPoints = umap.UMAP(random_state=1).fit(pcaPoints)

# genes = ["CCR7"]


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), # fig size
                     (2, 2) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    plotFactors(factors, lupus_tensor, ax[0:3], reorder=(0, 2), trim=(2,), saveGenes=False)

    # print("DataDF: \n\n", dataDF)
    # print("ProjDF: \n\n", projDF)
    
    # plotGeneUMAP(genes, "Pf2", pf2Points, dataDF, ax[0:])
    # plotGeneUMAP(genes, "PCA", pcaPoints, dataDF, ax[2:])

    return f
