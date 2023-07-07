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
import warnings



# load data (need to change filepath once dataset gets uploaded to opt/...)
# lupus_data = anndata.read_h5ad("/home/seanp/scCP/GSE174188_CLUES1_adjusted.h5ad")
lupus_data = anndata.read_h5ad("/opt/andrew/lupus/lupus.h5ad")

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


def lupus_data(offset: float = 1.0, third_axis= "ind_cov") -> anndata.AnnData:
    """Import Thompson lab PBMC dataset."""
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus.h5ad")
    scalingfactor = 1000

    assert np.all(np.isfinite(X.X.data)) # yeah this should be true

    X = X[:, np.mean(X.X > 0, axis=0) > 0.001] # leaving this in for now; only 1978 of 1999 fit this criteron
                                               # but that is most so maybe don't worry... talk to andrew
    # X.X /= np.sum(X.X, axis=0) don't need to normalize because "we" already did <3

    # X.X = np.log10((scalingfactor * X.X) + 1) not worried abt this
    #means = np.mean(X.X, axis=0)
    #cv = np.std(X.X, axis=0) / means

    #logmean = np.log10(means + 1e-10)  can't even really do this bc our means are 0 <3
    #logstd = np.log10(cv + 1e-10)

    if offset != 1.0:
        slope, intercept, _, _, _ = linregress(logmean, logstd)

        above_idx = logstd > logmean * slope + intercept + np.log10(offset)
        X = X[:, above_idx]

    # Center the genes [already done]
    # X.X -= np.mean(X.X, axis=0)

    # Assign cells a count per-experiment so we can reindex
    return tensorFy(X, third_axis)

####################################################################################



lupus_tensor = lupus_data()

_, factors, projs, _ = parafac2_nd(lupus_tensor, 
                                   rank = 10, 
                                   random_state = 1, 
                                   verbose=True)


dataDF, projDF, _ = flattenData(lupus_tensor, factors, projs)

# UMAP dimension reduction
pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))

# PCA dimension reduction
pc = PCA(n_components=rank)
pcaPoints = pc.fit_transform(lupus_tensor.unfold())
pcaPoints = umap.UMAP(random_state=1).fit(pcaPoints)

genes = ["CCR7"]


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), # fig size
                     (1, 2) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    print("DataDF: \n\n", dataDF)
    print("ProjDF: \n\n", projDF)
    
    plotGeneUMAP(genes, "Pf2", pf2Points, dataDF, ax[0])
    plotGeneUMAP(genes, "PCA", pcaPoints, dataDF, ax[1])



    return f
