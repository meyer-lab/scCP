"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotUMAP import plotGeneUMAP
import seaborn as sns
import matplotlib.colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from matplotlib.patches import Patch
import anndata



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    rank = 30
    X = openPf2(rank, dataName="Thomson")
    
    cd4 = ["IL7R"]
    cd8 = ["CD8A", "CD8B"]
    nk = ["GNLY", "NKG7"]
    mono1 = ["CD14", "LYZ", "MS4A7"]
    mono2 = ["FCGR3A", "CST3"]
    dc = ["CCR7", "HLA-DQA1", "GPR183"]
    b = ["MS4A1", "CD79A"]
    genes = np.concatenate((cd4, cd8, nk, mono1, mono2, dc, b))

    for i, gene in enumerate(genes):
        plotCmpUMAPDiv(gene, X, ax[i])
 
 

    return f



def plotCmpUMAPDiv(gene, X, ax):
    """Scatterplot of UMAP visualization weighted by
    projections for a component and cell state"""
    geneList = X[:, X.var_names.isin([gene])].X.flatten()
    geneList = geneList + np.min(geneList)
    geneList /= np.max(geneList)
    geneList = np.clip(geneList, None, np.quantile(geneList, 0.99))
    cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
    umap1 = X.obsm["embedding"][:, 0]
    umap2 = X.obsm["embedding"][:, 1]
    cellSkip = 20 
    umap1 = umap1[::cellSkip]
    umap2 = umap2[::cellSkip]
    geneList = geneList[::cellSkip]

    ax.scatter(
            umap1,
            umap2,
            c=geneList,
            cmap=cmap,
            s=0.2,
        )
    psm = plt.pcolormesh([[0, 1], [0, 1]], cmap=cmap)
    plt.colorbar(psm, ax=ax)
    # colorbar = plt.colorbar(psm, ax=plot)
    ax.set(
        title=f"{gene}-Based Decomposition",
        ylabel="UMAP2",
        xlabel="UMAP1",
    )
    
    # umap1 = pf2Points[::cellSkip, 0]
    # umap2 = pf2Points[::cellSkip, 1]
    # weightedProjs = allP[:, cellState-1] * factors[1][cellState-1, cmp-1]
    # weightedProjs = weightedProjs[::cellSkip]
    # weightedProjs = weightedProjs / np.max(np.abs(weightedProjs))
    # cmap = sns.diverging_palette(240, 10, as_cmap=True)
    # weightedProjs = weightedProjs / np.max(np.abs(weightedProjs))
    # psm = plt.pcolormesh([[-1, 1],[-1, 1]], cmap=cmap)


    # plt.colorbar(psm, ax=ax)




# def plotGeneUMAP(
#     gene: str,
#     decompType: str,
#     X: anndata.AnnData,
#     ax: Axes
# ):
#     """Scatterplot of UMAP visualization weighted by gene"""
 