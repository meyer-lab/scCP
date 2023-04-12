"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs: investigating UMAP
"""
import numpy as np
import pandas as pd 
import seaborn as sns
from .common import (
    subplotLabel,
    getSetup,
    flattenData,
    plotDrugDimReduc,
    plotGeneDimReduc,
)
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
import umap
from sklearn.decomposition import PCA
from matplotlib import gridspec, pyplot as plt


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 10), (4, 4))

    # Add subplot labels
    subplotLabel(ax)
    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes(offset=1.0)
    rank = 30
    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
    )
    dataDF, projDF = flattenData(data, factors, projs)
    
    # UMAP dimension reduction
    cmpNames = [f"Cmp. {i}" for i in np.arange(1, factors[0].shape[1] + 1)]
    umapReduc = umap.UMAP()
    pf2Points = umapReduc.fit_transform(projDF[cmpNames].to_numpy())
    # print(projDF[cmpNames])
    
    cmpName = ["Cmp. 3"]
    plotCmpDimReduc(projDF, cmpName, "Pf2", pf2Points, f, ax[0:1])
    
    # PCA dimension reduction
    # pc = PCA(n_components=rank)
    # pcaPoints = pc.fit_transform(dataDF[data.variable_labels].to_numpy())
    # pcaPoints = umapReduc.fit_transform(pcaPoints)
     
    # # NK, CD4, B, CD8
    # genes = ["NKG7", "IL7R", "MS4A1", "CD8A"]
    # plotGeneDimReduc(genes, "Pf2", pf2Points, dataDF, f, ax[0:4])
    # plotGeneDimReduc(genes, "PCA", pcaPoints, dataDF, f, ax[4:8])

    # # Find cells associated with drugs
    # drugs = [
    #     "Triamcinolone Acetonide",
    #     "Alprostadil",
    #     "Budesonide",
    #     "Betamethasone Valerate",
    # ]
    
    # plotDrugDimReduc(drugs, "Pf2", dataDF["Drug"].values, pf2Points, ax[8:12])
    # plotDrugDimReduc(drugs, "PCA", dataDF["Drug"].values, pcaPoints, ax[12:16])

    return f


def plotCmpDimReduc(projDF, projName, decomp, points, f, axs):
    """Scatterplot of UMAP visualization weighted by gene"""
    umap1 = points[::20, 0]
    umap2 = points[::20, 1]
    for i, proj in enumerate(projName):
        projs = projDF[proj].values
        cmap=plt.cm.get_cmap('plasma')
        tl = axs[i].scatter(
            umap1, umap2, c=projs[::20], cmap=cmap.reversed(), s=1,
        )
        f.colorbar(tl, ax=axs[i])
        axs[i].set(
            title=proj + "-" + decomp + "-Based Decomposition",
            ylabel="UMAP2",
            xlabel="UMAP1",
            xticks=np.linspace(np.min(umap1), 
                         np.max(umap1),
                         num=5),
            yticks=np.linspace(np.min(umap2), 
                         np.max(umap2),
                         num=5)
            )
        axs[i].axes.xaxis.set_ticklabels([])
        axs[i].axes.yaxis.set_ticklabels([])

    return