"""
Lupus: UMAP only progen cell 
"""
from .common import getSetup, openPf2
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.axes import Axes
from pacmap import PaCMAP
from .commonFuncs.plotGeneral import plotGenePerCategStatus



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 15), (5, 6))

    X = openPf2(rank=40, dataName="Lupus")
    cmp = 13
    
    X = X[X.obs["louvain"].isin(["23"])]
        
    # X = X[X.obsm["weighted_projections"][:, cmp-1] > .1, :]
    X.obsm["embedding"] = PaCMAP(random_state=1).fit_transform(X.obsm["projections"])
    print(X)
    
    plotPartialCmpUMAP(X, cmp, ax[0])
    plotPartialLabelUMAP(X, ax[1])

    # plotGeneFactors(i + 1, X, ax[2 * i : 2 * i + 2], geneAmount=5)
    
    
    X.obs["Cell Type"] = X.obs["louvain"]
    plotGenePerCategStatus(X, cmp, ax[2:27], geneAmount=12)
    # plotGenePerCategStatus(X, cmp, 40, "Lupus", ax[2:], geneAmount=12)

    
 


    return f


def plotPartialCmpUMAP(X, cmp: int, ax: Axes):
    """Scatterplot of UMAP visualization weighted by projections for a component"""
    weightedProjs = X.obsm["weighted_projections"]
    weightedProjs = weightedProjs[:, cmp - 1]
    weightedProjs = weightedProjs / np.max(np.abs(weightedProjs))
    weightedProjs[0] = -1
    weightedProjs[1] = 1
    
    cmap = sns.diverging_palette(240, 10, as_cmap=True, s=100)

    ax.scatter(
            X.obsm["embedding"][:, 0],
            X.obsm["embedding"][:, 1],
            c=weightedProjs,
            cmap=cmap,
            s=3,
        )
    psm = plt.pcolormesh([[-1, 1], [-1, 1]], cmap=cmap)
    ax.set(ylabel="UMAP2", xlabel="UMAP1", title="Cmp. " + str(cmp),
        xticks=np.linspace(np.min(X.obsm["embedding"][:, 0]), np.max(X.obsm["embedding"][:, 0]), num=5),
        yticks=np.linspace(np.min(X.obsm["embedding"][:, 1]), np.max(X.obsm["embedding"][:, 1]), num=5))
    plt.colorbar(psm, ax=ax, label="Cell Specific Weight")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])


def plotPartialLabelUMAP(X, ax: Axes):
    sns.scatterplot(x=X.obsm["embedding"][:, 0], y=X.obsm["embedding"][:, 1], hue=X.obs["SLE_status"], s=5, palette="muted", ax=ax)
    ax.set(ylabel="UMAP2", xlabel="UMAP1", 
        xticks=np.linspace(np.min(X.obsm["embedding"][:, 0]), np.max(X.obsm["embedding"][:, 0]), num=5),
        yticks=np.linspace(np.min(X.obsm["embedding"][:, 1]), np.max(X.obsm["embedding"][:, 1]), num=5))
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
